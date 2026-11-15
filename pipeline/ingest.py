"""Ingesta de focos de calor desde NASA FIRMS.

Dos caminos:
1. Archivo histórico (recomendado para 20+ años): descargar los CSV por país
   desde FIRMS y ponerlos en data/raw/. `load_archive()` los lee todos.
2. API de FIRMS con MAP_KEY: `fetch_api()` baja bloques de días y los guarda
   en caché, para no repetir llamadas si el proceso se corta.

Ambos devuelven el mismo esquema normalizado (ver `normalize`).
"""
from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

STD_COLUMNS = ["sensor", "satellite", "lat", "lon", "date", "time", "frp",
               "confidence", "conf_ok", "daynight", "type", "scan", "track"]


def _detect_sensor(df: pd.DataFrame, name: str = "") -> str:
    if "instrument" in df.columns and df["instrument"].notna().any():
        val = str(df["instrument"].dropna().iloc[0]).upper()
        if "MODIS" in val:
            return "MODIS"
        if "VIIRS" in val:
            return "VIIRS"
    if "bright_ti4" in df.columns:
        return "VIIRS"
    if "brightness" in df.columns:
        return "MODIS"
    low = name.lower()
    return "MODIS" if "modis" in low else "VIIRS"


def normalize(df: pd.DataFrame, sensor: str | None = None, name: str = "") -> pd.DataFrame:
    """Lleva un CSV de FIRMS (MODIS o VIIRS, API o archivo) a un esquema común."""
    if df.empty:
        return pd.DataFrame(columns=STD_COLUMNS)
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    sensor = sensor or _detect_sensor(df, name)

    out = pd.DataFrame({
        "sensor": sensor,
        "satellite": df.get("satellite", pd.Series(["?"] * len(df))).astype(str),
        "lat": pd.to_numeric(df["latitude"], errors="coerce"),
        "lon": pd.to_numeric(df["longitude"], errors="coerce"),
        "date": pd.to_datetime(df["acq_date"], errors="coerce"),
        "time": pd.to_numeric(df.get("acq_time", 0), errors="coerce").fillna(0).astype(int),
        "frp": pd.to_numeric(df.get("frp", np.nan), errors="coerce"),
        "daynight": df.get("daynight", pd.Series(["?"] * len(df))).astype(str),
        "type": pd.to_numeric(df.get("type", 0), errors="coerce").fillna(0).astype(int),
        "scan": pd.to_numeric(df.get("scan", np.nan), errors="coerce"),
        "track": pd.to_numeric(df.get("track", np.nan), errors="coerce"),
    })

    conf = df.get("confidence", pd.Series([None] * len(df)))
    if sensor == "MODIS":
        conf_num = pd.to_numeric(conf, errors="coerce")
        out["confidence"] = conf_num.astype("string")
        out["conf_ok"] = conf_num >= config.MODIS_MIN_CONFIDENCE
    else:
        conf_txt = conf.astype(str).str.strip().str.lower()
        out["confidence"] = conf_txt
        out["conf_ok"] = conf_txt.isin(config.VIIRS_KEEP_CONFIDENCE)

    # Para VIIRS solo usamos Suomi-NPP (el que tiene registro desde 2012).
    if sensor == "VIIRS":
        sat = out["satellite"].str.upper()
        is_snpp = sat.isin({"N", "SNPP", "SUOMI NPP", "S-NPP", "NPP", "?"})
        out = out[is_snpp]

    return out.dropna(subset=["lat", "lon", "date"]).reset_index(drop=True)


def load_archive(folder: Path = config.DATA_RAW) -> pd.DataFrame:
    """Lee todos los CSV de FIRMS de una carpeta (descarga de archivo histórico)."""
    files = sorted(Path(folder).glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No hay CSV en {folder}. Descarga el archivo histórico de FIRMS "
            "(MODIS y VIIRS S-NPP para República Dominicana) y ponlo ahí.")
    parts = []
    for f in files:
        raw = pd.read_csv(f, low_memory=False)
        parts.append(normalize(raw, name=f.name))
        print(f"  leído {f.name}: {len(raw):,} filas")
    return pd.concat(parts, ignore_index=True)


def _fetch_block(url: str, max_wait_retries: int = 40, wait_s: float = 45.0) -> str:
    """Descarga un bloque. Si FIRMS devuelve el límite de transacciones agotado
    (HTTP 400 + "Invalid MAP_KEY" pese a que la clave es válida, o 429), espera a
    que la ventana de 10 min se libere y reintenta, en vez de abortar la corrida."""
    for intento in range(max_wait_retries + 1):
        resp = requests.get(url, timeout=60)
        text = resp.text
        rate_limited = resp.status_code == 429 or (
            resp.status_code == 400 and "invalid map_key" in text.lower())
        if rate_limited and intento < max_wait_retries:
            print(f"    cupo de transacciones agotado; espero {wait_s:.0f}s "
                  f"(reintento {intento + 1}/{max_wait_retries})…", flush=True)
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        if text.lower().startswith(("invalid", "error")):
            raise RuntimeError(f"FIRMS respondió: {text[:200]}")
        return text
    raise RuntimeError("FIRMS siguió reportando cupo agotado tras varios reintentos")


def fetch_api(sensor: str, start: date, end: date, map_key: str,
              day_range: int = 5, use_country: bool = True,
              pause_s: float = 0.4) -> pd.DataFrame:
    """Baja focos de la API de FIRMS por bloques de `day_range` días, con caché.

    use_country=True usa el endpoint por país (ya recortado a RD);
    False usa la caja BBOX (incluye parte de Haití; se filtra después).
    Si se agota el cupo de transacciones (5000/10 min), espera y reintenta.
    """
    source = config.SOURCES[sensor]
    cache_dir = config.DATA_CACHE / source
    cache_dir.mkdir(parents=True, exist_ok=True)
    area = config.COUNTRY_ISO3 if use_country else ",".join(map(str, config.BBOX))
    kind = "country" if use_country else "area"

    parts, day = [], start
    while day <= end:
        n = min(day_range, (end - day).days + 1)
        cache = cache_dir / f"{kind}_{day.isoformat()}_{n}.csv"
        if cache.exists():
            text = cache.read_text()
        else:
            url = f"{config.FIRMS_API}/{kind}/csv/{map_key}/{source}/{area}/{n}/{day.isoformat()}"
            text = _fetch_block(url)
            cache.write_text(text)
            time.sleep(pause_s)
        if text.strip():
            parts.append(pd.read_csv(io.StringIO(text)))
        day += timedelta(days=n)

    if not parts:
        return pd.DataFrame(columns=STD_COLUMNS)
    return normalize(pd.concat(parts, ignore_index=True), sensor=sensor)


def check_key(map_key: str) -> dict:
    """Consulta cuántas transacciones quedan en la MAP_KEY."""
    url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={map_key}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()
