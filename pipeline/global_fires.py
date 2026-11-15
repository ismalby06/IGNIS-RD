"""Módulo de detección global de focos de calor en tiempo real (NASA FIRMS).

Descarga y procesa datos satelitales en vivo (NRT - Near Real Time) a nivel mundial
directamente de los satélites MODIS (Terra & Aqua) y VIIRS (Suomi-NPP, NOAA-20, NOAA-21)
a través de los feeds abiertos y la API oficial de NASA EOSDIS / LANCE FIRMS.
"""
from __future__ import annotations

import io
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

# Feeds abiertos NRT de NASA FIRMS (actualizados cada pocos minutos, sin necesidad de clave)
FIRMS_NRT_BASE = "https://firms.modaps.eosdis.nasa.gov/data/active_fire"

URLS_NRT = {
    "MODIS_24h": f"{FIRMS_NRT_BASE}/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv",
    "MODIS_48h": f"{FIRMS_NRT_BASE}/modis-c6.1/csv/MODIS_C6_1_Global_48h.csv",
    "MODIS_7d": f"{FIRMS_NRT_BASE}/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv",
    "VIIRS_N20_24h": f"{FIRMS_NRT_BASE}/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_24h.csv",
    "VIIRS_N20_48h": f"{FIRMS_NRT_BASE}/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_48h.csv",
    "VIIRS_N20_7d": f"{FIRMS_NRT_BASE}/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv",
    "VIIRS_N21_24h": f"{FIRMS_NRT_BASE}/noaa-21-viirs-c2/csv/J2_VIIRS_C2_Global_24h.csv",
    "VIIRS_SNPP_24h": f"{FIRMS_NRT_BASE}/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv",
}

# Regiones geográficas predefinidas para el visor mundial
WORLD_REGIONS = {
    "🌐 Todo el Planeta (Global)": {
        "center": [15.0, 0.0],
        "zoom": 2,
        "bbox": None,
        "desc": "Vista orbital completa de la Tierra con los focos más intensos"
    },
    "🇩🇴 República Dominicana (En Vivo)": {
        "center": [18.75, -70.45],
        "zoom": 8,
        "bbox": [-72.10, 17.40, -68.25, 20.05],
        "desc": "Focos satelitales en tiempo real dentro del territorio dominicano"
    },
    "🌎 América Central y el Caribe": {
        "center": [16.5, -75.0],
        "zoom": 5,
        "bbox": [-93.0, 7.0, -59.0, 24.5],
        "desc": "Caribe insular, Centroamérica e istmo mesoamericano"
    },
    "🌎 América del Sur (Amazonía / Gran Chaco)": {
        "center": [-14.0, -56.0],
        "zoom": 4,
        "bbox": [-82.0, -56.0, -34.0, 13.0],
        "desc": "Cuenca amazónica, Pantanal, Cerrado brasileño y Cono Sur"
    },
    "🌎 América del Norte (EE.UU., Canadá, México)": {
        "center": [44.0, -98.0],
        "zoom": 4,
        "bbox": [-168.0, 14.0, -50.0, 72.0],
        "desc": "Bosques boreales de Canadá, costas oeste y este de EE.UU. y México"
    },
    "🌍 Europa y Cuenca del Mediterráneo": {
        "center": [47.0, 15.0],
        "zoom": 4,
        "bbox": [-15.0, 34.0, 45.0, 71.0],
        "desc": "Península Ibérica, Grecia, Italia, Francia y Europa del Este"
    },
    "🌍 África Subsahariana y Central": {
        "center": [-1.0, 22.0],
        "zoom": 4,
        "bbox": [-18.0, -35.0, 52.0, 22.0],
        "desc": "Sabana africana, cuenca del Congo y cuerno de África (máxima densidad de quemas)"
    },
    "🌏 Asia del Sur y Sudeste Asiático": {
        "center": [16.0, 96.0],
        "zoom": 4,
        "bbox": [65.0, -11.0, 142.0, 38.0],
        "desc": "India, Indochina, Indonesia, Malasia y Filipinas"
    },
    "🌏 Asia Oriental y Siberia": {
        "center": [54.0, 108.0],
        "zoom": 3,
        "bbox": [70.0, 32.0, 175.0, 75.0],
        "desc": "Taiga siberiana, China oriental y Japón"
    },
    "🌏 Australia y Oceanía": {
        "center": [-25.0, 135.0],
        "zoom": 4,
        "bbox": [110.0, -48.0, 180.0, -10.0],
        "desc": "Outback australiano, costas de Nueva Gales del Sur y Nueva Zelanda"
    }
}


def normalize_global_df(df: pd.DataFrame, sensor_name: str) -> pd.DataFrame:
    """Estandariza columnas de cualquier feed de FIRMS."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    lat = pd.to_numeric(df.get("latitude"), errors="coerce")
    lon = pd.to_numeric(df.get("longitude"), errors="coerce")
    frp = pd.to_numeric(df.get("frp", np.nan), errors="coerce").fillna(0.0)
    
    # Temperatura de brillo
    if "bright_ti4" in df.columns:
        bright = pd.to_numeric(df["bright_ti4"], errors="coerce")
    elif "brightness" in df.columns:
        bright = pd.to_numeric(df["brightness"], errors="coerce")
    else:
        bright = pd.Series([np.nan] * len(df))

    date_col = pd.to_datetime(df.get("acq_date"), errors="coerce")
    time_val = pd.to_numeric(df.get("acq_time", 0), errors="coerce").fillna(0).astype(int)
    time_str = time_val.apply(lambda t: f"{t:04d}"[:2] + ":" + f"{t:04d}"[2:] + " UTC")

    sat = df.get("satellite", pd.Series(["?"] * len(df))).astype(str).str.strip()
    # Satélites legibles
    sat_map = {"T": "Terra", "A": "Aqua", "N": "Suomi-NPP", "N20": "NOAA-20", "N21": "NOAA-21",
               "1": "NOAA-20", "2": "NOAA-21"}
    sat_clean = sat.map(lambda s: sat_map.get(s, s))

    conf = df.get("confidence", pd.Series(["?"] * len(df))).astype(str)
    daynight = df.get("daynight", pd.Series(["?"] * len(df))).astype(str)

    scan = pd.to_numeric(df.get("scan", 1.0), errors="coerce").fillna(1.0)
    track = pd.to_numeric(df.get("track", 1.0), errors="coerce").fillna(1.0)

    res_km = "1.0 km" if "MODIS" in sensor_name else "375 m"

    out = pd.DataFrame({
        "lat": lat,
        "lon": lon,
        "frp": frp,
        "brightness_k": bright,
        "date": date_col,
        "time_utc": time_str,
        "satellite": sat_clean,
        "sensor": sensor_name,
        "resolution": res_km,
        "confidence": conf,
        "daynight": daynight,
        "scan": scan,
        "track": track
    })
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def fetch_live_feed(feed_key: str, cache_dir: Path | None = None, max_age_seconds: int = 1800) -> pd.DataFrame:
    """Descarga o recupera de caché un archivo CSV global NRT de NASA FIRMS."""
    url = URLS_NRT.get(feed_key)
    if not url:
        return pd.DataFrame()
    
    cache_path = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{feed_key}.csv"

    text = None
    if cache_path and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < max_age_seconds:
            try:
                text = cache_path.read_text(encoding="utf-8")
            except Exception:
                text = None

    if not text:
        try:
            resp = requests.get(url, timeout=35)
            if resp.status_code == 200 and resp.text.strip():
                text = resp.text
                if cache_path:
                    cache_path.write_text(text, encoding="utf-8")
        except Exception as e:
            if cache_path and cache_path.exists():
                text = cache_path.read_text(encoding="utf-8")
            else:
                return pd.DataFrame()

    if not text or not text.strip():
        return pd.DataFrame()

    try:
        raw = pd.read_csv(io.StringIO(text), low_memory=False)
        sensor_type = "MODIS" if "MODIS" in feed_key else "VIIRS"
        return normalize_global_df(raw, sensor_type)
    except Exception:
        return pd.DataFrame()


def query_firms_api_custom(map_key: str, source: str, area_coords: str | tuple,
                           day_range: int = 1) -> pd.DataFrame:
    """Consulta directa a la API de FIRMS usando una MAP_KEY para cualquier zona o país del mundo.
    
    area_coords puede ser un código ISO3 de país (ej. 'USA', 'BRA', 'ESP', 'DOM')
    o una tupla BBOX (w, s, e, n) / string 'w,s,e,n'.
    """
    if isinstance(area_coords, (list, tuple)):
        area_str = ",".join(f"{float(c):.4f}" for c in area_coords)
        endpoint_type = "area"
    elif "," in str(area_coords):
        area_str = str(area_coords).strip()
        endpoint_type = "area"
    else:
        area_str = str(area_coords).strip().upper()
        endpoint_type = "country"

    url = f"{config.FIRMS_API}/{endpoint_type}/csv/{map_key}/{source}/{area_str}/{day_range}"
    resp = requests.get(url, timeout=40)
    resp.raise_for_status()
    text = resp.text
    if text.lower().startswith(("invalid", "error")):
        raise RuntimeError(f"Respuesta FIRMS API: {text[:200]}")
    raw = pd.read_csv(io.StringIO(text))
    sensor_name = "MODIS" if "MODIS" in source else "VIIRS"
    return normalize_global_df(raw, sensor_name)


def filter_by_bbox(df: pd.DataFrame, bbox: list[float] | tuple[float, float, float, float] | None) -> pd.DataFrame:
    """Filtra puntos dentro de [oeste, sur, este, norte]."""
    if df.empty or bbox is None:
        return df
    w, s, e, n = bbox
    return df[(df["lon"].between(w, e)) & (df["lat"].between(s, n))].copy()
