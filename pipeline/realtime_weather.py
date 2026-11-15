"""
pipeline/realtime_weather.py
============================
Meteorología en TIEMPO REAL para el dashboard IGNIS-RD.

Fuente: **Open-Meteo API** (gratis, sin clave, cobertura global).
  - Condiciones actuales (temperatura, humedad, viento, precipitación)
  - Pronóstico horario 72 h
  - Pronóstico diario 7 días

Con estos datos se calcula el **Canadian FWI en vivo** reutilizando
las ecuaciones de Van Wagner & Pickett (1985) validadas en nasa_power.py.

Arquitectura:
    1. fetch_all_nodes()       → descarga (con caché TTL) el clima de ~37 nodos
                                  a 0.5° que cubren las 465 celdas de fuego de RD.
    2. compute_realtime_fwi()  → calcula FFMC, DMC, DC, ISI, BUI, FWI secuencial-
                                  mente por nodo (14 días de warmup + 7 de pronóstico).
    3. get_current_conditions()→ condiciones actuales por celda 0.1° con FWI, provincia.
    4. get_forecast_daily()    → pronóstico 7 días por celda y provincia.
    5. get_hourly_forecast()   → pronóstico horario 72 h para gráficos temporales.

NOTA: Open-Meteo entrega viento en km/h, temperatura en °C y precipitación en mm,
que es exactamente lo que necesitan las ecuaciones del FWI canadiense.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

# Reutilizar las ecuaciones FWI validadas de nasa_power.py
from pipeline.nasa_power import _fwi_sequence

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
CACHE_DIR = config.DATA_CACHE / "realtime_weather"
CACHE_TTL_SECONDS = 900          # 15 min para condiciones actuales
OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"

# Escala canadiense estándar de peligro FWI
FWI_LEVELS = [
    (0,  5,  "🟢 Bajo",     "#22c55e"),
    (5,  10, "🟡 Moderado", "#eab308"),
    (10, 20, "🟠 Alto",     "#f97316"),
    (20, 30, "🔴 Muy Alto", "#ef4444"),
    (30, 999,"⚫ Extremo",  "#1e1b4b"),
]


def fwi_level(val: float) -> tuple[str, str]:
    """Devuelve (etiqueta, color_hex) para un valor FWI."""
    if pd.isna(val):
        return ("—", "#64748b")
    for lo, hi, label, color in FWI_LEVELS:
        if lo <= val < hi:
            return label, color
    return "⚫ Extremo", "#1e1b4b"


def wind_direction_label(deg: float) -> str:
    """Convierte grados a rosa de vientos (N, NE, E, SE, S, SW, W, NW)."""
    if pd.isna(deg):
        return "—"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((deg + 22.5) % 360) / 45)
    return dirs[idx]


# ---------------------------------------------------------------------------
# 1. Nodos de consulta (malla 0.5° sobre RD)
# ---------------------------------------------------------------------------
def _get_query_nodes() -> pd.DataFrame:
    """Nodos 0.5° que cubren las celdas de fuego. Misma malla que NASA POWER."""
    cells_path = config.DATA_OUT / "cells.parquet"
    if cells_path.exists():
        cells = pd.read_parquet(cells_path)[["cell_lat", "cell_lon"]].drop_duplicates()
        cells["plat"] = (cells["cell_lat"] * 2).round() / 2
        cells["plon"] = (cells["cell_lon"] * 2).round() / 2
        nodes = cells[["plat", "plon"]].drop_duplicates().reset_index(drop=True)
    else:
        # Fallback: rejilla regular que cubre el BBOX de RD
        lats = np.arange(17.5, 20.5, 0.5)
        lons = np.arange(-72.0, -68.0, 0.5)
        nodes = pd.DataFrame(
            [(la, lo) for la in lats for lo in lons],
            columns=["plat", "plon"]
        )
    return nodes


def _cell_to_node_map() -> pd.DataFrame:
    """Mapa celda de fuego (0.1°) → nodo Open-Meteo (0.5°)."""
    cells_path = config.DATA_OUT / "cells.parquet"
    if not cells_path.exists():
        return pd.DataFrame(columns=["cell_lat", "cell_lon", "plat", "plon"])
    cells = pd.read_parquet(cells_path)[["cell_lat", "cell_lon"]].drop_duplicates()
    cells["plat"] = (cells["cell_lat"] * 2).round() / 2
    cells["plon"] = (cells["cell_lon"] * 2).round() / 2
    return cells


# ---------------------------------------------------------------------------
# 2. Descarga con caché TTL
# ---------------------------------------------------------------------------
def _cache_path(lat: float, lon: float) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"ometeo_{lat:+.2f}_{lon:+.2f}.json"


def _cache_valid(fp: Path) -> bool:
    if not fp.exists():
        return False
    return (time.time() - fp.stat().st_mtime) < CACHE_TTL_SECONDS


def _fetch_node(lat: float, lon: float, past_days: int = 14,
                forecast_days: int = 7, timeout: int = 10) -> dict:
    """Descarga (con caché TTL) clima actual + horario + diario de Open-Meteo."""
    fp = _cache_path(lat, lon)
    if _cache_valid(fp):
        try:
            return json.loads(fp.read_text())
        except Exception:
            pass

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "wind_direction_10m", "precipitation",
        ]),
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "wind_direction_10m", "precipitation",
        ]),
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
        ]),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "America/Santo_Domingo",
        "wind_speed_unit": "kmh",
    }

    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(OPEN_METEO_API, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if "current" in data:
                fp.write_text(json.dumps(data))
                return data
            last_err = data
        except Exception as e:
            last_err = e
        time.sleep(0.5 * (attempt + 1))

    # Fallback a caché expirada
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            pass
    raise RuntimeError(f"Open-Meteo falló ({lat},{lon}): {last_err}")


def fetch_all_nodes(verbose: bool = False) -> list[dict]:
    """Descarga clima para todos los nodos de RD. Devuelve lista de {plat, plon, data}."""
    nodes = _get_query_nodes()
    results = []
    for i, row in nodes.iterrows():
        lat, lon = float(row["plat"]), float(row["plon"])
        if verbose:
            print(f"  Open-Meteo {i + 1}/{len(nodes)} ({lat}, {lon})")
        try:
            data = _fetch_node(lat, lon)
            results.append({"plat": lat, "plon": lon, "data": data})
        except Exception as e:
            if verbose:
                print(f"    ⚠️ {e}")
    return results


# ---------------------------------------------------------------------------
# 3. Cálculo del FWI en tiempo real
# ---------------------------------------------------------------------------
def compute_realtime_fwi(results: list[dict]) -> pd.DataFrame:
    """Calcula FWI secuencial por nodo: 14 días de warmup + 7 de pronóstico.
    Devuelve tabla diaria con FFMC, DMC, DC, ISI, BUI, FWI por nodo."""
    records = []
    for node in results:
        plat, plon = node["plat"], node["plon"]
        hourly = node["data"].get("hourly", {})
        if not hourly or "time" not in hourly:
            continue

        times = pd.to_datetime(hourly["time"])
        temp = np.array(hourly.get("temperature_2m", []), dtype=float)
        rh = np.array(hourly.get("relative_humidity_2m", []), dtype=float)
        wind = np.array(hourly.get("wind_speed_10m", []), dtype=float)
        precip = np.array(hourly.get("precipitation", []), dtype=float)
        wind_dir = np.array(hourly.get("wind_direction_10m", []), dtype=float)

        if len(temp) == 0:
            continue

        # Agregar horario → diario (el FWI es un índice diario)
        df = pd.DataFrame({
            "time": times, "temp": temp, "rh": rh,
            "wind": wind, "precip": precip, "wind_dir": wind_dir,
        })
        df["date"] = df["time"].dt.normalize()
        daily = df.groupby("date").agg(
            t2m=("temp", "mean"),
            t2m_max=("temp", "max"),
            rh=("rh", "mean"),
            rh_min=("rh", "min"),
            wind_kmh=("wind", "mean"),
            wind_max_kmh=("wind", "max"),
            precip_mm=("precip", "sum"),
            wind_dir=("wind_dir", lambda s: float(s.median())),
        ).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily["rh"] = daily["rh"].clip(0, 100)

        if len(daily) < 2:
            continue

        # FWI secuencial (Van Wagner & Pickett 1985)
        ffmc, dmc, dc, isi, bui, fwi = _fwi_sequence(
            daily["t2m"].to_numpy(),
            daily["rh"].to_numpy(),
            daily["wind_kmh"].to_numpy(),
            daily["precip_mm"].to_numpy(),
            daily["date"].dt.month.to_numpy(),
        )
        daily["ffmc"] = np.round(ffmc, 1)
        daily["dmc"] = np.round(dmc, 1)
        daily["dc"] = np.round(dc, 1)
        daily["isi"] = np.round(isi, 1)
        daily["bui"] = np.round(bui, 1)
        daily["fwi"] = np.round(fwi, 1)
        daily["plat"] = plat
        daily["plon"] = plon
        records.append(daily)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Condiciones actuales por celda de fuego
# ---------------------------------------------------------------------------
def get_current_conditions() -> pd.DataFrame:
    """Condiciones actuales + FWI por celda (~11 km), con provincia oficial.
    Retorna un DataFrame listo para el mapa y las tablas del dashboard."""
    results = fetch_all_nodes(verbose=False)
    if not results:
        return pd.DataFrame()

    daily_fwi = compute_realtime_fwi(results)
    if daily_fwi.empty:
        return pd.DataFrame()

    # Día más reciente por nodo (hoy o ayer si aún no hay datos completos)
    today = pd.Timestamp.now().normalize()
    recent = daily_fwi[daily_fwi["date"] >= today - pd.Timedelta(days=1)]
    if recent.empty:
        recent = daily_fwi.sort_values("date").groupby(["plat", "plon"]).tail(1)
    else:
        recent = recent.sort_values("date").groupby(["plat", "plon"]).last().reset_index()

    # Inyectar condiciones instantáneas de la API ("current")
    for node in results:
        cur = node["data"].get("current", {})
        mask = (recent["plat"] == node["plat"]) & (recent["plon"] == node["plon"])
        if mask.any() and cur:
            recent.loc[mask, "temp_now"] = cur.get("temperature_2m", np.nan)
            recent.loc[mask, "rh_now"] = cur.get("relative_humidity_2m", np.nan)
            recent.loc[mask, "wind_now"] = cur.get("wind_speed_10m", np.nan)
            recent.loc[mask, "wind_dir_now"] = cur.get("wind_direction_10m", np.nan)
            recent.loc[mask, "precip_now"] = cur.get("precipitation", np.nan)

    # Mapear a celdas de fuego (0.1°)
    cell_map = _cell_to_node_map()
    if not cell_map.empty:
        mapped = cell_map.merge(recent, on=["plat", "plon"], how="left")
    else:
        mapped = recent.copy()
        mapped["cell_lat"] = mapped["plat"]
        mapped["cell_lon"] = mapped["plon"]

    # Provincia oficial
    import pipeline.geo as geo_mod
    mapped["provincia"] = [
        geo_mod.point_province(float(la), float(lo)) or "Zona Fronteriza / Costera"
        for la, lo in zip(mapped["cell_lat"], mapped["cell_lon"])
    ]

    # Nivel de peligro FWI
    if "fwi" in mapped.columns:
        levels = mapped["fwi"].apply(lambda x: fwi_level(x))
        mapped["fwi_level"] = [l[0] for l in levels]
        mapped["fwi_color"] = [l[1] for l in levels]

    # Dirección del viento en texto
    if "wind_dir_now" in mapped.columns:
        mapped["wind_dir_label"] = mapped["wind_dir_now"].apply(wind_direction_label)

    mapped["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return mapped


# ---------------------------------------------------------------------------
# 5. Pronóstico diario 7 días por celda
# ---------------------------------------------------------------------------
def get_forecast_daily() -> pd.DataFrame:
    """Pronóstico diario 7 días con FWI por celda y provincia."""
    results = fetch_all_nodes(verbose=False)
    if not results:
        return pd.DataFrame()

    daily_fwi = compute_realtime_fwi(results)
    if daily_fwi.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()
    forecast = daily_fwi[daily_fwi["date"] >= today].copy()
    if forecast.empty:
        return pd.DataFrame()

    cell_map = _cell_to_node_map()
    if not cell_map.empty:
        mapped = cell_map.merge(forecast, on=["plat", "plon"], how="inner")
    else:
        mapped = forecast.copy()
        mapped["cell_lat"] = mapped["plat"]
        mapped["cell_lon"] = mapped["plon"]

    import pipeline.geo as geo_mod
    lat_col = "cell_lat" if "cell_lat" in mapped.columns else "plat"
    lon_col = "cell_lon" if "cell_lon" in mapped.columns else "plon"
    mapped["provincia"] = [
        geo_mod.point_province(float(la), float(lo)) or "Zona Fronteriza / Costera"
        for la, lo in zip(mapped[lat_col], mapped[lon_col])
    ]

    if "fwi" in mapped.columns:
        levels = mapped["fwi"].apply(lambda x: fwi_level(x))
        mapped["fwi_level"] = [l[0] for l in levels]
        mapped["fwi_color"] = [l[1] for l in levels]

    return mapped


# ---------------------------------------------------------------------------
# 6. Pronóstico horario 72 h (para gráficos de evolución)
# ---------------------------------------------------------------------------
def get_hourly_forecast() -> pd.DataFrame:
    """Pronóstico horario 72 h: temp, rh, viento, FWI estimado por nodo."""
    results = fetch_all_nodes(verbose=False)
    if not results:
        return pd.DataFrame()

    rows = []
    now = pd.Timestamp.now()
    for node in results:
        plat, plon = node["plat"], node["plon"]
        hourly = node["data"].get("hourly", {})
        if not hourly or "time" not in hourly:
            continue
        times = pd.to_datetime(hourly["time"])
        mask_future = times >= now - pd.Timedelta(hours=1)
        mask_72h = times <= now + pd.Timedelta(hours=72)
        mask = mask_future & mask_72h

        temp = np.array(hourly.get("temperature_2m", []))[mask]
        rh = np.array(hourly.get("relative_humidity_2m", []))[mask]
        wind = np.array(hourly.get("wind_speed_10m", []))[mask]
        precip = np.array(hourly.get("precipitation", []))[mask]
        t_sel = times[mask]

        for j in range(len(t_sel)):
            rows.append({
                "time": t_sel.iloc[j] if hasattr(t_sel, 'iloc') else t_sel[j],
                "plat": plat, "plon": plon,
                "temp": float(temp[j]),
                "rh": float(rh[j]),
                "wind_kmh": float(wind[j]),
                "precip_mm": float(precip[j]),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Rankings y resúmenes
# ---------------------------------------------------------------------------
def province_fwi_ranking(current_df: pd.DataFrame) -> pd.DataFrame:
    """Ranking de provincias por peligro FWI actual."""
    if current_df.empty or "fwi" not in current_df.columns:
        return pd.DataFrame()

    agg = current_df.dropna(subset=["fwi"]).groupby("provincia").agg(
        fwi_medio=("fwi", "mean"),
        fwi_max=("fwi", "max"),
        temp_media=("t2m", "mean") if "t2m" in current_df.columns else ("fwi", "count"),
        rh_media=("rh", "mean") if "rh" in current_df.columns else ("fwi", "count"),
        viento_medio=("wind_kmh", "mean") if "wind_kmh" in current_df.columns else ("fwi", "count"),
        precip_total=("precip_mm", "sum") if "precip_mm" in current_df.columns else ("fwi", "count"),
        celdas=("fwi", "count"),
    ).reset_index()

    for col in ["fwi_medio", "fwi_max", "temp_media", "rh_media", "viento_medio", "precip_total"]:
        if col in agg.columns:
            agg[col] = agg[col].round(1)

    agg["alerta"] = agg["fwi_max"].apply(
        lambda x: fwi_level(x)[0] if pd.notna(x) else "—"
    )
    return agg.sort_values("fwi_max", ascending=False).reset_index(drop=True)


def national_fwi_summary(current_df: pd.DataFrame) -> dict:
    """Resumen nacional: distribución de celdas por nivel de peligro FWI."""
    if current_df.empty or "fwi" not in current_df.columns:
        return {"total": 0, "levels": []}

    valid = current_df.dropna(subset=["fwi"])
    total = len(valid)
    summary = {
        "total": total,
        "fwi_nacional_medio": round(float(valid["fwi"].mean()), 1),
        "fwi_nacional_max": round(float(valid["fwi"].max()), 1),
        "temp_media": round(float(valid["t2m"].mean()), 1) if "t2m" in valid.columns else None,
        "rh_media": round(float(valid["rh"].mean()), 0) if "rh" in valid.columns else None,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M AST"),
    }

    levels = []
    for lo, hi, label, color in FWI_LEVELS:
        n = int(((valid["fwi"] >= lo) & (valid["fwi"] < hi)).sum())
        levels.append({"label": label, "color": color, "count": n, "pct": round(100 * n / max(total, 1), 1)})
    summary["levels"] = levels
    return summary


# ---------------------------------------------------------------------------
# Autotest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Test: descarga Open-Meteo y cálculo FWI en tiempo real ===")
    print("\n1. Nodos de consulta:")
    nodes = _get_query_nodes()
    print(f"   {len(nodes)} nodos a 0.5° cubriendo RD")

    print("\n2. Descargando clima (con caché 15 min)...")
    results = fetch_all_nodes(verbose=True)
    print(f"   {len(results)} nodos descargados con éxito")

    print("\n3. Calculando FWI en tiempo real...")
    daily = compute_realtime_fwi(results)
    print(f"   {len(daily)} días-nodo con FWI calculado")
    if not daily.empty:
        today = pd.Timestamp.now().normalize()
        t = daily[daily["date"] == today]
        if not t.empty:
            print(f"   FWI hoy: min={t['fwi'].min():.1f}  med={t['fwi'].mean():.1f}  max={t['fwi'].max():.1f}")

    print("\n4. Condiciones actuales por celda...")
    cc = get_current_conditions()
    print(f"   {len(cc)} celdas con datos actuales")
    if not cc.empty and "fwi" in cc.columns:
        ns = national_fwi_summary(cc)
        print(f"   FWI nacional: medio={ns['fwi_nacional_medio']}  max={ns['fwi_nacional_max']}")
        for lv in ns["levels"]:
            print(f"     {lv['label']}: {lv['count']} celdas ({lv['pct']}%)")

    print("\n5. Ranking provincial:")
    rk = province_fwi_ranking(cc)
    if not rk.empty:
        print(rk.head(10).to_string(index=False))
    print("\n✅ Test completado")
