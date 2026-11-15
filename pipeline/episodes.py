"""Episodios de alta actividad e indexación H3, calculados a partir de datos reales.

Nada aquí se inventa: las ubicaciones, focos, FRP y celdas H3 salen de
modis_points.parquet / viirs_events.parquet (reales o simulados, pero
siempre los que produjo el pipeline). No hay vectores de propagación,
infraestructura OSM ni "confianza bayesiana" porque no hay un modelo real
de eso en este proyecto.
"""
from __future__ import annotations

import base64
import json
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

try:
    import h3
except Exception:
    h3 = None
import numpy as np
import pandas as pd
import requests

import config

TOPOJSON = config.ROOT / "static" / "world_50m.json"
TILE_CACHE = config.ROOT / "data" / "cache" / "basemap_tiles"


def _tile_xy(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """Devuelve (norte, oeste, sur, este) en grados para la tesela (x, y, zoom)."""
    n = 2 ** zoom

    def corner(xt, yt):
        lon = xt / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))
        return lat, lon

    north, west = corner(x, y)
    south, east = corner(x + 1, y + 1)
    return north, west, south, east


def _fetch_one_tile(x: int, y: int, z: int) -> str | None:
    """Descarga una tesela satelital (server-side) y la guarda en caché local en disco.

    El servidor logra conectarse a ArcGIS (solo que con latencia alta en algunos
    entornos); al bajarla una sola vez aquí y servirla como data URL, el navegador
    nunca necesita contactar al servidor de teselas directamente.
    """
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = TILE_CACHE / f"{z}_{x}_{y}.png"
    if cache_file.exists():
        return base64.b64encode(cache_file.read_bytes()).decode()
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and r.content:
            cache_file.write_bytes(r.content)
            return base64.b64encode(r.content).decode()
    except requests.RequestException:
        pass
    return None


def fetch_basemap_tiles(lat: float, lon: float, zoom: int = 12, grid: int = 3) -> list[dict]:
    """Descarga (o reusa de caché) un cuadro de `grid`x`grid` teselas satelitales
    alrededor de (lat, lon) y devuelve cada una con su posición geográfica real,
    lista para dibujarse como imagen superpuesta (sin que el navegador tenga que
    conectarse a ningún servidor externo)."""
    cx, cy = _tile_xy(lat, lon, zoom)
    half = grid // 2
    coords = [(cx + i, cy + j) for i in range(-half, half + 1) for j in range(-half, half + 1)]
    with ThreadPoolExecutor(max_workers=min(9, len(coords))) as ex:
        b64s = list(ex.map(lambda xy: _fetch_one_tile(xy[0], xy[1], zoom), coords))
    out = []
    for (x, y), b64 in zip(coords, b64s):
        if b64 is None:
            continue
        north, west, south, east = _tile_bounds(x, y, zoom)
        out.append({"bounds": [[south, west], [north, east]], "data_url": f"data:image/png;base64,{b64}"})
    return out


@lru_cache(maxsize=1)
def country_outline_rings() -> list[list[list[float]]]:
    """Contorno real de RD como anillos [[lat, lon], ...], para dibujar de fondo en el
    mapa de la pestaña 1 sin depender de teselas satelitales externas (mismo TopoJSON
    local que ya usa pipeline/clean.py para recortar los focos al territorio dominicano).
    """
    topo = json.loads(TOPOJSON.read_text())
    sx, sy = topo["transform"]["scale"]; tx, ty = topo["transform"]["translate"]

    def arc_coords(i):
        arc = np.cumsum(np.array(topo["arcs"][i if i >= 0 else ~i], float), axis=0)
        xy = np.c_[arc[:, 0] * sx + tx, arc[:, 1] * sy + ty]
        return xy if i >= 0 else xy[::-1]

    geom = next(g for g in topo["objects"]["countries"]["geometries"] if g.get("id") == config.COUNTRY_ISO3)
    polys = geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]]
    rings = []
    for poly in polys:
        for ring in poly:
            xy = np.vstack([arc_coords(i) for i in ring])
            rings.append([[float(lat), float(lon)] for lon, lat in xy])
    return rings


def zone_label(lat: float, lon: float) -> str:
    """Provincia oficial que contiene el punto (point-in-polygon), o el país."""
    from pipeline import geo
    return geo.point_province(lat, lon) or "República Dominicana"


def modis_footprint(row: pd.Series) -> tuple[float, float]:
    """Medio ancho/alto en grados a partir de scan/track (km) de FIRMS.

    scan y track son el tamaño real del píxel MODIS en esa pasada (crece
    hacia el borde del barrido, el efecto bow-tie). Si faltan, se usa el
    tamaño nominal de 1 km.
    """
    scan = row.get("scan"); track = row.get("track")
    scan = 1.0 if pd.isna(scan) else float(scan)
    track = 1.0 if pd.isna(track) else float(track)
    dlat = track / 2 / 110.57
    dlon = scan / 2 / (111.32 * max(np.cos(np.radians(row["lat"])), 0.2))
    return dlat, dlon


def top_locations(cells: pd.DataFrame, modis_pts: pd.DataFrame, viirs_ev_pts: pd.DataFrame,
                   n: int = 6, radius_deg: float = 0.07) -> list[dict]:
    """Las `n` celdas de la grilla con más actividad combinada en todo el registro.

    Para cada una, recorta los puntos MODIS/VIIRS reales dentro de un radio
    y calcula celdas H3 (resolución 8, ~0.7 km²) reales con su conteo y FRP.
    """
    if cells.empty:
        return []
    act = (cells.groupby(["cell_lat", "cell_lon"])
           .agg(modis=("modis", "sum"), viirs_ev=("viirs_ev", "sum"), viirs_px=("viirs_px", "sum"))
           .reset_index())
    act["total"] = act["modis"] + act["viirs_ev"]
    top = act.sort_values("total", ascending=False).head(n)

    out = []
    for _, r in top.iterrows():
        lat0, lon0 = r["cell_lat"], r["cell_lon"]
        m = modis_pts[(modis_pts["lat"].between(lat0 - radius_deg, lat0 + radius_deg)) &
                      (modis_pts["lon"].between(lon0 - radius_deg, lon0 + radius_deg))]
        v = viirs_ev_pts[(viirs_ev_pts["lat"].between(lat0 - radius_deg, lat0 + radius_deg)) &
                         (viirs_ev_pts["lon"].between(lon0 - radius_deg, lon0 + radius_deg))]
        out.append({
            "lat": float(lat0), "lon": float(lon0),
            "zona": zone_label(lat0, lon0),
            "modis_total": int(r["modis"]), "viirs_ev_total": int(r["viirs_ev"]),
            "viirs_px_total": int(r["viirs_px"]),
            "modis_pts": m.reset_index(drop=True),
            "viirs_pts": v.reset_index(drop=True),
            "h3_cells": h3_cells_real(m, v),
        })
    return out


def h3_cells_real(modis_pts: pd.DataFrame, viirs_pts: pd.DataFrame, res: int = 8) -> list[dict]:
    """Agrega focos reales MODIS y eventos VIIRS en celdas H3 reales (res-8).

    Suma el FRP crudo de cada sensor por separado (no se mezclan sin
    armonizar); la armonización real del pipeline opera a nivel mensual,
    no por evento individual, así que aquí solo se muestran conteos y FRP
    observados, sin un "FRP armonizado" ficticio.
    """
    rows = []
    if h3 is not None:
        for df, sensor in ((modis_pts, "modis"), (viirs_pts, "viirs")):
            if df.empty:
                continue
            for _, p in df.iterrows():
                cell = h3.latlng_to_cell(p["lat"], p["lon"], res)
                rows.append({"h3": cell, "sensor": sensor, "frp": float(p.get("frp", 0) or 0)})
        if not rows:
            return []
        t = pd.DataFrame(rows)
        g = t.pivot_table(index="h3", columns="sensor", values="frp", aggfunc=["count", "sum"], fill_value=0)
        out = []
        for cell in g.index:
            lat, lon = h3.cell_to_latlng(cell)
            boundary = [[b[0], b[1]] for b in h3.cell_to_boundary(cell)]
            modis_n = int(g.loc[cell, ("count", "modis")]) if ("count", "modis") in g.columns else 0
            viirs_n = int(g.loc[cell, ("count", "viirs")]) if ("count", "viirs") in g.columns else 0
            modis_frp = float(g.loc[cell, ("sum", "modis")]) if ("sum", "modis") in g.columns else 0.0
            viirs_frp = float(g.loc[cell, ("sum", "viirs")]) if ("sum", "viirs") in g.columns else 0.0
            out.append({
                "h3Index": cell, "lat": lat, "lon": lon, "boundary": boundary,
                "modis_n": modis_n, "viirs_n": viirs_n, "modis_frp": modis_frp, "viirs_frp": viirs_frp,
            })
        return out

    # Fallback si la librería H3 C-extension está restringida por el sistema operativo
    r_deg = 0.00414  # radio aproximado de celda H3 res-8 (~460m)
    for df, sensor in ((modis_pts, "modis"), (viirs_pts, "viirs")):
        if df.empty:
            continue
        for _, p in df.iterrows():
            plat, plon = float(p["lat"]), float(p["lon"])
            grid_y = round(plat / (r_deg * 1.5))
            grid_x = round(plon / (r_deg * math.sqrt(3)))
            cell_id = f"hex_{grid_y}_{grid_x}"
            center_lat = grid_y * (r_deg * 1.5)
            center_lon = grid_x * (r_deg * math.sqrt(3))
            rows.append({
                "h3": cell_id, "lat": center_lat, "lon": center_lon,
                "sensor": sensor, "frp": float(p.get("frp", 0) or 0)
            })
    if not rows:
        return []
    t = pd.DataFrame(rows)
    g = t.pivot_table(index=["h3", "lat", "lon"], columns="sensor", values="frp", aggfunc=["count", "sum"], fill_value=0)
    out = []
    cos_lat = math.cos(math.radians(18.7))
    angles = [math.radians(a) for a in [30, 90, 150, 210, 270, 330]]
    for (cell, lat, lon) in g.index:
        boundary = [
            [lat + r_deg * math.sin(a), lon + (r_deg / max(cos_lat, 0.1)) * math.cos(a)]
            for a in angles
        ]
        modis_n = int(g.loc[(cell, lat, lon), ("count", "modis")]) if ("count", "modis") in g.columns else 0
        viirs_n = int(g.loc[(cell, lat, lon), ("count", "viirs")]) if ("count", "viirs") in g.columns else 0
        modis_frp = float(g.loc[(cell, lat, lon), ("sum", "modis")]) if ("sum", "modis") in g.columns else 0.0
        viirs_frp = float(g.loc[(cell, lat, lon), ("sum", "modis")]) if ("sum", "modis") in g.columns else 0.0
        out.append({
            "h3Index": cell, "lat": lat, "lon": lon, "boundary": boundary,
            "modis_n": modis_n, "viirs_n": viirs_n, "modis_frp": modis_frp, "viirs_frp": viirs_frp,
        })
    return out
