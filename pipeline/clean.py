"""Limpieza: confianza, tipo FIRMS, recorte a RD y fuentes persistentes."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from matplotlib.path import Path
import pandas as pd

import config

TOPOJSON = config.ROOT / "static" / "world_50m.json"   # mismo archivo que usa el mapa de la app


@lru_cache(maxsize=1)
def dr_outline() -> Path:
    """Contorno de RD (escala 1:50 millones, precisión ~2–4 km) desde el TopoJSON local.

    Solo se usa cuando los datos vienen de una caja (bbox) o son simulados.
    El endpoint por país de FIRMS y el archivo histórico por país ya vienen
    recortados con la frontera oficial, que es más precisa.
    """
    topo = json.loads(TOPOJSON.read_text())
    sx, sy = topo["transform"]["scale"]; tx, ty = topo["transform"]["translate"]

    def arc_coords(i):
        arc = np.cumsum(np.array(topo["arcs"][i if i >= 0 else ~i], float), axis=0)
        xy = np.c_[arc[:, 0] * sx + tx, arc[:, 1] * sy + ty]
        return xy if i >= 0 else xy[::-1]

    geom = next(g for g in topo["objects"]["countries"]["geometries"] if g.get("id") == config.COUNTRY_ISO3)
    polys = geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]]
    verts, codes = [], []
    for poly in polys:
        for ring in poly:
            xy = np.vstack([arc_coords(i) for i in ring])
            verts.append(xy)
            codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
    return Path(np.vstack(verts), codes)


def inside_dr(lat, lon) -> np.ndarray:
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if lat.size == 0:
        return np.zeros(0, bool)
    return dr_outline().contains_points(np.c_[lon, lat])


def persistent_cells(df: pd.DataFrame, min_calendar_months: int = 9) -> pd.DataFrame:
    """Celdas de ~1 km que, junto con sus 8 vecinas, tienen focos en una fracción
    alta de todos los meses del registro Y en casi todos los meses del calendario.

    Mirar las vecinas evita que una fuente en el borde de dos celdas se parta en
    dos y pase el filtro. Exigir presencia casi todo el año evita borrar quemas
    agrícolas estacionales (caña, conucos) que se repiten cada temporada seca.
    """
    empty = pd.DataFrame(columns=["pc_lat", "pc_lon", "months", "fraction"])
    if df.empty:
        return empty
    d = config.PERSISTENT_CELL_DEG
    key = pd.DataFrame({
        "pc_lat": np.floor(df["lat"] / d).astype(int),
        "pc_lon": np.floor(df["lon"] / d).astype(int),
        "ym": df["date"].dt.to_period("M"),
    }).drop_duplicates()
    n_months = key["ym"].nunique()

    spread = pd.concat([key.assign(pc_lat=key.pc_lat + i, pc_lon=key.pc_lon + j)
                        for i in (-1, 0, 1) for j in (-1, 0, 1)]).drop_duplicates()
    g = spread.groupby(["pc_lat", "pc_lon"])["ym"]
    stats = pd.DataFrame({"months": g.nunique(),
                          "cal_months": g.apply(lambda s: s.dt.month.nunique())}).reset_index()
    stats["fraction"] = stats["months"] / max(n_months, 1)
    flagged = stats[(stats["fraction"] >= config.PERSISTENT_MONTH_FRACTION)
                    & (stats["cal_months"] >= min_calendar_months)]
    # Solo celdas que de verdad tienen focos (no las vecinas vacías)
    real = key[["pc_lat", "pc_lon"]].drop_duplicates()
    return flagged.merge(real, on=["pc_lat", "pc_lon"])[["pc_lat", "pc_lon", "months", "fraction"]]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica los filtros a un sensor y devuelve (datos limpios, informe)."""
    report = {"inicial": len(df)}
    df = df[df["conf_ok"]]
    report["tras_confianza"] = len(df)
    df = df[df["type"].isin(config.KEEP_FIRMS_TYPE)]
    report["tras_tipo_firms"] = len(df)
    df = df[inside_dr(df["lat"], df["lon"])]
    report["tras_recorte_rd"] = len(df)

    pers = persistent_cells(df)
    if not pers.empty:
        d = config.PERSISTENT_CELL_DEG
        k = list(zip(np.floor(df["lat"] / d).astype(int), np.floor(df["lon"] / d).astype(int)))
        bad = set(zip(pers["pc_lat"], pers["pc_lon"]))
        df = df[[x not in bad for x in k]]
    report["fuentes_persistentes"] = int(len(pers))
    report["final"] = len(df)
    return df.reset_index(drop=True), report
