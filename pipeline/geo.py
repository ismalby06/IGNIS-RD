"""Polígonos oficiales de provincias de República Dominicana (geoBoundaries ADM1).

Reemplaza los rectángulos aproximados por límites reales para filtrar y etiquetar
por provincia con point-in-polygon (igual algoritmo que el recorte de frontera en
pipeline/clean.py). El GeoJSON vive en static/ para funcionar sin internet.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from matplotlib.path import Path

import config

PROVINCES_FILE = config.ROOT / "static" / "dom_provinces.geojson"


@lru_cache(maxsize=1)
def _provinces() -> dict:
    """{nombre: {"paths": [Path, ...], "bbox": (w, s, e, n)}} por provincia."""
    data = json.loads(PROVINCES_FILE.read_text())
    out = {}
    for f in data["features"]:
        name = f["properties"]["name"]
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        paths, pts = [], []
        for poly in polys:
            ring = np.asarray(poly[0], float)      # anillo exterior
            paths.append(Path(ring))
            pts.append(ring)
        arr = np.vstack(pts)
        bbox = (float(arr[:, 0].min()), float(arr[:, 1].min()),
                float(arr[:, 0].max()), float(arr[:, 1].max()))
        out[name] = {"paths": paths, "bbox": bbox}
    return out


def province_names() -> list[str]:
    return sorted(_provinces())


def point_province(lat: float, lon: float) -> str | None:
    """Provincia que contiene el punto, o None si cae fuera (mar/Haití)."""
    for name, p in _provinces().items():
        w, s, e, n = p["bbox"]
        if w <= lon <= e and s <= lat <= n:
            if any(path.contains_point((lon, lat)) for path in p["paths"]):
                return name
    return None


def mask_points(lat, lon, name: str) -> np.ndarray:
    """Máscara booleana de los puntos dentro de la provincia `name`."""
    p = _provinces().get(name)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if p is None:
        return np.ones(len(lat), dtype=bool)
    pts = np.c_[lon, lat]
    inside = np.zeros(len(pts), dtype=bool)
    for path in p["paths"]:
        inside |= path.contains_points(pts)
    return inside


def zone_bbox(name: str) -> tuple[float, float, float, float] | None:
    p = _provinces().get(name)
    return p["bbox"] if p else None
