"""Grilla común, agrupación de píxeles VIIRS en eventos y tabla celda-mes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import config


def cell_center(v: pd.Series, deg: float = config.GRID_DEG) -> pd.Series:
    return (np.floor(v / deg) * deg + deg / 2).round(3)


def viirs_events(px: pd.DataFrame, radius_km: float = config.VIIRS_EVENT_RADIUS_KM) -> pd.DataFrame:
    """Une píxeles VIIRS del mismo día y paso (día/noche) a menos de `radius_km`.

    Un fuego que MODIS ve como un foco aparece como varios píxeles VIIRS;
    contar eventos en vez de píxeles quita gran parte de esa inflación.
    """
    if px.empty:
        return px.assign(n_px=[])
    out = []
    for (_, _), g in px.groupby([px["date"], px["daynight"]], sort=False):
        lat0 = np.radians(g["lat"].mean())
        xy = np.c_[g["lon"].to_numpy() * 111.32 * np.cos(lat0), g["lat"].to_numpy() * 110.57]
        if len(g) == 1:
            labels = np.array([0])
        else:
            pairs = cKDTree(xy).query_pairs(radius_km, output_type="ndarray")
            n = len(g)
            adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            _, labels = connected_components(adj, directed=False)
        agg = (g.assign(ev=labels)
                .groupby("ev")
                .agg(lat=("lat", "mean"), lon=("lon", "mean"), frp=("frp", "sum"),
                     n_px=("lat", "size"), date=("date", "first"), daynight=("daynight", "first")))
        out.append(agg)
    ev = pd.concat(out, ignore_index=True)
    ev["sensor"] = "VIIRS"
    return ev


def cell_month_table(modis: pd.DataFrame, viirs_px: pd.DataFrame, viirs_ev: pd.DataFrame) -> pd.DataFrame:
    """Una fila por celda y mes con conteos y FRP de cada sensor (nunca sumados entre sí)."""
    def agg(df, name, frp=True):
        if df.empty:
            return pd.DataFrame(columns=["month", "cell_lat", "cell_lon", name])
        t = pd.DataFrame({
            "month": df["date"].dt.to_period("M").dt.to_timestamp(),
            "cell_lat": cell_center(df["lat"]), "cell_lon": cell_center(df["lon"]),
            "frp": df["frp"].fillna(0),
        })
        g = t.groupby(["month", "cell_lat", "cell_lon"])
        r = g.size().rename(name).to_frame()
        if frp:
            r[f"{name}_frp"] = g["frp"].sum()
        return r.reset_index()

    keys = ["month", "cell_lat", "cell_lon"]
    tab = (agg(modis, "modis")
           .merge(agg(viirs_px, "viirs_px", frp=False), on=keys, how="outer")
           .merge(agg(viirs_ev, "viirs_ev"), on=keys, how="outer"))
    num = ["modis", "modis_frp", "viirs_px", "viirs_ev", "viirs_ev_frp"]
    tab[num] = tab[num].fillna(0)
    for c in ["modis", "viirs_px", "viirs_ev"]:
        tab[c] = tab[c].astype(int)
    return tab.sort_values(keys).reset_index(drop=True)


def zone_mask(tab: pd.DataFrame, zone: str) -> pd.Series:
    if zone == "Todo el país":
        return pd.Series(True, index=tab.index)
    bbox = config.ZONES.get(zone)
    if bbox is not None:                      # rectángulo (compatibilidad)
        w, s, e, n = bbox
        return tab["cell_lon"].between(w, e) & tab["cell_lat"].between(s, n)
    # provincia: point-in-polygon sobre los centros de celda
    from pipeline import geo
    return pd.Series(geo.mask_points(tab["cell_lat"].to_numpy(), tab["cell_lon"].to_numpy(), zone),
                     index=tab.index)


def monthly_series(tab: pd.DataFrame, zone: str = "Todo el país") -> pd.DataFrame:
    """Serie mensual completa (meses sin focos = 0) para una zona."""
    sub = tab[zone_mask(tab, zone)]
    cols = ["modis", "modis_frp", "viirs_px", "viirs_ev", "viirs_ev_frp"]
    s = sub.groupby("month")[cols].sum()
    full = pd.date_range(tab["month"].min(), tab["month"].max(), freq="MS")
    s = s.reindex(full, fill_value=0)
    s.index.name = "month"
    return s.reset_index()
