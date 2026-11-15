"""Calendario de quemas (mes × año), climatología, anomalías y frases resumen."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


def calendar_matrix(h: pd.DataFrame, col: str = "armonizado") -> pd.DataFrame:
    t = h.assign(year=h["month"].dt.year, m=h["month"].dt.month)
    return t.pivot_table(index="year", columns="m", values=col, aggfunc="sum").reindex(columns=range(1, 13))


def climatology(h: pd.DataFrame, col: str = "armonizado") -> pd.DataFrame:
    t = h.assign(m=h["month"].dt.month)
    c = t.groupby("m")[col].agg(["mean", "std", "median"]).reindex(range(1, 13))
    c["mes"] = config.MONTHS_ES
    c["share"] = c["mean"] / c["mean"].sum()
    return c


def anomalies(h: pd.DataFrame, col: str = "armonizado", z_min: float = 2.0, min_value: float = 5) -> pd.DataFrame:
    t = h.assign(m=h["month"].dt.month)
    stats = t.groupby("m")[col].agg(["mean", "std"])
    t = t.join(stats, on="m")
    t["z"] = (t[col] - t["mean"]) / t["std"].replace(0, np.nan)
    t["veces_normal"] = t[col] / t["mean"].replace(0, np.nan)
    out = t[(t["z"] >= z_min) & (t[col] >= min_value)]
    return out[["month", col, "mean", "z", "veces_normal"]].sort_values("z", ascending=False)


def annual(h: pd.DataFrame) -> pd.DataFrame:
    y = h.assign(year=h["month"].dt.year).groupby("year").agg(
        modis=("modis", "sum"), viirs_px=("viirs_px", "sum"), viirs_ev=("viirs_ev", "sum"),
        armonizado=("armonizado", "sum"), arm_lo=("arm_lo", "sum"), arm_hi=("arm_hi", "sum"))
    return y.reset_index()


def peak_window(clim: pd.DataFrame, width: int = 2) -> tuple[list[str], float]:
    vals = clim["mean"].to_numpy()
    best, best_i = -1, 0
    for i in range(12):
        s = sum(vals[(i + k) % 12] for k in range(width))
        if s > best:
            best, best_i = s, i
    months = [config.MONTHS_FULL_ES[(best_i + k) % 12] for k in range(width)]
    return months, float(best / max(vals.sum(), 1e-9))


def summary_sentences(h: pd.DataFrame, zone: str, jump: dict | None = None) -> list[str]:
    clim = climatology(h)
    months, share = peak_window(clim, 2)
    y = annual(h)
    full = y[(y["year"] < h["month"].max().year) | (h["month"].max().month == 12)]
    top = full.sort_values("armonizado", ascending=False).iloc[0]
    mean_y = full["armonizado"].mean()
    zona = "todo el país" if zone == "Todo el país" else zone.replace(" (aprox.)", "")
    out = [
        f"En {zona}, {months[0]} y {months[1]} concentran el "
        f"{share:.0%} de los focos de un año típico.",
        f"El año con más actividad fue {int(top['year'])}, con unos "
        f"{_fmt(top['armonizado'])} focos equivalentes MODIS, "
        f"{top['armonizado'] / max(mean_y, 1):.1f} veces el promedio anual.",
    ]
    if jump:
        out.append(
            f"Unir los sensores sin ajustar inventa un aumento de ×{jump['union_sin_ajuste']:.1f} "
            f"en 2012; la serie armonizada queda en ×{jump['armonizado']:.2f}, "
            f"cerca de lo que MODIS sigue viendo (×{jump['modis_real']:.2f}).")
    return out
