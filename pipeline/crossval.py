"""Validación cruzada entre plataformas VIIRS: Suomi-NPP vs NOAA-20.

S-NPP (2012+) y NOAA-20 (2018+) son dos sensores VIIRS casi idénticos en órbitas
distintas. Comparar sus detecciones mensuales en el periodo en que ambos vuelan
(2018+) comprueba que la señal de fuego que usa FuegoRD —y por tanto la
calibración VIIRS→MODIS— es consistente entre satélites, no un artefacto de uno
solo. No se suman: NOAA-20 sirve de contraste independiente.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import config
from pipeline import clean as cl, grid, ingest

NOAA20_SOURCE = "VIIRS_NOAA20_SP"
NOAA20_START = date(2018, 4, 1)


def _monthly_events(ev: pd.DataFrame, name: str) -> pd.DataFrame:
    m = (ev.assign(month=ev["date"].dt.to_period("M").dt.to_timestamp())
         .groupby("month").size().rename(name).to_frame())
    return m


def build_crossval(map_key: str, end: date | None = None, save: bool = True) -> dict:
    """Descarga/lee NOAA-20 (caché), lo limpia y lo compara con los eventos S-NPP ya
    procesados. Devuelve stats y guarda data/processed/noaa20_cross.parquet."""
    end = end or date.today()
    config.SOURCES.setdefault("VIIRS_N20", NOAA20_SOURCE)

    raw = ingest.fetch_api("VIIRS_N20", NOAA20_START, end, map_key, use_country=False)
    n20, rep = cl.clean(raw)
    n20_ev = grid.viirs_events(n20)

    snpp_ev = pd.read_parquet(config.DATA_OUT / "viirs_events.parquet")

    a = _monthly_events(snpp_ev, "snpp_ev")
    b = _monthly_events(n20_ev, "noaa20_ev")
    cmp = a.join(b, how="outer").fillna(0)
    cmp = cmp[cmp.index >= pd.Timestamp(NOAA20_START)].reset_index()
    cmp[["snpp_ev", "noaa20_ev"]] = cmp[["snpp_ev", "noaa20_ev"]].astype(int)

    corr = float(np.corrcoef(cmp["snpp_ev"], cmp["noaa20_ev"])[0, 1]) if len(cmp) > 2 else float("nan")
    ratio = (cmp["noaa20_ev"] / cmp["snpp_ev"].clip(lower=1))
    stats = {
        "noaa20_filas": int(len(raw)), "noaa20_limpios": int(rep["final"]),
        "noaa20_eventos": int(len(n20_ev)),
        "meses_comparados": int(len(cmp)),
        "desde": str(cmp["month"].min().date()) if len(cmp) else None,
        "hasta": str(cmp["month"].max().date()) if len(cmp) else None,
        "correlacion_mensual": corr,
        "ratio_mediano_noaa20_snpp": float(ratio.median()),
        "snpp_total": int(cmp["snpp_ev"].sum()), "noaa20_total": int(cmp["noaa20_ev"].sum()),
    }
    if save:
        cmp.to_parquet(config.DATA_OUT / "noaa20_cross.parquet", index=False)
        import json
        (config.DATA_OUT / "noaa20_cross.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    import os
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("Falta FIRMS_MAP_KEY")
    s = build_crossval(key, end=date(2026, 7, 31))
    print("Validación cruzada S-NPP ↔ NOAA-20:")
    for k, v in s.items():
        print(f"  {k}: {v}")
