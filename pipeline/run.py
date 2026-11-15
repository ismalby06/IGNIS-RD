"""Ejecuta el pipeline completo y guarda las salidas que usa la app.

Uso (desde la carpeta fuegord/):
    python -m pipeline.run --sample                     # datos simulados, sin MAP_KEY
    python -m pipeline.run --archive data/raw           # CSV del archivo histórico de FIRMS
    python -m pipeline.run --api --key TU_MAP_KEY --start 2001-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime

import pandas as pd

import config
from pipeline import clean as cl, grid, harmonize as hz, ingest


def process(modis_raw: pd.DataFrame, viirs_raw: pd.DataFrame, origen: str, save: bool = True) -> dict:
    print("Limpieza…")
    modis, rep_m = cl.clean(modis_raw)
    viirs, rep_v = cl.clean(viirs_raw)
    print(f"  MODIS {rep_m['inicial']:,} → {rep_m['final']:,} | VIIRS {rep_v['inicial']:,} → {rep_v['final']:,}")

    print("Agrupando píxeles VIIRS en eventos…")
    ev = grid.viirs_events(viirs)
    print(f"  {len(viirs):,} píxeles → {len(ev):,} eventos")

    print("Grilla y tabla celda-mes…")
    cells = grid.cell_month_table(modis, viirs, ev)
    series = grid.monthly_series(cells)

    print("Calibración y validación deja-un-año-fuera…")
    model, preds = hz.build_model(series)
    h = hz.harmonize(series, model)
    jump = hz.jump_metrics(h)
    for m in model["metricas"]:
        print(f"  {m['nombre']:<42} MAE mensual {m['mae_mensual']:6.1f} | error anual {m['error_anual_pct']:5.1f} %")
    print(f"  Elegido: {model['nombre']}")
    print(f"  Salto 2012 (conteos) → sin ajuste ×{jump['union_sin_ajuste']:.2f} | armonizado ×{jump['armonizado']:.2f} "
          f"| MODIS real ×{jump['modis_real']:.2f}")

    print("Armonización de FRP (potencia radiativa)…")
    frp_model, _ = hz.build_frp_model(series)
    hf = hz.harmonize_frp(series, frp_model)
    frp_jump = hz.frp_jump_metrics(hf)
    model["frp"] = frp_model
    print(f"  Elegido: {frp_model['nombre']}")
    print(f"  Salto 2012 (FRP) → sin ajuste ×{frp_jump['union_sin_ajuste']:.2f} | armonizado ×{frp_jump['armonizado']:.2f} "
          f"| MODIS real ×{frp_jump['modis_real']:.2f}")

    meta = {
        "origen": origen, "creado": datetime.now().isoformat(timespec="seconds"),
        "desde": str(series["month"].min().date()), "hasta": str(series["month"].max().date()),
        "limpieza": {"MODIS": rep_m, "VIIRS": rep_v},
        "viirs_pixeles": int(len(viirs)), "viirs_eventos": int(len(ev)),
    }
    out = {"cells": cells, "model": model, "preds": preds, "meta": meta,
           "modis_pts": modis[["lat", "lon", "date", "frp", "satellite", "scan", "track"]],
           "viirs_ev_pts": ev[["lat", "lon", "date", "frp", "n_px"]]}

    if save:
        config.DATA_OUT.mkdir(parents=True, exist_ok=True)
        cells.to_parquet(config.DATA_OUT / "cells.parquet", index=False)
        out["modis_pts"].to_parquet(config.DATA_OUT / "modis_points.parquet", index=False)
        out["viirs_ev_pts"].to_parquet(config.DATA_OUT / "viirs_events.parquet", index=False)
        pd.concat([p.assign(metodo=m) for m, p in preds.items()]).to_parquet(
            config.DATA_OUT / "loyo_preds.parquet", index=False)
        (config.DATA_OUT / "model_national.json").write_text(json.dumps(model, indent=2, ensure_ascii=False))
        (config.DATA_OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"Guardado en {config.DATA_OUT}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Pipeline FuegoRD")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="usar datos simulados")
    g.add_argument("--archive", nargs="?", const=str(config.DATA_RAW), help="carpeta con CSV de FIRMS")
    g.add_argument("--api", action="store_true", help="descargar de la API de FIRMS")
    ap.add_argument("--key", default=os.environ.get("FIRMS_MAP_KEY"), help="MAP_KEY (o variable FIRMS_MAP_KEY)")
    ap.add_argument("--start", default="2001-01-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--use-country", action="store_true",
                    help="usar endpoint por país (DOM); por defecto se usa el de área/BBOX")
    a = ap.parse_args()

    if a.sample:
        from pipeline.synthetic import simulate
        print("Generando datos SIMULADOS…")
        m, v = simulate()
        process(ingest.normalize(m, "MODIS"), ingest.normalize(v, "VIIRS"), origen="simulado")
    elif a.archive:
        df = ingest.load_archive(a.archive)
        process(df[df.sensor == "MODIS"], df[df.sensor == "VIIRS"], origen="FIRMS archivo")
    else:
        if not a.key:
            raise SystemExit("Falta la MAP_KEY: usa --key o la variable FIRMS_MAP_KEY")
        s, e = date.fromisoformat(a.start), date.fromisoformat(a.end)
        uc = a.use_country
        print("Descargando MODIS…"); m = ingest.fetch_api("MODIS", s, e, a.key, use_country=uc)
        print("Descargando VIIRS…"); v = ingest.fetch_api("VIIRS", max(s, date(2012, 1, 20)), e, a.key, use_country=uc)
        process(m, v, origen="FIRMS API")
        try:
            print("Validación cruzada VIIRS NOAA-20 ↔ S-NPP…")
            from pipeline import crossval
            cs = crossval.build_crossval(a.key, end=e)
            print(f"  NOAA-20 {cs['noaa20_eventos']:,} eventos | "
                  f"correlación mensual con S-NPP {cs['correlacion_mensual']:.3f} "
                  f"| ratio NOAA-20/S-NPP {cs['ratio_mediano_noaa20_snpp']:.2f}")
        except Exception as ex:
            print(f"  (validación NOAA-20 omitida: {ex})")


if __name__ == "__main__":
    main()
