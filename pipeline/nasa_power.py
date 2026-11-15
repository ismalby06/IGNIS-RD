"""
pipeline/nasa_power.py
======================
Meteorología REAL para el modelo de pronóstico, desde **NASA POWER**
(Prediction Of Worldwide Energy Resources, https://power.larc.nasa.gov).

NASA POWER es un producto NASA, gratuito y sin clave de API, que sirve
observaciones/reanálisis diarios (basados en MERRA-2 y datos de nubes CERES/GEOS)
desde 1981 hasta hace pocos días. Lo usamos para sustituir el "clima sintético"
del modelo anterior por temperatura, humedad, viento y lluvia **reales**, y con
ellos calcular el **Canadian Fire Weather Index (FWI) REAL** (no un proxy).

Flujo:
    1. fetch_power_grid()      -> descarga (con caché) el clima diario de una malla
                                   de puntos 0.5° que cubre República Dominicana.
    2. compute_fwi_system()    -> calcula FFMC, DMC, DC, ISI, BUI, FWI secuencial-
                                   mente por punto (ecuaciones Van Wagner & Pickett 1985).
    3. monthly_weather_by_cell -> agrega a mes × celda de fuego (0.1°) para fusionar
                                   con el panel de focos del pronóstico.

Las ecuaciones del FWI están validadas en __main__ contra el ejemplo clásico de
Van Wagner (FFMC≈87.7, DMC≈8.5, DC≈19.0, ISI≈10.9, BUI≈8.0, FWI≈10).

NOTA honesta: el FWI canónico usa condiciones del mediodía; NASA POWER entrega
medias diarias. Usamos la media diaria (práctica habitual con reanálisis), lo que
suaviza algo los extremos. Es clima REAL, a diferencia del modelo sintético.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import requests

import config

POWER_DIR = config.DATA_CACHE / "power"
POWER_MONTHLY = config.DATA_OUT / "power_monthly_cells.parquet"
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parámetros NASA POWER (comunidad AG = agroclimatología)
#   T2M         temperatura media a 2 m (°C)
#   T2M_MAX     temperatura máxima a 2 m (°C)
#   RH2M        humedad relativa a 2 m (%)
#   WS10M       velocidad del viento a 10 m (m/s)
#   PRECTOTCORR precipitación corregida (mm/día)
POWER_PARAMS = ["T2M", "T2M_MAX", "RH2M", "WS10M", "PRECTOTCORR"]
FILL = -999.0

# Arrays de longitud de día del sistema FWI (hemisferio norte), por mes (ene..dic)
DMC_LE = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
DC_LF = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


# ---------------------------------------------------------------------------
# 1. Descarga desde NASA POWER (con caché por punto)
# ---------------------------------------------------------------------------
def power_nodes_from_cells(cells_path: Path | None = None) -> pd.DataFrame:
    """Puntos de consulta NASA POWER (malla 0.5°) que cubren las celdas de fuego.
    Cada celda de fuego (0.1°) se asocia al nodo POWER 0.5° más cercano."""
    if cells_path is None:
        cells_path = config.DATA_OUT / "cells.parquet"
    cells = pd.read_parquet(cells_path)[["cell_lat", "cell_lon"]].drop_duplicates()
    cells["plat"] = (cells["cell_lat"] * 2).round() / 2
    cells["plon"] = (cells["cell_lon"] * 2).round() / 2
    nodes = cells[["plat", "plon"]].drop_duplicates().reset_index(drop=True)
    return nodes, cells


def _fetch_point(lat: float, lon: float, start: str, end: str,
                 cache_dir: Path = POWER_DIR, timeout: int = 90) -> dict:
    """Descarga (y cachea) un punto NASA POWER. Devuelve el dict de parámetros."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"power_{lat:+.2f}_{lon:+.2f}_{start}_{end}.json"
    fp = cache_dir / key
    if fp.exists():
        try:
            return json.loads(fp.read_text())["properties"]["parameter"]
        except Exception:
            pass
    url = (f"{POWER_API}?parameters={','.join(POWER_PARAMS)}&community=AG"
           f"&latitude={lat}&longitude={lon}&start={start}&end={end}&format=JSON")
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if "properties" in data and "parameter" in data["properties"]:
                fp.write_text(json.dumps(data))
                return data["properties"]["parameter"]
            last_err = data.get("messages", data)
        except Exception as e:
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"NASA POWER falló en ({lat},{lon}): {last_err}")


def fetch_power_grid(start: str = "20000101", end: str | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Descarga el clima diario real de todos los nodos POWER que cubren RD.
    Devuelve un DataFrame largo: [date, plat, plon, t2m, t2m_max, rh, wind_kmh, precip_mm]."""
    if end is None:
        end = pd.Timestamp.now().strftime("%Y%m%d")
    nodes, _ = power_nodes_from_cells()
    frames = []
    for i, row in nodes.iterrows():
        plat, plon = float(row["plat"]), float(row["plon"])
        if verbose:
            print(f"  NASA POWER {i + 1}/{len(nodes)}  ({plat}, {plon})")
        params = _fetch_point(plat, plon, start, end)
        idx = pd.to_datetime(list(params["T2M"].keys()), format="%Y%m%d")
        df = pd.DataFrame({
            "date": idx,
            "plat": plat, "plon": plon,
            "t2m": list(params["T2M"].values()),
            "t2m_max": list(params["T2M_MAX"].values()),
            "rh": list(params["RH2M"].values()),
            "wind_ms": list(params["WS10M"].values()),
            "precip_mm": list(params["PRECTOTCORR"].values()),
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Limpiar valores de relleno (-999) de días aún no disponibles
    for c in ["t2m", "t2m_max", "rh", "wind_ms", "precip_mm"]:
        out.loc[out[c] <= FILL + 1, c] = np.nan
    out = out.dropna(subset=["t2m", "rh", "wind_ms", "precip_mm"]).reset_index(drop=True)
    out["wind_kmh"] = out["wind_ms"] * 3.6
    out["rh"] = out["rh"].clip(0, 100)
    return out


# ---------------------------------------------------------------------------
# 2. Canadian Fire Weather Index (FWI) REAL — Van Wagner & Pickett (1985)
# ---------------------------------------------------------------------------
def _fwi_sequence(temp, rh, wind, rain, months,
                  ffmc0=85.0, dmc0=6.0, dc0=15.0):
    """Calcula FFMC, DMC, DC, ISI, BUI, FWI día a día para UNA serie temporal."""
    n = len(temp)
    FFMC = np.empty(n); DMC = np.empty(n); DC = np.empty(n)
    ISI = np.empty(n); BUI = np.empty(n); FWI = np.empty(n)
    ffmc_p, dmc_p, dc_p = ffmc0, dmc0, dc0

    for i in range(n):
        t = float(temp[i]); h = float(rh[i]); w = float(wind[i]); r = float(rain[i])
        mth = int(months[i])

        # ---- FFMC (humedad del combustible fino) ----
        mo = 147.2 * (101.0 - ffmc_p) / (59.5 + ffmc_p)
        if r > 0.5:
            rf = r - 0.5
            if mo <= 150.0:
                mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
            else:
                mo = (mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
                      + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf))
            mo = min(mo, 250.0)
        ed = (0.942 * h ** 0.679 + 11.0 * np.exp((h - 100.0) / 10.0)
              + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h)))
        if mo > ed:
            ko = 0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - (h / 100.0) ** 8)
            kd = ko * 0.581 * np.exp(0.0365 * t)
            m = ed + (mo - ed) * 10.0 ** (-kd)
        else:
            ew = (0.618 * h ** 0.753 + 10.0 * np.exp((h - 100.0) / 10.0)
                  + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h)))
            if mo < ew:
                kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - ((100.0 - h) / 100.0) ** 8)
                kw = kl * 0.581 * np.exp(0.0365 * t)
                m = ew - (ew - mo) * 10.0 ** (-kw)
            else:
                m = mo
        ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        ffmc = min(max(ffmc, 0.0), 101.0)

        # ---- DMC (humedad de capa media) ----
        td = max(t, -1.1)
        rk = 1.894 * (td + 1.1) * (100.0 - h) * DMC_LE[mth - 1] * 1e-4
        if r > 1.5:
            rw = 0.92 * r - 1.27
            wmi = 20.0 + np.exp(5.6348 - dmc_p / 43.43)
            if dmc_p <= 33.0:
                b = 100.0 / (0.5 + 0.3 * dmc_p)
            elif dmc_p <= 65.0:
                b = 14.0 - 1.3 * np.log(dmc_p)
            else:
                b = 6.2 * np.log(dmc_p) - 17.2
            wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
            pr = 43.43 * (5.6348 - np.log(max(wmr - 20.0, 1e-6)))
            pr = max(pr, 0.0)
        else:
            pr = dmc_p
        dmc = pr + rk
        dmc = max(dmc, 0.0)

        # ---- DC (sequía profunda) ----
        tdc = max(t, -2.8)
        pe = (0.36 * (tdc + 2.8) + DC_LF[mth - 1]) / 2.0
        pe = max(pe, 0.0)
        if r > 2.8:
            rw = 0.83 * r - 1.27
            smi = 800.0 * np.exp(-dc_p / 400.0)
            dr = dc_p - 400.0 * np.log(1.0 + 3.937 * rw / smi)
            dr = max(dr, 0.0)
            dc = dr + pe
        else:
            dc = dc_p + pe
        dc = max(dc, 0.0)

        # ---- ISI (propagación inicial) ----
        mm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        ff = 91.9 * np.exp(-0.1386 * mm) * (1.0 + mm ** 5.31 / 4.93e7)
        isi = 0.208 * np.exp(0.05039 * w) * ff

        # ---- BUI (combustible disponible) ----
        if dmc == 0.0 and dc == 0.0:
            bui = 0.0
        else:
            bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
        if bui < dmc:
            p = 0.0 if dmc == 0 else (dmc - bui) / dmc
            cc = 0.92 + (0.0114 * dmc) ** 1.7
            bui = max(dmc - cc * p, 0.0)

        # ---- FWI ----
        if bui <= 80.0:
            bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
        else:
            bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
        if bb <= 1.0:
            fwi = bb
        else:
            fwi = np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647)

        FFMC[i], DMC[i], DC[i] = ffmc, dmc, dc
        ISI[i], BUI[i], FWI[i] = isi, bui, fwi
        ffmc_p, dmc_p, dc_p = ffmc, dmc, dc

    return FFMC, DMC, DC, ISI, BUI, FWI


def compute_fwi_system(daily: pd.DataFrame) -> pd.DataFrame:
    """Añade FFMC, DMC, DC, ISI, BUI, FWI reales a la tabla diaria, por nodo POWER."""
    daily = daily.sort_values(["plat", "plon", "date"]).reset_index(drop=True)
    parts = []
    for (plat, plon), g in daily.groupby(["plat", "plon"], sort=False):
        g = g.copy()
        ff, dmc, dc, isi, bui, fwi = _fwi_sequence(
            g["t2m"].to_numpy(), g["rh"].to_numpy(),
            g["wind_kmh"].to_numpy(), g["precip_mm"].to_numpy(),
            g["date"].dt.month.to_numpy(),
        )
        g["ffmc"], g["dmc"], g["dc"] = ff, dmc, dc
        g["isi"], g["bui"], g["fwi"] = isi, bui, fwi
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Agregación mensual por celda de fuego
# ---------------------------------------------------------------------------
def monthly_weather_by_cell(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Descarga POWER, calcula FWI y agrega a mes × celda de fuego (0.1°).
    Cachea el resultado en data/processed/power_monthly_cells.parquet."""
    if POWER_MONTHLY.exists() and not force:
        return pd.read_parquet(POWER_MONTHLY)

    if verbose:
        print("Descargando clima real de NASA POWER (con caché)...")
    daily = fetch_power_grid(verbose=verbose)
    if verbose:
        print(f"  {len(daily):,} días-nodo descargados. Calculando FWI real...")
    daily = compute_fwi_system(daily)

    daily["month"] = daily["date"].values.astype("datetime64[M]")
    agg = daily.groupby(["plat", "plon", "month"]).agg(
        wx_t2m_mean=("t2m", "mean"),
        wx_t2m_max=("t2m_max", "max"),
        wx_rh_mean=("rh", "mean"),
        wx_rh_min=("rh", "min"),
        wx_wind_mean=("wind_kmh", "mean"),
        wx_precip_sum=("precip_mm", "sum"),
        wx_dry_days=("precip_mm", lambda s: int((s < 1.0).sum())),
        wx_ffmc_mean=("ffmc", "mean"),
        wx_isi_mean=("isi", "mean"),
        wx_fwi_mean=("fwi", "mean"),
        wx_fwi_max=("fwi", "max"),
    ).reset_index()

    # Mapear cada celda de fuego a su nodo POWER
    _, cell_map = power_nodes_from_cells()
    merged = cell_map.merge(agg, on=["plat", "plon"], how="left")
    cols = ["cell_lat", "cell_lon", "month"] + [c for c in agg.columns if c.startswith("wx_")]
    out = merged[cols].copy()

    POWER_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(POWER_MONTHLY, index=False)
    if verbose:
        print(f"  Guardado: {POWER_MONTHLY}  ({out['month'].min()} a {out['month'].max()}, "
              f"{out[['cell_lat','cell_lon']].drop_duplicates().shape[0]} celdas)")
    return out


WX_FEATURES = [
    "wx_t2m_mean", "wx_t2m_max", "wx_rh_mean", "wx_rh_min", "wx_wind_mean",
    "wx_precip_sum", "wx_dry_days", "wx_ffmc_mean", "wx_isi_mean",
    "wx_fwi_mean", "wx_fwi_max",
]


def _selftest_fwi():
    """Valida las ecuaciones FWI contra el ejemplo clásico de Van Wagner (1 día)."""
    ff, dmc, dc, isi, bui, fwi = _fwi_sequence(
        temp=[17.0], rh=[42.0], wind=[25.0], rain=[0.0], months=[4],
        ffmc0=85.0, dmc0=6.0, dc0=15.0,
    )
    exp = dict(FFMC=87.7, DMC=8.5, DC=19.0, ISI=10.9, BUI=8.0, FWI=10.1)
    got = dict(FFMC=ff[0], DMC=dmc[0], DC=dc[0], ISI=isi[0], BUI=bui[0], FWI=fwi[0])
    print("Autotest FWI (Van Wagner):")
    ok = True
    for k in exp:
        delta = abs(got[k] - exp[k])
        flag = "✓" if delta < 0.6 else "✗"
        if delta >= 0.6:
            ok = False
        print(f"  {k:<5} esperado≈{exp[k]:>6}  obtenido={got[k]:>7.2f}  {flag}")
    print("  => " + ("VALIDADO ✅" if ok else "REVISAR ❌"))
    return ok


if __name__ == "__main__":
    import sys
    _selftest_fwi()
    if "--fetch" in sys.argv:
        df = monthly_weather_by_cell(force="--force" in sys.argv)
        print("\nFeatures meteorológicas mensuales (muestra):")
        print(df.dropna().tail(6).to_string(index=False))
