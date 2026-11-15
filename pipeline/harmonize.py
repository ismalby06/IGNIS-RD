"""Armonización: VIIRS → focos equivalentes MODIS, con validación por años.

Se comparan cuatro métodos con validación "deja un año fuera" (LOYO) en el
periodo de solapamiento y se usa el de menor error mensual. La banda de
incertidumbre sale de los errores de esa misma validación, así que refleja
cuánto se equivoca el método en años que no vio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

METHODS = {
    "razon_px": "Razón simple, píxeles VIIRS",
    "razon_ev": "Razón simple, eventos VIIRS",
    "razon_ev_temporada": "Razón por temporada, eventos VIIRS",
    "loglineal_ev_temporada": "Log-lineal por temporada, eventos VIIRS",
}


def season(month_ts: pd.Series) -> pd.Series:
    return np.where(month_ts.dt.month.isin(config.DRY_MONTHS), "seca", "lluviosa")


def overlap(series: pd.DataFrame) -> pd.DataFrame:
    s = series[series["month"].dt.year >= config.OVERLAP_START].copy()
    s = s[s["month"] >= pd.Timestamp("2012-02-01")]           # primer mes completo de VIIRS
    return s


# ---------- ajuste y predicción por método ------------------------------------

def fit(method: str, train: pd.DataFrame) -> dict:
    x_col = "viirs_px" if method == "razon_px" else "viirs_ev"
    if method in ("razon_px", "razon_ev"):
        k = train["modis"].sum() / max(train[x_col].sum(), 1)
        return {"method": method, "x": x_col, "k": {"todas": float(k)}}
    seas = season(train["month"])
    if method == "razon_ev_temporada":
        ks = {}
        for s in ("seca", "lluviosa"):
            t = train[seas == s]
            ks[s] = float(t["modis"].sum() / max(t[x_col].sum(), 1))
        return {"method": method, "x": x_col, "k": ks}
    # log-lineal por temporada con corrección de sesgo (smearing de Duan)
    params = {}
    for s in ("seca", "lluviosa"):
        t = train[seas == s]
        x, y = np.log1p(t[x_col].to_numpy(float)), np.log1p(t["modis"].to_numpy(float))
        b, a = np.polyfit(x, y, 1) if len(t) > 2 else (1.0, 0.0)
        smear = float(np.mean(np.exp(y - (a + b * x)))) if len(t) else 1.0
        params[s] = {"a": float(a), "b": float(b), "smear": smear}
    return {"method": method, "x": x_col, "params": params}


def predict(model: dict, df: pd.DataFrame) -> np.ndarray:
    x = df[model["x"]].to_numpy(float)
    m = model["method"]
    if m in ("razon_px", "razon_ev"):
        return model["k"]["todas"] * x
    seas = season(df["month"])
    if m == "razon_ev_temporada":
        k = np.where(seas == "seca", model["k"]["seca"], model["k"]["lluviosa"])
        return k * x
    out = np.zeros(len(df))
    for s in ("seca", "lluviosa"):
        p = model["params"][s]; idx = seas == s
        out[idx] = np.exp(p["a"] + p["b"] * np.log1p(x[idx])) * p["smear"] - 1
    return np.clip(out, 0, None)


# ---------- validación deja-un-año-fuera ---------------------------------------

def loyo(method: str, ov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in sorted(ov["month"].dt.year.unique()):
        train, test = ov[ov["month"].dt.year != y], ov[ov["month"].dt.year == y]
        if test.empty or train.empty:
            continue
        pred = predict(fit(method, train), test)
        rows.append(test.assign(pred=pred, year=y)[["month", "year", "modis", "pred"]])
    return pd.concat(rows, ignore_index=True)


def evaluate(ov: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Error LOYO de cada método. Devuelve tabla de métricas y predicciones por método."""
    metrics, preds = [], {}
    for m in METHODS:
        p = loyo(m, ov)
        preds[m] = p
        ann = p.groupby("year")[["modis", "pred"]].sum()
        ann_err = (ann["pred"] - ann["modis"]).abs() / ann["modis"].clip(lower=1)
        metrics.append({
            "metodo": m, "nombre": METHODS[m],
            "mae_mensual": float((p["pred"] - p["modis"]).abs().mean()),
            "error_anual_pct": float(100 * ann_err.median()),
            "sesgo_pct": float(100 * (ann["pred"].sum() - ann["modis"].sum()) / max(ann["modis"].sum(), 1)),
            "corr_mensual": float(np.corrcoef(p["pred"], p["modis"])[0, 1]) if p["pred"].std() > 0 else np.nan,
        })
    met = pd.DataFrame(metrics).sort_values("mae_mensual").reset_index(drop=True)
    return met, preds


def band_from_loyo(p: pd.DataFrame, q=(0.1, 0.9)) -> dict:
    """Cuantiles del cociente observado/predicho en años no vistos (mensual y anual)."""
    r_m = (p["modis"] + 1) / (p["pred"] + 1)
    ann = p.groupby("year")[["modis", "pred"]].sum()
    r_a = (ann["modis"] + 1) / (ann["pred"] + 1)
    return {"mensual": [float(r_m.quantile(q[0])), float(r_m.quantile(q[1]))],
            "anual": [float(r_a.quantile(q[0])), float(r_a.quantile(q[1]))]}


# ---------- serie armonizada ---------------------------------------------------

def build_model(series: pd.DataFrame, force_method: str | None = None) -> dict:
    ov = overlap(series)
    met, preds = evaluate(ov)
    best = force_method or met.iloc[0]["metodo"]
    model = fit(best, ov)
    model["nombre"] = METHODS[best]
    model["banda"] = band_from_loyo(preds[best])
    model["metricas"] = met.to_dict(orient="records")
    model["solapamiento"] = [str(ov["month"].min().date()), str(ov["month"].max().date())]
    model["n_modis_solapamiento"] = int(ov["modis"].sum())
    return model, preds


def harmonize(series: pd.DataFrame, model: dict, cutover: str = "2012-02-01") -> pd.DataFrame:
    """MODIS observado antes del corte; VIIRS convertido (con banda) desde el corte."""
    s = series.copy()
    s["viirs_meq"] = predict(model, s)
    lo, hi = model["banda"]["mensual"]
    s["viirs_meq_lo"] = np.clip((s["viirs_meq"] + 1) * lo - 1, 0, None)
    s["viirs_meq_hi"] = (s["viirs_meq"] + 1) * hi - 1
    cut = s["month"] >= pd.Timestamp(cutover)
    s.loc[~cut, ["viirs_meq", "viirs_meq_lo", "viirs_meq_hi"]] = np.nan
    s["armonizado"] = np.where(cut, s["viirs_meq"], s["modis"])
    s["arm_lo"] = np.where(cut, s["viirs_meq_lo"], s["modis"])
    s["arm_hi"] = np.where(cut, s["viirs_meq_hi"], s["modis"])
    s["fuente"] = np.where(cut, "VIIRS → equivalente MODIS", "MODIS observado")
    return s


def jump_metrics(h: pd.DataFrame, before=(2007, 2011), after=(2012, 2016)) -> dict:
    """Cuánto 'sube' la serie en 2012 según cómo se unan los sensores."""
    y = h.assign(year=h["month"].dt.year).groupby("year").sum(numeric_only=True)
    b = y.loc[before[0]:before[1], "modis"].mean()
    a = y.loc[after[0]:after[1]]
    b = max(b, 1)
    return {
        "union_sin_ajuste": float(a["viirs_px"].mean() / b),
        "armonizado": float(a["armonizado"].mean() / b),
        "modis_real": float(a["modis"].mean() / b),
        "periodo_antes": list(before), "periodo_despues": list(after),
    }


# ======================================================================
#  Armonización de FRP (Potencia Radiativa del Fuego, MW)
# ----------------------------------------------------------------------
#  Igual que los conteos, pero calibrando la energía radiada: convierte el
#  FRP total mensual de VIIRS (suma por eventos) a "FRP equivalente MODIS".
#  Estructura paralela a la de conteos para no alterar ese camino ya
#  validado ni el esquema que lee la app. Predictor único: viirs_ev_frp.
# ======================================================================

FRP_METHODS = {
    "razon_frp": "Razón simple de FRP (eventos VIIRS)",
    "razon_frp_temporada": "Razón de FRP por temporada",
    "loglineal_frp_temporada": "Log-lineal de FRP por temporada (Duan)",
}
_FRP_X, _FRP_Y = "viirs_ev_frp", "modis_frp"


def fit_frp(method: str, train: pd.DataFrame) -> dict:
    x, y = _FRP_X, _FRP_Y
    if method == "razon_frp":
        k = train[y].sum() / max(train[x].sum(), 1e-9)
        return {"method": method, "x": x, "k": {"todas": float(k)}}
    seas = season(train["month"])
    if method == "razon_frp_temporada":
        ks = {s: float(train[seas == s][y].sum() / max(train[seas == s][x].sum(), 1e-9))
              for s in ("seca", "lluviosa")}
        return {"method": method, "x": x, "k": ks}
    params = {}
    for s in ("seca", "lluviosa"):
        t = train[seas == s]
        lx, ly = np.log1p(t[x].to_numpy(float)), np.log1p(t[y].to_numpy(float))
        b, a = np.polyfit(lx, ly, 1) if len(t) > 2 else (1.0, 0.0)
        smear = float(np.mean(np.exp(ly - (a + b * lx)))) if len(t) else 1.0
        params[s] = {"a": float(a), "b": float(b), "smear": smear}
    return {"method": method, "x": x, "params": params}


def predict_frp(model: dict, df: pd.DataFrame) -> np.ndarray:
    x = df[model["x"]].to_numpy(float)
    m = model["method"]
    if m == "razon_frp":
        return model["k"]["todas"] * x
    seas = season(df["month"])
    if m == "razon_frp_temporada":
        k = np.where(seas == "seca", model["k"]["seca"], model["k"]["lluviosa"])
        return k * x
    out = np.zeros(len(df))
    for s in ("seca", "lluviosa"):
        p = model["params"][s]; idx = seas == s
        out[idx] = np.exp(p["a"] + p["b"] * np.log1p(x[idx])) * p["smear"] - 1
    return np.clip(out, 0, None)


def _loyo_frp(method: str, ov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in sorted(ov["month"].dt.year.unique()):
        train, test = ov[ov["month"].dt.year != yr], ov[ov["month"].dt.year == yr]
        if test.empty or train.empty:
            continue
        pred = predict_frp(fit_frp(method, train), test)
        rows.append(test.assign(pred=pred, year=yr)[["month", "year", _FRP_Y, "pred"]])
    return pd.concat(rows, ignore_index=True)


def build_frp_model(series: pd.DataFrame, force_method: str | None = None) -> tuple[dict, dict]:
    ov = overlap(series)
    metrics, preds = [], {}
    for m in FRP_METHODS:
        p = _loyo_frp(m, ov)
        preds[m] = p
        ann = p.groupby("year")[[_FRP_Y, "pred"]].sum()
        ann_err = (ann["pred"] - ann[_FRP_Y]).abs() / ann[_FRP_Y].clip(lower=1)
        metrics.append({
            "metodo": m, "nombre": FRP_METHODS[m],
            "mae_mensual": float((p["pred"] - p[_FRP_Y]).abs().mean()),
            "error_anual_pct": float(100 * ann_err.median()),
            "corr_mensual": float(np.corrcoef(p["pred"], p[_FRP_Y])[0, 1]) if p["pred"].std() > 0 else np.nan,
        })
    met = pd.DataFrame(metrics).sort_values("mae_mensual").reset_index(drop=True)
    best = force_method or met.iloc[0]["metodo"]
    model = fit_frp(best, ov)
    model["nombre"] = FRP_METHODS[best]
    model["metricas"] = met.to_dict(orient="records")
    # Banda de incertidumbre desde los cocientes observado/predicho (años no vistos).
    p = preds[best]
    r_m = (p[_FRP_Y] + 1) / (p["pred"] + 1)
    ann = p.groupby("year")[[_FRP_Y, "pred"]].sum()
    r_a = (ann[_FRP_Y] + 1) / (ann["pred"] + 1)
    model["banda"] = {"mensual": [float(r_m.quantile(0.1)), float(r_m.quantile(0.9))],
                      "anual": [float(r_a.quantile(0.1)), float(r_a.quantile(0.9))]}
    return model, preds


def harmonize_frp(series: pd.DataFrame, model: dict, cutover: str = "2012-02-01") -> pd.DataFrame:
    """MODIS FRP observado antes del corte; VIIRS FRP convertido desde el corte."""
    s = series.copy()
    s["viirs_frp_meq"] = predict_frp(model, s)
    lo, hi = model["banda"]["mensual"]
    s["frp_lo"] = np.clip((s["viirs_frp_meq"] + 1) * lo - 1, 0, None)
    s["frp_hi"] = (s["viirs_frp_meq"] + 1) * hi - 1
    cut = s["month"] >= pd.Timestamp(cutover)
    s.loc[~cut, ["viirs_frp_meq", "frp_lo", "frp_hi"]] = np.nan
    s["frp_armonizado"] = np.where(cut, s["viirs_frp_meq"], s[_FRP_Y])
    s["frp_arm_lo"] = np.where(cut, s["frp_lo"], s[_FRP_Y])
    s["frp_arm_hi"] = np.where(cut, s["frp_hi"], s[_FRP_Y])
    return s


def frp_jump_metrics(h: pd.DataFrame, before=(2007, 2011), after=(2012, 2016)) -> dict:
    """Salto de FRP en 2012: crudo (sumar VIIRS tal cual) vs armonizado vs MODIS real."""
    y = h.assign(year=h["month"].dt.year).groupby("year").sum(numeric_only=True)
    b = max(y.loc[before[0]:before[1], _FRP_Y].mean(), 1e-6)
    a = y.loc[after[0]:after[1]]
    return {
        "union_sin_ajuste": float(a[_FRP_X].mean() / b),
        "armonizado": float(a["frp_armonizado"].mean() / b),
        "modis_real": float(a[_FRP_Y].mean() / b),
        "periodo_antes": list(before), "periodo_despues": list(after),
    }
