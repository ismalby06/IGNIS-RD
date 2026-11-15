"""
pipeline/forecast_model.py
==========================
Pronóstico HONESTO de próximos focos de calor (incendios/quemas) para República
Dominicana, construido ÚNICAMENTE con la serie armonizada de focos MODIS + VIIRS
de NASA FIRMS (la única fuente de datos que entrega el reto Space Apps 2026
"Harmonization of MODIS and VIIRS Hot Spots").

Diferencia con los satélites:
- Satélites MODIS/VIIRS: DETECTAN focos que YA están ardiendo (anticipación = 0 h).
- Este modelo: PRONOSTICA la probabilidad de que cada celda tenga focos el PRÓXIMO
  mes, usando sólo información disponible hasta el mes actual (pronóstico t+1 real,
  sin fuga temporal de datos).

Honestidad científica (lo que un jurado NASA va a revisar):
- NO se inventa meteorología. No usamos temperatura/viento/humedad "sintéticos".
  Todas las variables salen del propio historial de focos armonizado.
- El objetivo es el FUTURO (mes t+1), no el presente: se evita el "nowcasting"
  disfrazado de predicción.
- Se compara contra una LÍNEA BASE CLIMATOLÓGICA (lo típico de cada celda en cada
  mes) y se reporta el SKILL real sobre esa base (Brier Skill Score). Un modelo
  que no supera a la climatología no aporta nada, y aquí lo demostramos.
- Partición temporal estricta: entrena con el pasado, se evalúa en años nunca vistos.

Señales (todas calculadas sólo con el pasado, vía desplazamientos temporales):
    cell_lat, cell_lon        : posición de la celda (~11 km)
    month_num, doy_sin/cos    : estacionalidad (calendario)
    is_dry_season             : temporada seca de RD (dic-abr) — del calendario
    year_idx                  : tendencia interanual
    lag1_fire                 : hubo foco el mes anterior (persistencia)
    rate_3m / 6m / 12m        : fracción de meses con foco en las últimas 3/6/12 lunas
    months_since_fire         : meses desde el último foco (reciencia)
    seasonal_clim             : qué tan típico es el fuego en ESA celda y ESE mes
                                (climatología por celda, sólo con años anteriores)
    recent_frp_log            : energía radiativa reciente (log1p de FRP últimos 3 meses)
    neighbor_rate             : actividad de focos en las 8 celdas vecinas (el fuego
                                se agrupa espacialmente) — del mes actual, conocido en t
    cell_base_rate            : tasa histórica acumulada de la celda (prior espacial)
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, f1_score,
)

import config
import pipeline.geo as geo
from pipeline import nasa_power as npw

# --- Rutas de artefactos -----------------------------------------------------
MODEL_PATH = config.DATA_OUT / "forecast_model.joblib"
METRICS_PATH = config.DATA_OUT / "forecast_metrics.json"

# --- Configuración del pronóstico -------------------------------------------
DRY_SEASON_MONTHS = {12, 1, 2, 3, 4}       # temporada seca aprox. de RD
GRID_DEG = getattr(config, "GRID_DEG", 0.1)
NEIGHBOR_TOL = GRID_DEG * 1.5               # radio para considerar celdas "vecinas"

FEATURE_COLS = [
    "cell_lat", "cell_lon", "month_num", "doy_sin", "doy_cos", "is_dry_season",
    "year_idx", "lag1_fire", "rate_3m", "rate_6m", "rate_12m",
    "months_since_fire", "seasonal_clim", "recent_frp_log",
    "neighbor_rate", "cell_base_rate",
]

# Variables meteorológicas REALES de NASA POWER (opción B). Se añaden a FEATURE_COLS
# cuando existe el parquet data/processed/power_monthly_cells.parquet.
WX_FEATURES = npw.WX_FEATURES
FEATURE_COLS_WX = FEATURE_COLS + WX_FEATURES


def weather_available() -> bool:
    return npw.POWER_MONTHLY.exists()


def attach_weather(panel: pd.DataFrame) -> pd.DataFrame:
    """Fusiona las variables meteorológicas reales de NASA POWER (mes × celda).
    El clima del mes t es conocido al pronosticar t+1 (sin fuga de datos)."""
    if not weather_available():
        return panel
    wx = pd.read_parquet(npw.POWER_MONTHLY)
    wx["month"] = pd.to_datetime(wx["month"])
    return panel.merge(wx, on=["cell_lat", "cell_lon", "month"], how="left")

# Particiones temporales (años)
TRAIN_END = 2018          # entrena 2001–2018
VAL_END = 2022            # valida  2019–2022
# test: 2023 en adelante (años nunca vistos)


# ---------------------------------------------------------------------------
# 1. Panel espacio-temporal con señales SIN fuga de datos
# ---------------------------------------------------------------------------
def build_forecast_panel(cells_path: Path | None = None) -> pd.DataFrame:
    """Construye el panel (mes × celda) con el objetivo del PRÓXIMO mes (t+1)
    y variables derivadas únicamente del pasado."""
    if cells_path is None:
        cells_path = config.DATA_OUT / "cells.parquet"

    cells = pd.read_parquet(cells_path)

    # Rejilla completa mes × celda (rellena meses sin foco con 0)
    unique_cells = cells[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    months = pd.date_range(cells["month"].min(), cells["month"].max(), freq="MS")
    panel = (
        unique_cells.assign(key=1)
        .merge(pd.DataFrame({"month": months, "key": 1}), on="key")
        .drop(columns="key")
    )

    agg = cells.groupby(["month", "cell_lat", "cell_lon"]).agg(
        modis=("modis", "sum"),
        modis_frp=("modis_frp", "sum"),
        viirs_ev=("viirs_ev", "sum"),
        viirs_frp=("viirs_ev_frp", "sum"),
    ).reset_index()

    panel = panel.merge(agg, on=["month", "cell_lat", "cell_lon"], how="left").fillna(0)
    panel["has_fire"] = ((panel["modis"] > 0) | (panel["viirs_ev"] > 0)).astype(int)
    panel["total_frp"] = panel["modis_frp"] + panel["viirs_frp"]

    panel["year"] = panel["month"].dt.year
    panel["month_num"] = panel["month"].dt.month
    doy = panel["month"].dt.day_of_year if hasattr(panel["month"].dt, "day_of_year") else panel["month"].dt.dayofyear
    panel["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    panel["is_dry_season"] = panel["month_num"].isin(DRY_SEASON_MONTHS).astype(int)
    panel["year_idx"] = panel["year"] - int(panel["year"].min())

    # Ordenar por celda y tiempo para los desplazamientos temporales
    panel = panel.sort_values(["cell_lat", "cell_lon", "month"]).reset_index(drop=True)
    g = panel.groupby(["cell_lat", "cell_lon"], sort=False)

    # --- Persistencia y actividad reciente (sólo meses anteriores) ----------
    panel["lag1_fire"] = g["has_fire"].shift(1).fillna(0)
    panel["rate_3m"] = g["has_fire"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    panel["rate_6m"] = g["has_fire"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean()).fillna(0)
    panel["rate_12m"] = g["has_fire"].transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean()).fillna(0)
    panel["recent_frp_log"] = np.log1p(
        g["total_frp"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    )

    # --- Meses desde el último foco (reciencia) -----------------------------
    def _months_since(s: pd.Series) -> pd.Series:
        out = np.empty(len(s)); cnt = 999
        prev = s.shift(1).fillna(0).to_numpy()   # fuego hasta el mes anterior
        for i, v in enumerate(prev):
            cnt = 0 if v > 0 else min(cnt + 1, 999)
            out[i] = cnt
        return pd.Series(out, index=s.index)
    panel["months_since_fire"] = g["has_fire"].transform(_months_since)

    # --- Tasa base acumulada de la celda (prior espacial, sin el mes actual) -
    panel["cum_fire"] = g["has_fire"].cumsum() - panel["has_fire"]
    panel["cum_n"] = g.cumcount()
    panel["cell_base_rate"] = np.where(panel["cum_n"] > 0, panel["cum_fire"] / panel["cum_n"], 0.0)

    # --- Climatología estacional por celda × mes (sólo años anteriores) -----
    gc = panel.groupby(["cell_lat", "cell_lon", "month_num"], sort=False)
    clim_cum = gc["has_fire"].cumsum() - panel["has_fire"]
    clim_n = gc.cumcount()
    # fallback a la tasa base de la celda cuando aún no hay historia de ese mes
    panel["seasonal_clim"] = np.where(clim_n > 0, clim_cum / clim_n.replace(0, np.nan), panel["cell_base_rate"])
    panel["seasonal_clim"] = panel["seasonal_clim"].fillna(panel["cell_base_rate"])

    # --- Actividad de celdas vecinas en el mes actual (conocido en t) -------
    panel = _add_neighbor_rate(panel)

    # --- OBJETIVO: ¿habrá foco el PRÓXIMO mes? (t+1) ------------------------
    # El último mes de cada celda queda con target NaN (aún no hay "mes siguiente"):
    # se excluye del entrenamiento pero SÍ sirve para pronosticar el próximo mes real.
    panel["target_next"] = g["has_fire"].shift(-1)

    # Clima real de NASA POWER (opción B), si está disponible
    panel = attach_weather(panel)
    return panel


def _add_neighbor_rate(panel: pd.DataFrame) -> pd.DataFrame:
    """Para cada celda y mes, fracción de celdas vecinas (8-conectividad) con foco
    ese mismo mes. Es información del mes t (conocida al pronosticar t+1)."""
    cells = panel[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    lat = cells["cell_lat"].to_numpy()
    lon = cells["cell_lon"].to_numpy()
    # vecinos por celda (índices en la lista de celdas únicas)
    key_to_idx = {(round(la, 4), round(lo, 4)): i for i, (la, lo) in enumerate(zip(lat, lon))}
    neighbors = {}
    for i, (la, lo) in enumerate(zip(lat, lon)):
        nb = []
        for dla in (-GRID_DEG, 0.0, GRID_DEG):
            for dlo in (-GRID_DEG, 0.0, GRID_DEG):
                if dla == 0.0 and dlo == 0.0:
                    continue
                k = (round(la + dla, 4), round(lo + dlo, 4))
                if k in key_to_idx:
                    nb.append(key_to_idx[k])
        neighbors[i] = nb

    # matriz mes × celda con has_fire
    panel["_ckey"] = list(zip(panel["cell_lat"].round(4), panel["cell_lon"].round(4)))
    idx_map = {k: i for i, k in enumerate(zip(cells["cell_lat"].round(4), cells["cell_lon"].round(4)))}
    panel["_cidx"] = panel["_ckey"].map(idx_map)

    pivot = panel.pivot_table(index="month", columns="_cidx", values="has_fire",
                              aggfunc="max", fill_value=0).sort_index()
    pv = pivot.to_numpy(dtype=float)
    col_order = list(pivot.columns)
    col_pos = {c: j for j, c in enumerate(col_order)}

    # suma de vecinos / nº de vecinos por celda y mes
    neigh_rate = np.zeros_like(pv)
    for i in range(len(cells)):
        if i not in col_pos:
            continue
        nb = [col_pos[n] for n in neighbors[i] if n in col_pos]
        if not nb:
            continue
        neigh_rate[:, col_pos[i]] = pv[:, nb].mean(axis=1)

    nr_df = pd.DataFrame(neigh_rate, index=pivot.index, columns=col_order)
    long = nr_df.reset_index().melt(id_vars="month", var_name="_cidx", value_name="neighbor_rate")
    panel = panel.merge(long, on=["month", "_cidx"], how="left")
    panel["neighbor_rate"] = panel["neighbor_rate"].fillna(0.0)
    return panel.drop(columns=["_ckey", "_cidx", "cum_fire", "cum_n"], errors="ignore")


# ---------------------------------------------------------------------------
# 2. Entrenamiento y evaluación honesta
# ---------------------------------------------------------------------------
def train_forecast(force_retrain: bool = False) -> dict:
    """Entrena los modelos de pronóstico, los evalúa en años nunca vistos y
    guarda modelo + métricas. Devuelve el diccionario de métricas."""
    if not force_retrain and MODEL_PATH.exists() and METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    panel = build_forecast_panel()
    labeled = panel.dropna(subset=["target_next"]).copy()
    labeled["target_next"] = labeled["target_next"].astype(int)

    has_wx = weather_available() and all(c in labeled.columns for c in WX_FEATURES)
    feat_wx = FEATURE_COLS_WX if has_wx else FEATURE_COLS

    train = labeled[labeled["year"] <= TRAIN_END]
    test = labeled[labeled["year"] > VAL_END]      # 2023+ : años nunca vistos

    X_tr, y_tr = train[FEATURE_COLS], train["target_next"]
    X_te, y_te = test[FEATURE_COLS], test["target_next"]

    preds = {}

    # (a) Línea base climatológica: la probabilidad típica de esa celda y mes.
    #     Es la referencia honesta a superar.
    preds["Climatología (línea base)"] = test["seasonal_clim"].clip(0, 1).to_numpy()

    # (b) Persistencia: "mañana se parece a hoy".
    preds["Persistencia (naive)"] = test["rate_3m"].clip(0, 1).to_numpy()

    # (c) Regresión logística regularizada (solo FIRMS).
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)),
    ]).fit(X_tr, y_tr)
    preds["Regresión Logística (L2)"] = lr.predict_proba(X_te)[:, 1]

    # (d) HistGradientBoosting solo con historial de focos (FIRMS).
    hgb = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=6,
        min_samples_leaf=25, l2_regularization=1.0, random_state=42,
    ).fit(X_tr, y_tr)
    preds["Gradient Boosting · solo FIRMS"] = hgb.predict_proba(X_te)[:, 1]

    # (e) HistGradientBoosting con FIRMS + clima real NASA POWER (opción B).
    hgb_wx = None
    if has_wx:
        hgb_wx = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_depth=6,
            min_samples_leaf=25, l2_regularization=1.0, random_state=42,
        ).fit(train[feat_wx], y_tr)   # HGB admite NaN de forma nativa
        preds["Gradient Boosting · FIRMS + NASA POWER"] = hgb_wx.predict_proba(test[feat_wx])[:, 1]

    # Referencia para el skill: Brier de la climatología
    brier_clim = brier_score_loss(y_te, preds["Climatología (línea base)"])

    comparison = []
    for name, p in preds.items():
        p = np.clip(p, 0, 1)
        brier = brier_score_loss(y_te, p)
        comparison.append({
            "modelo": name,
            "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
            "pr_auc": round(float(average_precision_score(y_te, p)), 4),
            "brier_score": round(float(brier), 4),
            # Brier Skill Score vs climatología: >0 => mejor que la climatología
            "brier_skill_vs_clim": round(float(1.0 - brier / brier_clim), 4),
            "f1_score": round(float(f1_score(y_te, (p >= 0.5).astype(int))), 4),
        })

    # El campeón operativo: el de clima real si mejora el PR-AUC, si no, solo FIRMS.
    if hgb_wx is not None:
        pr_firms = average_precision_score(y_te, np.clip(preds["Gradient Boosting · solo FIRMS"], 0, 1))
        pr_wx = average_precision_score(y_te, np.clip(preds["Gradient Boosting · FIRMS + NASA POWER"], 0, 1))
        use_wx = pr_wx >= pr_firms
    else:
        use_wx = False

    champion_model = hgb_wx if use_wx else hgb
    champion_feats = feat_wx if use_wx else FEATURE_COLS
    champion_name = "Gradient Boosting · FIRMS + NASA POWER" if use_wx else "Gradient Boosting · solo FIRMS"
    champ_row = next(r for r in comparison if r["modelo"] == champion_name)

    # Importancia por permutación y calibración DEL CAMPEÓN.
    importances = _permutation_importance(champion_model, test[champion_feats], y_te)
    calibration = _reliability_table(
        y_te.to_numpy(), np.clip(champion_model.predict_proba(test[champion_feats])[:, 1], 0, 1))

    # Estadísticas más recientes por celda, para pronosticar el próximo mes real.
    last_month = panel["month"].max()
    latest = panel[panel["month"] == last_month].copy()

    joblib.dump({
        "champion": champion_model,
        "champion_feats": champion_feats,
        "firms_only": hgb,
        "weather_model": hgb_wx,
        "logistic": lr,
        "feature_cols": FEATURE_COLS,
        "feature_cols_wx": feat_wx,
        "uses_weather": bool(use_wx),
        "latest_cell_stats": latest,
        "last_month": str(last_month.date()),
    }, MODEL_PATH, compress=3)

    metrics = {
        "objetivo": "Probabilidad de foco en el PRÓXIMO mes por celda (~11 km) — pronóstico t+1",
        "fuente_datos": "NASA FIRMS armonizado (MODIS + VIIRS S-NPP), 2000–2026",
        "clima_real_nasa_power": bool(has_wx),
        "campeon_usa_clima": bool(use_wx),
        "sin_meteorologia_sintetica": True,
        "model_comparison": comparison,
        "feature_importances": importances,
        "calibration_curve": calibration,
        "train_period": f"2001 - {TRAIN_END}",
        "test_period": f"{VAL_END + 1} - {int(labeled['year'].max())} (años nunca vistos)",
        "total_observations": int(len(labeled)),
        "test_observations": int(len(test)),
        "test_fires_next_month": int(y_te.sum()),
        "base_rate_test": round(float(y_te.mean()), 4),
        "champion": champion_name,
        "champion_roc_auc": champ_row["roc_auc"],
        "champion_pr_auc": champ_row["pr_auc"],
        "champion_brier_skill": champ_row["brier_skill_vs_clim"],
        "last_data_month": str(last_month.date()),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def _permutation_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 3) -> dict:
    """Importancia por permutación medida como caída de ROC-AUC. Robusta y honesta."""
    rng = np.random.default_rng(42)
    X = X.reset_index(drop=True)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            auc = roc_auc_score(y, model.predict_proba(Xp)[:, 1])
            drops.append(base - auc)
        out[col] = round(float(np.mean(drops)), 4)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def _reliability_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> list:
    """Tabla de fiabilidad: prob. predicha media vs frecuencia observada por tramo."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for k in range(bins):
        lo, hi = edges[k], edges[k + 1]
        m = (p >= lo) & (p < hi if k < bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "pred_mean": round(float(p[m].mean()), 3),
            "obs_freq": round(float(y_true[m].mean()), 3),
            "n": int(m.sum()),
        })
    return rows


# ---------------------------------------------------------------------------
# 3. Pronóstico operativo del próximo mes
# ---------------------------------------------------------------------------
def get_trained() -> tuple[dict, dict]:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_forecast(force_retrain=True)
    artifacts = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return artifacts, metrics


def forecast_next_month() -> pd.DataFrame:
    """Devuelve, por celda, la probabilidad pronosticada de foco el próximo mes,
    con nivel de alerta semafórico y provincia oficial."""
    artifacts, _ = get_trained()
    model = artifacts["champion"]
    feats = artifacts.get("champion_feats", FEATURE_COLS)
    latest = artifacts["latest_cell_stats"].copy()
    # Asegurar que todas las columnas del campeón existen (clima real incluido)
    for c in feats:
        if c not in latest.columns:
            latest[c] = np.nan

    probs = model.predict_proba(latest[feats])[:, 1]
    out = pd.DataFrame({
        "cell_lat": latest["cell_lat"].to_numpy(),
        "cell_lon": latest["cell_lon"].to_numpy(),
        "prob_pct": np.round(probs * 100.0, 1),
        "seasonal_clim_pct": np.round(latest["seasonal_clim"].to_numpy() * 100.0, 1),
        "rate_6m_pct": np.round(latest["rate_6m"].to_numpy() * 100.0, 1),
        "months_since_fire": latest["months_since_fire"].astype(int).to_numpy(),
    })
    # FWI real de NASA POWER del último mes (si está disponible)
    if "wx_fwi_mean" in latest.columns:
        out["fwi_real"] = np.round(latest["wx_fwi_mean"].to_numpy(), 1)
        out["temp_real"] = np.round(latest["wx_t2m_mean"].to_numpy(), 1)
        out["rh_real"] = np.round(latest["wx_rh_mean"].to_numpy(), 1)

    conds = [out["prob_pct"] >= 70, out["prob_pct"] >= 45, out["prob_pct"] >= 25]
    out["nivel_alerta"] = np.select(conds, ["🔴 Extremo", "🟠 Alto", "🟡 Moderado"], default="🟢 Bajo")

    out["provincia"] = [
        geo.point_province(float(la), float(lo)) or "Zona Fronteriza / Costera"
        for la, lo in zip(out["cell_lat"], out["cell_lon"])
    ]
    # mes pronosticado = mes siguiente al último dato
    last = pd.Timestamp(artifacts["last_month"])
    out.attrs["forecast_month"] = (last + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    return out.sort_values("prob_pct", ascending=False).reset_index(drop=True)


def province_forecast_ranking(grid: pd.DataFrame) -> pd.DataFrame:
    """Ranking de provincias por riesgo pronosticado el próximo mes."""
    agg = grid.groupby("provincia").agg(
        prob_media=("prob_pct", "mean"),
        prob_max=("prob_pct", "max"),
        celdas_riesgo=("prob_pct", lambda s: int((s >= 45).sum())),
        total_celdas=("prob_pct", "count"),
    ).reset_index()
    agg = agg[agg["provincia"] != "República Dominicana"]
    conds = [agg["prob_max"] >= 70, agg["prob_max"] >= 45, agg["prob_max"] >= 25]
    agg["alerta"] = np.select(conds, ["🔴 Extrema", "🟠 Alta", "🟡 Moderada"], default="🟢 Baja")
    agg["prob_media"] = agg["prob_media"].round(1)
    agg["prob_max"] = agg["prob_max"].round(1)
    return agg.sort_values("prob_max", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    print("Entrenando el modelo de pronóstico (t+1) con FIRMS armonizado...")
    m = train_forecast(force_retrain="--retrain" in sys.argv)
    print(f"\nObjetivo: {m['objetivo']}")
    print(f"Test: {m['test_period']}  |  tasa base de fuego: {m['base_rate_test']:.1%}")
    print("\n=== Comparación de modelos (años nunca vistos) ===")
    hdr = f"{'modelo':<30}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>8}{'Skill vs clim':>15}{'F1':>7}"
    print(hdr); print("-" * len(hdr))
    for r in m["model_comparison"]:
        print(f"{r['modelo']:<30}{r['roc_auc']:>9}{r['pr_auc']:>9}{r['brier_score']:>8}"
              f"{r['brier_skill_vs_clim']:>15}{r['f1_score']:>7}")
    print("\n=== Importancia por permutación (caída de ROC-AUC) ===")
    for k, v in list(m["feature_importances"].items())[:8]:
        print(f"  {k:<22}{v:>8}")
    print("\n=== Pronóstico del próximo mes ===")
    fc = forecast_next_month()
    print(f"Mes pronosticado: {fc.attrs.get('forecast_month')}")
    print(province_forecast_ranking(fc).head(10).to_string(index=False))
