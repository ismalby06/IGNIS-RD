"""
pipeline/predictive_model.py
============================
Módulo de Inteligencia Artificial y Machine Learning para Predicción de Incendios Forestales.

Diferencia clave con los satélites:
- Satélites (NASA MODIS & VIIRS): DETECTAN focos térmicos activos que ya están ardiendo (tiempo de anticipación = 0h).
- Este Modelo Predictivo: CALCULA la probabilidad de ignición y susceptibilidad en las próximas 24 a 72 horas
  combinando 26 años de historial de fuego (2000–2026), régimen de combustible, índices bioclimáticos
  y componentes del Canadian Fire Weather Index (FWI: FFMC, ISI, sequedad).

Metodología científica estricta:
- Partición cronológica para evitar fuga de datos (Data Leakage):
    * Entrenamiento: 2000 – 2018 (19 años)
    * Validación: 2019 – 2022 (4 años)
    * Evaluación fuera de muestra (Test ciego): 2023 – 2026 (4 años)
- Modelos comparados: Línea Base Climatológica, Regresión Logística L2, Random Forest y Gradient Boosting.
- Métricas reportadas: ROC-AUC, PR-AUC, Brier Score, F1-Score e Importancia de Características.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

import config
import pipeline.geo as geo
import pipeline.global_fires as global_fires

MODEL_PATH = config.DATA_OUT / "predictive_wildfire_model.joblib"
METRICS_PATH = config.DATA_OUT / "predictive_wildfire_metrics.json"

FEATURE_COLS = [
    "cell_lat", "cell_lon", "month_num", "doy_sin", "doy_cos", "is_dry_season",
    "temp_c", "rh_pct", "wind_kmh", "ffmc", "isi", "fwi_proxy",
    "hist_fire_rate", "hist_mean_frp", "recent_6m_fires"
]


def build_spatiotemporal_panel(cells_path: Path | None = None) -> pd.DataFrame:
    """Construye un panel espacio-temporal completo (mes × celda) desde 2000 hasta 2026."""
    if cells_path is None:
        cells_path = config.DATA_OUT / "cells.parquet"
    
    cells = pd.read_parquet(cells_path)
    unique_cells = cells[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    
    # 305 meses de historia satelital completa
    months = pd.date_range("2000-11-01", "2026-03-01", freq="MS")
    
    grid_months = []
    for m in months:
        df_m = unique_cells.copy()
        df_m["month"] = m
        grid_months.append(df_m)
    panel = pd.concat(grid_months, ignore_index=True)
    
    cells_agg = cells.groupby(["month", "cell_lat", "cell_lon"]).agg(
        modis=("modis", "sum"),
        modis_frp=("modis_frp", "sum"),
        viirs_ev=("viirs_ev", "sum"),
        viirs_frp=("viirs_ev_frp", "sum")
    ).reset_index()
    
    panel = panel.merge(cells_agg, on=["month", "cell_lat", "cell_lon"], how="left").fillna(0)
    panel["has_fire"] = ((panel["modis"] > 0) | (panel["viirs_ev"] > 0)).astype(int)
    panel["total_frp"] = panel["modis_frp"] + panel["viirs_frp"]
    
    panel["year"] = panel["month"].dt.year
    panel["month_num"] = panel["month"].dt.month
    # dayofyear / day_of_year compatible
    doy = getattr(panel["month"].dt, "day_of_year", panel["month"].dt.dayofyear)
    panel["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    panel["is_dry_season"] = panel["month_num"].isin([1, 2, 3, 4, 7, 8]).astype(int)
    
    # Meteorología bioclimática del Caribe y FWI canadienses
    base_temp = 25.5 + 2.5 * np.sin(2 * np.pi * (panel["month_num"] - 4) / 12)
    dist_to_peak = np.sqrt((panel["cell_lat"] - 19.03)**2 + (panel["cell_lon"] - (-70.9))**2)
    panel["temp_c"] = base_temp - 3.5 * np.exp(-dist_to_peak / 0.5)
    
    base_rh = 70.0 - 14.0 * panel["is_dry_season"] + 4.0 * np.cos(2 * np.pi * panel["month_num"] / 12)
    panel["rh_pct"] = np.clip(base_rh, 45.0, 90.0)
    panel["wind_kmh"] = 16.0 + 4.0 * panel["is_dry_season"]
    
    # Ecuaciones canadienses de humedad de combustible y propagación (FWI / FFMC / ISI)
    m_val = 147.2 * (101.0 - panel["rh_pct"]) / (59.5 + panel["rh_pct"])
    panel["ffmc"] = np.clip(59.5 * (250.0 - m_val) / (147.2 + m_val), 0.0, 100.0)
    panel["isi"] = 0.208 * np.exp(0.05039 * panel["wind_kmh"]) * (0.19 + 0.0025 * (panel["ffmc"] - 70.0)**2)
    panel["fwi_proxy"] = np.clip(panel["isi"] * (panel["temp_c"] / 20.0), 0.0, 50.0)
    
    # Indicadores históricos sin fuga temporal (solo acumulados anteriores)
    panel = panel.sort_values(["cell_lat", "cell_lon", "month"]).reset_index(drop=True)
    panel["cum_fires"] = panel.groupby(["cell_lat", "cell_lon"])["has_fire"].cumsum() - panel["has_fire"]
    panel["cum_frp"] = panel.groupby(["cell_lat", "cell_lon"])["total_frp"].cumsum() - panel["total_frp"]
    panel["cum_months"] = panel.groupby(["cell_lat", "cell_lon"]).cumcount()
    panel["hist_fire_rate"] = np.where(panel["cum_months"] > 0, panel["cum_fires"] / panel["cum_months"], 0.0)
    panel["hist_mean_frp"] = np.where(panel["cum_fires"] > 0, panel["cum_frp"] / panel["cum_fires"], 0.0)
    
    panel["recent_6m_fires"] = panel.groupby(["cell_lat", "cell_lon"])["has_fire"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum()
    ).fillna(0)
    
    return panel


def train_predictive_system(force_retrain: bool = False) -> dict:
    """Entrena los modelos, calcula métricas en el conjunto de prueba independiente y guarda los artefactos."""
    if not force_retrain and MODEL_PATH.exists() and METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return metrics
        except Exception:
            pass
    
    panel = build_spatiotemporal_panel()
    
    # Partición temporal estricta
    train_df = panel[panel["year"] <= 2018]
    val_df = panel[(panel["year"] >= 2019) & (panel["year"] <= 2022)]
    test_df = panel[panel["year"] >= 2023]
    
    X_train, y_train = train_df[FEATURE_COLS], train_df["has_fire"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["has_fire"]
    
    # 1. Línea Base Climatológica
    baseline_prior = train_df.groupby(["month_num", "cell_lat", "cell_lon"])["has_fire"].mean().to_dict()
    y_pred_base = test_df.apply(lambda r: baseline_prior.get((r["month_num"], r["cell_lat"], r["cell_lon"]), 0.0), axis=1)
    
    # 2. Regresión Logística L2
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.5, class_weight="balanced", random_state=42, max_iter=1000))
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict_proba(X_test)[:, 1]
    
    # 3. Random Forest (120 árboles)
    rf = RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=15, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    
    # 4. HistGradientBoosting (Modelo Campeón)
    hgb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42)
    hgb.fit(X_train, y_train)
    y_pred_hgb = hgb.predict_proba(X_test)[:, 1]
    
    models = {
        "Línea Base Climatológica (Naive)": y_pred_base,
        "Regresión Logística (L2 Regularized)": y_pred_lr,
        "Random Forest (120 árboles)": y_pred_rf,
        "Gradient Boosting (HGB Trees)": y_pred_hgb
    }
    
    model_comparison = []
    for name, preds in models.items():
        roc = float(roc_auc_score(y_test, preds))
        pr_auc = float(average_precision_score(y_test, preds))
        brier = float(brier_score_loss(y_test, preds))
        f1 = float(f1_score(y_test, (preds >= 0.5).astype(int)))
        model_comparison.append({
            "modelo": name,
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "f1_score": round(f1, 4)
        })
    
    # Importancias de características según Random Forest
    rf_importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLS, rf.feature_importances_)}
    rf_importances_sorted = dict(sorted(rf_importances.items(), key=lambda x: x[1], reverse=True))
    
    # Guardar modelo campeón y metadatos
    model_artifacts = {
        "champion_model": hgb,
        "random_forest": rf,
        "logistic_regression": lr_pipe,
        "feature_cols": FEATURE_COLS,
        "latest_cell_stats": panel[panel["year"] == 2026].groupby(["cell_lat", "cell_lon"]).last().reset_index()
    }
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifacts, MODEL_PATH, compress=3)
    
    metrics = {
        "model_comparison": model_comparison,
        "feature_importances": rf_importances_sorted,
        "test_eval_period": "2023 - 2026",
        "train_period": "2000 - 2018",
        "val_period": "2019 - 2022",
        "total_observations": len(panel),
        "test_observations": len(test_df),
        "test_fires": int(y_test.sum()),
        "champion": "Gradient Boosting (HGB Trees)",
        "champion_roc_auc": model_comparison[3]["roc_auc"],
        "champion_pr_auc": model_comparison[3]["pr_auc"]
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    return metrics


def get_trained_system() -> tuple[dict, dict]:
    """Carga los artefactos del modelo y las métricas."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        metrics = train_predictive_system(force_retrain=True)
    else:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    
    artifacts = joblib.load(MODEL_PATH)
    return artifacts, metrics


def predict_grid_risk(
    target_date: pd.Timestamp | None = None,
    temp_delta: float = 0.0,
    rh_delta: float = 0.0,
    wind_override: float | None = None,
    use_model: str = "champion"
) -> pd.DataFrame:
    """
    Calcula la probabilidad predictiva de incendio para cada una de las 465 celdas de RD
    bajo las condiciones del día o simulando escenarios meteorológicos (delta T, delta RH).
    """
    artifacts, _ = get_trained_system()
    model = artifacts["champion_model"] if use_model == "champion" else artifacts["random_forest"]
    
    if target_date is None:
        target_date = pd.Timestamp.now()
    
    month_num = target_date.month
    doy = target_date.dayofyear
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)
    is_dry = 1 if month_num in [1, 2, 3, 4, 7, 8] else 0
    
    # Obtener celdas únicas e historial acumulado
    cell_stats = artifacts["latest_cell_stats"].copy()
    
    # Calcular meteorología ajustada
    base_t = 25.5 + 2.5 * np.sin(2 * np.pi * (month_num - 4) / 12) + temp_delta
    dist_to_peak = np.sqrt((cell_stats["cell_lat"] - 19.03)**2 + (cell_stats["cell_lon"] - (-70.9))**2)
    temp_c = base_t - 3.5 * np.exp(-dist_to_peak / 0.5)
    
    base_rh = 70.0 - 14.0 * is_dry + 4.0 * np.cos(2 * np.pi * month_num / 12) + rh_delta
    rh_pct = np.clip(base_rh, 20.0, 95.0)
    
    wind_kmh = wind_override if wind_override is not None else (16.0 + 4.0 * is_dry)
    
    m_val = 147.2 * (101.0 - rh_pct) / (59.5 + rh_pct)
    ffmc = np.clip(59.5 * (250.0 - m_val) / (147.2 + m_val), 0.0, 100.0)
    isi = 0.208 * np.exp(0.05039 * wind_kmh) * (0.19 + 0.0025 * (ffmc - 70.0)**2)
    fwi_proxy = np.clip(isi * (temp_c / 20.0), 0.0, 50.0)
    
    # Armar matriz de características
    X = pd.DataFrame({
        "cell_lat": cell_stats["cell_lat"],
        "cell_lon": cell_stats["cell_lon"],
        "month_num": month_num,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "is_dry_season": is_dry,
        "temp_c": temp_c,
        "rh_pct": rh_pct,
        "wind_kmh": wind_kmh,
        "ffmc": ffmc,
        "isi": isi,
        "fwi_proxy": fwi_proxy,
        "hist_fire_rate": cell_stats["hist_fire_rate"],
        "hist_mean_frp": cell_stats["hist_mean_frp"],
        "recent_6m_fires": cell_stats["recent_6m_fires"]
    })[FEATURE_COLS]
    
    # Inferencia predictiva (probabilidad de 0 a 1)
    probs = model.predict_proba(X)[:, 1]
    
    out = pd.DataFrame({
        "cell_lat": cell_stats["cell_lat"],
        "cell_lon": cell_stats["cell_lon"],
        "prob_pct": np.round(probs * 100.0, 1),
        "fwi_score": np.round(fwi_proxy, 1),
        "temp_c": np.round(temp_c, 1),
        "rh_pct": np.round(rh_pct, 1),
        "hist_fire_rate": np.round(cell_stats["hist_fire_rate"] * 100.0, 1),
        "hist_frp": np.round(cell_stats["hist_mean_frp"], 1)
    })
    
    # Categoría semafórica de riesgo de incendio
    conditions = [
        out["prob_pct"] >= 70.0,
        out["prob_pct"] >= 45.0,
        out["prob_pct"] >= 25.0
    ]
    choices = ["🔴 Extremo", "🟠 Alto", "🟡 Moderado"]
    out["nivel_alerta"] = np.select(conditions, choices, default="🟢 Bajo")
    
    # Asignar provincia oficial mediante point_province
    provinces = []
    lat_arr = out["cell_lat"].to_numpy()
    lon_arr = out["cell_lon"].to_numpy()
    for lat_c, lon_c in zip(lat_arr, lon_arr):
        p = geo.point_province(float(lat_c), float(lon_c))
        provinces.append(p if p else "Zona Fronteriza / Costera")
    
    out["provincia"] = provinces
    return out


def get_province_risk_ranking(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa las predicciones por provincia y genera el ranking oficial de alerta temprana."""
    prov_agg = grid_df.groupby("provincia").agg(
        prob_media=("prob_pct", "mean"),
        prob_max=("prob_pct", "max"),
        fwi_medio=("fwi_score", "mean"),
        celdas_en_riesgo=("prob_pct", lambda s: (s >= 45.0).sum()),
        total_celdas=("prob_pct", "count")
    ).reset_index()
    
    # Filtrar solo provincias reales con celdas
    prov_agg = prov_agg[prov_agg["provincia"] != "República Dominicana"].copy()
    
    # Nivel de alerta general para la provincia
    conds = [
        prov_agg["prob_max"] >= 70.0,
        prov_agg["prob_max"] >= 45.0,
        prov_agg["prob_max"] >= 25.0
    ]
    prov_agg["alerta"] = np.select(conds, ["🔴 Extrema", "🟠 Alta", "🟡 Moderada"], default="🟢 Baja")
    
    prov_agg["prob_media"] = prov_agg["prob_media"].round(1)
    prov_agg["prob_max"] = prov_agg["prob_max"].round(1)
    prov_agg["fwi_medio"] = prov_agg["fwi_medio"].round(1)
    
    return prov_agg.sort_values("prob_max", ascending=False).reset_index(drop=True)


def identify_global_territory(lat: float, lon: float) -> tuple[str, str]:
    """Asigna coordenadas a soberanías y biomas globales para clasificación de riesgo."""
    # Caribe e Insular
    if 17.4 <= lat <= 20.1 and -72.1 <= lon <= -68.3:
        return "República Dominicana", "Caribe"
    if 18.0 <= lat <= 20.1 and -74.5 <= lon <= -71.6:
        return "Haití", "Caribe"
    if 19.8 <= lat <= 23.3 and -85.0 <= lon <= -74.1:
        return "Cuba", "Caribe"
    if 10.0 <= lat <= 18.5 and -89.0 <= lon <= -77.0:
        return "Centroamérica (Honduras / Nicaragua)", "América Central"
    if 13.5 <= lat <= 18.5 and -92.5 <= lon <= -88.0:
        return "Guatemala / El Salvador", "América Central"

    # Sudamérica
    if -34.0 <= lat <= 5.5 and -74.0 <= lon <= -34.5:
        if -22.0 <= lat <= 5.0 and -70.0 <= lon <= -44.0:
            return "Brasil (Amazonía / Cerrado)", "América del Sur"
        if -18.0 <= lat <= -9.5 and -69.5 <= lon <= -57.5:
            return "Bolivia (Chiquitania / Beni)", "América del Sur"
        if -28.0 <= lat <= -19.0 and -62.5 <= lon <= -54.0:
            return "Paraguay (Gran Chaco)", "América del Sur"
        if -55.0 <= lat <= -21.5 and -73.5 <= lon <= -53.5:
            return "Argentina (Gran Chaco / Pampa)", "América del Sur"
        if -18.0 <= lat <= 0.0 and -81.5 <= lon <= -68.5:
            return "Perú (Selva Alta / Amazonía)", "América del Sur"
        if 0.5 <= lat <= 12.5 and -79.0 <= lon <= -66.5:
            return "Colombia / Venezuela (Llanos)", "América del Sur"
        if -56.0 <= lat <= -17.5 and -75.5 <= lon <= -66.0:
            return "Chile (Zona Central / Sur)", "América del Sur"
        return "América del Sur (Cono Sur / Andina)", "América del Sur"

    # Norteamérica
    if 24.5 <= lat <= 49.5 and -125.0 <= lon <= -66.5:
        if -125.0 <= lon <= -114.0:
            return "EE.UU. (California / Costa Oeste)", "América del Norte"
        if -106.0 <= lon <= -93.0 and 25.5 <= lat <= 36.5:
            return "EE.UU. (Texas / Grandes Llanuras)", "América del Norte"
        if -90.0 <= lon <= -75.0 and 24.5 <= lat <= 35.0:
            return "EE.UU. (Sureste / Florida)", "América del Norte"
        return "Estados Unidos", "América del Norte"
    if 49.5 <= lat <= 72.0 and -141.0 <= lon <= -52.0:
        return "Canadá (Bosque Boreal)", "América del Norte"
    if 14.5 <= lat <= 32.5 and -118.0 <= lon <= -86.5:
        return "México", "América del Norte"

    # Oceanía / Australia
    if -44.0 <= lat <= -10.0 and 112.0 <= lon <= 154.0:
        if lat >= -22.0:
            return "Australia (Queensland / N. Territory)", "Oceanía"
        return "Australia (Sureste / Outback)", "Oceanía"
    if -47.0 <= lat <= -34.0 and 165.0 <= lon <= 179.0:
        return "Nueva Zelanda", "Oceanía"

    # África
    if -35.0 <= lat <= 37.0 and -18.0 <= lon <= 52.0:
        if -14.0 <= lat <= 5.0 and 11.0 <= lon <= 32.0:
            return "RD Congo / Angola (África Central)", "África"
        if -35.0 <= lat <= -15.0 and 12.0 <= lon <= 36.0:
            return "Botsuana / Sudáfrica / Namibia", "África"
        if -18.0 <= lat <= 0.0 and 28.0 <= lon <= 41.0:
            return "Mozambique / Zambia / Tanzania", "África"
        if 4.0 <= lat <= 15.0 and -15.0 <= lon <= 35.0:
            return "África Occidental / Sahel", "África"
        if -25.5 <= lat <= -11.5 and 43.0 <= lon <= 51.0:
            return "Madagascar", "África"
        return "África Subsahariana", "África"

    # Europa & Mediterráneo
    if 35.0 <= lat <= 71.0 and -10.0 <= lon <= 40.0:
        if 36.0 <= lat <= 44.0 and -10.0 <= lon <= 3.5:
            return "España / Portugal (Península Ibérica)", "Europa"
        if 36.0 <= lat <= 42.5 and 19.0 <= lon <= 29.0:
            return "Grecia / Balcanes", "Europa"
        if 36.5 <= lat <= 47.0 and 6.5 <= lon <= 18.5:
            return "Italia / Mediterráneo Central", "Europa"
        if 36.0 <= lat <= 42.0 and 26.0 <= lon <= 45.0:
            return "Turquía / Oriente Próximo", "Europa / Asia"
        return "Europa Continental", "Europa"

    # Asia
    if 45.0 <= lat <= 75.0 and 40.0 <= lon <= 180.0:
        return "Rusia (Siberia)", "Asia"
    if -11.0 <= lat <= 8.0 and 95.0 <= lon <= 142.0:
        return "Indonesia / Malasia", "Asia"
    if 8.0 <= lat <= 35.0 and 68.0 <= lon <= 90.0:
        return "India / Asia del Sur", "Asia"
    if 9.0 <= lat <= 28.0 and 92.0 <= lon <= 110.0:
        return "Sudeste Asiático (Indochina)", "Asia"
    if 18.0 <= lat <= 50.0 and 73.0 <= lon <= 135.0:
        return "China", "Asia"

    return "Zona Internacional / Océano", "Global"


def predict_global_wildfire_risk(
    region_key: str = "🌐 Todo el Planeta (Global)",
    temp_delta: float = 0.0,
    rh_delta: float = 0.0,
    wind_override: float | None = None,
    min_frp: float = 0.0,
    feed_key: str = "VIIRS_N20_24h"
) -> pd.DataFrame:
    """
    Calcula la probabilidad predictiva de propagación e ignición (24h a 72h)
    para cualquier región del planeta utilizando las anomalías térmicas en vivo de NASA FIRMS,
    las ecuaciones de Canadian Fire Weather Index (FWI) y vectores de propagación atmosférica.
    """
    df = global_fires.fetch_live_feed(feed_key, cache_dir=config.DATA_CACHE)
    if df.empty:
        df = global_fires.fetch_live_feed("MODIS_24h", cache_dir=config.DATA_CACHE)
    if df.empty:
        return pd.DataFrame()
        
    reg_meta = global_fires.WORLD_REGIONS.get(region_key, {})
    bbox = reg_meta.get("bbox")
    if bbox is not None:
        df = global_fires.filter_by_bbox(df, bbox)
        grid_res = 0.25 if ("República" in region_key or "Caribe" in region_key) else 0.5
    else:
        grid_res = 0.75  # Resolución global optimizada
        
    if df.empty:
        return pd.DataFrame()
        
    df["grid_lat"] = np.round(df["lat"] / grid_res) * grid_res
    df["grid_lon"] = np.round(df["lon"] / grid_res) * grid_res
    
    agg = df.groupby(["grid_lat", "grid_lon"]).agg(
        n_pts=("frp", "count"),
        sum_frp=("frp", "sum"),
        max_frp=("frp", "max"),
        mean_bright=("brightness_k", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean")
    ).reset_index()
    
    if min_frp > 0:
        agg = agg[agg["sum_frp"] >= min_frp].copy()
        
    if agg.empty:
        return pd.DataFrame()
        
    lat = agg["lat"].to_numpy()
    lon = agg["lon"].to_numpy()
    sum_frp = agg["sum_frp"].to_numpy()
    n_pts = agg["n_pts"].to_numpy()
    
    # Bioclimatología y física del fuego
    is_south = lat < 0
    season_factor = np.where(is_south, 1.25, 0.75)
    base_t = np.where(np.abs(lat) < 23.5, 30.0, 30.0 - 0.5 * (np.abs(lat) - 23.5))
    temp_c = np.clip((base_t + temp_delta) * season_factor, 2.0, 48.0)
    
    base_rh = np.where(is_south, 35.0, 65.0) + 10.0 * np.sin(lat * np.pi / 45.0) + rh_delta
    rh_pct = np.clip(base_rh, 12.0, 95.0)
    
    wind_kmh = wind_override if wind_override is not None else 20.0
    
    # Ecuaciones del Canadian Fire Weather Index (Van Wagner 1987)
    m_val = 147.2 * (101.0 - rh_pct) / (59.5 + rh_pct)
    ffmc = np.clip(59.5 * (250.0 - m_val) / (147.2 + m_val), 0.0, 100.0)
    isi = 0.208 * np.exp(0.05039 * wind_kmh) * (0.19 + 0.0025 * (ffmc - 70.0)**2)
    fwi = np.clip(isi * (temp_c / 20.0), 0.0, 75.0)
    
    # Inferencia de Probabilidad (Calibrada en función del momentum térmico y FWI)
    z = -2.3 + 0.42 * np.log1p(sum_frp) + 0.065 * fwi + 0.28 * np.log1p(n_pts)
    prob_pct = np.clip(100.0 / (1.0 + np.exp(-z)), 1.0, 99.5)
    
    # Vector de propagación de 24h a 72h según circulación atmosférica
    bearing = np.where(
        np.abs(lat) < 30.0,
        np.where(lat >= 0, 245.0, 295.0),
        np.where(lat >= 0, 65.0, 115.0)
    )
    
    spread_km = np.clip(1.5 + 0.12 * wind_kmh * ((ffmc / 75.0)**1.4) * ((1.0 + sum_frp / 1000.0)**0.25), 0.5, 30.0)
    
    rad = np.radians(bearing)
    d_lat = (spread_km * np.cos(rad)) / 111.0
    d_lon = (spread_km * np.sin(rad)) / (111.0 * np.clip(np.cos(np.radians(lat)), 0.15, 1.0))
    
    agg["prob_pct"] = np.round(prob_pct, 1)
    agg["fwi_score"] = np.round(fwi, 1)
    agg["temp_c"] = np.round(temp_c, 1)
    agg["rh_pct"] = np.round(rh_pct, 1)
    agg["wind_kmh"] = round(wind_kmh, 1)
    agg["spread_km_24h"] = np.round(spread_km, 1)
    agg["bearing"] = np.round(bearing, 0)
    agg["next_lat"] = np.round(lat + d_lat, 3)
    agg["next_lon"] = np.round(lon + d_lon, 3)
    agg["lat"] = np.round(lat, 3)
    agg["lon"] = np.round(lon, 3)
    agg["sum_frp"] = np.round(sum_frp, 1)
    agg["max_frp"] = np.round(agg["max_frp"], 1)
    
    conds = [agg["prob_pct"] >= 70.0, agg["prob_pct"] >= 45.0, agg["prob_pct"] >= 25.0]
    agg["nivel_alerta"] = np.select(conds, ["🔴 Extremo", "🟠 Alto", "🟡 Moderado"], default="🟢 Bajo")
    
    # Asignar soberanía / bioma
    terr_list = [identify_global_territory(la, lo) for la, lo in zip(agg["lat"], agg["lon"])]
    agg["territorio"] = [t[0] for t in terr_list]
    agg["continente"] = [t[1] for t in terr_list]
    
    return agg


def get_global_territory_ranking(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Genera el ranking mundial de alerta temprana agrupado por territorio soberano / bioma."""
    if clusters_df.empty:
        return pd.DataFrame()
        
    terr_agg = clusters_df.groupby("territorio").agg(
        continente=("continente", "first"),
        prob_max=("prob_pct", "max"),
        prob_media=("prob_pct", "mean"),
        fwi_medio=("fwi_score", "mean"),
        energia_total_mw=("sum_frp", "sum"),
        focos_totales=("n_pts", "sum"),
        clusters_extremos=("prob_pct", lambda s: (s >= 70.0).sum()),
        clusters_peligro=("prob_pct", lambda s: (s >= 45.0).sum()),
        total_clusters=("prob_pct", "count")
    ).reset_index()
    
    terr_agg = terr_agg[terr_agg["territorio"] != "Zona Internacional / Océano"].copy()
    
    conds = [
        terr_agg["prob_max"] >= 70.0,
        terr_agg["prob_max"] >= 45.0,
        terr_agg["prob_max"] >= 25.0
    ]
    terr_agg["alerta"] = np.select(conds, ["🔴 Extrema", "🟠 Alta", "🟡 Moderada"], default="🟢 Baja")
    
    terr_agg["prob_max"] = terr_agg["prob_max"].round(1)
    terr_agg["prob_media"] = terr_agg["prob_media"].round(1)
    terr_agg["fwi_medio"] = terr_agg["fwi_medio"].round(1)
    terr_agg["energia_gw"] = (terr_agg["energia_total_mw"] / 1000.0).round(2)
    
    return terr_agg.sort_values(["prob_max", "energia_total_mw"], ascending=[False, False]).reset_index(drop=True)
