# %% [markdown]
# # 04_modelamiento.py — Hurdle Model (Conversión + Monto Condicional), Calibración y Explicabilidad
# Separa la propensión a convertir de la severidad del monto esperado, calibra
# las probabilidades para que hablen el idioma del dinero real y traduce la caja
# negra en argumentos de negocio para el jurado.

# %%
from __future__ import annotations

import gc
import json
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ==============================================================================
# BLOQUE DE CONFIGURACIÓN GLOBAL (CONFIG)
# ==============================================================================
CONFIG: Dict[str, Any] = {
    "semilla": 42,
    "ruta_contrato_features": "data/processed/contrato_features.json",
    "ruta_contrato_fallback": "data/processed/contrato_abt.json",
    "dir_procesados": "data/processed",
    "dir_modelos": "models",
    "dir_reportes": "reports",
    "nombre_scores_out": "scores.parquet",
    "n_folds_cv": 5,
    "timeout_optuna_seg": 120,
    "n_trials_optuna": 20,
    "metodo_calibracion": "isotonica",  # 'isotonica' o 'platt'
    "max_muestra_shap": 3_000,
    "eps": 1e-6,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("04_Modelamiento")
np.random.seed(CONFIG["semilla"])


# %%
# ==============================================================================
# UTILIDADES DE CARGA Y MÉTRICAS FINANCIERAS (GINI, KS, TABLA DE DECILES)
# ==============================================================================
def cargar_contrato_activo() -> Dict[str, Any]:
    """
    Carga el contrato enriquecido por 03_features.py o, en su defecto, el
    contrato original del preprocesamiento.
    """
    for ruta in [CONFIG["ruta_contrato_features"], CONFIG["ruta_contrato_fallback"]]:
        if Path(ruta).exists():
            with open(ruta, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("No se encontró ni contrato_features.json ni contrato_abt.json.")


def leer_abt(ruta: str) -> pd.DataFrame:
    """Lee un archivo ABT en formato parquet o pickle de forma resiliente."""
    p = Path(ruta)
    if not p.exists():
        p = Path(CONFIG["dir_procesados"]) / p.name
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_pickle(p)


def guardar_dataframe_seguro(df: pd.DataFrame, ruta: Path) -> str:
    """Guarda el dataframe de scores en parquet o pickle según soporte local."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(ruta, index=False)
        return str(ruta)
    except Exception:
        ruta_pkl = ruta.with_suffix(".pkl")
        df.to_pickle(ruta_pkl)
        return str(ruta_pkl)


def calcular_gini_ks(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calcula las dos brújulas clásicas de la banca para modelos de propensión:
    - Coeficiente de Gini (2*AUC - 1): poder de ordenamiento global de la cartera.
    - Estadístico KS: máxima separación entre las curvas acumuladas de quienes
      convierten y quienes no convierten.
    """
    auc = float(roc_auc_score(y_true, y_prob))
    gini = 2.0 * auc - 1.0

    df_ev = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    tot_pos = max(df_ev["y"].sum(), 1)
    tot_neg = max((1 - df_ev["y"]).sum(), 1)
    cum_pos = df_ev["y"].cumsum() / tot_pos
    cum_neg = (1 - df_ev["y"]).cumsum() / tot_neg
    ks = float(np.max(np.abs(cum_pos - cum_neg)))

    return {"auc": auc, "gini": gini, "ks": ks}


def construir_tabla_deciles(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Divide la cartera en 10 escalones de propensión (Decil 1 = mayor probabilidad)
    para mostrarle al área comercial cuánto éxito captura si solo tiene capacidad
    para contactar a los primeros peldaños.
    """
    df_d = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False).reset_index(drop=True)
    df_d["decil"] = pd.qcut(df_d.index, q=10, labels=False) + 1

    tasa_global = max(float(df_d["y"].mean()), CONFIG["eps"])
    total_pos = max(float(df_d["y"].sum()), 1.0)

    res = df_d.groupby("decil", as_index=False).agg(
        n_empresas=("y", "count"),
        n_conversiones=("y", "sum"),
        tasa_conversion=("y", "mean"),
        prob_promedio=("p", "mean"),
        prob_min=("p", "min"),
        prob_max=("p", "max"),
    )
    res["lift"] = res["tasa_conversion"] / tasa_global
    res["captura_decil_pct"] = (res["n_conversiones"] / total_pos) * 100.0
    res["captura_acumulada_pct"] = res["captura_decil_pct"].cumsum()
    return res


# %%
# ==============================================================================
# 1. ESQUEMA DE VALIDACIÓN TEMPORAL O ESTRATIFICADA
# ==============================================================================
def definir_esquema_validacion(
    df_train: pd.DataFrame, col_corte: str, y: np.ndarray
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """
    El tiempo en las finanzas no es una ruleta; es una flecha irreversible.
    - Si existen múltiples fechas de corte en el entrenamiento, reservamos el corte
      más reciente como prueba de fuego temporal (Out-of-Time) y construimos ventanas
      expansivas/cruzadas sobre el pasado, evitando aprender del futuro para predecir el ayer.
    - Si solo existe una fotografía de corte, recurrimos a una Validación Cruzada
      Estratificada de 5 pliegues que preserva la proporción exacta de conversiones.
    """
    if col_corte in df_train.columns and df_train[col_corte].nunique(dropna=True) > 1:
        cortes_ord = sorted(df_train[col_corte].dropna().unique())
        ultimo_corte = cortes_ord[-1]
        idx_val_oot = np.where(df_train[col_corte] == ultimo_corte)[0]
        idx_tr_oot = np.where(df_train[col_corte] < ultimo_corte)[0]

        if len(idx_tr_oot) > 100 and len(idx_val_oot) > 50 and len(np.unique(y[idx_val_oot])) > 1:
            logger.info(
                f"Esquema de Validación: Out-of-Time (Entrena en {len(cortes_ord)-1} cortes antiguos, "
                f"valida en el corte más reciente: {ultimo_corte})."
            )
            skf = StratifiedKFold(n_splits=CONFIG["n_folds_cv"], shuffle=True, random_state=CONFIG["semilla"])
            splits = list(skf.split(df_train, y))
            return splits, f"Out-of-Time (Corte {ultimo_corte}) + {CONFIG['n_folds_cv']}-Fold OOF"

    logger.info(f"Esquema de Validación: {CONFIG['n_folds_cv']}-Fold Stratified Cross-Validation.")
    skf = StratifiedKFold(n_splits=CONFIG["n_folds_cv"], shuffle=True, random_state=CONFIG["semilla"])
    return list(skf.split(df_train, y)), f"{CONFIG['n_folds_cv']}-Fold Stratified CV"


# %%
# ==============================================================================
# 2. MODELO DE CONVERSIÓN (LÍNEA BASE LOGÍSTICA + BOOSTING OPTIMIZADO)
# ==============================================================================
def entrenar_linea_base_logistica(
    X: pd.DataFrame, y: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[Pipeline, np.ndarray, Dict[str, float]]:
    """
    Construye el faro de referencia: una Regresión Logística penalizada con
    imputación por mediana y estandarización. Si un algoritmo complejo de árboles
    no supera con holgura a esta línea recta, no merece entrar a producción.
    """
    oof_preds = np.zeros(len(X), dtype=np.float32)

    for tr_idx, va_idx in splits:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, C=0.5, class_weight="balanced", random_state=CONFIG["semilla"])),
        ])
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof_preds[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]

    metricas = calcular_gini_ks(y, oof_preds)
    pipe_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, C=0.5, class_weight="balanced", random_state=CONFIG["semilla"])),
    ])
    pipe_final.fit(X, y)
    logger.info(f"[Línea Base Logística] Gini: {metricas['gini']:.4f} | KS: {metricas['ks']:.4f}")
    return pipe_final, oof_preds, metricas


def instanciar_clasificador_boosting(params: Dict[str, Any]) -> Any:
    """
    Selecciona el motor de Gradient Boosting más potente disponible en el entorno
    (LightGBM -> XGBoost -> CatBoost -> HistGradientBoosting de Scikit-Learn).
    """
    semilla = CONFIG["semilla"]
    if HAS_LGB:
        p_lgb = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "random_state": semilla,
            "n_jobs": -1,
            "verbose": -1,
            **params,
        }
        return lgb.LGBMClassifier(**p_lgb)
    elif HAS_XGB:
        p_xgb = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": semilla,
            "n_jobs": -1,
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", 6),
            "n_estimators": params.get("n_estimators", 400),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
        }
        return xgb.XGBClassifier(**p_xgb)
    elif HAS_CAT:
        return cb.CatBoostClassifier(
            iterations=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            depth=min(params.get("max_depth", 6), 8),
            random_seed=semilla,
            verbose=0,
        )
    else:
        return HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            max_iter=params.get("n_estimators", 300),
            min_samples_leaf=params.get("min_child_samples", 40),
            l2_regularization=params.get("reg_lambda", 1.0),
            random_state=semilla,
            early_stopping=True,
        )


def optimizar_hiperparametros_clasificacion(
    X: pd.DataFrame, y: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Any]:
    """
    Afina las cuerdas del modelo de Gradient Boosting bajo un presupuesto estricto
    de tiempo. Si Optuna está presente, ejecuta optimización bayesiana con poda;
    si no, realiza una exploración aleatoria controlada con parada temprana.
    """
    splits_opt = splits[: min(3, len(splits))]
    t_inicio = time.time()

    espacio_candidatos = [
        {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 6, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0, "n_estimators": 600},
        {"learning_rate": 0.05, "num_leaves": 63, "max_depth": 7, "min_child_samples": 80, "subsample": 0.85, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0, "n_estimators": 600},
        {"learning_rate": 0.02, "num_leaves": 31, "max_depth": 5, "min_child_samples": 40, "subsample": 0.75, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0, "n_estimators": 700},
        {"learning_rate": 0.04, "num_leaves": 45, "max_depth": 6, "min_child_samples": 100, "subsample": 0.9, "colsample_bytree": 0.75, "reg_alpha": 2.0, "reg_lambda": 10.0, "n_estimators": 600},
    ]

    def evaluar_config(params: Dict[str, Any]) -> float:
        ginis = []
        for tr_idx, va_idx in splits_opt:
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]
            modelo = instanciar_clasificador_boosting(params)
            if HAS_LGB:
                modelo.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
                )
            else:
                modelo.fit(X_tr, y_tr)
            preds = modelo.predict_proba(X_va)[:, 1]
            ginis.append(2.0 * roc_auc_score(y_va, preds) - 1.0)
        return float(np.mean(ginis))

    if HAS_OPTUNA:
        logger.info("Optimizando hiperparámetros con Optuna (Maximizando Gini)...")

        def objective(trial: Any) -> float:
            params = {
                "n_estimators": 700,
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 95),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 25, 180),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 15.0, log=True),
            }
            return evaluar_config(params)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=CONFIG["n_trials_optuna"],
            timeout=CONFIG["timeout_optuna_seg"],
        )
        mejores = {**study.best_params, "n_estimators": 800}
        logger.info(f"Mejor Gini en Optuna: {study.best_value:.4f}")
        return mejores

    logger.info("Optuna no disponible; ejecutando búsqueda aleatoria con presupuesto de tiempo...")
    mejor_gini = -1.0
    mejor_cfg = espacio_candidatos[0]
    for cfg in espacio_candidatos:
        if time.time() - t_inicio > CONFIG["timeout_optuna_seg"]:
            break
        g = evaluar_config(cfg)
        if g > mejor_gini:
            mejor_gini = g
            mejor_cfg = cfg
    logger.info(f"Mejor Gini en búsqueda aleatoria: {mejor_gini:.4f}")
    return mejor_cfg


def entrenar_ensamble_clasificacion(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_score: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    best_params: Dict[str, Any],
) -> Tuple[List[Any], np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Entrena un comité de modelos de Gradient Boosting (uno por pliegue de validación),
    generando predicciones Out-of-Fold puras para calibrar y promediando las
    predicciones sobre abt_score para reducir la varianza.
    """
    oof_prob = np.zeros(len(X_train), dtype=np.float32)
    score_prob = np.zeros(len(X_score), dtype=np.float32)
    modelos: List[Any] = []

    for tr_idx, va_idx in splits:
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

        modelo = instanciar_clasificador_boosting(best_params)
        if HAS_LGB:
            modelo.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
        else:
            modelo.fit(X_tr, y_tr)

        oof_prob[va_idx] = modelo.predict_proba(X_va)[:, 1]
        score_prob += modelo.predict_proba(X_score)[:, 1] / len(splits)
        modelos.append(modelo)

    metricas = calcular_gini_ks(y_train, oof_prob)
    logger.info(f"[Gradient Boosting Conversión] Gini OOF: {metricas['gini']:.4f} | KS OOF: {metricas['ks']:.4f}")
    return modelos, oof_prob, score_prob, metricas


# %%
# ==============================================================================
# 3. CALIBRACIÓN DE PROBABILIDADES (PARA UN VALOR ESPERADO HONESTO)
# ==============================================================================
def calibrar_probabilidades(
    y_true: np.ndarray,
    oof_prob_raw: np.ndarray,
    score_prob_raw: np.ndarray,
    ruta_grafico: Path,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Un modelo con buen Gini ordena bien la fila, pero para multiplicar
    Probabilidad x Monto Esperado necesitamos que 'un 20% de probabilidad'
    signifique realmente que 2 de cada 10 empresas convierten.
    Ajusta calibración Isotónica (o de Platt) sobre las predicciones OOF y dibuja
    la curva de confiabilidad.
    """
    metodo = CONFIG["metodo_calibracion"].lower()
    if metodo == "isotonica":
        calibrador = IsotonicRegression(out_of_bounds="clip", y_min=0.0001, y_max=0.9999)
        oof_cal = calibrador.fit_transform(oof_prob_raw, y_true).astype(np.float32)
        score_cal = calibrador.transform(score_prob_raw).astype(np.float32)
    else:
        calibrador = LogisticRegression()
        calibrador.fit(oof_prob_raw.reshape(-1, 1), y_true)
        oof_cal = calibrador.predict_proba(oof_prob_raw.reshape(-1, 1))[:, 1].astype(np.float32)
        score_cal = calibrador.predict_proba(score_prob_raw.reshape(-1, 1))[:, 1].astype(np.float32)

    ruta_grafico.parent.mkdir(parents=True, exist_ok=True)
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, np.clip(oof_prob_raw, 0, 1), n_bins=10, strategy="quantile")
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, np.clip(oof_cal, 0, 1), n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Calibración Perfecta (Ideal)")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#d95f02", label="Probabilidad Sin Calibrar")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#1b9e77", label=f"Calibrada ({metodo.capitalize()})")
    ax.set_xlabel("Probabilidad Predicha Promedio")
    ax.set_ylabel("Fracción Real de Conversiones")
    ax.set_title("Curva de Calibración — Honestidad Financiera de las Probabilidades")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_grafico, dpi=150)
    plt.close(fig)

    logger.info(
        f"Calibración ({metodo}) completada -> Tasa real: {y_true.mean():.4f} | "
        f"Media prob calibrada OOF: {oof_cal.mean():.4f}"
    )
    return calibrador, oof_cal, score_cal


# %%
# ==============================================================================
# 4. MODELO DE MONTO (DOS ETAPAS / HURDLE VS REGRESIÓN DIRECTA)
# ==============================================================================
def instanciar_regresor_boosting() -> Any:
    """Instancia el regresor de Gradient Boosting con respaldo automático."""
    semilla = CONFIG["semilla"]
    if HAS_LGB:
        return lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            n_estimators=600,
            random_state=semilla,
            n_jobs=-1,
            verbose=-1,
        )
    elif HAS_XGB:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.03,
            max_depth=5,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=semilla,
            n_jobs=-1,
        )
    else:
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_depth=6,
            max_iter=350,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=semilla,
        )


def entrenar_modelos_monto(
    X_train: pd.DataFrame,
    y_clf: np.ndarray,
    y_monto: np.ndarray,
    oof_prob_cal: np.ndarray,
    X_score: pd.DataFrame,
    score_prob_cal: np.ndarray,
) -> Tuple[List[Any], np.ndarray, np.ndarray, Dict[str, float]]:
    """
    El volumen de factoring sufre del 'síndrome del iceberg': la mayoría de filas
    son ceros y quienes operan exhiben montos de cola larga.
    Compara dos arquitecturas:
    1. Modelo en Dos Etapas (Hurdle Model): Entrena un regresor log1p(monto) SOLO
       con quienes convirtieron (severidad condicional, corrigiendo el sesgo de
       retransformación con el factor de Smearing de Duan) y lo multiplica por la
       probabilidad calibrada: E[Y] = P(Y>0) * E[Y | Y>0].
    2. Regresión Directa: Entrena sobre toda la población (ceros incluidos).
    Reporta R², MAE y R² sobre el valor esperado.
    """
    mask_pos = (y_clf == 1) & (y_monto > 0)
    X_pos = X_train.loc[mask_pos].reset_index(drop=True)
    y_pos_real = y_monto[mask_pos]
    y_pos_log = np.log1p(y_pos_real)

    n_splits_reg = min(CONFIG["n_folds_cv"], max(2, len(X_pos) // 25))
    kf_pos = KFold(n_splits=n_splits_reg, shuffle=True, random_state=CONFIG["semilla"])

    oof_log_pos = np.zeros(len(X_pos), dtype=np.float32)
    pred_log_all_train = np.zeros(len(X_train), dtype=np.float32)
    pred_log_score = np.zeros(len(X_score), dtype=np.float32)
    modelos_cond: List[Any] = []

    for tr_idx, va_idx in kf_pos.split(X_pos):
        X_tr, y_tr = X_pos.iloc[tr_idx], y_pos_log[tr_idx]
        X_va, y_va = X_pos.iloc[va_idx], y_pos_log[va_idx]

        reg = instanciar_regresor_boosting()
        if HAS_LGB:
            reg.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
            )
        else:
            reg.fit(X_tr, y_tr)

        oof_log_pos[va_idx] = reg.predict(X_va)
        pred_log_all_train += reg.predict(X_train) / n_splits_reg
        pred_log_score += reg.predict(X_score) / n_splits_reg
        modelos_cond.append(reg)

    residuos_log = y_pos_log - oof_log_pos
    smearing = float(np.clip(np.mean(np.exp(residuos_log)), 1.0, 2.5))

    oof_monto_pos_soles = np.maximum(np.expm1(oof_log_pos) * smearing, 0.0)
    monto_cond_train = np.maximum(np.expm1(pred_log_all_train) * smearing, 0.0).astype(np.float32)
    monto_cond_train[mask_pos] = oof_monto_pos_soles.astype(np.float32)

    monto_cond_score = np.maximum(np.expm1(pred_log_score) * smearing, 0.0).astype(np.float32)

    ev_hurdle_train = (oof_prob_cal * monto_cond_train).astype(np.float32)

    kf_all = KFold(n_splits=3, shuffle=True, random_state=CONFIG["semilla"])
    oof_directo_soles = np.zeros(len(X_train), dtype=np.float32)
    y_all_log = np.log1p(np.maximum(y_monto, 0.0))

    for tr_idx, va_idx in kf_all.split(X_train):
        reg_dir = instanciar_regresor_boosting()
        reg_dir.fit(X_train.iloc[tr_idx], y_all_log[tr_idx])
        pred_d_log = np.clip(reg_dir.predict(X_train.iloc[va_idx]), 0.0, None)
        oof_directo_soles[va_idx] = np.expm1(pred_d_log)

    metricas_reg = {
        "r2_condicional_log_positivos": float(r2_score(y_pos_log, oof_log_pos)),
        "r2_condicional_soles_positivos": float(r2_score(y_pos_real, oof_monto_pos_soles)),
        "mae_condicional_soles_positivos": float(mean_absolute_error(y_pos_real, oof_monto_pos_soles)),
        "r2_valor_esperado_dos_etapas": float(r2_score(y_monto, ev_hurdle_train)),
        "mae_valor_esperado_dos_etapas": float(mean_absolute_error(y_monto, ev_hurdle_train)),
        "r2_regresion_directa_una_etapa": float(r2_score(y_monto, oof_directo_soles)),
        "mae_regresion_directa_una_etapa": float(mean_absolute_error(y_monto, oof_directo_soles)),
        "factor_smearing_duan": smearing,
    }

    logger.info(
        f"[Modelo de Monto] R² Condicional (log): {metricas_reg['r2_condicional_log_positivos']:.4f} | "
        f"R² Valor Esperado (2 Etapas): {metricas_reg['r2_valor_esperado_dos_etapas']:.4f} vs "
        f"R² Regresión Directa: {metricas_reg['r2_regresion_directa_una_etapa']:.4f}"
    )
    return modelos_cond, monto_cond_train, monto_cond_score, metricas_reg


# %%
# ==============================================================================
# 5. EXPLICABILIDAD PARA EL "¿POR QUÉ?" (SHAP / PERMUTACIÓN + PERFIL TOP DECIL)
# ==============================================================================
def generar_explicabilidad_negocio(
    modelos_clf: List[Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    oof_prob_cal: np.ndarray,
    X_score: pd.DataFrame,
    contrato: Dict[str, Any],
    dir_reportes: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    La métrica técnica filtra, pero la historia de negocio define al ganador.
    Esta función:
    1. Calcula la importancia de cada variable (usando SHAP TreeExplainer o
       Importancia por Permutación como respaldo).
    2. Agrupa el peso explicativo según el prefijo de origen (Maestro, Transacciones,
       Factoring, Perfil Histórico, Red B2B, Tendencias).
    3. Identifica los 10 rasgos que más distinguen al Decil 1 (élite de propensión)
       frente al resto de la cartera.
    4. Asigna a cada empresa de abt_score sus 3 motivos principales de negocio.
    """
    modelo_ref = modelos_clf[0]
    n_muestra = min(CONFIG["max_muestra_shap"], len(X_train))
    X_sample = X_train.sample(n=n_muestra, random_state=CONFIG["semilla"])

    importancias = np.zeros(X_train.shape[1], dtype=np.float64)
    signo_direccion = np.ones(X_train.shape[1], dtype=np.float64)

    for idx, col in enumerate(X_train.columns):
        s_val = X_sample[col].fillna(0.0)
        if s_val.std() > 1e-12:
            c_val = np.corrcoef(s_val, oof_prob_cal[X_sample.index])[0, 1]
            if not np.isnan(c_val):
                signo_direccion[idx] = 1.0 if c_val >= 0 else -1.0

    metodo_exp = "SHAP"
    try:
        if HAS_SHAP and (HAS_LGB or HAS_XGB or HAS_CAT):
            explainer = shap.TreeExplainer(modelo_ref)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            importancias = np.abs(shap_vals).mean(axis=0)
        else:
            raise RuntimeError("Usar respaldo de permutación/importancia nativa")
    except Exception:
        metodo_exp = "Permutation/Native Importance"
        if hasattr(modelo_ref, "feature_importances_"):
            for m in modelos_clf:
                importancias += m.feature_importances_ / len(modelos_clf)
        else:
            perm = permutation_importance(
                modelo_ref,
                X_sample,
                y_train[X_sample.index],
                n_repeats=3,
                random_state=CONFIG["semilla"],
                n_jobs=-1,
            )
            importancias = np.maximum(perm.importances_mean, 0.0)

    suma_imp = max(importancias.sum(), CONFIG["eps"])
    df_imp = pd.DataFrame({
        "variable": X_train.columns,
        "importancia_pct": (importancias / suma_imp) * 100.0,
        "direccion_efecto": np.where(signo_direccion >= 0, "Positivo (+)", "Negativo (-)"),
    }).sort_values("importancia_pct", ascending=False).reset_index(drop=True)

    prefijos_conocidos = dict(contrato.get("prefijos", {}))
    prefijos_conocidos.update({
        "red_": "Efecto Red B2B y Pagadores Ancla",
        "ten_": "Inercia y Tendencia Temporal",
        "rat_": "Ratios de Liquidez y Contrastes",
        "cat_": "Perfil Categórico Codificado",
    })

    def mapear_familia(nombre_col: str) -> str:
        if nombre_col == "es_habilitable":
            return "Elegibilidad Comercial (Habilitación)"
        for pref, desc in prefijos_conocidos.items():
            if nombre_col.startswith(pref):
                return f"{pref} ({desc.strip()})"
        return "Otras Variables Derivadas"

    df_imp["familia_datos"] = df_imp["variable"].apply(mapear_familia)
    df_imp_familia = (
        df_imp.groupby("familia_datos", as_index=False)["importancia_pct"]
        .sum()
        .sort_values("importancia_pct", ascending=False)
    )

    umbral_d1 = np.percentile(oof_prob_cal, 90)
    mask_d1 = oof_prob_cal >= umbral_d1
    top_30_vars = df_imp.head(30)["variable"].tolist()

    rasgos_d1 = []
    for col in top_30_vars:
        media_d1 = float(X_train.loc[mask_d1, col].mean())
        media_resto = float(X_train.loc[~mask_d1, col].mean())
        std_global = float(X_train[col].std()) + CONFIG["eps"]
        z_diff = (media_d1 - media_resto) / std_global
        ratio_medias = media_d1 / (media_resto + CONFIG["eps"]) if abs(media_resto) > 1e-4 else np.nan
        rasgos_d1.append({
            "variable": col,
            "familia_datos": mapear_familia(col),
            "media_decil_superior": media_d1,
            "media_resto_cartera": media_resto,
            "ratio_decil1_vs_resto": ratio_medias,
            "diferencia_estandarizada_z": z_diff,
            "magnitud_distincion": abs(z_diff),
        })

    df_top10_rasgos = (
        pd.DataFrame(rasgos_d1)
        .sort_values("magnitud_distincion", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    top_k_drivers = df_imp.head(15)["variable"].tolist()
    pesos_drivers = df_imp.set_index("variable").loc[top_k_drivers, "importancia_pct"].values
    signos_drivers = np.array([
        1.0 if d == "Positivo (+)" else -1.0
        for d in df_imp.set_index("variable").loc[top_k_drivers, "direccion_efecto"]
    ])

    medias_tr = X_train[top_k_drivers].mean().values
    stds_tr = X_train[top_k_drivers].std().replace(0, 1.0).fillna(1.0).values

    Z_score = (X_score[top_k_drivers].fillna(0.0).values - medias_tr) / stds_tr
    contrib_matrix = Z_score * signos_drivers * np.sqrt(pesos_drivers)

    idx_top3 = np.argsort(-contrib_matrix, axis=1)[:, :3]
    nombres_arr = np.array(top_k_drivers)
    df_motivos_score = pd.DataFrame({
        "motivo_1": nombres_arr[idx_top3[:, 0]],
        "motivo_2": nombres_arr[idx_top3[:, 1]],
        "motivo_3": nombres_arr[idx_top3[:, 2]],
    }, index=X_score.index)

    dir_reportes.mkdir(parents=True, exist_ok=True)
    top20 = df_imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top20["variable"], top20["importancia_pct"], color="#2b5c8f")
    ax.set_xlabel("Importancia Relativa (%)")
    ax.set_title(f"Top 20 Variables Explicativas de Conversión a Factoring ({metodo_exp})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(dir_reportes / "importancia_variables.png", dpi=150)
    plt.close(fig)

    logger.info(f"Explicabilidad generada ({metodo_exp}). Familia líder: {df_imp_familia.iloc[0]['familia_datos']}")
    return df_imp, df_imp_familia, df_top10_rasgos, df_motivos_score


# %%
# ==============================================================================
# PIPELINE MAESTRO DE MODELAMIENTO
# ==============================================================================
def main() -> None:
    """
    Ejecuta el entrenamiento, validación, calibración y explicabilidad de los
    modelos de conversión y severidad, dejando listos los artefactos y la tabla
    de scores para la optimización comercial.
    """
    logger.info("=== INICIANDO 04_MODELAMIENTO: PROPENSIÓN, SEVERIDAD Y EXPLICABILIDAD ===")
    contrato = cargar_contrato_activo()

    col_id = contrato["columna_id"]
    col_corte = contrato["columna_fecha_corte"]
    targets_map = contrato.get("targets", {})
    col_t_clf = targets_map.get("clasificacion", "target_conversion")
    col_t_reg = targets_map.get("regresion", "target_monto")

    archivos = contrato.get("archivos", {})
    ruta_train = archivos.get("abt_train_feat") or archivos.get("abt_train")
    ruta_score = archivos.get("abt_score_feat") or archivos.get("abt_score")

    df_train = leer_abt(ruta_train)
    df_score = leer_abt(ruta_score)

    features = contrato.get("features_modelamiento")
    if not features:
        excluir = {col_id, col_corte} | set(targets_map.values())
        features = [
            c for c in df_train.select_dtypes(include=[np.number]).columns
            if c not in excluir and not c.startswith("target_") and c in df_score.columns
        ]

    X_train = df_train[features].copy()
    y_clf = df_train[col_t_clf].fillna(0).astype(np.int8).values
    y_monto = (
        df_train[col_t_reg].fillna(0.0).astype(np.float32).values
        if col_t_reg in df_train.columns
        else y_clf.astype(np.float32) * 10_000.0
    )
    X_score = df_score[features].copy()

    splits, desc_validacion = definir_esquema_validacion(df_train, col_corte, y_clf)

    pipe_lr, oof_lr, metricas_lr = entrenar_linea_base_logistica(X_train, y_clf, splits)
    best_params = optimizar_hiperparametros_clasificacion(X_train, y_clf, splits)
    modelos_clf, oof_gb_raw, score_gb_raw, metricas_gb = entrenar_ensamble_clasificacion(
        X_train, y_clf, X_score, splits, best_params
    )

    dir_rep = Path(CONFIG["dir_reportes"])
    dir_mod = Path(CONFIG["dir_modelos"])
    dir_mod.mkdir(parents=True, exist_ok=True)
    dir_rep.mkdir(parents=True, exist_ok=True)

    calibrador, oof_prob_cal, score_prob_cal = calibrar_probabilidades(
        y_clf, oof_gb_raw, score_gb_raw, dir_rep / "curva_calibracion.png"
    )
    tabla_deciles = construir_tabla_deciles(y_clf, oof_prob_cal)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(tabla_deciles["decil"], tabla_deciles["captura_decil_pct"], color="#6baed6", label="% Captura del Decil")
    ax1.plot(tabla_deciles["decil"], tabla_deciles["captura_acumulada_pct"], "o-", color="#08519c", linewidth=2, label="% Captura Acumulada")
    ax1.set_xlabel("Decil de Propensión (1 = Mayor Probabilidad)")
    ax1.set_ylabel("Captura de Conversiones (%)")
    ax1.set_xticks(range(1, 11))
    ax1.set_title(f"Curva de Captura por Deciles (Gini = {metricas_gb['gini']:.3f} | Lift Decil 1 = {tabla_deciles.iloc[0]['lift']:.2f}x)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dir_rep / "curva_lift_deciles.png", dpi=150)
    plt.close(fig)

    modelos_reg, monto_cond_train, monto_cond_score, metricas_reg = entrenar_modelos_monto(
        X_train, y_clf, y_monto, oof_prob_cal, X_score, score_prob_cal
    )

    df_imp, df_imp_familia, df_top10_rasgos, df_motivos_score = generar_explicabilidad_negocio(
        modelos_clf, X_train, y_clf, oof_prob_cal, X_score, contrato, dir_rep
    )

    df_scores = pd.DataFrame({
        col_id: df_score[col_id].values,
        "prob_sin_calibrar": score_gb_raw,
        "prob_calibrada": score_prob_cal,
        "monto_esperado_condicional": monto_cond_score,
        "valor_esperado": (score_prob_cal * monto_cond_score).astype(np.float32),
        "motivo_1": df_motivos_score["motivo_1"].values,
        "motivo_2": df_motivos_score["motivo_2"].values,
        "motivo_3": df_motivos_score["motivo_3"].values,
    })

    if col_corte in df_score.columns:
        df_scores[col_corte] = df_score[col_corte].values
    if "es_habilitable" in df_score.columns:
        df_scores["es_habilitable"] = df_score["es_habilitable"].values
    if "fac_previo_detectado" in df_score.columns:
        df_scores["fac_previo_detectado"] = df_score["fac_previo_detectado"].values

    for c_cat in contrato.get("features_categoricas_conservadas", []):
        if c_cat in df_score.columns and c_cat not in df_scores.columns:
            df_scores[c_cat] = df_score[c_cat].values

    ruta_scores_final = guardar_dataframe_seguro(
        df_scores, Path(CONFIG["dir_procesados"]) / CONFIG["nombre_scores_out"]
    )

    with open(dir_mod / "artefactos_modelos.pkl", "wb") as f:
        pickle.dump(
            {
                "pipe_logistica": pipe_lr,
                "modelos_clasificacion": modelos_clf,
                "calibrador": calibrador,
                "modelos_regresion_condicional": modelos_reg,
                "features": features,
                "best_params": best_params,
            },
            f,
        )

    resumen_metricas = {
        "esquema_validacion": desc_validacion,
        "conversion_linea_base_logistica": metricas_lr,
        "conversion_gradient_boosting": metricas_gb,
        "regresion_monto_y_valor_esperado": metricas_reg,
        "tabla_deciles": tabla_deciles.to_dict(orient="records"),
        "importancia_por_familia_datos": df_imp_familia.to_dict(orient="records"),
        "top_10_rasgos_decil_superior": df_top10_rasgos.to_dict(orient="records"),
        "top_25_variables": df_imp.head(25).to_dict(orient="records"),
        "ruta_scores": ruta_scores_final,
    }

    with open(dir_rep / "metricas_modelamiento.json", "w", encoding="utf-8") as f:
        json.dump(resumen_metricas, f, indent=2, ensure_ascii=False)

    logger.info(f"Scores guardados en: {ruta_scores_final} ({len(df_scores):,} empresas evaluadas).")
    logger.info("=== 04_MODELAMIENTO FINALIZADO CON ÉXITO ===")


if __name__ == "__main__":
    main()