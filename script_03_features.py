# %% [markdown]
# # 03_features.py — Orfebrería de Señales, Topología B2B y Elegibilidad Comercial
# Transforma la fotografía estática en una radiografía dinámica de liquidez,
# inercia temporal y efecto red sin escribir un solo nombre de columna a mano.

# %%
from __future__ import annotations

import gc
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# BLOQUE DE CONFIGURACIÓN GLOBAL (CONFIG)
# ==============================================================================
CONFIG: Dict[str, Any] = {
    "semilla": 42,
    "ruta_contrato_entrada": "data/processed/contrato_abt.json",
    "ruta_contrato_salida": "data/processed/contrato_features.json",
    "dir_procesados": "data/processed",
    "nombre_abt_train_out": "abt_train_feat.parquet",
    "nombre_abt_score_out": "abt_score_feat.parquet",
    "umbral_nulos_descarte": 0.98,
    "umbral_correlacion_descarte": 0.98,
    "max_filas_corr_muestra": 50_000,
    "umbral_asimetria_log1p": 2.0,
    "max_cardinalidad_codigo_a_cat": 30,
    "suavizado_target_encoding": 20.0,
    "eps": 1e-6,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("03_Features")
np.random.seed(CONFIG["semilla"])


# %%
# ==============================================================================
# UTILIDADES DE LECTURA, ESCRITURA Y FECHAS SEGURAS
# ==============================================================================
def cargar_contrato(ruta: str) -> Dict[str, Any]:
    """
    Abre la brújula del proyecto: el contrato JSON que dicta cómo se llaman
    las piezas del rompecabezas sin atar el código a supuestos frágiles.
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise FileNotFoundError(f"No se encontró el contrato en: {ruta}")
    with open(ruta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def leer_tabla_segura(ruta: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Lee archivos tabulares (.parquet, .pkl, .csv) con tolerancia a fallos.
    Si una fuente secundaria no llegó en el USB, avisa con elegancia y permite
    que el río siga su curso.
    """
    if not ruta:
        return None
    candidatos = [Path(ruta), Path(CONFIG["dir_procesados"]) / Path(ruta).name]
    ruta_real = next((p for p in candidatos if p.exists()), None)
    if ruta_real is None:
        logger.warning(f"Tabla no encontrada en disco ({ruta}). Se omitirá su bloque.")
        return None

    ext = ruta_real.suffix.lower()
    try:
        if ext == ".parquet":
            return pd.read_parquet(ruta_real)
        elif ext in (".pkl", ".pickle"):
            return pd.read_pickle(ruta_real)
        elif ext in (".csv", ".txt"):
            return pd.read_csv(ruta_real)
        else:
            return pd.read_parquet(ruta_real)
    except Exception as e:
        logger.warning(f"No se pudo leer {ruta_real} ({e}). Continuando sin ella.")
        return None


def guardar_tabla_segura(df: pd.DataFrame, ruta_destino: Path) -> str:
    """
    Persiste el tablón enriquecido en disco. Si el motor de Parquet no está
    disponible en el entorno local, recurre a Pickle como bote salvavidas.
    """
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        if ruta_destino.suffix.lower() == ".parquet":
            df.to_parquet(ruta_destino, index=False)
        else:
            df.to_pickle(ruta_destino)
        return str(ruta_destino)
    except Exception as e:
        ruta_alt = ruta_destino.with_suffix(".pkl")
        logger.warning(f"Fallo al guardar parquet ({e}); guardando en {ruta_alt}")
        df.to_pickle(ruta_alt)
        return str(ruta_alt)


def parsear_fecha_vectorizada(serie: pd.Series) -> pd.Series:
    """
    Traduce cualquier dialecto temporal (fechas ISO, timestamps o enteros AAAAMM)
    al lenguaje universal de datetime64 sin iterar fila por fila.
    """
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    s_str = serie.astype(str).str.strip()
    muestra = s_str.dropna().head(20)
    if len(muestra) > 0 and muestra.str.match(r"^\d{6}$").all():
        return pd.to_datetime(s_str + "01", format="%Y%m%d", errors="coerce")
    return pd.to_datetime(serie, errors="coerce")


def optimizar_memoria_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adelgaza la representación numérica en RAM para que millones de registros
    quepan en memoria sin perder precisión financiera relevante.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        c_min, c_max = df[col].min(), df[col].max()
        if c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    return df


# %%
# ==============================================================================
# 1. RATIOS Y CONTRASTES ENTRE VARIABLES HERMANAS (POR SUFIJO)
# ==============================================================================
def construir_ratios_hermanas(
    df: pd.DataFrame, cols_protegidas: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Busca familias de variables que comparten la misma raíz semántica pero
    distinto sufijo estadístico (_sum, _mean, _max, _std, _ult, _meses_activos).
    Como un analista financiero que lee entre líneas, construye señales de:
    - Impulso reciente (_ult / _mean): ¿la empresa acelera o se apaga?
    - Volatilidad relativa (_std / _mean): ¿su flujo de caja es un reloj o una tormenta?
    - Intensidad de picos (_max / _mean): ¿opera por campañas abruptas?
    - Velocidad mensual (_sum / _meses_activos): caudal real cuando está activa.
    """
    sufijos_objetivo = [
        "_n_registros",
        "_meses_activos",
        "_recencia_dias",
        "_antiguedad_dias",
        "_sum",
        "_mean",
        "_max",
        "_std",
        "_ult",
    ]
    familias: Dict[str, Dict[str, str]] = {}
    cols_num = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in cols_protegidas and not c.startswith("target_")
    ]

    for col in cols_num:
        for suf in sufijos_objetivo:
            if col.endswith(suf):
                raiz = col[: -len(suf)]
                familias.setdefault(raiz, {})[suf] = col
                break

    nuevas: Dict[str, pd.Series] = {}
    eps = CONFIG["eps"]

    for raiz, mapa in familias.items():
        # 1. Impulso reciente: fotografía final vs película promedio
        if "_ult" in mapa and "_mean" in mapa:
            c_ult, c_mean = mapa["_ult"], mapa["_mean"]
            denom = np.where(np.abs(df[c_mean]) < eps, np.nan, df[c_mean])
            nuevas[f"rat_{raiz}_impulso_ult_mean"] = (df[c_ult] / denom).astype(np.float32)
            nuevas[f"rat_{raiz}_brecha_ult_mean"] = (df[c_ult] - df[c_mean]).astype(np.float32)

        # 2. Coeficiente de variación: riesgo e irregularidad operativa
        if "_std" in mapa and "_mean" in mapa:
            c_std, c_mean = mapa["_std"], mapa["_mean"]
            denom = np.abs(df[c_mean]) + eps
            nuevas[f"rat_{raiz}_cv_std_mean"] = (df[c_std] / denom).astype(np.float32)

        # 3. Concentración de picos: valor máximo frente al promedio habitual
        if "_max" in mapa and "_mean" in mapa:
            c_max, c_mean = mapa["_max"], mapa["_mean"]
            denom = np.abs(df[c_mean]) + eps
            nuevas[f"rat_{raiz}_pico_max_mean"] = (df[c_max] / denom).astype(np.float32)

        # 4. Densidad por mes activo: caudal efectivo en los meses que sí opera
        if "_sum" in mapa and "_meses_activos" in mapa:
            c_sum, c_mes = mapa["_sum"], mapa["_meses_activos"]
            denom = np.where(df[c_mes] <= 0, np.nan, df[c_mes])
            nuevas[f"rat_{raiz}_caudal_por_mes_activo"] = (df[c_sum] / denom).astype(np.float32)

        # 5. Ticket medio implícito: suma entre número de registros
        if "_sum" in mapa and "_n_registros" in mapa:
            c_sum, c_n = mapa["_sum"], mapa["_n_registros"]
            denom = np.where(df[c_n] <= 0, np.nan, df[c_n])
            nuevas[f"rat_{raiz}_ticket_medio"] = (df[c_sum] / denom).astype(np.float32)

        # 6. Ratio de enfriamiento: cuánta de su antigüedad se ha vuelto silencio reciente
        if "_recencia_dias" in mapa and "_antiguedad_dias" in mapa:
            c_rec, c_ant = mapa["_recencia_dias"], mapa["_antiguedad_dias"]
            denom = np.maximum(df[c_ant], 1.0)
            nuevas[f"rat_{raiz}_silencio_relativo"] = (df[c_rec] / denom).astype(np.float32)

    if not nuevas:
        return df, []

    df_nuevas = pd.DataFrame(nuevas, index=df.index)
    df_out = pd.concat([df, df_nuevas], axis=1)
    logger.info(f"Se destilaron {len(nuevas)} variables de contraste entre hermanas.")
    return df_out, list(nuevas.keys())


# %%
# ==============================================================================
# 2. TENDENCIA TEMPORAL DESDE TABLAS LIMPIAS (SIN FUGA DE INFORMACIÓN)
# ==============================================================================
def extraer_tendencias_temporales(
    pares_corte: pd.DataFrame,
    contrato: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Acude a las tablas limpias (Perfil Histórico y Transacciones) para medir
    el pulso dinámico antes del corte: compara los últimos 1, 3 y 6 meses contra
    el pasado inmediato y calcula la pendiente lineal exacta de las series mensuales
    usando covarianza vectorizada, respetando de forma sagrada fecha <= fecha_corte.
    """
    col_id = contrato["columna_id"]
    col_corte = contrato["columna_fecha_corte"]
    roles_dict = contrato.get("tablas_limpias_roles", {})
    archivos_dict = contrato.get("archivos", {})

    pares = pares_corte[[col_id, col_corte]].drop_duplicates().copy()
    pares["_corte_dt"] = parsear_fecha_vectorizada(pares[col_corte])

    bloques_calculados: List[pd.DataFrame] = []

    for nombre_rol, meta in roles_dict.items():
        llave_archivo = f"limpio_{nombre_rol}" if f"limpio_{nombre_rol}" in archivos_dict else nombre_rol
        ruta_tabla = archivos_dict.get(llave_archivo)
        if not ruta_tabla:
            for k_arch, v_arch in archivos_dict.items():
                if nombre_rol.lower() in k_arch.lower() and "limpio" in k_arch.lower():
                    ruta_tabla = v_arch
                    break

        col_fecha_t = meta.get("fecha")
        id_tabla = meta.get("id")
        if not ruta_tabla or not col_fecha_t:
            continue

        df_t = leer_tabla_segura(ruta_tabla)
        if df_t is None or col_fecha_t not in df_t.columns:
            continue

        if not id_tabla or id_tabla not in df_t.columns:
            entidades = [e for e in meta.get("entidades", []) if e in df_t.columns]
            if col_id in df_t.columns:
                id_tabla = col_id
            elif entidades:
                id_tabla = entidades[0]
            else:
                continue

        candidatas_num = []
        if meta.get("monto") and meta["monto"] in df_t.columns:
            candidatas_num.append(meta["monto"])
        for c_num in meta.get("numericas", []):
            if c_num in df_t.columns and c_num not in candidatas_num:
                candidatas_num.append(c_num)
        candidatas_num = [
            c for c in candidatas_num[:4]
            if pd.api.types.is_numeric_dtype(df_t[c]) and not c.startswith("target_")
        ]
        if not candidatas_num:
            continue

        logger.info(f"Calculando cinemática temporal en '{nombre_rol}' sobre {candidatas_num}...")
        df_sub = df_t[[id_tabla, col_fecha_t] + candidatas_num].copy()
        df_sub.rename(columns={id_tabla: col_id}, inplace=True)
        df_sub["_fecha_dt"] = parsear_fecha_vectorizada(df_sub[col_fecha_t])
        df_sub = df_sub.dropna(subset=[col_id, "_fecha_dt"])

        df_sub["_ym"] = df_sub["_fecha_dt"].dt.to_period("M").dt.to_timestamp()
        df_mensual = (
            df_sub.groupby([col_id, "_ym"], as_index=False)[candidatas_num]
            .sum()
        )
        del df_sub
        gc.collect()

        merged = pares.merge(df_mensual, on=col_id, how="inner")
        merged = merged[merged["_ym"] <= merged["_corte_dt"]].copy()
        if merged.empty:
            continue

        merged["_meses_atras"] = (
            (merged["_corte_dt"].dt.year - merged["_ym"].dt.year) * 12
            + (merged["_corte_dt"].dt.month - merged["_ym"].dt.month)
        ).clip(lower=0)

        merged["_t"] = -merged["_meses_atras"].astype(np.float32)

        pref = nombre_rol[:3].lower()
        llaves_grupo = [col_id, col_corte]

        m_1m = merged["_meses_atras"] <= 1
        m_3m = merged["_meses_atras"] <= 3
        m_p3m = (merged["_meses_atras"] > 3) & (merged["_meses_atras"] <= 6)
        m_6m = merged["_meses_atras"] <= 6
        m_p6m = (merged["_meses_atras"] > 6) & (merged["_meses_atras"] <= 12)

        base_grupo = pares[[col_id, col_corte]].copy()

        for col_v in candidatas_num:
            merged["_v_1m"] = np.where(m_1m, merged[col_v], 0.0)
            merged["_v_3m"] = np.where(m_3m, merged[col_v], 0.0)
            merged["_v_p3m"] = np.where(m_p3m, merged[col_v], 0.0)
            merged["_v_6m"] = np.where(m_6m, merged[col_v], 0.0)
            merged["_v_p6m"] = np.where(m_p6m, merged[col_v], 0.0)
            merged["_ty"] = merged["_t"] * merged[col_v]
            merged["_t2"] = merged["_t"] ** 2

            agg_df = merged.groupby(llaves_grupo, as_index=False).agg(
                s_1m=("_v_1m", "sum"),
                s_3m=("_v_3m", "sum"),
                s_p3m=("_v_p3m", "sum"),
                s_6m=("_v_6m", "sum"),
                s_p6m=("_v_p6m", "sum"),
                mean_t=("_t", "mean"),
                mean_y=(col_v, "mean"),
                mean_ty=("_ty", "mean"),
                mean_t2=("_t2", "mean"),
                n_pts=("_t", "count"),
            )

            eps = CONFIG["eps"]
            var_t = np.maximum(agg_df["mean_t2"] - agg_df["mean_t"] ** 2, 0.0)
            cov_ty = agg_df["mean_ty"] - agg_df["mean_t"] * agg_df["mean_y"]
            pendiente = np.where(
                (agg_df["n_pts"] >= 2) & (var_t > eps),
                cov_ty / (var_t + eps),
                0.0,
            )

            res_col = pd.DataFrame({
                col_id: agg_df[col_id],
                col_corte: agg_df[col_corte],
                f"ten_{pref}_{col_v}_acel_1m_vs_3m": (
                    (agg_df["s_1m"] * 3.0) / (agg_df["s_3m"] + eps)
                ).astype(np.float32),
                f"ten_{pref}_{col_v}_acel_3m_vs_prev3m": (
                    (agg_df["s_3m"] + 1.0) / (agg_df["s_p3m"] + 1.0)
                ).astype(np.float32),
                f"ten_{pref}_{col_v}_acel_6m_vs_prev6m": (
                    (agg_df["s_6m"] + 1.0) / (agg_df["s_p6m"] + 1.0)
                ).astype(np.float32),
                f"ten_{pref}_{col_v}_pendiente_mensual": pendiente.astype(np.float32),
            })
            base_grupo = base_grupo.merge(res_col, on=llaves_grupo, how="left")

        bloques_calculados.append(base_grupo)
        del merged, df_mensual
        gc.collect()

    if not bloques_calculados:
        return None

    df_final_ten = bloques_calculados[0]
    for b in bloques_calculados[1:]:
        df_final_ten = df_final_ten.merge(b, on=[col_id, col_corte], how="outer")
    return optimizar_memoria_df(df_final_ten)


# %%
# ==============================================================================
# 3. TOPOLOGÍA DE RED B2B Y PROVEEDORES DE ACEPTANTES ANCLA
# ==============================================================================
def construir_variables_red(
    pares_corte: pd.DataFrame,
    contrato: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Reconstruye el tejido comercial entre empresas. En el negocio de Factoring,
    una empresa no opera en el vacío: si le vende a grandes pagadores (aceptantes)
    que el banco ya conoce y califica en la base de Factoring, la fricción de riesgo
    desaparece. Esta función mide:
    - Grado de la red (número de contrapartes distintas).
    - Concentración del flujo en la contraparte principal (dependencia comercial).
    - Exposición a Pagadores Ancla de Factoring (la llave maestra para 'Habilitar').
    """
    col_id = contrato["columna_id"]
    col_corte = contrato["columna_fecha_corte"]
    roles = contrato.get("tablas_limpias_roles", {})
    archivos = contrato.get("archivos", {})

    rol_trx = next((k for k in roles if "trx" in k.lower() or "transac" in k.lower()), None)
    rol_fac = next((k for k in roles if "fac" in k.lower()), None)
    if not rol_trx:
        return None

    meta_trx = roles[rol_trx]
    ruta_trx = archivos.get(f"limpio_{rol_trx}") or archivos.get(rol_trx)
    if not ruta_trx:
        for k, v in archivos.items():
            if "transac" in k.lower() or "trx" in k.lower():
                ruta_trx = v
                break

    df_trx = leer_tabla_segura(ruta_trx)
    if df_trx is None:
        return None

    entidades_trx = [e for e in meta_trx.get("entidades", []) if e in df_trx.columns]
    id_trx = meta_trx.get("id") if meta_trx.get("id") in df_trx.columns else None
    if id_trx and id_trx not in entidades_trx:
        entidades_trx = [id_trx] + entidades_trx

    if len(entidades_trx) < 2:
        logger.warning("Transacciones no tiene al menos 2 columnas de entidad para armar red.")
        return None

    col_fecha_trx = meta_trx.get("fecha")
    col_monto_trx = meta_trx.get("monto")
    if not col_monto_trx or col_monto_trx not in df_trx.columns:
        nums = [c for c in meta_trx.get("numericas", []) if c in df_trx.columns]
        col_monto_trx = nums[0] if nums else None

    df_trx = df_trx.copy()
    if col_fecha_trx and col_fecha_trx in df_trx.columns:
        df_trx["_fecha_dt"] = parsear_fecha_vectorizada(df_trx[col_fecha_trx])
    else:
        df_trx["_fecha_dt"] = pd.Timestamp("1970-01-01")

    if col_monto_trx:
        df_trx["_monto"] = pd.to_numeric(df_trx[col_monto_trx], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df_trx["_monto"] = 1.0

    df_fac = None
    entidades_fac: List[str] = []
    col_fecha_fac = None
    if rol_fac:
        meta_fac = roles[rol_fac]
        ruta_fac = archivos.get(f"limpio_{rol_fac}") or archivos.get(rol_fac)
        if not ruta_fac:
            for k, v in archivos.items():
                if "fac" in k.lower() and "limpio" in k.lower():
                    ruta_fac = v
                    break
        df_fac = leer_tabla_segura(ruta_fac)
        if df_fac is not None:
            entidades_fac = [e for e in meta_fac.get("entidades", []) if e in df_fac.columns]
            if meta_fac.get("id") and meta_fac["id"] in df_fac.columns and meta_fac["id"] not in entidades_fac:
                entidades_fac.append(meta_fac["id"])
            col_fecha_fac = meta_fac.get("fecha")
            if col_fecha_fac and col_fecha_fac in df_fac.columns:
                df_fac["_fecha_dt"] = parsear_fecha_vectorizada(df_fac[col_fecha_fac])
            else:
                df_fac["_fecha_dt"] = pd.Timestamp("1970-01-01")

    cortes_unicos = pares_corte[col_corte].dropna().unique()
    resultados_cortes: List[pd.DataFrame] = []

    col_e1, col_e2 = entidades_trx[0], entidades_trx[1]

    for corte_val in cortes_unicos:
        corte_dt = parsear_fecha_vectorizada(pd.Series([corte_val])).iloc[0]
        empresas_corte = pares_corte.loc[pares_corte[col_corte] == corte_val, [col_id, col_corte]].drop_duplicates()

        sub_trx = df_trx[df_trx["_fecha_dt"] <= corte_dt]
        if sub_trx.empty:
            continue

        nodos_ancla_fac = set()
        if df_fac is not None and entidades_fac:
            sub_fac = df_fac[df_fac["_fecha_dt"] <= corte_dt]
            col_aceptante = entidades_fac[-1]
            nodos_ancla_fac = set(sub_fac[col_aceptante].dropna().unique())

        par_a = (
            sub_trx.groupby([col_e1, col_e2], as_index=False)["_monto"]
            .agg(monto_par="sum", n_ops_par="count")
            .rename(columns={col_e1: col_id, col_e2: "_contraparte"})
        )
        par_a["_es_ancla"] = par_a["_contraparte"].isin(nodos_ancla_fac).astype(np.int8)
        par_a["_monto_ancla"] = par_a["monto_par"] * par_a["_es_ancla"]

        res_a = par_a.groupby(col_id, as_index=False).agg(
            red_rol1_n_contrapartes=("_contraparte", "nunique"),
            red_rol1_monto_total=("monto_par", "sum"),
            red_rol1_monto_max_cp=("monto_par", "max"),
            red_rol1_n_anclas_fac=("_es_ancla", "sum"),
            red_rol1_monto_anclas_fac=("_monto_ancla", "sum"),
        )
        eps = CONFIG["eps"]
        res_a["red_rol1_concentracion_top1"] = (
            res_a["red_rol1_monto_max_cp"] / (res_a["red_rol1_monto_total"] + eps)
        ).astype(np.float32)
        res_a["red_rol1_pct_monto_anclas_fac"] = (
            res_a["red_rol1_monto_anclas_fac"] / (res_a["red_rol1_monto_total"] + eps)
        ).astype(np.float32)

        par_b = (
            sub_trx.groupby([col_e2, col_e1], as_index=False)["_monto"]
            .agg(monto_par="sum", n_ops_par="count")
            .rename(columns={col_e2: col_id, col_e1: "_contraparte"})
        )
        par_b["_es_ancla"] = par_b["_contraparte"].isin(nodos_ancla_fac).astype(np.int8)
        par_b["_monto_ancla"] = par_b["monto_par"] * par_b["_es_ancla"]

        res_b = par_b.groupby(col_id, as_index=False).agg(
            red_rol2_n_contrapartes=("_contraparte", "nunique"),
            red_rol2_monto_total=("monto_par", "sum"),
            red_rol2_monto_max_cp=("monto_par", "max"),
            red_rol2_n_anclas_fac=("_es_ancla", "sum"),
            red_rol2_monto_anclas_fac=("_monto_ancla", "sum"),
        )
        res_b["red_rol2_concentracion_top1"] = (
            res_b["red_rol2_monto_max_cp"] / (res_b["red_rol2_monto_total"] + eps)
        ).astype(np.float32)
        res_b["red_rol2_pct_monto_anclas_fac"] = (
            res_b["red_rol2_monto_anclas_fac"] / (res_b["red_rol2_monto_total"] + eps)
        ).astype(np.float32)

        res_corte = empresas_corte.merge(res_a, on=col_id, how="left").merge(res_b, on=col_id, how="left")
        cols_red = [c for c in res_corte.columns if c.startswith("red_")]
        res_corte[cols_red] = res_corte[cols_red].fillna(0.0)

        res_corte["red_total_anclas_fac"] = (
            res_corte["red_rol1_n_anclas_fac"] + res_corte["red_rol2_n_anclas_fac"]
        ).astype(np.float32)
        res_corte["red_tiene_vinculo_ancla_fac"] = (res_corte["red_total_anclas_fac"] > 0).astype(np.int8)
        res_corte["red_balance_comercial_relativo"] = (
            (res_corte["red_rol1_monto_total"] - res_corte["red_rol2_monto_total"])
            / (res_corte["red_rol1_monto_total"] + res_corte["red_rol2_monto_total"] + eps)
        ).astype(np.float32)

        resultados_cortes.append(res_corte)

    if not resultados_cortes:
        return None

    df_red_total = pd.concat(resultados_cortes, axis=0, ignore_index=True)
    logger.info(f"Topología de red B2B tejida: {df_red_total.shape[1] - 2} variables de ecosistema creadas.")
    return optimizar_memoria_df(df_red_total)


# %%
# ==============================================================================
# 4. BANDERA DE ELEGIBILIDAD ('es_habilitable') Y MULTIPLICADOR DE BASE
# ==============================================================================
def construir_bandera_habilitable(
    df: pd.DataFrame,
    contrato: Dict[str, Any],
    umbrales_ref: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    El primer mandato del caso de negocio es 'Habilitar': descubrir empresas que
    hoy no usan factoring con el banco (clientes vírgenes en el producto), pero que
    respiran actividad transaccional, comercial o vínculos de red suficientes para
    ser sujetos de adelanto de facturas.
    Calcula y reporta cuántas veces (ej. 2.5x, 3.2x) se amplía la base comercial.
    """
    prefijos = contrato.get("prefijos", {})
    pref_fac = next((k for k, v in prefijos.items() if "fac" in k.lower() or "fac" in v.lower()), "fac_")
    pref_trx = next((k for k, v in prefijos.items() if "trx" in k.lower() or "trans" in v.lower()), "trx_")
    pref_per = next((k for k, v in prefijos.items() if "per" in k.lower() or "hist" in v.lower()), "per_")

    cols_fac = [c for c in df.columns if c.startswith(pref_fac) and not c.startswith("target_")]
    cols_fac_pos = [
        c for c in cols_fac
        if any(s in c for s in ("_tiene", "_n_registros", "_sum", "_mean", "_max", "_meses_activos"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if cols_fac_pos:
        matriz_fac = df[cols_fac_pos].fillna(0.0)
        ya_uso_fac = (matriz_fac.max(axis=1) > 0).astype(np.int8)
    else:
        ya_uso_fac = pd.Series(0, index=df.index, dtype=np.int8)

    df["fac_previo_detectado"] = ya_uso_fac

    cols_potencial = [
        c for c in df.columns
        if (c.startswith(pref_trx) or c.startswith(pref_per) or c.startswith("red_"))
        and any(s in c for s in ("_sum", "_mean", "_n_registros", "_meses_activos", "monto_total", "anclas_fac"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if umbrales_ref is None:
        umbrales_ref = {}
        mask_clientes = ya_uso_fac == 1
        for c in cols_potencial:
            serie_ref = df.loc[mask_clientes, c].dropna() if mask_clientes.sum() > 20 else df[c].dropna()
            serie_pos = serie_ref[serie_ref > 0]
            umbrales_ref[c] = float(np.percentile(serie_pos, 20)) if len(serie_pos) > 0 else 0.0

    conteo_senales = pd.Series(0, index=df.index, dtype=np.int16)
    for c, umb in umbrales_ref.items():
        if c in df.columns:
            conteo_senales += (df[c].fillna(0.0) > max(umb, 0.0)).astype(np.int16)

    tiene_vinculo_ancla = (
        df["red_tiene_vinculo_ancla_fac"] > 0
        if "red_tiene_vinculo_ancla_fac" in df.columns
        else pd.Series(False, index=df.index)
    )

    umbral_min_senales = 1 if len(cols_potencial) <= 3 else 2
    potencial_comercial = (conteo_senales >= umbral_min_senales) | tiene_vinculo_ancla

    df["n_senales_potencial_habilitacion"] = conteo_senales.astype(np.float32)
    df["es_habilitable"] = ((ya_uso_fac == 0) & potencial_comercial).astype(np.int8)

    n_ya_usaron = int((ya_uso_fac == 1).sum())
    n_habilitables = int(df["es_habilitable"].sum())
    mult_expansion = (n_ya_usaron + n_habilitables) / max(n_ya_usaron, 1)
    ratio_nuevos_vs_actuales = n_habilitables / max(n_ya_usaron, 1)

    umbrales_ref["_meta_n_ya_usaron"] = float(n_ya_usaron)
    umbrales_ref["_meta_n_habilitables"] = float(n_habilitables)
    umbrales_ref["_meta_multiplicador_base_total"] = float(mult_expansion)
    umbrales_ref["_meta_ratio_nuevos_sobre_actuales"] = float(ratio_nuevos_vs_actuales)

    logger.info(
        f"[HABILITACIÓN] Base con factoring previo: {n_ya_usaron:,} | "
        f"Nuevas empresas habilitables: {n_habilitables:,} | "
        f"Ampliación de base: {mult_expansion:.2f}x (Nuevos/Actuales = {ratio_nuevos_vs_actuales:.2f}x)"
    )
    return df, umbrales_ref


# %%
# ==============================================================================
# 5. TRANSFORMADOR FIT/TRANSFORM (SIN FUGA: AJUSTADO SOLO EN TRAIN)
# ==============================================================================
def ajustar_transformador_features(
    df_train: pd.DataFrame,
    contrato: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aprende las reglas de higiene y codificación exclusivamente sobre el conjunto
    de entrenamiento:
    - Detecta códigos disfrazados de números de baja cardinalidad.
    - Identifica variables numéricas de cola larga para aplicar log1p.
    - Aprende codificación de frecuencia y Target Encoding suavizado (Empirical Bayes).
    - Descarta variables mudas (constantes), desérticas (>98% nulos) o redundantes (corr > 0.98).
    """
    col_id = contrato["columna_id"]
    col_corte = contrato["columna_fecha_corte"]
    targets_map = contrato.get("targets", {})
    col_target_clf = targets_map.get("clasificacion", "target_conversion")
    cols_protegidas = {col_id, col_corte} | set(targets_map.values())

    codigos_num = [c for c in contrato.get("features_numericas_que_parecen_codigo", []) if c in df_train.columns]
    codigos_a_cat: List[str] = []
    for c in codigos_num:
        if df_train[c].nunique(dropna=True) <= CONFIG["max_cardinalidad_codigo_a_cat"]:
            codigos_a_cat.append(c)

    cat_cols = list(
        set(
            [c for c in contrato.get("features_categoricas", []) if c in df_train.columns]
            + [c for c in df_train.select_dtypes(include=["object", "category"]).columns if c not in cols_protegidas]
            + codigos_a_cat
        )
    )

    prior_global = float(df_train[col_target_clf].mean()) if col_target_clf in df_train.columns else 0.1
    mapas_freq: Dict[str, Dict[str, float]] = {}
    mapas_te: Dict[str, Dict[str, float]] = {}
    m_smooth = CONFIG["suavizado_target_encoding"]

    for c in cat_cols:
        s_cat = df_train[c].astype(str).fillna("SIN_DATO")
        vc = s_cat.value_counts(normalize=True)
        mapas_freq[c] = vc.to_dict()

        if col_target_clf in df_train.columns:
            stats = df_train.groupby(s_cat)[col_target_clf].agg(["count", "mean"])
            te_vals = (stats["count"] * stats["mean"] + m_smooth * prior_global) / (stats["count"] + m_smooth)
            mapas_te[c] = te_vals.to_dict()

    num_cols_raw = [
        c for c in df_train.select_dtypes(include=[np.number]).columns
        if c not in cols_protegidas and c not in cat_cols and not c.startswith("target_")
    ]

    cols_log1p: List[str] = []
    for c in num_cols_raw:
        if any(p in c for p in ("_sum", "_mean", "_max", "_ult", "monto", "caudal", "ticket")):
            s_clean = df_train[c].dropna()
            if len(s_clean) > 100 and s_clean.min() >= 0 and s_clean.skew() > CONFIG["umbral_asimetria_log1p"]:
                cols_log1p.append(c)

    return {
        "cols_protegidas": list(cols_protegidas),
        "codigos_a_cat": codigos_a_cat,
        "cat_cols": cat_cols,
        "prior_global": prior_global,
        "mapas_freq": mapas_freq,
        "mapas_te": mapas_te,
        "cols_log1p": cols_log1p,
    }


def aplicar_transformaciones_base(
    df: pd.DataFrame,
    estado: Dict[str, Any],
    es_train: bool,
    col_target_clf: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aplica de manera idéntica las transformaciones matemáticas y codificaciones
    tanto a abt_train como a abt_score, preservando la simetría entre entrenamiento
    y producción.
    """
    df = df.copy()
    nuevas_cols: Dict[str, pd.Series] = {}

    for c in estado["cols_log1p"]:
        if c in df.columns:
            nuevas_cols[f"{c}_log1p"] = np.log1p(np.maximum(df[c].fillna(0.0), 0.0)).astype(np.float32)

    prior = estado["prior_global"]
    for c in estado["cat_cols"]:
        if c not in df.columns:
            continue
        s_cat = df[c].astype(str).fillna("SIN_DATO")
        mapa_f = estado["mapas_freq"].get(c, {})
        nuevas_cols[f"cat_{c}_freq"] = s_cat.map(mapa_f).fillna(0.0).astype(np.float32)

        if c in estado["mapas_te"]:
            mapa_t = estado["mapas_te"][c]
            nuevas_cols[f"cat_{c}_te"] = s_cat.map(mapa_t).fillna(prior).astype(np.float32)

    if nuevas_cols:
        df = pd.concat([df, pd.DataFrame(nuevas_cols, index=df.index)], axis=1)

    return optimizar_memoria_df(df)


def seleccionar_features_limpias(
    df_train: pd.DataFrame,
    estado: Dict[str, Any],
) -> Tuple[List[str], Dict[str, int]]:
    """
    Realiza una poda higiénica aprendida solo en train: elimina variables con más de
    98% de vacíos, variables de varianza cero y variables colineales (> 0.98) que
    repiten la misma historia con distinto nombre.
    """
    protegidas = set(estado["cols_protegidas"])
    cat_originales = set(estado["cat_cols"])

    candidatas = [
        c for c in df_train.select_dtypes(include=[np.number]).columns
        if c not in protegidas and c not in cat_originales and not c.startswith("target_")
    ]

    tasa_nulos = df_train[candidatas].isnull().mean()
    descarte_nulos = set(tasa_nulos[tasa_nulos > CONFIG["umbral_nulos_descarte"]].index)
    vivas = [c for c in candidatas if c not in descarte_nulos]

    stds = df_train[vivas].std(ddof=0)
    descarte_const = set(stds[(stds.isna()) | (stds <= 1e-12)].index)
    vivas = [c for c in vivas if c not in descarte_const]

    if len(df_train) > CONFIG["max_filas_corr_muestra"]:
        df_muestra = df_train[vivas].sample(n=CONFIG["max_filas_corr_muestra"], random_state=CONFIG["semilla"])
    else:
        df_muestra = df_train[vivas]

    corr_matrix = df_muestra.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    descarte_corr = set(
        [col for col in upper.columns if any(upper[col] > CONFIG["umbral_correlacion_descarte"])]
    )

    descarte_corr.discard("es_habilitable")
    features_finales = [c for c in vivas if c not in descarte_corr]
    if "es_habilitable" in df_train.columns and "es_habilitable" not in features_finales:
        features_finales.append("es_habilitable")

    resumen_poda = {
        "candidatas_iniciales": len(candidatas),
        "descartadas_por_nulos": len(descarte_nulos),
        "descartadas_por_constantes": len(descarte_const),
        "descartadas_por_correlacion": len(descarte_corr),
        "features_finales_seleccionadas": len(features_finales),
    }
    logger.info(f"Poda de variables completada: {resumen_poda}")
    return features_finales, resumen_poda


# %%
# ==============================================================================
# PIPELINE MAESTRO DE INGENIERÍA DE VARIABLES
# ==============================================================================
def main() -> None:
    """
    Orquesta de principio a fin la creación de variables, garantizando que
    abt_train y abt_score pasen por el mismo molde exacto y actualizando el
    contrato para los scripts de modelamiento y optimización.
    """
    logger.info("=== INICIANDO 03_FEATURES: INGENIERÍA DE VARIABLES AGNÓSTICA ===")
    contrato = cargar_contrato(CONFIG["ruta_contrato_entrada"])

    col_id = contrato["columna_id"]
    col_corte = contrato["columna_fecha_corte"]
    archivos = contrato.get("archivos", {})
    targets_map = contrato.get("targets", {})
    cols_protegidas = [col_id, col_corte] + list(targets_map.values())

    df_train = leer_tabla_segura(archivos.get("abt_train"))
    df_score = leer_tabla_segura(archivos.get("abt_score"))
    if df_train is None or df_score is None:
        raise RuntimeError("No se pudieron cargar abt_train o abt_score desde las rutas del contrato.")

    logger.info(f"Dimensiones iniciales -> Train: {df_train.shape} | Score: {df_score.shape}")

    df_train, nuevas_rat = construir_ratios_hermanas(df_train, cols_protegidas)
    df_score, _ = construir_ratios_hermanas(df_score, cols_protegidas)

    pares_totales = pd.concat(
        [df_train[[col_id, col_corte]], df_score[[col_id, col_corte]]],
        axis=0,
        ignore_index=True,
    ).drop_duplicates()

    df_tendencias = extraer_tendencias_temporales(pares_totales, contrato)
    if df_tendencias is not None:
        df_train = df_train.merge(df_tendencias, on=[col_id, col_corte], how="left")
        df_score = df_score.merge(df_tendencias, on=[col_id, col_corte], how="left")
        del df_tendencias
        gc.collect()

    df_red = construir_variables_red(pares_totales, contrato)
    if df_red is not None:
        df_train = df_train.merge(df_red, on=[col_id, col_corte], how="left")
        df_score = df_score.merge(df_red, on=[col_id, col_corte], how="left")
        del df_red
        gc.collect()

    df_train, umbrales_hab = construir_bandera_habilitable(df_train, contrato, umbrales_ref=None)
    df_score, meta_hab_score = construir_bandera_habilitable(df_score, contrato, umbrales_ref=umbrales_hab.copy())

    estado_prep = ajustar_transformador_features(df_train, contrato)
    col_target_clf = targets_map.get("clasificacion")
    df_train = aplicar_transformaciones_base(df_train, estado_prep, es_train=True, col_target_clf=col_target_clf)
    df_score = aplicar_transformaciones_base(df_score, estado_prep, es_train=False)

    features_finales, resumen_poda = seleccionar_features_limpias(df_train, estado_prep)

    for f in features_finales:
        if f not in df_score.columns:
            df_score[f] = np.nan

    cat_conservar = [c for c in estado_prep["cat_cols"] if c in df_train.columns and c in df_score.columns]
    cols_base_train = [c for c in [col_id, col_corte] + list(targets_map.values()) if c in df_train.columns]
    cols_base_score = [c for c in [col_id, col_corte] if c in df_score.columns]

    cols_orden_train = list(dict.fromkeys(cols_base_train + cat_conservar + features_finales))
    cols_orden_score = list(dict.fromkeys(cols_base_score + cat_conservar + features_finales))

    df_train_out = df_train[cols_orden_train]
    df_score_out = df_score[cols_orden_score]

    dir_out = Path(CONFIG["dir_procesados"])
    ruta_train_out = guardar_tabla_segura(df_train_out, dir_out / CONFIG["nombre_abt_train_out"])
    ruta_score_out = guardar_tabla_segura(df_score_out, dir_out / CONFIG["nombre_abt_score_out"])

    contrato_actualizado = dict(contrato)
    contrato_actualizado["archivos"] = dict(contrato.get("archivos", {}))
    contrato_actualizado["archivos"]["abt_train_feat"] = ruta_train_out
    contrato_actualizado["archivos"]["abt_score_feat"] = ruta_score_out
    contrato_actualizado["features_modelamiento"] = features_finales
    contrato_actualizado["features_categoricas_conservadas"] = cat_conservar
    contrato_actualizado["resumen_ingenieria_variables"] = {
        **resumen_poda,
        "habilitacion_train": {k: v for k, v in umbrales_hab.items() if k.startswith("_meta_")},
        "habilitacion_score": {k: v for k, v in meta_hab_score.items() if k.startswith("_meta_")},
    }

    ruta_contrato_out = Path(CONFIG["ruta_contrato_salida"])
    ruta_contrato_out.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_contrato_out, "w", encoding="utf-8") as f:
        json.dump(contrato_actualizado, f, indent=2, ensure_ascii=False)

    logger.info(f"Contrato actualizado guardado en: {ruta_contrato_out}")
    logger.info(f"ABT Train final: {df_train_out.shape} | ABT Score final: {df_score_out.shape}")
    logger.info("=== 03_FEATURES FINALIZADO CON ÉXITO ===")


if __name__ == "__main__":
    main()