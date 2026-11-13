# -*- coding: utf-8 -*-
"""
02_prepro.py  ·  Preprocesamiento
==================================
DataFest 2026 · Caso Factoring: "¿A qué empresas debemos contactar?"

Si el EDA fue escuchar, esto es ordenar la casa antes de recibir visitas.
Entran cuatro bases crudas, cada una con su propio desorden; sale una sola
mesa servida: UNA FILA POR EMPRESA, lista para que el modelo se siente a comer.

El recorrido:
  1. Limpieza      · duplicados, columnas muertas, llaves armonizadas, textos parejos.
  2. Fecha de corte· la raya en el piso que separa lo que ya se sabía de lo que vino después.
  3. Compresión    · cada tabla larga (millones de filas) se resume a nivel empresa.
  4. Ensamble      · todo se cuelga del maestro para formar la ABT (tabla analítica base).
  5. Target        · conversión (0/1) y monto en la ventana futura, sin mirar el futuro al
                     construir variables: quien espía el examen no aprende nada.
  6. Contrato      · un JSON que le dice a la etapa siguiente (features/optimización/modelo)
                     qué columna es qué, para que nadie tenga que adivinar nombres.

Salidas en data/processed/:
  limpio_<tabla>.parquet   → las 4 bases ya aseadas (formato largo)
  abt_train.parquet        → empresas × fecha de corte, con variables y target
  abt_score.parquet        → empresas con toda su historia, SIN target (la campaña que viene)
  contrato_abt.json        → el diccionario de roles y columnas
  bitacora_prepro.txt      → qué se hizo y cuánto se movió en cada paso

Uso:  python 02_prepro.py
"""
# %% Preparativos
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from datafest_comun import (CONFIG, adelgazar, armonizar_llave, asegurar_carpetas, cargar_todo, decir,
                            es_fecha, es_numero, es_texto, guardar_json, guardar_tabla,
                            levantar_acta_de_roles, titulo)

DIR = Path(CONFIG["DIR_SALIDA"])
BITACORA: list[str] = []
PREFIJO = {"transacciones": "trx", "factoring": "fac", "perfil": "per"}


def apuntar(texto: str, nivel: int = 1) -> None:
    """Todo cambio deja huella: lo que no se anota, mañana nadie lo puede defender ante el jurado."""
    BITACORA.append(("   " * nivel) + texto)
    decir(texto, nivel)


# %% 1. Limpieza · quitar el polvo sin mover los muebles
def limpiar(tabla: str, df: pd.DataFrame, rol: dict, universo: set | None) -> pd.DataFrame:
    """
    Limpieza conservadora. La regla de oro: se corrige lo que es error, se
    respeta lo que es rareza. Un monto gigante puede ser justo el cliente que
    el banco quiere conocer; por eso los extremos se reportan, no se amputan
    (salvo que CONFIG['PERCENTIL_WINSOR'] diga lo contrario).
    """
    apuntar(f"▸ {tabla.upper()}  ({len(df):,} filas de entrada)", 0)
    n0 = len(df)

    # a) Duplicados exactos: el mismo hecho contado dos veces infla todo lo que toca.
    df = df.drop_duplicates(ignore_index=True)
    apuntar(f"duplicados exactos eliminados: {n0 - len(df):,}")

    # b) Columnas muertas: totalmente vacías o con un único valor. No dicen nada, solo ocupan sitio.
    muertas = [c for c in df.columns if c not in rol["entidades"] and df[c].nunique(dropna=True) <= 1]
    if muertas:
        df = df.drop(columns=muertas)
        for k in ("numericas", "categoricas", "fechas"):
            rol[k] = [c for c in rol[k] if c not in muertas]
        if rol["fecha"] in muertas:
            rol["fecha"] = rol["fechas"][0] if rol["fechas"] else None
        if rol["monto"] in muertas:
            rol["monto"] = None
        apuntar(f"columnas vacías o constantes eliminadas: {muertas}")

    # c) Llaves: mismo idioma en todas las tablas, y fuera las filas sin dueño.
    for c in rol["entidades"]:
        df[c] = armonizar_llave(df[c])
    if rol["id"]:
        sin_llave = int(df[rol["id"]].isna().sum())
        if sin_llave:
            df = df[df[rol["id"]].notna()].reset_index(drop=True)
        apuntar(f"filas sin llave eliminadas: {sin_llave:,}")
        if tabla == "maestro":
            repetidas = int(df.duplicated(rol["id"]).sum())
            if repetidas:
                # Si una empresa tiene dos fichas, nos quedamos con la más completa.
                orden = df.notna().sum(axis=1).sort_values(ascending=False, kind="stable").index
                df = df.loc[orden].drop_duplicates(rol["id"]).sort_index().reset_index(drop=True)
            apuntar(f"empresas con ficha repetida resueltas: {repetidas:,}")
        elif universo:
            huerfanas = ~df[rol["id"]].isin(universo)
            apuntar(f"filas cuya empresa no figura en el maestro: {int(huerfanas.sum()):,} ({huerfanas.mean():.2%})")
            if CONFIG["DESCARTAR_HUERFANOS"] and huerfanas.any():
                df = df[~huerfanas].reset_index(drop=True)
                apuntar("→ descartadas (DESCARTAR_HUERFANOS=True)")

    # d) Fechas: en tablas de eventos, un registro sin fecha es una carta sin matasellos.
    if tabla != "maestro" and rol["fecha"]:
        sin_fecha = int(df[rol["fecha"]].isna().sum())
        if sin_fecha:
            df = df[df[rol["fecha"]].notna()].reset_index(drop=True)
        apuntar(f"filas sin fecha eliminadas: {sin_fecha:,}")
        futuras = int((df[rol["fecha"]] > pd.Timestamp.today() + pd.Timedelta(days=366)).sum())
        if futuras:
            apuntar(f"⚠ fechas a más de un año en el futuro: {futuras:,} (revisar; no se eliminan)")

    # e) Textos: 'Lima', ' lima ' y 'LIMA' son el mismo lugar. El vacío recibe nombre propio.
    for c in rol["categoricas"]:
        s = df[c].astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
        vacio = df[c].isna() | s.isin(["", "NAN", "NONE", "NULL", "N/A", "NA", "<NA>", "-"])
        df[c] = s.where(~vacio, "SIN_DATO")

    # f) Números: infinitos fuera; negativos y extremos, al reporte.
    for c in rol["numericas"]:
        s = df[c]
        inf = int(np.isinf(s.astype("float64")).sum())
        if inf:
            df[c] = s.replace([np.inf, -np.inf], np.nan)
        neg = int((s < 0).sum())
        if neg and c == rol["monto"]:
            apuntar(f"⚠ {c}: {neg:,} valores negativos (¿extornos?, ¿notas de crédito?). Se conservan.")
        p = CONFIG["PERCENTIL_WINSOR"]
        if p and s.nunique() > 20:
            techo = s.quantile(p)
            recortados = int((s > techo).sum())
            df[c] = s.clip(upper=techo)
            apuntar(f"{c}: {recortados:,} valores recortados al percentil {p}")

    apuntar(f"filas de salida: {len(df):,}  ({len(df) / max(n0, 1):.1%} de la entrada)")
    return adelgazar(df)


# %% 2. Fecha de corte · la raya en el piso
def definir_cortes(tablas: dict, roles: dict) -> tuple[list[pd.Timestamp], pd.Timestamp, int]:
    """
    Para aprender a predecir hay que simular el momento de la decisión: nos
    paramos en una fecha pasada, miramos solo hacia atrás para describir a la
    empresa y solo hacia adelante para saber si convirtió.

    Devuelve (cortes de entrenamiento, corte de scoring, meses de ventana).
    El corte de scoring es 'hoy': la última fecha conocida en toda la data.
    """
    ventana = int(CONFIG["MESES_VENTANA_TARGET"])
    ultimas = [tablas[t][r["fecha"]].max() for t, r in roles.items() if t != "maestro" and r["fecha"]]
    if not ultimas:
        apuntar("No hay fechas en ninguna tabla: se arma una sola foto, sin corte temporal.", 0)
        return [], pd.Timestamp.today().normalize(), ventana
    hoy = max(ultimas)

    rf = roles.get("factoring")
    if not rf or not rf["fecha"] or not rf["id"] or CONFIG["TARGET_EXTERNO"]:
        return [], hoy, ventana

    f = tablas["factoring"][rf["fecha"]]
    fin, inicio = f.max(), f.min()
    meses_historia = (fin.year - inicio.year) * 12 + (fin.month - inicio.month) + 1
    if meses_historia < 2 * ventana:
        ventana = max(1, meses_historia // 2)
        apuntar(f"Historia de factoring corta ({meses_historia} meses): ventana ajustada a {ventana}.", 0)

    if CONFIG["FECHAS_CORTE"]:
        cortes = [pd.Timestamp(c) for c in CONFIG["FECHAS_CORTE"]]
    else:
        cortes = [fin - pd.DateOffset(months=ventana)]
    apuntar(f"Historia de factoring: {inicio:%Y-%m-%d} → {fin:%Y-%m-%d} ({meses_historia} meses)", 0)
    apuntar(f"Cortes de entrenamiento: {[f'{c:%Y-%m-%d}' for c in cortes]} · ventana target: {ventana} meses", 0)
    apuntar(f"Corte de scoring (toda la historia): {hoy:%Y-%m-%d}", 0)
    return cortes, hoy, ventana


# %% 3. Compresión · de millones de filas a una fila por empresa
def resumir_por_empresa(df: pd.DataFrame, col: str, rol: dict, prefijo: str, corte: pd.Timestamp) -> pd.DataFrame:
    """
    El corazón del preprocesamiento. Una empresa con 4 000 transacciones no cabe
    en una fila; hay que destilarla. De cada tabla larga extraemos, por empresa:

      · cuánta actividad tuvo            (n registros, meses activos)
      · de qué magnitud                  (suma, media, máximo, dispersión de cada número)
      · hace cuánto                      (recencia y antigüedad respecto al corte)
      · con cuántos                      (contrapartes distintas: su red de negocios)
      · de qué tipo                      (reparto entre categorías de pocas opciones)
      · cómo terminó                     (último valor conocido, si la tabla es un panel mensual)

    Son agregaciones deliberadamente básicas: el ladrillo, no la arquitectura.
    Tendencias, ratios y variables de red quedan para la etapa de features.
    """
    p = f"{prefijo}_{col}" if col != rol["id"] else prefijo
    g = df.groupby(col, sort=False, observed=True)
    piezas = [g.size().rename(f"{p}_n_registros")]

    nums = rol["numericas"]
    if nums:
        ag = g[nums].agg(["sum", "mean", "max", "std"])
        ag.columns = [f"{p}_{c}_{f}" for c, f in ag.columns]
        piezas.append(ag)

    f = rol["fecha"]
    if f:
        fechas = g[f].agg(["min", "max"])
        piezas.append(((corte - fechas["max"]).dt.days).rename(f"{p}_recencia_dias"))
        piezas.append(((corte - fechas["min"]).dt.days).rename(f"{p}_antiguedad_dias"))
        meses = df[f].dt.year * 12 + df[f].dt.month
        piezas.append(meses.groupby(df[col], sort=False, observed=True).nunique().rename(f"{p}_meses_activos"))

        # Si es un panel (≈ una fila por empresa y periodo), el último valor es la foto más fresca.
        if nums and len(df) and df.duplicated([col, f]).mean() < 0.05 and g.ngroups < len(df):
            ult = df.sort_values(f, kind="stable").groupby(col, sort=False, observed=True)[nums].last()
            ult.columns = [f"{p}_{c}_ult" for c in ult.columns]
            piezas.append(ult)

    for otra in rol["entidades"]:
        if otra != col:
            piezas.append(g[otra].nunique().rename(f"{p}_n_{otra}_distintos"))

    chicas = [c for c in rol["categoricas"] if 2 <= df[c].nunique() <= CONFIG["MAX_NIVELES_CATEGORIA"]][:5]
    for c in chicas:
        reparto = pd.crosstab(df[col], df[c], normalize="index")
        reparto.columns = [f"{p}_{c}_pct_{str(v).lower().replace(' ', '_')[:20]}" for v in reparto.columns]
        piezas.append(reparto)

    out = pd.concat(piezas, axis=1)
    out.index.name = "_llave"
    return out


# %% 4 y 5. Ensamble + Target · la mesa servida
def construir_abt(tablas: dict, roles: dict, corte: pd.Timestamp, ventana: int, con_target: bool) -> pd.DataFrame:
    """
    Arma la tabla analítica para UNA fecha de corte.
    Todo lo que entra como variable ocurrió en o antes del corte; el target,
    estrictamente después. Esa disciplina es lo que hace creíble al Gini.
    """
    rm = roles.get("maestro")
    if rm and rm["id"]:
        m = tablas["maestro"]
        base = m.rename(columns={rm["id"]: "id_empresa"}).copy()
        # Fechas del maestro (constitución, alta como cliente…) → antigüedad en días al corte.
        for c in rm["fechas"]:
            base[f"mae_{c}_antiguedad_dias"] = (corte - base[c]).dt.days
            base = base.drop(columns=c)
        # Textos casi únicos (razón social, dirección) no generalizan: se dejan fuera.
        for c in rm["categoricas"]:
            if base[c].nunique() > 0.5 * len(base) and len(base) > 100:
                base = base.drop(columns=c)
        base = base.rename(columns={c: f"mae_{c}" for c in base.columns if c != "id_empresa" and not c.startswith("mae_")})
    else:
        # Sin maestro: el universo es la unión de todas las empresas vistas.
        ids = pd.unique(np.concatenate([tablas[t][c].dropna().unique() for t, r in roles.items() for c in r["entidades"]]))
        base = pd.DataFrame({"id_empresa": ids})

    for tabla in ("transacciones", "factoring", "perfil"):
        rol = roles.get(tabla)
        if tabla not in tablas or not rol or not rol["entidades"]:
            continue
        df = tablas[tabla]
        if rol["fecha"]:
            df = df[df[rol["fecha"]] <= corte]          # ← la aduana anti-fuga de información
        for col in rol["entidades"]:
            resumen = resumir_por_empresa(df, col, rol, PREFIJO[tabla], corte)
            base = base.merge(resumen, how="left", left_on="id_empresa", right_index=True)
            marca = [c for c in resumen.columns if c.endswith("_n_registros")][0]
            # Ausencia estructural: si no hay registros, la actividad es cero, no "desconocida".
            ceros = [c for c in resumen.columns if c.endswith(("_n_registros", "_sum", "_meses_activos", "_distintos"))
                     or "_pct_" in c]
            base[ceros] = base[ceros].fillna(0)
            base[marca.replace("_n_registros", "_tiene")] = (base[marca] > 0).astype("int8")
            base = base.copy()      # desfragmenta la tabla tras cada tanda de columnas nuevas

    base.insert(1, "fecha_corte", corte)

    if con_target:
        rf = roles["factoring"]
        fac = tablas["factoring"]
        fin = corte + pd.DateOffset(months=ventana)
        futuro = fac[(fac[rf["fecha"]] > corte) & (fac[rf["fecha"]] <= fin)]
        g = futuro.groupby(rf["id"], sort=False, observed=True)
        y = g.size().rename("target_n_ops").to_frame()
        if rf["monto"]:
            y["target_monto"] = g[rf["monto"]].sum()
        base = base.merge(y, how="left", left_on="id_empresa", right_index=True)
        base["target_n_ops"] = base["target_n_ops"].fillna(0).astype("int32")
        if "target_monto" in base:
            base["target_monto"] = base["target_monto"].fillna(0.0)
        base["target_conversion"] = (base["target_n_ops"] > 0).astype("int8")
    return adelgazar(base)


def adjuntar_target_externo(abt: pd.DataFrame, tablas: dict, roles: dict) -> pd.DataFrame:
    """Si el organizador ya trae el target hecho, se usa el suyo: no se discute con el árbitro."""
    te = CONFIG["TARGET_EXTERNO"]
    t, c = te["tabla"], te["columna"]
    if t == "maestro" and f"mae_{c}" in abt.columns:
        return abt.rename(columns={f"mae_{c}": "target_conversion"})
    y = tablas[t].groupby(roles[t]["id"], sort=False, observed=True)[c].max().rename("target_conversion")
    return abt.merge(y, how="left", left_on="id_empresa", right_index=True)


# %% 6. Contrato · el mapa que se le entrega a la etapa siguiente
def redactar_contrato(abt_train, abt_score, roles, archivos, cortes, hoy, ventana) -> dict:
    """
    El contrato evita el mal que origina este diseño: código que solo corre con
    los nombres de columna de quien lo escribió. Quien modele NO debe escribir
    nombres; debe leer este JSON y obedecerlo.
    """
    ref = abt_train if abt_train is not None else abt_score
    targets = [c for c in ref.columns if c.startswith("target_")]
    reservadas = {"id_empresa", "fecha_corte", *targets}
    feats = [c for c in ref.columns if c not in reservadas]
    contrato = {
        "caso": "DataFest 2026 · Factoring · ¿A qué empresas debemos contactar?",
        "grano": "una fila por empresa y fecha de corte",
        "columna_id": "id_empresa",
        "columna_fecha_corte": "fecha_corte",
        "targets": {
            "clasificacion": "target_conversion" if "target_conversion" in targets else None,
            "regresion": "target_monto" if "target_monto" in targets else None,
            "conteo": "target_n_ops" if "target_n_ops" in targets else None,
        },
        "ventana_target_meses": ventana,
        "fechas_corte_train": [f"{c:%Y-%m-%d}" for c in cortes],
        "fecha_corte_score": f"{hoy:%Y-%m-%d}",
        "features_numericas": [c for c in feats if es_numero(ref[c])],
        "features_categoricas": [c for c in feats if es_texto(ref[c])],
        # Números del maestro con pocos valores distintos: quizá son códigos (rubro 1-20, ubigeo)
        # y conviene tratarlos como categorías. La decisión se deja a quien modele.
        "features_numericas_que_parecen_codigo": [
            c for c in feats if c.startswith("mae_") and es_numero(ref[c]) and ref[c].nunique() <= 60],
        "prefijos": {"mae_": "maestro de empresas", "trx_": "transacciones",
                     "fac_": "historial de factoring ANTERIOR al corte", "per_": "perfil histórico"},
        "nulos": "NaN numérico = sin información (no imputado). Categoría vacía = 'SIN_DATO'.",
        "archivos": archivos,
        "tablas_limpias_roles": roles,
        "filas": {"abt_train": None if abt_train is None else len(abt_train), "abt_score": len(abt_score)},
        "tasa_conversion_train": None if abt_train is None or "target_conversion" not in abt_train
        else round(float(abt_train["target_conversion"].mean()), 6),
    }
    return contrato


# %% Orquesta
def main() -> None:
    t0 = time.time()
    asegurar_carpetas(CONFIG)
    tablas = cargar_todo(CONFIG)
    roles = levantar_acta_de_roles(tablas, CONFIG)

    titulo("1 · LIMPIEZA")
    universo = None
    if "maestro" in tablas and roles["maestro"]["id"]:
        tablas["maestro"] = limpiar("maestro", tablas["maestro"], roles["maestro"], None)
        universo = set(tablas["maestro"][roles["maestro"]["id"]])
    for tabla in [t for t in tablas if t != "maestro"]:
        tablas[tabla] = limpiar(tabla, tablas[tabla], roles[tabla], universo)

    archivos = {}
    for tabla, df in tablas.items():
        archivos[f"limpio_{tabla}"] = guardar_tabla(df, DIR / f"limpio_{tabla}")

    titulo("2 · FECHA DE CORTE")
    cortes, hoy, ventana = definir_cortes(tablas, roles)

    titulo("3-5 · COMPRESIÓN, ENSAMBLE Y TARGET")
    abt_train = None
    if cortes:
        fotos = []
        for corte in cortes:
            foto = construir_abt(tablas, roles, corte, ventana, con_target=True)
            apuntar(f"foto al {corte:%Y-%m-%d}: {len(foto):,} empresas × {foto.shape[1]} columnas · "
                    f"conversión {foto['target_conversion'].mean():.2%}")
            fotos.append(foto)
        abt_train = pd.concat(fotos, ignore_index=True) if len(fotos) > 1 else fotos[0]
    abt_score = construir_abt(tablas, roles, hoy, ventana, con_target=False)
    if CONFIG["TARGET_EXTERNO"]:
        abt_train = adjuntar_target_externo(abt_score.copy(), tablas, roles)
        apuntar(f"target externo adjuntado desde {CONFIG['TARGET_EXTERNO']}")
    apuntar(f"abt_score al {hoy:%Y-%m-%d}: {len(abt_score):,} empresas × {abt_score.shape[1]} columnas")
    if abt_train is None:
        apuntar("⚠ No se pudo construir target (falta factoring con fecha). Se entrega solo abt_score.")

    # Verificación final: una empresa, una fila por corte. Si esto falla, nada de lo demás vale.
    for nombre, abt in (("abt_train", abt_train), ("abt_score", abt_score)):
        if abt is not None:
            assert not abt.duplicated(["id_empresa", "fecha_corte"]).any(), f"{nombre}: hay empresas repetidas"

    titulo("6 · GUARDADO Y CONTRATO")
    if abt_train is not None:
        archivos["abt_train"] = guardar_tabla(abt_train, DIR / "abt_train")
    archivos["abt_score"] = guardar_tabla(abt_score, DIR / "abt_score")
    contrato = redactar_contrato(abt_train, abt_score, roles, archivos, cortes, hoy, ventana)
    guardar_json(contrato, DIR / "contrato_abt.json")
    apuntar(f"archivos: {archivos}", 0)
    apuntar(f"features numéricas: {len(contrato['features_numericas'])} · "
            f"categóricas: {len(contrato['features_categoricas'])}", 0)
    apuntar(f"tiempo total: {time.time() - t0:,.0f} s", 0)
    (DIR / "bitacora_prepro.txt").write_text("\n".join(BITACORA), encoding="utf-8")
    titulo(f"PREPRO terminado · todo quedó en {DIR}/")


if __name__ == "__main__":
    main()
