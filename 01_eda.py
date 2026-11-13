# -*- coding: utf-8 -*-
"""
01_eda.py  ·  Análisis exploratorio
====================================
DataFest 2026 · Caso Factoring: "¿A qué empresas debemos contactar?"

Antes de modelar hay que escuchar. Este script es la primera conversación con
los datos: no les pide nada todavía, solo los deja hablar. Recorre las cuatro
bases —la ficha de cada empresa, su red de pagos, su historia con el factoring
y su evolución mes a mes— y deja por escrito lo que encuentra.

Qué responde, en orden:
  1. Radiografía     · ¿De qué tamaño es cada tabla y qué tan completa viene?
  2. Variables       · ¿Cómo se reparte cada número y cada categoría?
  3. Tiempo          · ¿Qué periodo cubre y con qué ritmo llegan los registros?
  4. Puentes         · ¿Las tablas se reconocen entre sí por su llave?
  5. Embudo          · ¿Cuántas empresas hay, cuántas transan, cuántas ya usan factoring?
  6. Concentración   · ¿El negocio está repartido o vive en pocas manos?
  7. Señales         · ¿Qué rasgos del maestro separan a quien usa factoring de quien no?

Uso:   python 01_eda.py          (o celda por celda en VS Code con los "# %%")
Salida: reports/eda/  →  INFORME_EDA.md, figuras .png y tablas .csv

No hay un solo nombre de columna escrito a mano: todo sale del acta de roles.
"""
# %% Preparativos
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # dibuja a archivo; no necesita pantalla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datafest_comun import (CONFIG, armonizar_llave, asegurar_carpetas, cargar_todo, decir,
                            es_entero, levantar_acta_de_roles, titulo)

# Paleta sobria: un azul para contar, un naranja para señalar. El color acentúa, no decora.
AZUL, NARANJA, GRIS, TINTA = "#2a78d6", "#eb6834", "#c9c8c2", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRIS, "axes.labelcolor": TINTA, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.color": TINTA, "ytick.color": TINTA,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9,
})

DIR = Path(CONFIG["DIR_REPORTES"])
INFORME: list[str] = []          # aquí se va tejiendo el informe en Markdown


def anotar(texto: str = "") -> None:
    INFORME.append(texto)


def guardar_fig(nombre: str) -> None:
    plt.savefig(DIR / f"{nombre}.png")
    plt.close()
    anotar(f"![{nombre}]({nombre}.png)\n")


def tabla_md(df: pd.DataFrame, max_filas: int = 40) -> str:
    """Convierte un DataFrame chico en tabla Markdown (sin depender de 'tabulate')."""
    d = df.head(max_filas).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:,.4g}")
    filas = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * d.shape[1]]
    filas += ["| " + " | ".join(map(str, f)) + " |" for f in d.astype(str).itertuples(index=False)]
    return "\n".join(filas) + "\n"


def muestra(df: pd.DataFrame) -> pd.DataFrame:
    """Para dibujar basta una cucharada bien revuelta; para contar, usamos la olla entera."""
    n = CONFIG["FILAS_MUESTRA_GRAFICOS"]
    return df if len(df) <= n else df.sample(n, random_state=CONFIG["SEMILLA"])


# %% 1. Radiografía · tamaño, tipos, vacíos y repetidos
def radiografia(tabla: str, df: pd.DataFrame, rol: dict) -> pd.DataFrame:
    """
    El chequeo de signos vitales. Una columna con 60 % de nulos no es una
    variable: es un rumor. Mejor saberlo ahora que descubrirlo en el modelo.
    """
    n = len(df)
    resumen = pd.DataFrame({
        "columna": df.columns,
        "tipo": [str(t) for t in df.dtypes],
        "pct_nulos": (df.isna().mean() * 100).round(2).values,
        "n_unicos": [df[c].nunique(dropna=True) for c in df.columns],
    })
    resumen["rol"] = [
        "llave" if c == rol["id"] else "empresa" if c in rol["entidades"] else
        "fecha" if c in rol["fechas"] else "numerica" if c in rol["numericas"] else "categorica"
        for c in df.columns]
    resumen["ejemplo"] = [str(df[c].dropna().iloc[0])[:28] if df[c].notna().any() else "" for c in df.columns]
    resumen["alerta"] = np.select(
        [resumen["pct_nulos"] >= 99.9, resumen["n_unicos"] <= 1, resumen["pct_nulos"] > 50],
        ["vacía", "constante", "muy incompleta"], default="")
    resumen.to_csv(DIR / f"{tabla}_radiografia.csv", index=False)

    duplicadas = int(df.duplicated().sum()) if n <= 3_000_000 else int(df.head(3_000_000).duplicated().sum())
    anotar(f"## {tabla.capitalize()}\n")
    anotar(f"- **Filas:** {n:,} · **Columnas:** {df.shape[1]} · "
           f"**Memoria:** {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
    anotar(f"- **Filas duplicadas exactas:** {duplicadas:,} ({duplicadas / max(n, 1):.2%})")
    anotar(f"- **Llave:** `{rol['id']}` · **Fecha:** `{rol['fecha']}` · **Monto:** `{rol['monto']}`\n")
    anotar(tabla_md(resumen))
    decir(f"{tabla}: {n:,} filas, {duplicadas:,} duplicadas, "
          f"{int((resumen['pct_nulos'] > 0).sum())} columnas con nulos")

    con_nulos = resumen[resumen["pct_nulos"] > 0].sort_values("pct_nulos").tail(25)
    if not con_nulos.empty:
        plt.figure(figsize=(6.5, max(1.8, 0.28 * len(con_nulos))))
        plt.barh(con_nulos["columna"], con_nulos["pct_nulos"], color=AZUL, height=0.6)
        plt.xlabel("% de valores nulos"); plt.xlim(0, 100)
        plt.title(f"{tabla} · ¿Dónde hay silencios? (nulos por columna)")
        guardar_fig(f"{tabla}_nulos")
    return resumen


# %% 2. Variables · cómo se reparte cada una
def variables_numericas(tabla: str, df: pd.DataFrame, rol: dict) -> None:
    """
    Los montos en banca casi nunca siguen la campana: son una llanura con una
    cordillera al fondo. Por eso miramos percentiles y asimetría, y cuando la
    cola es larga dibujamos en escala logarítmica para que el valle también se vea.
    """
    nums = rol["numericas"]
    if not nums:
        return
    d = df[nums]
    desc = d.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    desc["asimetria"] = d.skew(numeric_only=True)
    desc["pct_ceros"] = (d == 0).mean() * 100
    desc["pct_negativos"] = (d < 0).mean() * 100
    desc = desc.round(4)
    desc.to_csv(DIR / f"{tabla}_numericas.csv")
    anotar(f"### {tabla} · Variables numéricas\n")
    anotar(tabla_md(desc.reset_index().rename(columns={"index": "variable"})[
        ["variable", "mean", "50%", "1%", "99%", "max", "asimetria", "pct_ceros", "pct_negativos"]]))

    m = muestra(df)
    elegidas = sorted(nums, key=lambda c: -df[c].notna().mean())[:12]
    cols = 3
    filas = int(np.ceil(len(elegidas) / cols))
    fig, ejes = plt.subplots(filas, cols, figsize=(10, 2.5 * filas), squeeze=False)
    for ax, c in zip(ejes.ravel(), elegidas):
        v = m[c].dropna().astype("float64")
        cola_larga = len(v) > 0 and v.min() >= 0 and abs(v.skew()) > 2 and v.nunique() > 20
        if cola_larga:
            v = np.log10(v[v > 0])
            ax.set_xlabel("log10 (cola larga)")
        ax.hist(v, bins=40, color=AZUL, edgecolor="white", linewidth=0.4)
        ax.set_title(c[:32], fontsize=9)
    for ax in ejes.ravel()[len(elegidas):]:
        ax.set_visible(False)
    fig.suptitle(f"{tabla} · Distribución de variables numéricas", fontweight="bold", y=1.0)
    fig.tight_layout()
    guardar_fig(f"{tabla}_distribuciones")

    # Correlaciones: buscamos variables que cantan la misma canción (redundancia).
    utiles = [c for c in nums if df[c].nunique() > 1][:25]
    if len(utiles) >= 3:
        corr = m[utiles].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(0.36 * len(utiles) + 3, 0.36 * len(utiles) + 2.2))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(utiles))); ax.set_xticklabels([u[:18] for u in utiles], rotation=90)
        ax.set_yticks(range(len(utiles))); ax.set_yticklabels([u[:18] for u in utiles])
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman")
        ax.set_title(f"{tabla} · ¿Quién se mueve con quién?")
        guardar_fig(f"{tabla}_correlacion")
        pares = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
        gemelas = pares[pares.abs() >= 0.9].sort_values(key=np.abs, ascending=False).head(15)
        if not gemelas.empty:
            anotar("**Pares casi gemelos (|ρ| ≥ 0.90), candidatos a quedarse con uno solo:**\n")
            anotar(tabla_md(gemelas.rename("spearman").reset_index().rename(
                columns={"level_0": "var_a", "level_1": "var_b"})))


def variables_categoricas(tabla: str, df: pd.DataFrame, rol: dict) -> None:
    """Cada categoría es un barrio: importa saber cuáles están poblados y cuáles son casas sueltas."""
    cats = [c for c in rol["categoricas"] if 1 < df[c].nunique() <= 5000][:9]
    if not cats:
        return
    anotar(f"### {tabla} · Variables categóricas\n")
    cols = 3
    filas = int(np.ceil(len(cats) / cols))
    fig, ejes = plt.subplots(filas, cols, figsize=(11, 2.9 * filas), squeeze=False)
    for ax, c in zip(ejes.ravel(), cats):
        vc = df[c].fillna("(nulo)").value_counts()
        top = vc.head(10)[::-1]
        ax.barh([str(i)[:22] for i in top.index], top.values / len(df) * 100, color=AZUL, height=0.65)
        ax.set_title(f"{c[:26]} · {len(vc):,} niveles", fontsize=9)
        ax.set_xlabel("% de filas")
        anotar(f"- `{c}`: {len(vc):,} niveles · el más común es **{vc.index[0]}** ({vc.iloc[0] / len(df):.1%}) · "
               f"niveles con < 1 % de filas: {int((vc / len(df) < 0.01).sum()):,}")
    for ax in ejes.ravel()[len(cats):]:
        ax.set_visible(False)
    fig.suptitle(f"{tabla} · Categorías más frecuentes", fontweight="bold", y=1.0)
    fig.tight_layout()
    anotar()
    guardar_fig(f"{tabla}_categorias")


# %% 3. Tiempo · el pulso de cada tabla
def linea_de_tiempo(tabla: str, df: pd.DataFrame, rol: dict) -> None:
    """
    Una tabla con fechas es una película, no una foto. Aquí vemos si la cinta
    está completa: meses con huecos, saltos bruscos o un final truncado son
    pistas de cómo armar (o no) la ventana de entrenamiento.
    """
    f = rol["fecha"]
    if not f:
        return
    fechas = df[f].dropna()
    if fechas.empty:
        return
    por_mes = fechas.dt.to_period("M").value_counts().sort_index()
    eje = por_mes.index.to_timestamp()
    anotar(f"### {tabla} · Línea de tiempo (`{f}`)\n")
    anotar(f"- **Desde** {fechas.min():%Y-%m-%d} **hasta** {fechas.max():%Y-%m-%d} · {len(por_mes)} meses con datos")
    huecos = pd.period_range(por_mes.index.min(), por_mes.index.max(), freq="M").difference(por_mes.index)
    anotar(f"- **Meses sin registros dentro del rango:** {len(huecos)} {list(map(str, huecos[:6])) if len(huecos) else ''}\n")

    paneles = [("Registros por mes", por_mes.values)]
    if rol["monto"]:
        suma = df.groupby(df[f].dt.to_period("M"))[rol["monto"]].sum().sort_index()
        paneles.append((f"Suma mensual de {rol['monto']}", suma.reindex(por_mes.index).values))
    fig, ejes = plt.subplots(len(paneles), 1, figsize=(8.5, 2.5 * len(paneles)), sharex=True, squeeze=False)
    for ax, (nombre, valores) in zip(ejes.ravel(), paneles):
        ax.plot(eje, valores, color=AZUL, linewidth=2, marker="o", markersize=3)
        ax.set_title(f"{tabla} · {nombre}")
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    guardar_fig(f"{tabla}_tiempo")

    # ¿Es un panel (una fila por empresa y mes)? Lo decimos porque cambia cómo se agrega.
    if rol["id"]:
        por_empresa = df.groupby(rol["id"], sort=False, observed=True)[f].nunique()
        anotar(f"- **Periodos por empresa:** mediana {por_empresa.median():.0f} · "
               f"mín {por_empresa.min()} · máx {por_empresa.max()}\n")


# %% 4. Puentes · ¿las tablas se reconocen entre sí?
def puentes(tablas: dict, roles: dict) -> dict[str, set]:
    """
    Cuatro bases son cuatro islas hasta que la llave tiende el puente.
    Medimos cuántas empresas del maestro aparecen en cada tabla y cuántas
    llaves de cada tabla son 'huérfanas' (no figuran en el maestro).
    """
    anotar("## Puentes entre tablas (integridad de llaves)\n")
    conjuntos: dict[str, set] = {}
    for tabla, rol in roles.items():
        for col in rol["entidades"]:
            conjuntos[f"{tabla}.{col}"] = set(armonizar_llave(tablas[tabla][col]).dropna().unique())
    base_nombre = next((k for k in conjuntos if k.startswith("maestro.")), None)
    if not base_nombre:
        anotar("_No hay maestro (o no se detectó su llave): no se puede medir cobertura._\n")
        return conjuntos
    base = conjuntos[base_nombre]
    filas = []
    for nombre, cj in conjuntos.items():
        if nombre == base_nombre:
            continue
        comun = len(cj & base)
        filas.append({"columna": nombre, "empresas_distintas": len(cj),
                      "pct_del_maestro_presente": round(comun / max(len(base), 1) * 100, 2),
                      "pct_huerfanas": round((len(cj) - comun) / max(len(cj), 1) * 100, 2)})
    cobertura = pd.DataFrame(filas)
    cobertura.to_csv(DIR / "puentes_cobertura.csv", index=False)
    anotar(f"Empresas en el maestro: **{len(base):,}**\n")
    anotar(tabla_md(cobertura))
    decir("Cobertura de llaves:\n" + cobertura.to_string(index=False))
    return conjuntos


# %% 5. Embudo · de todas las empresas a las que ya usan factoring
def embudo(conjuntos: dict[str, set], roles: dict) -> None:
    """
    El caso pregunta '¿cuánto se amplía la base?'. Para responder hay que saber
    primero de dónde partimos: del universo total, cuántas empresas dejan huella
    transaccional y cuántas ya cruzaron la puerta del factoring. La brecha entre
    ambos escalones es, literalmente, el terreno por conquistar.
    """
    def de(tabla, solo_principal=False):
        rol = roles.get(tabla)
        if not rol:
            return set()
        cols = [rol["id"]] if solo_principal and rol["id"] else rol["entidades"]
        return set().union(*[conjuntos.get(f"{tabla}.{c}", set()) for c in cols]) if cols else set()

    universo = de("maestro") or set().union(*conjuntos.values())
    pasos = [("Universo de empresas", universo),
             ("Con perfil histórico", de("perfil") & universo),
             ("Con transacciones", de("transacciones") & universo),
             ("Ya usó factoring (protagonista)", de("factoring", True) & universo)]
    pasos = [(n, c) for n, c in pasos if n == "Universo de empresas" or c]
    df = pd.DataFrame({"etapa": [p[0] for p in pasos], "empresas": [len(p[1]) for p in pasos]})
    df["pct_del_universo"] = (df["empresas"] / max(len(universo), 1) * 100).round(2)
    df.to_csv(DIR / "embudo.csv", index=False)
    anotar("## Embudo de oportunidad\n")
    anotar(tabla_md(df))

    con_trx, con_fac = de("transacciones") & universo, de("factoring", True) & universo
    if con_trx and con_fac:
        campo = len(con_trx - con_fac)
        anotar(f"**Lectura de negocio:** {campo:,} empresas transan pero nunca usaron factoring. "
               f"Ese es el campo sin sembrar: equivale a **{campo / max(len(con_fac), 1):.1f}×** la base que ya lo usa.\n")

    plt.figure(figsize=(7, 0.6 * len(df) + 1))
    colores = [AZUL] * (len(df) - 1) + [NARANJA]
    plt.barh(df["etapa"][::-1], df["empresas"][::-1], color=colores[::-1], height=0.6)
    for y, (v, p) in enumerate(zip(df["empresas"][::-1], df["pct_del_universo"][::-1])):
        plt.text(v, y, f"  {v:,}  ({p:.1f} %)", va="center", fontsize=8, color=TINTA)
    plt.xlim(0, df["empresas"].max() * 1.3)
    plt.title("Embudo · del universo a quienes ya usan factoring")
    plt.xlabel("Número de empresas")
    guardar_fig("embudo")


# %% 6. Concentración · ¿pocas manos, mucho monto?
def concentracion(tablas: dict, roles: dict) -> None:
    """
    La vieja regla del 80/20 puesta a prueba. Si un puñado de empresas mueve
    casi todo el monto, 'cliente más valioso' y 'cliente más probable' pueden
    ser personas distintas, y eso decide cómo se prioriza la fuerza comercial.
    """
    anotar("## Concentración del monto\n")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hubo = False
    for (tabla, color) in (("transacciones", AZUL), ("factoring", NARANJA)):
        rol = roles.get(tabla)
        if not rol or not rol["id"] or not rol["monto"]:
            continue
        por_emp = tablas[tabla].groupby(rol["id"], sort=False, observed=True)[rol["monto"]].sum()
        por_emp = por_emp[por_emp > 0].sort_values(ascending=False)
        if por_emp.empty:
            continue
        acum = por_emp.cumsum() / por_emp.sum()
        x = np.arange(1, len(acum) + 1) / len(acum)
        ax.plot(x * 100, acum.values * 100, color=color, linewidth=2, label=f"{tabla} ({rol['monto']})")
        top10 = acum.iloc[max(int(len(acum) * 0.10) - 1, 0)]
        top1 = acum.iloc[max(int(len(acum) * 0.01) - 1, 0)]
        anotar(f"- **{tabla}**: el 1 % de empresas concentra el **{top1:.1%}** del monto; el 10 %, el **{top10:.1%}**.")
        hubo = True
    if not hubo:
        plt.close()
        anotar("_No se identificó llave y monto para medir concentración._\n")
        return
    ax.plot([0, 100], [0, 100], color=GRIS, linewidth=1, linestyle="--", label="reparto parejo")
    ax.set_xlabel("% de empresas (de mayor a menor monto)")
    ax.set_ylabel("% acumulado del monto")
    ax.set_title("Curva de concentración")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    anotar()
    guardar_fig("concentracion")


# %% 7. Señales · qué rasgos del maestro acompañan al uso de factoring
def senales(tablas: dict, roles: dict) -> None:
    """
    Primer vistazo al 'por qué' que pide el jurado. Cruzamos cada rasgo de la
    ficha de la empresa con un hecho simple: ¿alguna vez usó factoring?
    No es causalidad, es brújula: indica hacia dónde mirar con el modelo.
    """
    if "maestro" not in tablas or "factoring" not in tablas:
        return
    rm, rf = roles["maestro"], roles["factoring"]
    if not rm["id"] or not rf["id"]:
        return
    usuarios = set(armonizar_llave(tablas["factoring"][rf["id"]]).dropna().unique())
    m = tablas["maestro"].drop_duplicates(rm["id"])
    uso = armonizar_llave(m[rm["id"]]).isin(usuarios).astype(int)
    tasa_base = uso.mean()
    anotar("## Señales: rasgos del maestro vs. uso histórico de factoring\n")
    anotar(f"Tasa base (empresas del maestro que alguna vez usaron factoring): **{tasa_base:.2%}**\n")

    # Categóricas, y numéricas discretas que se comportan como categorías (segmento 1-5, flags).
    cats = [c for c in rm["categoricas"] if 2 <= m[c].nunique() <= 30]
    cats += [c for c in rm["numericas"] if 2 <= m[c].nunique() <= 12 and es_entero(m[c])]
    cats = cats[:6]
    filas_resumen = []
    if cats:
        cols = 2
        filas = int(np.ceil(len(cats) / cols))
        fig, ejes = plt.subplots(filas, cols, figsize=(10.5, 3.1 * filas), squeeze=False)
        for ax, c in zip(ejes.ravel(), cats):
            g = pd.DataFrame({"cat": m[c].astype(object).where(m[c].notna(), "(nulo)").astype(str), "uso": uso})
            t = g.groupby("cat")["uso"].agg(["mean", "size"])
            t = t[t["size"] >= max(30, 0.002 * len(m))].sort_values("mean").tail(12)
            if t.empty:
                ax.set_visible(False)
                continue
            ax.barh([str(i)[:24] for i in t.index], t["mean"] * 100, color=AZUL, height=0.65)
            ax.axvline(tasa_base * 100, color=NARANJA, linewidth=1.5, linestyle="--")
            ax.set_title(f"{c[:30]}", fontsize=9)
            ax.set_xlabel("% que usó factoring  (línea naranja = tasa base)")
            filas_resumen.append({"variable": c, "nivel_mas_propenso": t.index[-1],
                                  "tasa_pct": round(t["mean"].iloc[-1] * 100, 2),
                                  "veces_la_base": round(t["mean"].iloc[-1] / max(tasa_base, 1e-9), 2)})
        for ax in ejes.ravel()[len(cats):]:
            ax.set_visible(False)
        fig.suptitle("¿Dónde se concentra el uso de factoring?", fontweight="bold", y=1.0)
        fig.tight_layout()
        guardar_fig("senales_categoricas")

    # Numéricas continuas: tasa de uso por quintil (¿sube la tasa cuando sube la variable?).
    for c in [c for c in rm["numericas"] if m[c].nunique() > 12][:8]:
        try:
            q = pd.qcut(m[c], 5, duplicates="drop")
        except (ValueError, TypeError):
            continue
        t = uso.groupby(q, observed=True).mean()
        if len(t) >= 3:
            filas_resumen.append({"variable": c, "nivel_mas_propenso": f"quintil {int(np.argmax(t.values)) + 1} de {len(t)}",
                                  "tasa_pct": round(t.max() * 100, 2),
                                  "veces_la_base": round(t.max() / max(tasa_base, 1e-9), 2)})
    if filas_resumen:
        res = pd.DataFrame(filas_resumen).sort_values("veces_la_base", ascending=False)
        res.to_csv(DIR / "senales_resumen.csv", index=False)
        anotar(tabla_md(res))


# %% Orquesta · todo junto, en orden
def main() -> None:
    asegurar_carpetas(CONFIG)
    tablas = cargar_todo(CONFIG)
    roles = levantar_acta_de_roles(tablas, CONFIG)

    anotar("# Informe EDA · DataFest 2026 · Caso Factoring\n")
    anotar("> Documento generado por `01_eda.py`. Las cifras salen de la data completa; "
           "los gráficos de distribución, de una muestra aleatoria.\n")

    titulo("EDA · Tabla por tabla")
    for tabla, df in tablas.items():
        rol = roles[tabla]
        radiografia(tabla, df, rol)
        variables_numericas(tabla, df, rol)
        variables_categoricas(tabla, df, rol)
        linea_de_tiempo(tabla, df, rol)

    titulo("EDA · Mirada cruzada")
    conjuntos = puentes(tablas, roles)
    embudo(conjuntos, roles)
    concentracion(tablas, roles)
    senales(tablas, roles)

    (DIR / "INFORME_EDA.md").write_text("\n".join(INFORME), encoding="utf-8")
    titulo(f"EDA terminado · abre {DIR / 'INFORME_EDA.md'}")


if __name__ == "__main__":
    main()
