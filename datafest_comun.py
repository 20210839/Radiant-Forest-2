# -*- coding: utf-8 -*-
"""
datafest_comun.py  ·  La caja de herramientas compartida
=========================================================
DataFest 2026 · Caso Factoring: "¿A qué empresas debemos contactar?"

Este módulo es el cimiento silencioso de la casa: nadie lo aplaude, pero si
falla, todo lo demás se inclina. Aquí vive lo que el EDA y el preprocesamiento
necesitan por igual: encontrar los archivos, leerlos sin tropezar, y —sobre
todo— ADIVINAR CON CRITERIO qué papel cumple cada columna.

La idea que gobierna todo el código: NO se escribe ningún nombre de columna
a mano. Los nombres de columna son como los apodos: cambian de barrio en
barrio. Lo que no cambia es el comportamiento del dato (una llave se repite
entre tablas, una fecha se deja leer como fecha, un monto es un número con
cola larga). Por eso detectamos por conducta y no por apellido.

Si la detección se equivoca (puede pasar: es un detective, no un oráculo),
el bloque CONFIG permite corregirla en una línea, sin tocar la lógica.

Dependencias: solo pandas, numpy y matplotlib. Nada exótico, porque el día
del evento los datos llegan por USB y no conviene depender del internet.
"""
from __future__ import annotations

import gzip
import json
import re
import unicodedata
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIG · El tablero de mando
#    Todo lo que un humano podría querer cambiar está aquí y solo aquí.
#    Con todo en None, el código decide por su cuenta.
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # --- Dónde vive cada cosa -------------------------------------------------
    "DIR_DATOS": "data/raw",            # carpeta donde se copian las 4 bases del USB
    "DIR_SALIDA": "data/processed",     # aquí se guardan tablas limpias y la ABT
    "DIR_REPORTES": "reports/eda",      # aquí caen figuras, CSV e informe del EDA

    # --- Qué archivo es qué tabla --------------------------------------------
    # None = que lo deduzca por el nombre del archivo.
    # Acepta una ruta, o una lista de rutas si la tabla viene partida en trozos.
    "ARCHIVOS": {
        "maestro": None,          # ¿Quién es la empresa?
        "transacciones": None,    # ¿Con quién hace negocios?
        "factoring": None,        # ¿Ya usó factoring?
        "perfil": None,           # ¿Cómo le ha ido en el tiempo?
    },

    # --- Correcciones manuales al detective ----------------------------------
    # Se llenan SOLO si el "acta de roles" que imprime el código sale mal.
    # Los nombres van como quedan tras normalizar (minúsculas, sin tildes, con _).
    "ROLES": {
        "maestro":       {"id": None},
        "transacciones": {"id": None, "entidades": None, "fecha": None, "monto": None},
        "factoring":     {"id": None, "entidades": None, "fecha": None, "monto": None},
        "perfil":        {"id": None, "fecha": None},
    },

    # --- El reloj del problema ------------------------------------------------
    # Entrenamos mirando el pasado y preguntando por el futuro. La fecha de corte
    # es la raya en el piso: de ahí hacia atrás son variables, hacia adelante es
    # el target. None = se calcula sola (última fecha de factoring − ventana).
    "FECHAS_CORTE": None,          # p. ej. ["2025-06-30", "2025-09-30"] para varias fotos
    "MESES_VENTANA_TARGET": 3,     # cuántos meses de futuro definen "convirtió"

    # --- Si el organizador ya entrega el target hecho -------------------------
    # p. ej. {"tabla": "maestro", "columna": "flag_conversion"}
    "TARGET_EXTERNO": None,

    # --- Higiene --------------------------------------------------------------
    "DESCARTAR_HUERFANOS": False,  # True = botar filas cuya empresa no está en el maestro
    "PERCENTIL_WINSOR": None,      # p. ej. 0.999 para recortar colas; None = solo reportar
    "MAX_NIVELES_CATEGORIA": 6,    # categóricas transaccionales con ≤ N niveles se pivotean

    # --- EDA ------------------------------------------------------------------
    "FILAS_MUESTRA_GRAFICOS": 300_000,  # los gráficos se dibujan con muestra; los conteos, no
    "SEMILLA": 42,
}

# Pistas de vocabulario. No son reglas: son corazonadas que desempatan.
PISTAS_TABLA = {
    "maestro":       ["maestro", "master", "ficha", "empresa", "cliente"],
    "transacciones": ["transac", "trx", "txn", "pagos", "movim"],
    "factoring":     ["factoring", "factor", "desemb", "operac"],
    "perfil":        ["perfil", "histor", "evoluc", "mensual", "panel"],
}
PISTAS_ID = ["id", "ruc", "cod", "key", "empresa", "cliente", "doc", "cuc", "proveedor",
             "aceptante", "girador", "pagador", "origen", "destino", "emisor", "receptor"]
PISTAS_FECHA = ["fecha", "fec", "date", "periodo", "codmes", "mes", "dia", "time"]
PISTAS_MONTO = ["monto", "mto", "importe", "imp", "saldo", "venta", "valor", "desemb",
                "amount", "soles", "usd"]
PISTAS_NO_LLAVE = PISTAS_MONTO + ["cant", "num", "frec", "dias", "tasa", "plazo", "ctd", "n_"]
# En factoring, ¿quién es la empresa que adelanta su factura (a quien contactamos)?
PISTAS_PROTAGONISTA = ["proveedor", "cedente", "cliente", "pyme", "emisor", "benef", "empresa"]
PISTAS_CONTRAPARTE = ["aceptante", "pagador", "deudor", "girador", "contraparte", "destino"]

FORMATOS_FECHA = [
    "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y%m%d", "%Y-%m", "%Y%m",
    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%d.%m.%Y",
]
EXTENSIONES = {".csv", ".txt", ".tsv", ".dat", ".gz", ".zip", ".parquet", ".pq",
               ".xlsx", ".xls", ".feather", ".pkl"}


# ════════════════════════════════════════════════════════════════════════════
# 2. Utilidades menudas
# ════════════════════════════════════════════════════════════════════════════
def decir(mensaje: str, nivel: int = 0) -> None:
    """Habla en consola con sangría. Un pipeline mudo es un pipeline sospechoso."""
    print(("   " * nivel) + mensaje, flush=True)


def titulo(texto: str) -> None:
    decir("\n" + "═" * 78)
    decir(texto)
    decir("═" * 78)


def sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(texto)) if not unicodedata.combining(c))


def nombre_limpio(texto: str) -> str:
    """'Monto Transado (S/)' → 'monto_transado_s'. Le quita el maquillaje al nombre."""
    t = sin_tildes(texto).lower().strip()
    t = re.sub(r"[^0-9a-z]+", "_", t).strip("_")
    return t or "col"


def tiene_pista(nombre: str, pistas: list[str]) -> bool:
    return any(p in nombre for p in pistas)


def es_texto(s: pd.Series) -> bool:
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)) and not pd.api.types.is_numeric_dtype(s)


def es_fecha(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def es_numero(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def es_entero(s: pd.Series) -> bool:
    """¿Número sin decimales? (aunque pandas lo haya guardado como float por los nulos)."""
    if not es_numero(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        return True
    v = s.dropna()
    if v.empty:
        return False
    v = v.iloc[:50_000]
    return bool(np.all(np.mod(v.to_numpy(dtype="float64"), 1) == 0))


def asegurar_carpetas(cfg: dict) -> None:
    for k in ("DIR_SALIDA", "DIR_REPORTES"):
        Path(cfg[k]).mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 3. Encontrar y leer los archivos
# ════════════════════════════════════════════════════════════════════════════
def encontrar_archivos(cfg: dict) -> dict[str, list[Path]]:
    """
    Empareja cada una de las 4 bases con su archivo.
    Es como repartir invitaciones leyendo solo el sobre: nos guiamos por el
    nombre del archivo, y si el sobre no dice nada claro, pedimos ayuda.
    """
    carpeta = Path(cfg["DIR_DATOS"])
    asignados: dict[str, list[Path]] = {}
    for tabla, ruta in cfg["ARCHIVOS"].items():
        if ruta:
            rutas = [ruta] if isinstance(ruta, (str, Path)) else list(ruta)
            asignados[tabla] = [Path(r) for r in rutas]

    if not carpeta.exists() and len(asignados) < len(cfg["ARCHIVOS"]):
        raise FileNotFoundError(
            f"No existe la carpeta de datos '{carpeta}'. Copia ahí las bases o ajusta CONFIG['DIR_DATOS'].")

    usados = {p.resolve() for ps in asignados.values() for p in ps}
    candidatos = [p for p in sorted(carpeta.rglob("*")) if p.is_file()
                  and p.suffix.lower() in EXTENSIONES and p.resolve() not in usados] if carpeta.exists() else []

    # Puntaje: cada pista que aparece en el nombre suma; las primeras pesan más.
    puntajes = []
    for p in candidatos:
        n = nombre_limpio(p.stem)
        for tabla, pistas in PISTAS_TABLA.items():
            if tabla in asignados:
                continue
            pts = sum((len(pistas) - i) for i, pista in enumerate(pistas) if pista in n)
            if pts:
                puntajes.append((pts, tabla, p))
    tomados: set[Path] = set()
    for pts, tabla, p in sorted(puntajes, key=lambda x: -x[0]):
        if tabla not in asignados and p not in tomados:
            asignados[tabla] = [p]
            tomados.add(p)

    faltan = [t for t in cfg["ARCHIVOS"] if t not in asignados]
    sobran = [p.name for p in candidatos if p not in tomados]
    if faltan:
        decir(f"⚠ No pude ubicar archivo para: {faltan}. Archivos sin dueño: {sobran}")
        decir("  → Indica la ruta en CONFIG['ARCHIVOS']. Sigo con lo que hay.")
    elif sobran:
        decir(f"ℹ Archivos que no usé (si alguno es parte de una tabla, agrégalo en CONFIG): {sobran}")
    return asignados


def _primeras_lineas(ruta: Path, n: int = 5) -> tuple[str, str]:
    """Abre apenas una rendija del archivo para olfatear separador y codificación."""
    suf = ruta.suffix.lower()
    if suf == ".gz":
        f = gzip.open(ruta, "rb")
    elif suf == ".zip":
        z = zipfile.ZipFile(ruta)
        f = z.open(z.namelist()[0])
    else:
        f = open(ruta, "rb")
    with f:
        crudo = b"".join(f.readline() for _ in range(n))
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return crudo.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return crudo.decode("latin-1", errors="replace"), "latin-1"


def leer_tabla(ruta: Path, nrows: int | None = None) -> pd.DataFrame:
    """Lee CSV/TXT (cualquier separador), Parquet, Excel, Feather o Pickle sin que se lo pidan dos veces."""
    ruta = Path(ruta)
    suf = ruta.suffix.lower()
    if suf in (".parquet", ".pq"):
        df = pd.read_parquet(ruta)
    elif suf in (".xlsx", ".xls"):
        df = pd.read_excel(ruta, nrows=nrows)
    elif suf == ".feather":
        df = pd.read_feather(ruta)
    elif suf == ".pkl":
        df = pd.read_pickle(ruta)
    else:
        texto, enc = _primeras_lineas(ruta)
        cabecera = texto.splitlines()[0] if texto else ""
        sep = max([",", ";", "|", "\t"], key=cabecera.count)
        try:
            df = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False, nrows=nrows)
        except UnicodeDecodeError:
            df = pd.read_csv(ruta, sep=sep, encoding="latin-1", low_memory=False, nrows=nrows)
    if nrows is not None and len(df) > nrows:
        df = df.head(nrows)
    return df


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Uniforma los nombres y resuelve homónimos (dos 'monto' → 'monto', 'monto_2')."""
    vistos: dict[str, int] = {}
    nuevos = []
    for c in df.columns:
        n = nombre_limpio(c)
        if n.startswith("unnamed"):
            n = "col_sin_nombre"
        vistos[n] = vistos.get(n, 0) + 1
        nuevos.append(n if vistos[n] == 1 else f"{n}_{vistos[n]}")
    df.columns = nuevos
    return df


# ════════════════════════════════════════════════════════════════════════════
# 4. Tipado · Enseñarle a cada columna a decir lo que realmente es
# ════════════════════════════════════════════════════════════════════════════
def _texto_a_numero(s: pd.Series) -> pd.Series | None:
    """
    Rescata números disfrazados de texto: 'S/ 1,234.50', '1.234,50', '12%'.
    Respeta los códigos con ceros a la izquierda ('01', '007'): esos son
    etiquetas con forma de número, no cantidades.
    """
    m = s.dropna()
    if m.empty:
        return None
    m = m.sample(min(len(m), 5000), random_state=7).astype(str).str.strip()
    limpio = m.str.replace(r"(S/\.?|US\$|[$€%\s])", "", regex=True)
    if limpio.str.match(r"^-?\(?\d[\d.,]*\)?$").mean() < 0.95:
        return None
    if limpio.str.match(r"^0\d").mean() > 0.05:
        return None
    coma_decimal = limpio.str.contains(r",\d{1,2}$").mean() > 0.3 and limpio.str.contains(r"\.\d{1,2}$").mean() < 0.3
    t = s.astype(str).str.strip().str.replace(r"(S/\.?|US\$|[$€%\s()])", "", regex=True)
    t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False) if coma_decimal \
        else t.str.replace(",", "", regex=False)
    return pd.to_numeric(t, errors="coerce")


def _texto_a_fecha(s: pd.Series) -> pd.Series | None:
    """Prueba un llavero de formatos y se queda con la llave que abre más cerraduras."""
    m = s.dropna()
    if m.empty:
        return None
    m = m.sample(min(len(m), 2000), random_state=7).astype(str).str.strip()
    largo = m.str.len().median()
    if largo < 6 or largo > 30 or m.str.contains(r"\d").mean() < 0.9:
        return None
    mejor_fmt, mejor = None, 0.0
    for fmt in FORMATOS_FECHA:
        ok = pd.to_datetime(m, format=fmt, errors="coerce").notna().mean()
        if ok > mejor:          # '>' estricto: ante empate gana día/mes, como se escribe en Perú
            mejor_fmt, mejor = fmt, ok
    if mejor >= 0.9:
        return pd.to_datetime(s.astype(str).str.strip(), format=mejor_fmt, errors="coerce")
    return None


def _numero_a_fecha(s: pd.Series, nombre: str) -> pd.Series | None:
    """Reconoce periodos escritos como números: 202503 (AAAAMM) o 20250331 (AAAAMMDD)."""
    if not es_entero(s):
        return None
    v = s.dropna()
    if v.empty:
        return None
    mn, mx = float(v.min()), float(v.max())
    pista = tiene_pista(nombre, PISTAS_FECHA)
    if 190001 <= mn and mx <= 210012:
        if bool(((v % 100) >= 1).all() and ((v % 100) <= 12).all()) and (pista or v.nunique() <= 240):
            return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m", errors="coerce")
    if 19000101 <= mn and mx <= 21001231:
        f = pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        if f.notna().sum() >= 0.95 * s.notna().sum():
            return f
    return None


def tipar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recorre la tabla columna por columna y le devuelve a cada una su oficio:
    número, fecha o texto. Es la aduana del pipeline: nada entra mal declarado.
    """
    for c in list(df.columns):
        s = df[c]
        if es_fecha(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.astype("int8")
            continue
        if es_texto(s):
            num = _texto_a_numero(s)
            if num is not None:
                df[c] = s = num
            else:
                fec = _texto_a_fecha(s)
                if fec is not None:
                    df[c] = fec
                else:
                    df[c] = s.astype(str).str.strip().where(s.notna(), None)
                continue
        if es_numero(s):
            fec = _numero_a_fecha(s, c)
            if fec is not None:
                df[c] = fec
    return df


def adelgazar(df: pd.DataFrame) -> pd.DataFrame:
    """Dieta de memoria: con millones de filas, cada byte que sobra pesa como piedra en la mochila."""
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            df[c] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, downcast="integer")
    return df


def armonizar_llave(s: pd.Series) -> pd.Series:
    """
    Deja las llaves hablando el mismo idioma en todas las tablas:
    123, 123.0, ' 123 ' y '000123' terminan siendo la misma persona.
    """
    if es_numero(s) and es_entero(s):
        t = s.astype("Int64").astype(str)
    else:
        t = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    t = t.str.replace(r"^0+(?=\d)", "", regex=True)
    return t.where(s.notna(), None).astype(object)


# ════════════════════════════════════════════════════════════════════════════
# 5. Carga completa
# ════════════════════════════════════════════════════════════════════════════
def cargar_todo(cfg: dict, nrows: int | None = None) -> dict[str, pd.DataFrame]:
    """Lee las bases disponibles, normaliza nombres y tipa. Devuelve {tabla: DataFrame}."""
    titulo("CARGA · Abriendo las bases")
    tablas: dict[str, pd.DataFrame] = {}
    for tabla, rutas in encontrar_archivos(cfg).items():
        trozos = [normalizar_columnas(leer_tabla(r, nrows)) for r in rutas]
        df = trozos[0] if len(trozos) == 1 else pd.concat(trozos, ignore_index=True)
        df = adelgazar(tipar(df))
        tablas[tabla] = df
        mb = df.memory_usage(deep=True).sum() / 1e6
        decir(f"✔ {tabla:<14} ← {', '.join(r.name for r in rutas)}  |  "
              f"{len(df):>12,} filas × {df.shape[1]:>3} columnas  |  {mb:,.0f} MB")
    if not tablas:
        raise RuntimeError("No se cargó ninguna tabla. Revisa CONFIG['DIR_DATOS'] y CONFIG['ARCHIVOS'].")
    return tablas


# ════════════════════════════════════════════════════════════════════════════
# 6. El acta de roles · ¿Quién es quién en cada tabla?
# ════════════════════════════════════════════════════════════════════════════
def _candidata_a_llave(s: pd.Series) -> bool:
    if es_fecha(s):
        return False
    if es_numero(s) and not es_entero(s):
        return False
    return True


def _id_del_maestro(df: pd.DataFrame) -> str | None:
    """La llave del maestro es la columna que no se repite: una empresa, una fila, un nombre propio."""
    n = len(df)
    mejores = []
    for i, c in enumerate(df.columns):
        s = df[c]
        if not _candidata_a_llave(s) or s.notna().sum() == 0:
            continue
        unicidad = s.nunique(dropna=True) / max(s.notna().sum(), 1)
        if unicidad >= 0.98 and s.notna().mean() >= 0.98:
            mejores.append((not tiene_pista(c, PISTAS_ID), i, c))
    if mejores:
        return sorted(mejores)[0][2]
    # Plan B: maestro con duplicados → la columna con pista de ID más diversa.
    cands = [(-(df[c].nunique() / max(n, 1)), c) for c in df.columns
             if _candidata_a_llave(df[c]) and tiene_pista(c, PISTAS_ID)]
    return sorted(cands)[0][1] if cands else None


def _entidades_de(df: pd.DataFrame, universo: set | None, id_maestro: str | None) -> list[str]:
    """
    Busca en una tabla las columnas que nombran empresas.
    La prueba es de parentesco, no de apellido: si los valores de la columna
    aparecen en el maestro, esa columna habla de empresas, se llame como se llame.
    """
    halladas = []
    for c in df.columns:
        s = df[c]
        if not _candidata_a_llave(s):
            continue
        con_pista = tiene_pista(c, PISTAS_ID) or c == id_maestro
        parece_medida = tiene_pista(c, PISTAS_NO_LLAVE) and not con_pista
        if universo:
            u = pd.Series(s.dropna().unique())
            if len(u) < 20:
                continue
            muestra = armonizar_llave(u.sample(min(len(u), 3000), random_state=7))
            parentesco = muestra.isin(universo).mean()
            if (con_pista and parentesco >= 0.3) or (not parece_medida and parentesco >= 0.8):
                halladas.append(c)
        elif con_pista and s.nunique() > 50:
            halladas.append(c)
    return halladas


def _elegir_fecha(df: pd.DataFrame) -> str | None:
    fechas = [c for c in df.columns if es_fecha(df[c])]
    if not fechas:
        return None
    # Preferimos la fecha más completa; el nombre desempata.
    return sorted(fechas, key=lambda c: (-df[c].notna().mean(), not tiene_pista(c, PISTAS_FECHA)))[0]


def _elegir_monto(df: pd.DataFrame, numericas: list[str]) -> str | None:
    if not numericas:
        return None
    con_pista = [c for c in numericas if tiene_pista(c, PISTAS_MONTO)]
    pool = con_pista or numericas
    # Entre candidatas, el monto suele ser la de mayor dispersión (la que más "respira").
    return sorted(pool, key=lambda c: -float(np.nan_to_num(df[c].std())))[0]


def levantar_acta_de_roles(tablas: dict[str, pd.DataFrame], cfg: dict) -> dict:
    """
    Levanta el 'acta de roles': para cada tabla dice cuál es la llave, cuáles
    son las columnas de empresa, la fecha que manda, el monto principal y qué
    columnas son números o categorías.

    ESTA ES LA SALIDA QUE UN HUMANO DEBE LEER ANTES DE CONFIAR EN TODO LO DEMÁS.
    Si un rol salió torcido, se endereza en CONFIG['ROLES'] y se vuelve a correr.
    """
    titulo("ACTA DE ROLES · Quién es quién (valídalo con tus ojos)")
    manual = cfg.get("ROLES", {})
    roles: dict[str, dict] = {}

    id_maestro, universo = None, None
    if "maestro" in tablas:
        id_maestro = (manual.get("maestro") or {}).get("id") or _id_del_maestro(tablas["maestro"])
        if id_maestro:
            universo = set(armonizar_llave(tablas["maestro"][id_maestro]).dropna().unique())

    if universo is None:
        # Sin maestro no hay padrón oficial: lo improvisamos con las columnas que, por su
        # nombre, declaran ser identificadores. Sirve para reconocer a sus parientes en otras tablas.
        semillas = [armonizar_llave(pd.Series(df[c].dropna().unique()))
                    for df in tablas.values() for c in df.columns
                    if _candidata_a_llave(df[c]) and tiene_pista(c, PISTAS_ID) and df[c].nunique() > 50]
        if semillas:
            universo = set(pd.concat(semillas).unique())
            decir("ℹ No hay maestro: el universo de empresas se armó con las columnas tipo ID de las demás tablas.")

    for tabla, df in tablas.items():
        m = manual.get(tabla) or {}
        if tabla == "maestro":
            entidades = [id_maestro] if id_maestro else []
            principal = id_maestro
        else:
            entidades = m.get("entidades") or _entidades_de(df, universo, id_maestro)
            principal = m.get("id")
            if principal and principal not in entidades:
                entidades = [principal] + entidades
            if not principal and entidades:
                # Protagonista: primero por vocabulario; si no, la columna con más empresas
                # distintas (los proveedores son multitud; los grandes pagadores, un puñado).
                def orden(c):
                    return (c != id_maestro, not tiene_pista(c, PISTAS_PROTAGONISTA),
                            tiene_pista(c, PISTAS_CONTRAPARTE), -df[c].nunique())
                principal = sorted(entidades, key=orden)[0]

        fecha = m.get("fecha") or _elegir_fecha(df)
        fechas = [c for c in df.columns if es_fecha(df[c])]
        numericas = [c for c in df.columns if es_numero(df[c]) and c not in entidades]
        categoricas = [c for c in df.columns if es_texto(df[c]) and c not in entidades]
        # Solo las tablas de eventos tienen un "monto principal"; en maestro y perfil no aplica.
        monto = m.get("monto") or (_elegir_monto(df, numericas) if tabla in ("transacciones", "factoring") else None)

        roles[tabla] = {"id": principal, "entidades": entidades, "fecha": fecha, "fechas": fechas,
                        "monto": monto, "numericas": numericas, "categoricas": categoricas}

        decir(f"▸ {tabla.upper()}")
        decir(f"llave principal : {principal}", 1)
        if tabla != "maestro":
            decir(f"columnas empresa: {entidades}", 1)
            decir(f"fecha que manda : {fecha}   (todas: {fechas})", 1)
            decir(f"monto principal : {monto}", 1)
        decir(f"numéricas ({len(numericas)}): {numericas[:8]}{' …' if len(numericas) > 8 else ''}", 1)
        decir(f"categóricas ({len(categoricas)}): {categoricas[:8]}{' …' if len(categoricas) > 8 else ''}", 1)
        if not principal:
            decir("⚠ SIN LLAVE DETECTADA → complétala en CONFIG['ROLES']", 1)
    return roles


# ════════════════════════════════════════════════════════════════════════════
# 7. Guardado
# ════════════════════════════════════════════════════════════════════════════
def guardar_tabla(df: pd.DataFrame, ruta_sin_extension: Path) -> str:
    """Guarda en Parquet si hay motor disponible; si no, en Pickle. Devuelve el nombre real del archivo."""
    ruta_sin_extension = Path(ruta_sin_extension)
    try:
        destino = ruta_sin_extension.with_suffix(".parquet")
        df.to_parquet(destino, index=False)
    except Exception:
        destino = ruta_sin_extension.with_suffix(".pkl")
        df.to_pickle(destino)
    return destino.name


def guardar_json(objeto: dict, ruta: Path) -> None:
    def _serializable(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.strftime("%Y-%m-%d")
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    Path(ruta).write_text(json.dumps(objeto, indent=2, ensure_ascii=False, default=_serializable), encoding="utf-8")
