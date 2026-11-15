"""Configuración central de FuegoRD.

Todo lo que un miembro del equipo podría querer ajustar está aquí:
área de estudio, fuentes FIRMS, filtros, tamaño de la grilla y zonas.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"          # CSV descargados de FIRMS (archivo histórico)
DATA_CACHE = ROOT / "data" / "cache"      # respuestas de la API, una por bloque de días
DATA_OUT = ROOT / "data" / "processed"    # salidas del pipeline que lee la app

# --- Área de estudio -------------------------------------------------------
COUNTRY_ISO3 = "DOM"
COUNTRY_NAME = "República Dominicana"
# Caja que contiene a RD (oeste, sur, este, norte). Incluye algo de Haití y mar;
# por eso se filtra luego con la frontera aproximada (ver pipeline/clean.py).
BBOX = (-72.10, 17.40, -68.25, 20.00)

# --- Fuentes FIRMS -----------------------------------------------------------
# Standard Processing (SP) = archivo científico, el adecuado para series largas.
SOURCES = {
    "MODIS": "MODIS_SP",          # Terra + Aqua, 1 km, desde nov 2000 / jul 2002
    "VIIRS": "VIIRS_SNPP_SP",     # Suomi-NPP, 375 m, desde ene 2012
}
FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api"

# --- Filtros de calidad ------------------------------------------------------
MODIS_MIN_CONFIDENCE = 30          # 0-100; <30 = baja confianza
VIIRS_KEEP_CONFIDENCE = {"n", "h", "nominal", "high"}
KEEP_FIRMS_TYPE = {0}              # 0 = fuego de vegetación presunto (descarta volcanes,
                                   # fuentes estáticas y costa afuera cuando la columna existe)

# Fuentes persistentes (vertederos, industrias): celda de ~1 km con focos en una
# fracción alta de todos los meses del registro.
PERSISTENT_CELL_DEG = 0.01
PERSISTENT_MONTH_FRACTION = 0.25

# --- Grilla y eventos --------------------------------------------------------
GRID_DEG = 0.1                     # celda de ~11 km para mapas y agregados
VIIRS_EVENT_RADIUS_KM = 1.0        # focos VIIRS a <1 km el mismo paso = un evento

# --- Armonización --------------------------------------------------------------
OVERLAP_START = 2012               # primer año completo de VIIRS S-NPP
DRY_MONTHS = {12, 1, 2, 3, 4}      # temporada seca (aprox.) para calibrar por estación
MIN_REGION_MODIS = 150             # focos MODIS mínimos para calibrar una zona por separado

# --- Zonas de interés -------------------------------------------------------
# El filtrado y las etiquetas por provincia usan los POLÍGONOS OFICIALES de
# República Dominicana (geoBoundaries ADM1) en static/dom_provinces.geojson,
# vía pipeline/geo.py (point-in-polygon). "Todo el país" = sin filtro.
# Este dict solo conserva la opción nacional; las provincias se cargan del
# GeoJSON. Admite además rectángulos (w,s,e,n) por compatibilidad si se añaden.
ZONES = {
    "Todo el país": None,
}

MONTHS_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
             "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MONTHS_FULL_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                  "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
