# IGNIS-RD 🔥🛰️

### Sistema Predictivo de Incendios Forestales & Meteorología en Tiempo Real (NASA FIRMS + NASA POWER + Open-Meteo)

Plataforma científica y operativa para República Dominicana y monitoreo global desarrollada para el **NASA Space Apps Challenge**.
Combina 26 años de observaciones satelitales armonizadas (MODIS y VIIRS de NASA FIRMS), climatología física reanalizada (NASA POWER), estaciones meteorológicas en tiempo real (Open-Meteo) y el cálculo matemático del Fire Weather Index (FWI) canadiense (Van Wagner 1987) para pronosticar focos de calor y riesgo de propagación.

## El problema en una línea

VIIRS (375 m, desde 2012) detecta varias veces más focos que MODIS (1 km, desde 2000).
Unir los dos registros tal cual inventa un aumento de incendios en 2012. FuegoRD convierte
VIIRS a "focos equivalentes MODIS", valida esa conversión con años que no usó para
calibrar y entrega una serie continua de 26 años con su banda de incertidumbre.

## Empezar en 3 pasos

```bash
pip install -r requirements.txt
python -m pipeline.run --sample      # datos SIMULADOS, no necesita MAP_KEY (~1 min)
streamlit run app.py                 # abre http://localhost:8501
```

La app ya trae **datos reales de NASA FIRMS** procesados en `data/processed/`
(MODIS y VIIRS S-NPP, nov 2000 – ago 2026), así que puedes saltarte el paso 2 y abrir
la app directamente. El paso 2 con `--sample` regenera la carpeta con datos *simulados*
(útil para desarrollar sin MAP_KEY); para volver a los datos reales usa `--api` o
`--archive` (ver abajo).

## Usar datos reales

**Opción A, archivo histórico (recomendada para 20+ años).**
1. Entra a la descarga de archivo de FIRMS (firms.modaps.eosdis.nasa.gov/download/).
2. Pide MODIS C6.1 y VIIRS S-NPP 375 m para *Dominican Republic*, de 2000 a hoy, en CSV.
3. Pon los CSV en `data/raw/` y ejecuta:

```bash
python -m pipeline.run --archive data/raw
```

**Opción B, API de FIRMS con MAP_KEY** (gratis en firms.modaps.eosdis.nasa.gov/api/map_key/).

```bash
export FIRMS_MAP_KEY=tu_clave        # en Windows: set FIRMS_MAP_KEY=tu_clave
python -m pipeline.run --api --start 2001-01-01 --end 2025-12-31
```

La API devuelve pocos días por consulta, así que 25 años son miles de llamadas. Las
respuestas se guardan en `data/cache/` y el proceso se puede retomar si se corta.

## Cómo funciona

| Paso | Archivo | Qué hace |
| --- | --- | --- |
| 1. Ingesta | `pipeline/ingest.py` | Lee CSV de FIRMS o la API y normaliza MODIS y VIIRS a un esquema común |
| 2. Limpieza | `pipeline/clean.py` | Confianza, tipo FIRMS, recorte a RD, fuentes persistentes (vertederos, industrias) |
| 3. Eventos y grilla | `pipeline/grid.py` | Une píxeles VIIRS a <1 km en eventos; tabla celda (0,1°) × mes |
| 4. Armonización | `pipeline/harmonize.py` | Prueba 4 métodos (conteos) + 3 (FRP), valida dejando un año fuera, elige el mejor y calcula la banda |
| 5. Calendario | `pipeline/calendar_tools.py` | Matriz mes × año, mes típico, meses anómalos, frases resumen |
| 6. Provincias | `pipeline/geo.py` | Polígonos oficiales de RD (geoBoundaries ADM1) para filtrar y etiquetar por provincia (point-in-polygon) |
| 7. Validación cruzada | `pipeline/crossval.py` | Compara VIIRS S-NPP vs NOAA-20 en 2018+ (robustez entre satélites) |
| 8. Monitor Global en Vivo | `pipeline/global_fires.py` | Detección planetaria NRT en tiempo real desde satélites MODIS y VIIRS (NASA FIRMS) |
| 9. Modelo Predictivo IA (Global & RD) | `pipeline/predictive_model.py` | Predicción proactiva 24h–72h con Canadian FWI, HistGradientBoosting, vectores de avance y cobertura mundial |
| App | `app.py` | Streamlit: detector global en vivo, modelo predictivo IA (Global & RD), serie 26 años, calendario, validación y metodología |

Todos los parámetros (área, filtros, grilla, temporada seca, zonas) están en `config.py`.

## Resultado con datos reales (NASA FIRMS, nov 2000 – ago 2026)

Tras limpieza: **30.653 focos MODIS** y **81.089 píxeles VIIRS → 55.917 eventos**.

| | Salto en 2012 |
| --- | --- |
| Unir sin ajustar | ×4,52 |
| Serie armonizada | ×0,92 |
| Lo que MODIS sigue viendo | ×0,92 |

La serie armonizada (×0,92) coincide **exactamente** con lo que el propio MODIS siguió
observando (×0,92): el salto ×4,52 era puro artefacto del cambio de sensor, no un
aumento de incendios en República Dominicana.

Método elegido por validación deja-un-año-fuera (LOYO): **razón simple de píxeles VIIRS**,
con error mediano del total anual en años no vistos del **7,1 %** y correlación mensual
0,96. Los cuatro métodos compiten y el de menor error mensual gana automáticamente; con
otro recorte temporal o espacial el método elegido puede cambiar.

## Más allá de los conteos

**Modelo Predictivo IA de Próximas Zonas en Riesgo (24h a 72h · Cobertura Mundial & RD).**
A diferencia de los satélites MODIS y VIIRS, que solo detectan anomalías térmicas cuando el fuego ya inició (0h de anticipación), el modelo predictivo de FuegoRD es proactivo:
- Integra el Canadian Fire Weather Index (FWI: humedad de combustible fino FFMC, propagación por viento ISI e índice de sequía).
- Emplea un clasificador *HistGradientBoosting* calibrado sobre 141.825 observaciones históricas (ROC-AUC 0.80, PR-AUC 0.66 en conjunto de prueba ciego independiente 2023–2026).
- Proyecta vectores de avance térmico y velocidad de propagación perimetral (km/24h) en cualquier región del planeta (Amazonía, Norteamérica, Europa mediterránea, África subsahariana, Australia y el Caribe).

**FRP armonizado (potencia radiativa).** Además de *cuántos* focos, armonizamos *cuánta
energía* radian. El salto crudo de 2012 es ×4,52 en conteos pero solo **×1,20 en FRP**:
VIIRS detecta muchos fuegos pequeños que MODIS no ve (inflan el conteo) pero aportan poca
potencia. Ya armonizado, el FRP queda en **×0,90** (MODIS real ×0,88). Ver pestaña 3.

**Provincias oficiales.** El filtrado y las etiquetas usan los polígonos oficiales de las
32 provincias de RD (geoBoundaries ADM1, `static/dom_provinces.geojson`) con
point-in-polygon real, no rectángulos aproximados.

**Validación cruzada VIIRS S-NPP ↔ NOAA-20.** Las detecciones mensuales de los dos
sensores VIIRS (2018+) correlacionan a **0,99**, confirmando que la señal de fuego —y la
calibración— es robusta entre satélites, no un artefacto de uno solo. Ver pestaña 6.

## Ideas para el hackathon (14–15 nov)

- Máscara de nubes para separar "sin fuego" de "no se pudo ver".
- Añadir VIIRS NOAA-21 (además de NOAA-20) para el periodo reciente.
- Versión en inglés de la app para los jueces globales.

## Notas

- `static/world_50m.json` es el contorno de países que usa el mapa (copiado del paquete
  `sane-topojson` que usa Plotly). Va dentro del proyecto para que el mapa funcione sin
  internet y para recortar los datos al territorio dominicano.
- `static/dom_provinces.geojson` son los polígonos oficiales de las 32 provincias de RD
  (geoBoundaries ADM1, licencia abierta). Van dentro del proyecto para filtrar y etiquetar
  por provincia sin depender de internet.
- Un foco no es un incendio ni área quemada: es un píxel caliente al paso del satélite.
