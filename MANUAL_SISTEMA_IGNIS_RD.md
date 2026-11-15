# 🛰️ MANUAL TÉCNICO Y CIENTÍFICO: SISTEMA FUEGORD / IGNIS-RD
### *Armonización Real de Focos de Calor MODIS & VIIRS para República Dominicana*
**NASA Space Apps Challenge 2026** · *Santo Domingo, República Dominicana 🇩🇴*

---

## 📑 TABLA DE CONTENIDOS
1. [Resumen Ejecutivo y Ficha Técnica](#1-resumen-ejecutivo-y-ficha-técnica)
2. [Nivel 1: Conceptos Fundamentales (Lo Más Básico)](#2-nivel-1-conceptos-fundamentales-lo-más-básico)
3. [Nivel 2: Ingesta, Filtrado y Limpieza de Datos](#3-nivel-2-ingesta-filtrado-y-limpieza-de-datos)
4. [Nivel 3: Agrupación Espaciotemporal y Malla Común](#4-nivel-3-agrupación-espaciotemporal-y-malla-común)
5. [Nivel 4: El Motor Matemático de Armonización (LOYO)](#5-nivel-4-el-motor-matemático-de-armonización-loyo)
6. [Nivel 5: Episodios Reales, Celdas H3 Genuinas y Huellas Satelitales](#6-nivel-5-episodios-reales-celdas-h3-genuinas-y-huellas-satelitales)
7. [Nivel 6: Comparación Física de Resolución (MODIS 1 km vs. VIIRS 375 m)](#7-nivel-6-comparación-física-de-resolución-modis-1-km-vs-viirs-375-m)
8. [Nivel 7: Infraestructura Offline, Caché de Teselas y Redundancia](#8-nivel-7-infraestructura-offline-caché-de-teselas-y-redundancia)
9. [Nivel 8: Arquitectura de Software y Código Fuente En Vivo](#9-nivel-8-arquitectura-de-software-y-código-fuente-en-vivo)
10. [Nivel 9: Manual de Usuario del Dashboard en Streamlit](#10-nivel-9-manual-de-usuario-del-dashboard-en-streamlit)
11. [Guía de Reproducibilidad y Comandos](#11-guía-de-reproducibilidad-y-comandos)

---

## 1. RESUMEN EJECUTIVO Y FICHA TÉCNICA

* **Proyecto:** FuegoRD / IGNIS-RD.
* **Reto NASA:** *Harmonization of MODIS and VIIRS Hot Spots* (NASA Space Apps Challenge 2026).
* **Filosofía Científica:** **100% de reproducibilidad, datos reales y cero simulación artificial.** Todo lo que se presenta en la aplicación se calcula directamente a partir de los datos procesados por el pipeline científico (NASA FIRMS MODIS C6.1 y VIIRS S-NPP, archivo Standard Processing, nov 2000 – ago 2026; NOAA-20/21 es una ampliación prevista). No se inventan coordenadas, ni modelos no entrenados, ni tokens falsos, ni backends inexistentes.
* **Objetivo:** Unificar 26 años de registros de anomalías térmicas (2000–2026) eliminando el salto artificial de detección ocurrido en 2012 mediante calibración estadística estacional y validación cruzada *Leave-One-Year-Out* (LOYO), proporcionando además un centro de análisis geoespacial con celdas hexagonales Uber H3 reales.
* **Beneficiarios:** Cuerpo de Bomberos Forestales (Ministerio de Medio Ambiente), Centro de Operaciones de Emergencias (COE), investigadores de cambio climático y gestores de áreas protegidas.

---

## 2. NIVEL 1: CONCEPTOS FUNDAMENTALES (LO MÁS BÁSICO)

### 2.1 ¿Qué es un foco de calor (*Active Fire / Hot Spot*)?
Un satélite no toma una "fotografía convencional" de un fuego. Mide la **radianza infrarroja de onda media (MWIR, ~3.9 µm)** emitida por la superficie terrestre. Cuando la temperatura de un punto supera un umbral estadístico respecto a sus píxeles vecinos fríos, el sensor lo clasifica como una **anomalía térmica** o foco de calor.

> **Principio de Rigor:** Un foco de calor satelital **no siempre es un incendio forestal destructivo**. Puede tratarse de una quema agrícola controlada de caña de azúcar, una fuente persistente urbana (vertedero de Duquesa) o una combustión de biomasa en un conuco.

### 2.2 ¿Qué es el FRP (*Fire Radiative Power*)?
El **FRP** es la **Potencia Radiativa del Fuego**, medida en **Megavatios (MW)**. Cuantifica la tasa instantánea de energía térmica radiada por el frente de fuego en el momento exacto del paso del satélite:
$$FRP = \frac{A_{píxel} \cdot \sigma}{a} \cdot (T_{MWIR}^b - T_{fondo}^b)$$
Un fuego con $FRP = 5\text{ MW}$ representa una quema menor de pastizal; fuegos $>80\text{ MW}$ indican incendios intensos de copa en bosques de montaña.

### 2.3 MODIS vs. VIIRS: Dos Generaciones de Sensores
| Característica | MODIS (Terra & Aqua) | VIIRS (Suomi-NPP & NOAA-20) |
| :--- | :--- | :--- |
| **Inicio Operativo** | Noviembre 2000 | Enero 2012 |
| **Resolución Espacial** | **1,000 metros (1 km)** en el nadir | **375 metros** en toda la franja |
| **Sensibilidad Térmica** | Detecta fuegos $\ge 100\text{ m}^2$ o $\ge 5\text{ MW}$ | Detecta fuegos pequeños $\ge 5\text{ m}^2$ o $\ge 0.5\text{ MW}$ |
| **Efecto Bow-Tie (Borde)** | Píxel se estira hasta **$2.0 \times 4.8\text{ km}$** | Corregido internamente por agregación de detectores |

### 2.4 El Problema Crítico: El Salto Artificial de 2012
Si se unen sin ajustar los registros crudos de MODIS (2000–2011) con los de VIIRS (2012–2026), se observa un **aumento ficticio de $\times 4.52$ en 2012** (dato real NASA FIRMS para RD).
* Este incremento no se debe al cambio climático ni a una ola repentina de incendios, sino al cambio de instrumento satelital por uno 7 veces más sensible y con píxeles mucho más pequeños.
* **Misión de FuegoRD:** Calibrar matemáticamente la serie para reconstruir una línea histórica de 26 años comparable en **focos equivalentes MODIS**, reduciendo el salto de $\times 4.52$ a **$\times 0.92$** (perfectamente alineado con el $\times 0.92$ observado por el MODIS real en el mismo periodo). Es decir, una vez corregido el cambio de instrumento, los incendios en RD **no aumentaron** en 2012: el salto era 100% artefacto del sensor.

---

## 3. NIVEL 2: INGESTA, FILTRADO Y LIMPIEZA DE DATOS

Implementado en `pipeline/ingest.py` y `pipeline/clean.py`:

```
                  DATOS CRUDOS FIRMS (NASA)
                             │
                             ▼
              [ 1. Filtro de Calidad/Confianza ]
                 MODIS ≥ 30% | VIIRS Nominal/Alto
                             │
                             ▼
              [ 2. Filtro de Tipo FIRMS ]
                 Solo tipo 0 (Vegetación Presunta)
                             │
                             ▼
              [ 3. Recorte Fronterizo Vectorial ]
                 Geofencing Polígono Oficial RD 50m
                             │
                             ▼
            [ 4. Remoción Fuentes Persistentes ]
                 Elimina Vertederos / Hornos (Duquesa)
                             │
                             ▼
                    DATOS DEPURADOS REALES
```

### 3.1 Filtros Científicos Aplicados
1. **Filtro de Confianza:** MODIS `confidence >= 30%`; VIIRS `confidence in {'n', 'h', 'nominal', 'high'}`.
2. **Filtro de Tipo (`type == 0`):** Se descartan llamaradas industriales y anomalías marinas, conservando únicamente fuego en vegetación terrestre.
3. **Recorte Territorial Vectorial (`inside_dr`):** Se aplica el algoritmo de *Ray Casting* (`Path.contains_points`) sobre el contorno oficial de República Dominicana (`static/world_50m.json`), descartando detecciones fuera de frontera.
4. **Filtro de Fuentes Persistentes (`persistent_cells`):** Identifica celdas de $0.01^\circ$ (~1 km) que emiten calor en más del 25% de todos los meses de los 26 años y en al menos 9 meses distintos del calendario (ej. Vertedero de Duquesa y chimeneas industriales), evitando que contaminen la serie de incendios forestales.

> **Zonificación por provincias oficiales (`pipeline/geo.py`):** El filtrado y las etiquetas por zona ya **no** usan rectángulos aproximados. Se emplean los polígonos oficiales de las **32 provincias** de República Dominicana (geoBoundaries ADM1, `static/dom_provinces.geojson`) con el mismo algoritmo de *Ray Casting* (`Path.contains_points`). Al seleccionar una provincia en el panel, los KPIs, la serie, el calendario y los mapas se recortan con su límite real, y cada episodio se etiqueta con la provincia exacta a la que pertenece.

---

## 4. NIVEL 3: AGRUPACIÓN ESPACIOTEMPORAL Y MALLA COMÚN

### 4.1 De Píxeles VIIRS a Eventos Térmicos (`pipeline/grid.py`)
Un incendio grande es detectado por VIIRS como 10–15 píxeles adyacentes de 375 m, mientras que MODIS solo observa 1 píxel de 1 km.
* FuegoRD agrupa los píxeles VIIRS de la misma pasada satelital a una distancia euclidiana $\le 1.0\text{ km}$ (`VIIRS_EVENT_RADIUS_KM = 1.0`).
* Los píxeles agrupados se condensan en **1 Evento Térmico VIIRS**, registrando el número de píxeles que lo formaron y sumando su FRP crudo.

### 4.2 Malla Regular Común ($0.10^\circ \approx 11\text{ km}$)
Se construye una cuadrícula común en todo el país de $0.1^\circ \times 0.1^\circ$ (~11 km de lado). Cada registro se asigna a su celda correspondiente y se agregan los conteos de forma mensual por sensor: **los datos crudos de ambos sensores nunca se suman directamente**.

---

## 5. NIVEL 4: EL MOTOR MATEMÁTICO DE ARMONIZACIÓN (LOYO)

Implementado en `pipeline/harmonize.py`. Utiliza los 14 años de solapamiento orbital (2012–2026) para calibrar la conversión de VIIRS a equivalente MODIS.

### 5.1 Modelos Estadísticos en Competencia
1. **Razón Simple de Píxeles (`razon_px`):** $\widehat{MODIS} = k \cdot VIIRS_{px}$.
2. **Razón Simple de Eventos (`razon_ev`):** $\widehat{MODIS} = k \cdot VIIRS_{ev}$.
3. **Razón Estacional de Eventos (`razon_ev_temporada`):** Calibra dos factores $k$ independientes: temporada seca (diciembre a abril) y temporada lluviosa (mayo a noviembre).
4. **Regresión Log-Lineal Estacional con Corrección de Sesgo de Duan (`loglineal_ev_temporada`):**  
   $$\ln(MODIS + 1) = a + b \cdot \ln(VIIRS_{ev} + 1)$$
   $$\text{Smear} = \frac{1}{N} \sum_{i=1}^N \exp(y_i - (a + b \cdot x_i))$$
   $$\widehat{MODIS} = \exp(a + b \cdot \ln(VIIRS_{ev} + 1)) \cdot \text{Smear} - 1$$

### 5.2 Validación Cruzada LOYO (*Leave-One-Year-Out*)
Para evitar el sobreajuste y la autocorrelación temporal:
* Para cada año del solapamiento ($2012 \dots 2026$), se entrena con los restantes años y se predice el año excluido sin haberlo visto.
* Se evalúan el MAE mensual, error porcentual anual mediano, sesgo acumulado y correlación de Pearson.
* **El sistema selecciona automáticamente el modelo con menor error para la zona.**
* **Banda de Incertidumbre (10% al 90%):** Se calcula a partir de los residuales empíricos en años no vistos.

### 5.3 Armonización de la Potencia Radiativa (FRP)
Además de armonizar el *conteo* de focos, FuegoRD armoniza la *energía* radiada. Con la misma metodología LOYO (razón simple, razón por temporada y log-lineal de Duan sobre el FRP mensual por eventos), convierte el FRP total de VIIRS a **FRP equivalente MODIS** (MW).

* **Hallazgo físico clave:** el salto artificial de 2012 es de **$\times 4.52$ en conteos** pero solo **$\times 1.20$ en FRP**. VIIRS detecta multitud de fuegos pequeños que MODIS no ve (inflan el conteo) pero aportan poca potencia radiativa; por eso la energía apenas salta. Ya armonizado, el FRP queda en **$\times 0.90$** (MODIS real $\times 0.88$), coherente con la serie de conteos.
* Implementado en `pipeline/harmonize.py` (`build_frp_model`, `harmonize_frp`), visible en el gráfico 4 de la Pestaña 3.

### 5.4 Validación Cruzada entre Plataformas VIIRS (S-NPP ↔ NOAA-20)
VIIRS vuela en dos satélites casi idénticos: Suomi-NPP (2012+) y NOAA-20 (2018+). `pipeline/crossval.py` descarga, limpia y agrupa en eventos el registro de NOAA-20 y lo compara con S-NPP en el periodo común.

* **Resultado:** la correlación de las detecciones mensuales es **0.99** (100 meses, 2018–2026), con un ratio NOAA-20/S-NPP de 1.06. Esto demuestra que la señal de fuego —y por tanto la calibración VIIRS→MODIS— es **robusta entre satélites**, no un artefacto de un solo sensor. **Los sensores no se suman:** NOAA-20 es un contraste independiente. Visible en la Pestaña 6.

---

## 6. NIVEL 5: EPISODIOS REALES, CELDAS H3 GENUINAS Y HUELLAS SATELITALES

Implementado en el nuevo módulo `pipeline/episodes.py`. Reemplaza cualquier dato simulado con cálculos reales derivados del dataset procesado.

### 6.1 Identificación de Ubicaciones de Alta Actividad (`top_locations`)
A partir de `cells.parquet`, el sistema agrupa los focos históricos y selecciona las $N$ celdas de la grilla ($0.1^\circ$) con mayor número de eventos combinados en todo el registro (ej. puntos críticos en la Cordillera Central y Sierra de Bahoruco).
* Para cada ubicación, se extraen los puntos reales de MODIS (`modis_points.parquet`) y eventos de VIIRS (`viirs_events.parquet`) dentro del radio de análisis.

### 6.2 Celdas Hexagonales Uber H3 Genuinas (`h3_cells_real`)
Utiliza la librería oficial de Uber **`h3` (v4.5+)** para proyectar cada coordenada geográfica real a su celda hexagonal de **resolución 8** (~0.46 km² de área, ~460 m de radio):
```python
cell = h3.latlng_to_cell(lat, lon, res=8)
boundary = h3.cell_to_boundary(cell)
```
* **Cero datos inventados:** Para cada celda H3, se calculan el conteo exacto de focos MODIS observados, conteo de eventos VIIRS, y la suma del FRP crudo observado por cada sensor por separado.

### 6.3 Huella Satelital Real MODIS según `scan` y `track` (`modis_footprint`)
En FIRMS, cada detección MODIS incluye los atributos físicos de la pasada:
* `scan`: Ancho transversal del píxel a lo largo del barrido del espejo (km).
* `track`: Altura del píxel a lo largo de la trayectoria orbital del satélite (km).

El sistema calcula el polígono de cobertura exacto en grados de latitud y longitud:
$$dlat = \frac{track / 2}{110.57\text{ km/°}}$$
$$dlon = \frac{scan / 2}{111.32\text{ km/°} \cdot \cos(lat)}$$
Esto dibuja el **rectángulo exacto del píxel real**, mostrando cómo se expande en el borde del barrido por el efecto *bow-tie*.

### 6.4 Monitor Satelital Global en Vivo (`pipeline/global_fires.py`)
Para extender la detección más allá de República Dominicana a escala planetaria, el sistema integra conexión directa con los feeds NRT (*Near Real Time*) de NASA LANCE / FIRMS:
* **Satélites en Vivo:** MODIS Collection 6.1 (Terra y Aqua, 1 km) y VIIRS (Suomi-NPP, NOAA-20 y NOAA-21, 375 m).
* **Cobertura Mundial:** Monitoreo continuo del planeta entero (más de 100,000 focos diarios en tiempo real) con filtros regionales predefinidos (América Central, Amazonía, América del Norte, Europa, África subsahariana, Asia, Australia) o coordenadas personalizadas (bounding box).
* **Telemetría Verdadera:** Cada detección satelital en el mapa interactivo incluye potencia radiativa (FRP en MW), temperatura de brillo infrarrojo (Kelvin y Celsius), fecha/hora UTC exacta del sobrevuelo orbital, ángulo de barrido y nivel de confianza oficial de la NASA.
* **Consulta por API Oficial:** Soporte nativo para consultar cualquier país del mundo vía endpoints REST de NASA FIRMS utilizando una `MAP_KEY`.

---

## 7. NIVEL 6: COMPARACIÓN FÍSICA DE RESOLUCIÓN (MODIS 1 KM VS. VIIRS 375 M)

La Pestaña 2 del dashboard abandona cualquier simulación de modelos de IA no entrenados y ofrece un **laboratorio de física satelital honesto**:
1. En cada ubicación de alta actividad, toma los eventos ocurridos durante el periodo de solapamiento (donde ambos satélites volaban simultáneamente).
2. Muestra la relación directa entre el número de focos que detectó MODIS y los eventos/píxeles que detectó VIIRS.
3. Evidencia visualmente por qué VIIRS divide un incendio grande en múltiples eventos o detecta fuegos de bajo FRP que caen por debajo del umbral de detección de MODIS.

---

## 8. NIVEL 7: INFRAESTRUCTURA OFFLINE, CACHÉ DE TESELAS Y REDUNDANCIA

Para resolver las limitaciones de red y tiempos de espera en servidores con latencia DNS o restricciones de CDN, FuegoRD implementa una arquitectura resiliente:

### 8.1 Leaflet Local Incrustado (`leaflet_assets`)
Los archivos `leaflet.js` (147 KB) y `leaflet.css` (14 KB) se almacenan localmente en `static/leaflet/` y se inyectan directamente en el componente HTML mediante `st.components.v1.html`.
* **Beneficio:** El mapa funciona sin depender de conexiones a `unpkg.com` o CDNs externos.

### 8.2 Descarga y Caché de Teselas Satelitales del Lado del Servidor (`fetch_basemap_tiles`)
El navegador no se conecta a servidores de mapas externos:
1. El servidor de Streamlit calcula matemáticamente las coordenadas $X, Y, Z$ de las teselas alrededor del centro del episodio mediante `_tile_xy(lat, lon, zoom)`.
2. Descarga concurrentemente las teselas con `ThreadPoolExecutor` desde ArcGIS Imagery y las guarda en disco en `data/cache/basemap_tiles/{z}_{x}_{y}.png`.
3. Convierte las teselas en imágenes codificadas en **Base64 (`data:image/png;base64,...`)** y las superpone como capas `L.imageOverlay` en el mapa.
4. **Beneficio:** Una vez descargadas, la carga es instantánea y no requiere que el navegador del usuario resuelva peticiones DNS externas.

### 8.3 Respaldo Offline Absoluto (Contorno Vectorial de RD)
Si el servidor no tiene acceso a internet o el usuario selecciona *"Sin imagen de fondo"*, la función `country_outline_rings()` lee el archivo local `static/world_50m.json` y dibuja el contorno de República Dominicana como polígonos vectoriales nativos. **El mapa es 100% operativo sin internet.**

---

## 9. NIVEL 8: ARQUITECTURA DE SOFTWARE Y CÓDIGO FUENTE EN VIVO

### 9.1 Árbol de Componentes del Proyecto
```
fuegord/
├── config.py                 # Parámetros globales y BBOX de RD
├── app.py                    # Dashboard en Streamlit (con KPIs dinámicos y Leaflet local)
├── requirements.txt          # Dependencias oficiales (incluye h3>=4.0)
├── iniciar.sh                # Script de inicio rápido en 1 clic
├── README.md                 # Documentación inicial
├── MANUAL_SISTEMA_IGNIS_RD.md# Este manual técnico y científico
├── vista_previa.html         # Vista geoespacial complementaria
├── static/
│   ├── world_50m.json        # TopoJSON de contorno territorial
│   ├── dom_provinces.geojson # Polígonos oficiales de las 32 provincias (geoBoundaries ADM1)
│   └── leaflet/              # Librería Leaflet local (js y css)
├── pipeline/
│   ├── episodes.py           # Cálculo de episodios reales, H3 y caché de teselas
│   ├── ingest.py             # Normalización de esquemas FIRMS
│   ├── clean.py              # Filtros de calidad, geofencing y persistencias
│   ├── grid.py               # Agrupación de eventos VIIRS y malla de 0.1°
│   ├── harmonize.py          # Calibración de conteos y FRP + validación LOYO
│   ├── geo.py                # Provincias oficiales: point-in-polygon y etiquetas
│   ├── crossval.py           # Validación cruzada VIIRS S-NPP vs NOAA-20
│   ├── calendar_tools.py     # Climatología, anomalías y matrices
│   ├── synthetic.py          # Simulador estocástico para desarrollo
│   └── run.py                # Orquestador del pipeline
└── data/
    ├── cache/basemap_tiles/  # Teselas satelitales cacheadas en disco
    └── processed/            # Parquet analítico (cells, points, events, loyo, noaa20_cross)
```

### 9.2 Inspección de Código en Vivo (`inspect.getsource`)
En la Pestaña 7 del dashboard, el código fuente mostrado no son fragmentos estáticos ni código simulado. Utiliza la función nativa `inspect.getsource()` de Python:
```python
st.code(inspect.getsource(hz), language="python")
st.code(inspect.getsource(ep), language="python")
```
Esto garantiza a los jueces del hackathon que **el código que están leyendo es exactamente el mismo que se está ejecutando en memoria**.

---

## 10. NIVEL 9: MANUAL DE USUARIO DEL DASHBOARD EN STREAMLIT

Al ingresar a **`http://localhost:8501/`**:

### 10.1 KPIs Dinámicos de Cabecera
Calculados en vivo según la zona y el rango de años seleccionados en la barra lateral:
1. **MODIS (1 km):** Focos observados en la selección.
2. **VIIRS eventos:** Eventos agrupados espaciotemporalmente.
3. **Celdas con actividad:** Celdas de $0.1^\circ$ con al menos un fuego en el periodo.
4. **Reducción px → evento:** Porcentaje de compactación logrado por la agrupación de eventos VIIRS.
5. **Clima en Vivo RD:** Velocidad del viento, temperatura y humedad en tiempo real vía Open-Meteo API.
6. **Salto 2012 Ajustado:** Muestra la relación ajustada ($\times 0.92$) vs la sin ajustar ($\times 4.52$).

### 10.2 Navegación por las 8 Pestañas

* **`🛰️ 1. Episodios reales + H3`**: Selector de ubicaciones reales de alta actividad, visualizador Leaflet local con capas de huellas MODIS (`scan/track`), eventos VIIRS y celdas H3 (res-8) con conteos de focos y FRP real por sensor. Alternativa para ver la malla nacional completa.
* **`📐 2. Comparación de resolución`**: Análisis físico directo de la sensibilidad y fragmentación de píxeles entre MODIS y VIIRS en las ubicaciones críticas durante el solapamiento.
* **`📈 3. Serie 26 Años & Armonización`**: Serie histórica continua 2000–2026, comparación del salto artificial vs serie corregida, banda de incertidumbre 10–90%, dispersión 1:1 y descarga en CSV.
* **`🗓️ 4. Calendario de Quemas`**: Matriz de calor anual/mensual, mes típico de quema y tabla de meses anómalos ($\ge 2\sigma$).
* **`📡 5. Reportes de terreno (demo)`**: Flujo de interacción para reporte de guardaparques, almacenado limpiamente en la sesión del navegador (`st.session_state`) sin falsos backends.
* **`🎯 6. Validación LOYO`**: Tabla con las métricas de error de los 4 métodos estadísticos y gráfico de barras año por año en datos no vistos.
* **`💻 7. Código & Arquitectura`**: Inspección en vivo del código de `harmonize.py` y `episodes.py` con `inspect.getsource()`.
* **`📖 8. Metodología`**: Documentación científica del reto NASA Space Apps 2026 y limitaciones del modelo.

---

## 11. GUÍA DE REPRODUCIBILIDAD Y COMANDOS

### 11.1 Instalación y Ejecución Rápida
```bash
# 1. Clonar o descomprimir fuegord_actualizado.zip
unzip fuegord_actualizado.zip -d fuegord
cd fuegord

# 2. Iniciar con el script automatizado
chmod +x iniciar.sh
./iniciar.sh
```

### 11.2 Comandos Manuales
```bash
# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias exactas
pip install -r requirements.txt

# Reprocesar el pipeline con datos de muestra
python -m pipeline.run --sample

# Ingestar archivo histórico real descargado de FIRMS
python -m pipeline.run --archive data/raw

# Ejecutar la aplicación de Streamlit
streamlit run app.py --server.port 8501
```

---

> **Acreditación:** Desarrollado para el **NASA Space Apps Challenge 2026** · Reto *Harmonization of MODIS and VIIRS Hot Spots*. Datos cortesía de NASA EOSDIS FIRMS, Open-Meteo y OpenStreetMap contributors. República Dominicana 🇩🇴.
