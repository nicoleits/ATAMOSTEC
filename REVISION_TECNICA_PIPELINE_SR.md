# Revisión Técnica del Pipeline de Procesamiento de Datos y Cálculo del Soiling Ratio (SR)

## PARTE 1: RESUMEN NARRATIVO (Estilo Artículo Científico)

### Procesamiento, Filtrado e Integración de Datos para el Cálculo del SR

El sistema de análisis de soiling implementado en este repositorio procesa datos provenientes de múltiples fuentes instrumentales para estimar el Soiling Ratio (SR) mediante metodologías eléctricas, ópticas y óptico-gravimétricas. El pipeline de procesamiento se estructura en etapas secuenciales que abarcan desde la lectura de archivos crudos hasta la generación de series temporales consolidadas para análisis estadístico.

**Fuentes de Datos y Lectura:** Los datos se leen desde archivos CSV almacenados en el directorio `datos/`, incluyendo curvas IV de módulos fotovoltaicos (`raw_pvstand_iv_data.csv`), mediciones de DustIQ (`raw_dustiq_data.csv`), irradiancias de celdas de referencia (`refcells_data.csv`), datos del Soiling Kit (`soiling_kit_raw_data.csv`), transmitancias de vidrios fotovoltaicos (`raw_pv_glasses_data.csv`), y datos de temperatura procesados (`data_temp.csv`). Todos los archivos utilizan timestamps en formato UTC, y las funciones de lectura (`pd.read_csv`) convierten automáticamente los índices temporales a `DatetimeIndex` con zona horaria UTC.

**Sincronización e Integración Temporal:** Las series temporales se sincronizan mediante remuestreo a frecuencias comunes (1 minuto para datos de alta resolución, diario y semanal para agregaciones). El sistema utiliza principalmente el cuantil 25 (Q25) para agregaciones semanales, minimizando el impacto de valores atípicos. Las ventanas horarias de análisis varían según el método: PVStand y RefCells utilizan franjas fijas (13:00-18:00 UTC), mientras que DustIQ y PV Glasses emplean ventanas relativas al mediodía solar calculadas mediante la clase `medio_dia_solar` con intervalos típicos de ±60 minutos alrededor del mediodía real.

**Filtros y Control de Calidad:** Se aplican múltiples capas de filtrado. Umbrales de irradiancia mínima (200-300 W/m² para RefCells y PV Glasses, 700 W/m² para limpieza en PVStand) eliminan datos nocturnos y condiciones de baja irradiancia. Los SR se filtran por rangos válidos (80-105% para RefCells, 70-101% para PVStand, >90% para Soiling Kit, >70% para DustIQ). La eliminación de outliers utiliza el método IQR (Interquartile Range) con factores de 1.5-3.0, y se aplican filtros de consistencia basados en valores mínimos de potencia (≥170W para Pmax en PVStand) y corriente (≥0.5A para Isc en PVStand).

**Cálculo de Parámetros Eléctricos y Correcciones:** Las curvas IV se procesan para extraer ISC, VOC, Pmpp, Imp y Vmp mediante funciones como `calculate_iv_parameters()` que calculan la potencia como P = V × I y identifican el punto de máxima potencia. Las correcciones a condiciones estándar (STC) siguen la norma IEC 60891, aplicando coeficientes de temperatura: α_ISC = 0.0004 K⁻¹ para corrección de corriente de cortocircuito y β_Pmax = -0.0037 K⁻¹ (o +0.0037 según configuración) para potencia máxima, con temperatura de referencia T_ref = 25°C.

**Definición y Cálculo del SR:** El SR se calcula de forma consistente como la razón entre el módulo/celda sucia y la referencia, multiplicada por 100. Para métodos eléctricos: SR_ISC = (Isc_sucio / Isc_referencia) × 100 y SR_Pmax = (Pmax_sucio / Pmax_referencia) × 100. DustIQ proporciona SR directamente desde el sensor (`SR_C11_Avg`). Para celdas de referencia: SR = (Irradiancia_sucia / Irradiancia_limpia) × 100. El método óptico-gravimétrico (PV Glasses) calcula SR como transmitancia relativa: SR = (R_FCi / REF) × 100, donde REF es el promedio de dos celdas de referencia limpias (R_FC1_Avg y R_FC2_Avg), y se correlaciona con masas depositadas medidas gravimétricamente.

**Integración Temporal:** Los SR instantáneos se agregan a escalas diarias y semanales. Para agregación diaria se utiliza la media aritmética (`resample('D').mean()`), mientras que para agregación semanal se emplea el cuantil 25 (`resample('1W').quantile(0.25)`) para robustez estadística. Algunos métodos aplican ventanas horarias específicas antes de la agregación (por ejemplo, 13:00-18:00 UTC para PVStand, o ventanas de mediodía solar para DustIQ y PV Glasses).

**Salida Final:** Las series finales de SR se guardan en formato CSV y Excel en el directorio `datos_procesados_analisis_integrado_py/`, organizadas por método (subdirectorios: `dustiq/`, `ref_cells/`, `pvstand/`, `soiling_kit/`, `pv_glasses/`). Los archivos consolidados incluyen series minutales filtradas, agregaciones diarias (media y Q25), y agregaciones semanales (Q25). Los scripts de análisis estadístico (`statistical_deviation_analyzer.py`, `soiling_intercomparison.py`) utilizan estos archivos CSV como entrada para realizar intercomparaciones mediante ANOVA, R², RMSE, MBE y CCC.

---

## PARTE 2: RESUMEN TÉCNICO DETALLADO

### 1. FUENTES DE DATOS Y LECTURA

#### Archivos de Entrada (CSV en `datos/`):
- **PVStand IV**: `raw_pvstand_iv_data.csv`
  - Columnas: `_time`, `_measurement` (o `module`), `Imax`, `Pmax`, `Umax`
  - Formato tiempo: `%Y-%m-%d %H:%M:%S%z` (UTC)
  - Función lectura: `pd.read_csv()` en `pvstand_analyzer.py:133`
  
- **DustIQ**: `raw_dustiq_data.csv`
  - Columnas: `timestamp`, `SR_C11_Avg`, `SR_C12_Avg`
  - Formato tiempo: `%Y-%m-%d %H:%M:%S` (UTC naive después de conversión)
  - Función lectura: `pd.read_csv(index_col='timestamp')` en `dustiq_analyzer.py:66`
  - Lectura por chunks: 50,000 filas por chunk para archivos grandes
  
- **Celdas de Referencia**: `refcells_data.csv`
  - Columnas: `timestamp`, `1RC409(w.m-2)`, `1RC410(w.m-2)`, `1RC411(w.m-2)`, `1RC412(w.m-2)`, etc.
  - Formato tiempo: `%Y-%m-%d %H:%M:%S%z` (UTC)
  - Función lectura: `pd.read_csv(index_col=settings.REFCELLS_TIME_COLUMN)` en `ref_cells_analyzer.py:45`
  
- **Soiling Kit**: `soiling_kit_raw_data.csv`
  - Columnas: `timestamp`, `Isc(e)`, `Isc(p)`, `Te(C)`, `Tp(C)`
  - Formato tiempo: `%Y-%m-%d %H:%M:%S%z` (UTC)
  - Función lectura: `pd.read_csv()` en `soiling_kit_analyzer.py:52`
  
- **PV Glasses**: `raw_pv_glasses_data.csv`
  - Columnas: `_time`, `R_FC1_Avg`, `R_FC2_Avg`, `R_FC3_Avg`, `R_FC4_Avg`, `R_FC5_Avg`
  - Formato tiempo: `%Y-%m-%d %H:%M:%S%z` (UTC)
  - Función lectura: `pl.read_csv()` (Polars) en `pv_glasses_analyzer.py`
  
- **Temperatura**: `data_temp.csv` (generado por `download_temp_only.ipynb`)
  - Columnas: `TIMESTAMP`, `1TE416(C)`, `1TE418(C)`
  - Formato tiempo: `%Y-%m-%d %H:%M:%S.%f` (UTC)
  - Función lectura: `pd.read_csv()` en `pvstand_analyzer.py:270`

#### Procesamiento de Curvas IV (IV600):
- **Archivo procesado**: `processed_iv600_data.csv`
  - Columnas: `fecha`, `modulo`, `voltajes` (array string), `corrientes` (array string)
  - Función procesamiento: `process_iv600_data()` en `preprocessing_iv600.py:146`
  - Extracción parámetros: `calculate_iv_parameters()` en `preprocessing_iv600.py:34`
    - `pmp = max(V × I)`
    - `isc = max(I)`
    - `voc = max(V)`
    - `imp = I[idx_pmp]` donde `idx_pmp = argmax(P)`
    - `vmp = V[idx_pmp]`

### 2. SINCRONIZACIÓN E INTEGRACIÓN DE SERIES

#### Remuestreo Temporal:
- **Frecuencia base**: 1 minuto (`PVSTAND_RESAMPLE_FREQ_MINUTES = 1`)
- **Agregación diaria**: `resample('D').mean()` o `resample('D').quantile(0.25)`
- **Agregación semanal**: `resample('1W', origin='start').quantile(0.25)` (Q25)
- **Agregación 3 días**: `resample('3D').quantile(0.25)`

#### Ventanas Horarias:
- **PVStand**: `between_time('13:00', '18:00')` UTC (config: `PVSTAND_FILTER_START_TIME`, `PVSTAND_FILTER_END_TIME`)
- **RefCells**: `between_time('13:00', '18:00')` UTC
- **Soiling Kit**: `between_time('06:00', '18:00')` UTC (config: `SOILING_KIT_FILTER_START_TIME`, `SOILING_KIT_FILTER_END_TIME`)
- **DustIQ**: Múltiples franjas:
  - Fijas: `10:00-11:00`, `12:00-13:00`, `14:00-15:00`, `16:00-17:00` UTC
  - Mediodía solar: ±60 minutos alrededor del mediodía real (calculado por `medio_dia_solar`)
- **PV Glasses**: Mediodía solar real con intervalo de ±60 minutos (`PV_GLASSES_SOLAR_NOON_INTERVAL_MINUTES = 60`)

#### Cálculo de Mediodía Solar:
- Clase: `medio_dia_solar` en `analysis/classes_codes.py`
- Utilidad: `UtilsMedioDiaSolar` en `utils/solar_time.py`
- Parámetros sitio: Latitud = -23.506°, Longitud = -69.079°, Altitud = 1380 m
- Ventana típica: ±2.5 horas (300 minutos) alrededor del mediodía

### 3. FILTROS Y CONTROL DE CALIDAD

#### Umbrales de Irradiancia:
- **RefCells**: `MIN_IRRADIANCE = 200 W/m²` (config: `settings.py:267`)
- **PV Glasses**: `PV_GLASSES_SR_IRRADIANCE_THRESHOLD = 300 W/m²` (config: `settings.py:162`)
- **PVStand**: `PVSTAND_GHI_THRESHOLD_CLEANING = 700 W/m²` (config: `settings.py:124`)

#### Filtros de SR:
- **RefCells**: `SR_MIN = 80.0%`, `SR_MAX = 105.0%` (config: `REFCELLS_SR_MIN_FILTER = 0.80`, `REFCELLS_SR_MAX_FILTER = 1.05`)
- **PVStand**: `SR_MIN = 70%`, `SR_MAX = 101%` (config: `PVSTAND_SR_MIN_FILTER_THRESHOLD = 0.7`, `PVSTAND_SR_MAX_FILTER_THRESHOLD = 1.01`)
- **Soiling Kit**: `SR > 90%` (config: `SOILING_KIT_SR_LOWER_THRESHOLD = 90.0`)
- **DustIQ**: `SR > 0%` (config: `DUSTIQ_SR_FILTER_THRESHOLD = 0`), pero en práctica se filtra `SR > 70%` en análisis intercomparación
- **IV600**: `SR >= 93%` y `SR <= 101%` (hardcoded en `analisis_iv600_fixed.py:317`)

#### Eliminación de Outliers (IQR):
- **Método**: `Q1 - k×IQR` a `Q3 + k×IQR`
- **PV Glasses**: `k = 1.5` (config: `PV_GLASSES_REMOVE_OUTLIERS_IQR = True`, implementado en `pv_glasses_analyzer.py:444-466`)
- **PVStand (temperatura)**: `k = 3.0` (hardcoded en `pvstand_analyzer.py:532`)
- **Análisis estadístico**: `k = 1.5` (config: `STATISTICAL_DEVIATION_IQR_FACTOR = 1.5`)

#### Filtros de Consistencia Eléctrica:
- **PVStand Pmax**: `Pmax >= 170 W` (hardcoded en `pvstand_analyzer.py:402, 539`)
- **PVStand Isc referencia**: `Isc_reference >= 0.5 A` (hardcoded en `pvstand_analyzer.py:409`)
- **Soiling Kit**: `Isc_soiled.abs() > 1e-6` y `Isc_ref > 0` (hardcoded en `soiling_kit_analyzer.py:184`)

#### Filtros de Factor de Corrección de Temperatura:
- **PVStand**: `0.5 < factor_corr < 2.0` (hardcoded en `pvstand_analyzer.py:523`)

#### Eliminación de Datos Nocturnos:
- Implícito mediante filtros de irradiancia mínima y ventanas horarias

### 4. CÁLCULO DE PARÁMETROS ELÉCTRICOS Y CORRECCIONES

#### Extracción de Parámetros IV:
- **Función**: `calculate_iv_parameters(voltages, currents)` en `preprocessing_iv600.py:34`
- **Cálculos**:
  ```python
  P = V × I
  pmp = max(P)
  isc = max(I)
  voc = max(V)
  idx_pmp = argmax(P)
  imp = I[idx_pmp]
  vmp = V[idx_pmp]
  ```

#### Correcciones a STC (IEC 60891):
- **Coeficientes de temperatura**:
  - `α_ISC = 0.0004 K⁻¹` (Soiling Kit: `SOILING_KIT_ALPHA_ISC_CORR`, PVStand: `PVSTAND_ALPHA_ISC_CORR = -0.0004`)
  - `β_Pmax = -0.0037 K⁻¹` (PVStand: `PVSTAND_BETA_PMAX_CORR = +0.0037` o `-0.0037` según configuración)
  - `T_ref = 25.0°C` (config: `SOILING_KIT_TEMP_REF_C`, `PVSTAND_TEMP_REF_CORRECTION_C`)

- **Fórmulas de corrección**:
  - **Isc corregida**: `Isc_corr = Isc × (1 + α_ISC × (T_ref - T_celda))`
    - Implementado en: `soiling_kit_analyzer.py:176-177`, `pvstand_analyzer.py:166-167`
  - **Pmax corregida**: `Pmax_corr = Pmax × (1 + β_Pmax × (T_ref - T_celda))`
    - Implementado en: `pvstand_analyzer.py:521-522`

#### Factor de Forma (Fill Factor):
- Calculado desde curvas IV: `FF = (Pmax) / (Isc × Voc)`
- Columna en datos PVStand: `FactorDeForma` (config: `PVSTAND_FILL_FACTOR_COLUMN`)

### 5. DEFINICIÓN Y CÁLCULO DEL SOILING RATIO (SR)

#### Métodos Eléctricos:

**Soiling Kit** (`soiling_kit_analyzer.py:180-191`):
- **SR Raw**: `SR_Raw = (Isc(p) / Isc(e)) × 100`
  - Donde `Isc(p)` = protegido (referencia), `Isc(e)` = expuesto (sucio)
- **SR Corregido**: `SR_TempCorrected = (Isc_Ref_Corrected / Isc_Soiled_Corrected) × 100`
  - Con corrección de temperatura aplicada previamente

**PVStand** (`pvstand_analyzer.py:415-450`):
- **SR Isc (sin corrección)**: `SR_ISC = (Isc_sucio / Isc_referencia) × 100`
- **SR Isc (corregido)**: `SR_ISC_Corrected = (Isc_sucio_corr / Isc_ref_corr) × 100`
- **SR Pmax (sin corrección)**: `SR_Pmax = (Pmax_sucio / Pmax_referencia) × 100`
- **SR Pmax (corregido)**: `SR_Pmax_Corrected = (Pmax_sucio_corr / Pmax_ref_corr) × 100`
- **Offset Pmax**: `SR_Pmax_Corrected_Final = SR_Pmax_Corrected + 3.0%` (config: `PVSTAND_PMAX_SR_OFFSET = 3.0`)
- **Normalización**: Si `PVSTAND_NORMALIZE_SR_FLAG = True`, se normaliza a 100% en fecha de referencia (`PVSTAND_NORMALIZE_SR_REF_DATE_STR = '2024-08-01'`)

**IV600** (`analisis_iv600_fixed.py:315-327`):
- **SR Pmp**: `SR_Pmp = (Pmax_sucio / Pmax_referencia) × 100`
- **SR Isc**: `SR_Isc = (Isc_sucio / Isc_referencia) × 100`
- Módulos: 1MD434 (sucio) vs 1MD439 (referencia), 1MD440 (sucio) vs 1MD439 (referencia)

#### Método Óptico (DustIQ):
- **SR directo**: `SR = SR_C11_Avg` (columna del sensor, ya en porcentaje)
- **Filtrado**: `SR > DUSTIQ_SR_FILTER_THRESHOLD` (típicamente 0, pero se aplica filtro > 70% en análisis)
- Implementado en: `dustiq_analyzer.py:112`

#### Método Óptico (Celdas de Referencia):
- **SR**: `SR = (Irradiancia_sucia / Irradiancia_limpia) × 100`
- **Celdas sucias**: `1RC410(w.m-2)`, `1RC411(w.m-2)` (config: `REFCELLS_SOILED_COLUMNS_TO_ANALYZE`)
- **Celda limpia (referencia)**: `1RC412(w.m-2)` (config: `REFCELLS_REFERENCE_COLUMN`)
- Implementado en: `ref_cells_analyzer.py:134`
- **Ajuste opcional**: Si `REFCELLS_ADJUST_TO_100_FLAG = True`, se ajusta el primer valor a 100% (`_adjust_series_start_to_100()` en `ref_cells_analyzer.py:17`)

#### Método Óptico-Gravimétrico (PV Glasses):
- **Cálculo SR**: `SR_R_FCi = (R_FCi_Avg / REF) × 100`
  - Donde `REF = (R_FC1_Avg + R_FC2_Avg) / 2` (promedio de dos celdas de referencia)
  - `R_FCi_Avg` son transmitancias de vidrios sucios (i = 3, 4, 5)
- **Filtro irradiancia**: `REF >= 300 W/m²` (config: `PV_GLASSES_SR_IRRADIANCE_THRESHOLD`)
- Implementado en: `pv_glasses_analyzer.py:1064-1080`, `pv_glasses_analyzer_q25.py:428-432`
- **Correlación con masa**: Los SR se correlacionan con masas depositadas medidas gravimétricamente:
  - `SR_R_FC3` ↔ `Masa_C_Referencia`
  - `SR_R_FC4` ↔ `Masa_B_Referencia`
  - `SR_R_FC5` ↔ `Masa_A_Referencia`
  - Mapeo en: `settings.py:164-168`, `pv_glasses_analyzer.py:768-784`

### 6. INTEGRACIÓN TEMPORAL (DIARIA / SEMANAL)

#### Agregación Diaria:
- **Media aritmética**: `resample('D').mean()`
  - Soiling Kit: `sr_daily_raw_mean`, `sr_daily_corrected_mean` (`soiling_kit_analyzer.py:220-221`)
  - RefCells: No se usa media diaria, solo Q25
- **Cuantil 25**: `resample('D').quantile(0.25)`
  - RefCells: `df_daily_sr_q25` (`ref_cells_analyzer.py:166`)
  - DustIQ: `sr_diario_en_franja` para mediodía solar (`dustiq_analyzer.py:452`)
  - PV Glasses: Implementado en análisis Q25

#### Agregación Semanal:
- **Cuantil 25**: `resample('1W', origin='start').quantile(0.25)`
  - Soiling Kit: `sr_weekly_raw_q25`, `sr_weekly_corrected_q25` (`soiling_kit_analyzer.py:228-229`)
  - RefCells: `df_weekly_sr_q25` con `resample('W-SUN').quantile(0.25)` (`ref_cells_analyzer.py:169`)
  - DustIQ: `sr_dustiq_semanal` (`dustiq_analyzer.py:322`), `sr_mediodia_semanal` (`dustiq_analyzer.py:576`)
  - PVStand: `resample('1W').quantile(graph_quantile)` donde `graph_quantile = 0.25` (`pvstand_analyzer.py:958`)
  - IV600: `resample('1W').quantile(0.25)` (`analisis_iv600_fixed.py`)

#### Agregación 3 Días:
- **Cuantil 25**: `resample('3D').quantile(0.25)`
  - PVStand: Implementado en gráficos (`pvstand_analyzer.py:972`)

#### Ventanas Horarias para Agregación:
- **PVStand**: Datos filtrados a 13:00-18:00 UTC antes de agregación
- **RefCells**: Datos filtrados a 13:00-18:00 UTC antes de agregación
- **DustIQ**: Múltiples opciones:
  - Franjas fijas: `between_time('14:00','18:00')` antes de `resample('1W')` (`dustiq_analyzer.py:322`)
  - Mediodía solar: Ventana ±60 minutos alrededor del mediodía real antes de agregación
- **PV Glasses**: Ventana de mediodía solar (±60 minutos) antes de agregación

#### Promedios Ponderados:
- **No se utilizan**: El sistema emplea promedios simples o cuantiles, no promedios ponderados por irradiancia

### 7. SALIDA FINAL PARA ANÁLISIS ESTADÍSTICO

#### Archivos CSV Generados:
- **Directorio base**: `datos_procesados_analisis_integrado_py/`
- **Subdirectorios por método**:
  - `dustiq/`: `dustiq_sr_semanal_q25.csv`, `dustiq_sr_franjas_semanal_q25.csv`, `dustiq_sr_mediodia_semanal_q25.csv`
  - `ref_cells/`: `ref_cells_sr_minutal_filtrado.csv`, `ref_cells_sr_diario_q25.csv`, `ref_cells_sr_semanal_q25.csv`
  - `pvstand/`: Múltiples archivos CSV con SR normalizados y absolutos
  - `soiling_kit/`: `soiling_kit_sr_minutal_filtered.csv`, `soiling_kit_sr_raw_daily_mean.csv`, `soiling_kit_sr_corrected_daily_mean.csv`, `soiling_kit_sr_raw_weekly_q25.csv`, `soiling_kit_sr_corrected_weekly_q25.csv`
  - `pv_glasses/`: Archivos CSV con SR y masas por periodo
  - `iv600/`: Archivos CSV con SR diarios y semanales

#### Archivos Excel Consolidados:
- **DustIQ**: `dustiq_datos_completos_agregados_Q25.xlsx` (`dustiq_analyzer.py:638`)
  - Hojas: Franjas horarias fijas, mediodía solar semanal, estadísticas generales
- **PVStand**: `pvstand_datos_completos_agregados_Q25.xlsx` (`pvstand_analyzer.py:945`)
  - Hojas: SR normalizados/absolutos semanales, 3 días, datos minutales
- **PVStand Solar Noon**: `pvstand_datos_completos_agregados_solar_noon_Q25.xlsx` (`pvstand_analyzer_solar_noon.py:886`)
- **IV600**: `soiling_ratios_iv600_completo.xlsx` (`analisis_iv600_fixed.py:856`)
  - Hojas: SR diarios (promedio y Q25), SR semanales Q25

#### Scripts de Análisis Estadístico:
- **Intercomparación**: `analysis/soiling_intercomparison.py`
  - Carga series SR de todos los métodos
  - Calcula métricas: R², RMSE, MBE, CCC
  - Genera gráficos comparativos
- **Desviaciones estadísticas**: `analysis/statistical_deviation_analyzer.py`
  - Detecta anomalías mediante Z-score, IQR, DBSCAN
  - Ventana móvil: 24 horas (`STATISTICAL_DEVIATION_ROLLING_WINDOW`)
- **Incertidumbre SR**: `analysis/sr_uncertainty.py`
  - Bootstrap por bloques: `BLOCK_SIZE_MIN = 10` minutos, `N_BOOT = 2000`
  - Incertidumbres de calibración: `U_SOILED = 0.5%`, `U_CLEAN = 0.5%`
- **Gráficos consolidados**: `analysis/consolidated_weekly_q25_plot.py`
  - Genera gráficos comparativos de todos los métodos usando series semanales Q25

#### Formato de Datos Finales:
- **Índice temporal**: `DatetimeIndex` en UTC (naive después de procesamiento)
- **Columnas**: Series de SR con nombres descriptivos (ej: `SR_Raw_Filtered`, `SR_TempCorrected_Filtered`, `SR_Pmax_Corrected`)
- **Valores faltantes**: NaN para datos filtrados o inválidos
- **Frecuencia**: Depende del nivel de agregación (minutal, diario, semanal)

#### Integración para Intercomparación:
- Los scripts de intercomparación cargan los archivos CSV semanales Q25 de cada método
- Sincronización temporal mediante `pd.merge()` o `pd.concat()` con alineación por fecha
- Filtrado común por rango de fechas: típicamente desde `2024-07-23` o `2024-08-01` hasta fecha de fin del análisis
- Normalización opcional: Algunos métodos normalizan SR a 100% en fecha de referencia antes de intercomparación

---

## NOTAS ADICIONALES

### Configuración Centralizada:
- Parámetros en `config/settings.py`
- Rutas en `config/paths.py`
- Logging en `config/logging_config.py`

### Preprocesamiento:
- `data_processing/data_preprocessor.py`: Coordina preprocesamiento
- `download/download_temp_only.ipynb`: Genera `data_temp.csv` (requerido para PVStand)
- `data_processing/preprocessing_iv600.py`: Procesa curvas IV completas a parámetros extraídos

### Utilidades:
- `utils/solar_time.py`: Cálculo de mediodía solar
- `utils/helpers.py`: Funciones auxiliares (normalización, guardado de gráficos)
- `utils/plot_utils.py`: Utilidades para visualización

### Orden de Ejecución Recomendado:
1. Ejecutar `download/download_temp_only.ipynb` para generar datos de temperatura
2. Ejecutar preprocesamiento (`main.py` opción 10 o automático)
3. Ejecutar análisis individuales por método (`main.py` opciones 1-8)
4. Ejecutar análisis consolidados e intercomparación (`main.py` opciones 9, 12-16)







