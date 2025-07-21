# README_INI - Sistema de Análisis de Soiling (VERSIÓN CORREGIDA)

## Descripción General

Este sistema integrado de análisis de soiling permite procesar y analizar datos de diferentes tipos de sensores y equipos para evaluar el efecto de ensuciamiento en sistemas fotovoltaicos. El sistema incluye múltiples analizadores especializados que procesan datos de diferentes fuentes y generan gráficos y reportes consolidados.

## Estructura del Sistema

### Menú Principal
El sistema se ejecuta a través de `main.py` y ofrece las siguientes opciones:

1. **Soiling Kit** - Análisis de datos del kit de ensuciamiento
2. **DustIQ** - Análisis de datos del sensor DustIQ
3. **Celdas de Referencia** - Análisis de celdas de referencia
4. **PVStand** - Análisis de datos del banco de pruebas PV
5. **PVStand - Mediodía Solar** - Análisis PVStand con filtro de mediodía solar
6. **PV Glasses** - Análisis de vidrios fotovoltaicos
7. **Calendario** - Análisis de calendario de muestras
8. **Análisis IV600 Filtrado** - Análisis de datos IV600 sin picos
9. **Gráfico Consolidado Semanal Q25** - Gráfico consolidado sin tendencia
10. **Ejecutar preprocesamiento solamente**
11. **Ejecutar todos los análisis**

---

## 1. Analizador Soiling Kit

### Descripción
Analiza datos del kit de ensuciamiento que compara módulos protegidos vs expuestos, calculando el Soiling Ratio (SR) con y sin corrección de temperatura.

### Inputs
- **Archivo**: `datos/soiling_kit_raw_data.csv` ✅
- **Columnas requeridas**:
  - `_time` - Timestamp en formato UTC
  - `Isc(e)` - Corriente de cortocircuito del módulo expuesto (A) ✅
  - `Isc(p)` - Corriente de cortocircuito del módulo protegido (A) ✅
  - `Te(C)` - Temperatura del módulo expuesto (°C) ✅
  - `Tp(C)` - Temperatura del módulo protegido (°C) ✅

### Outputs
- **CSV procesados**: `datos_procesados_analisis_integrado_py/soiling_kit/`
  - `soiling_kit_sr_raw_weekly_q25.csv` - SR semanal Q25 sin corregir
  - `soiling_kit_sr_corrected_weekly_q25.csv` - SR semanal Q25 corregido por temperatura
  - `soiling_kit_sr_minutal_filtered.csv` - Datos minutales filtrados
- **Gráficos**: `graficos_analisis_integrado_py/soiling_kit/`
  - Gráficos de SR diario y semanal
  - Comparación SR raw vs corregido
  - Análisis de tendencias

---

## 2. Analizador DustIQ

### Descripción
Procesa datos del sensor DustIQ que mide la transmitancia óptica para evaluar el ensuciamiento en tiempo real.

### Inputs
- **Archivo**: `datos/raw_dustiq_data.csv` ✅
- **Columnas requeridas**:
  - `_time` - Timestamp en formato UTC
  - `SR_C11_Avg` - Soiling Ratio promedio del sensor C11 (%) ✅
  - `SR_C12_Avg` - Soiling Ratio promedio del sensor C12 (%) ✅

### Outputs
- **CSV procesados**: `datos_procesados_analisis_integrado_py/dustiq/`
  - `dustiq_sr_semanal_q25.csv` - SR semanal Q25
  - `dustiq_sr_mediodia_semanal_q25.csv` - SR semanal Q25 mediodía solar
  - `dustiq_sr_franjas_semanal_q25.csv` - SR semanal Q25 por franjas horarias
- **Gráficos**: `graficos_analisis_integrado_py/dustiq/`
  - Gráficos de SR diario y semanal
  - Análisis por franjas horarias
  - Gráficos con tendencias

---

## 3. Analizador Celdas de Referencia

### Descripción
Analiza datos de celdas de referencia (RefCells) para evaluar el ensuciamiento comparando celdas limpias vs sucias.

### Inputs
- **Archivo**: `datos/refcells_data.csv` ✅
- **Columnas requeridas**:
  - `_time` - Timestamp en formato UTC
  - `1RC410(w.m-2)` - Irradiancia celda de referencia 410 (W/m²) ✅
  - `1RC411(w.m-2)` - Irradiancia celda de referencia 411 (W/m²) ✅
  - `1RC412(w.m-2)` - Irradiancia celda de referencia 412 (W/m²) ✅

### Outputs
- **CSV procesados**: `datos_procesados_analisis_integrado_py/ref_cells/`
  - `ref_cells_sr_semanal_q25.csv` - SR semanal Q25
  - `ref_cells_sr_diario_q25.csv` - SR diario Q25
  - `analisis_dias_nublados_solar_noon.csv` - Análisis de días nublados
- **Gráficos**: `graficos_analisis_integrado_py/ref_cells/`
  - Gráficos de SR combinado e individual
  - Análisis por condiciones climáticas
  - Gráficos de mediodía solar

---

## 4. Analizador PVStand

### Descripción
Analiza datos del banco de pruebas PV que incluye curvas IV completas para evaluar el rendimiento de módulos.

### Inputs
- **Archivo IV**: `datos/raw_pvstand_iv_data.csv` ✅
- **Archivo temperatura**: `datos/data_temp.csv` ✅
- **Columnas IV requeridas**:
  - `_time` - Timestamp en formato UTC
  - `PERC1_fixed_1MD43420160719_Imax` - Corriente máxima módulo sucio (A) ✅
  - `PERC1_fixed_1MD43420160719_Pmax` - Potencia máxima módulo sucio (W) ✅
  - `PERC2_fixed_1MD43920160719_Imax` - Corriente máxima módulo limpio (A) ✅
  - `PERC2_fixed_1MD43920160719_Pmax` - Potencia máxima módulo limpio (W) ✅
- **Columnas temperatura requeridas**:
  - `1TE416(C)` - Temperatura módulo sucio (°C) ✅
  - `1TE418(C)` - Temperatura módulo limpio (°C) ✅

### Outputs
- **CSV procesados**: `datos_procesados_analisis_integrado_py/pv_stand/`
  - `pvstand_sr_main_norm.csv` - SR normalizado principal
  - `pvstand_sr_no_norm_with_offset.csv` - SR no normalizado con offset
  - `pvstand_sr_raw_no_offset.csv` - SR raw sin offset
- **Gráficos**: `graficos_analisis_integrado_py/pvstand/`
  - Gráficos de SR normalizado y no normalizado
  - Análisis de potencias y corrientes
  - Gráficos de tendencias

---

## 5. Analizador PVStand - Mediodía Solar

### Descripción
Versión especializada del analizador PVStand que filtra datos solo durante el mediodía solar para análisis más precisos.

### Inputs
- **Mismos inputs que PVStand estándar**
- **Configuración adicional**: Filtro de mediodía solar (±2.5 horas)

### Outputs
- **CSV procesados**: `datos_procesados_analisis_integrado_py/pv_stand/solar_noon/`
  - `pvstand_sr_main_norm_solar_noon.csv` - SR normalizado mediodía solar
  - `pvstand_sr_no_norm_with_offset_solar_noon.csv` - SR no normalizado mediodía solar
  - `pvstand_sr_raw_no_offset_solar_noon.csv` - SR raw mediodía solar
- **Gráficos**: `graficos_analisis_integrado_py/pvstand_solar_noon/`
  - Gráficos específicos de mediodía solar
  - Análisis de potencias diarias promedio
  - Gráficos de tendencias

---

## 6. Analizador PV Glasses

### Descripción
Analiza datos de vidrios fotovoltaicos para evaluar la transmitancia y el efecto del ensuciamiento.

### Inputs
- **Archivo**: `datos/raw_pv_glasses_data.csv` ✅
- **Columnas requeridas**:
  - `_time` - Timestamp en formato UTC
  - `R_FC1_Avg` - Transmitancia vidrio FC1 (%)
  - `R_FC2_Avg` - Transmitancia vidrio FC2 (%)
  - `R_FC3_Avg` - Transmitancia vidrio FC3 (%)
  - `R_FC4_Avg` - Transmitancia vidrio FC4 (%)
  - `R_FC5_Avg` - Transmitancia vidrio FC5 (%)

### Outputs
- **CSV procesados**: `datos_procesados_analisis_integrado_py/pv_glasses/`
  - Datos de transmitancia procesados
  - Análisis de degradación temporal
- **Gráficos**: `graficos_analisis_integrado_py/pv_glasses/`
  - Gráficos de transmitancia vs tiempo
  - Análisis de correlación con condiciones ambientales

---

## 7. Analizador Calendario

### Descripción
Procesa información del calendario de muestras para correlacionar eventos de limpieza con cambios en el rendimiento.

### Inputs
- **Archivo**: `datos/calendario_muestras_seleccionado.csv` ✅
- **Columnas requeridas**:
  - `Fecha` - Fecha del evento
  - `Tipo_Evento` - Tipo de evento (limpieza, muestreo, etc.)
  - `Descripcion` - Descripción del evento

### Outputs
- **CSV procesados**: `datos_procesados_analisis_integrado_py/calendario/`
  - Eventos procesados y categorizados
- **Gráficos**: `graficos_analisis_integrado_py/calendario/`
  - Cronograma de eventos
  - Correlación con datos de rendimiento

---

## 8. Analizador IV600 Filtrado

### Descripción
Analiza datos del sistema IV600 con filtros especiales para eliminar picos y anomalías.

### Inputs
- **Archivo**: `datos/raw_iv600_data.csv` ✅
- **Columnas requeridas**:
  - `timestamp` - Timestamp en formato UTC
  - `Module` - Identificador del módulo
  - `Pmax` - Potencia máxima (W)
  - `Isc` - Corriente de cortocircuito (A)
  - `Voc` - Voltaje de circuito abierto (V)
  - `Imp` - Corriente en punto de máxima potencia (A)
  - `Vmp` - Voltaje en punto de máxima potencia (V)

### Outputs
- **CSV procesados**: `datos_procesados_analisis_integrado_py/iv600/`
  - Datos filtrados y procesados
  - SR calculados por módulo
- **Gráficos**: `graficos_analisis_integrado_py/iv600/`
  - Gráficos de SR filtrados
  - Comparaciones entre módulos
  - Análisis de tendencias

---

## 9. Gráfico Consolidado Semanal Q25

### Descripción
Genera un gráfico consolidado que combina los análisis semanales Q25 de todos los sistemas sin líneas de tendencia.

### Inputs
- **Archivos CSV semanales Q25** de todos los analizadores:
  - `ref_cells_sr_semanal_q25.csv`
  - `dustiq_sr_semanal_q25.csv`
  - `soiling_kit_sr_raw_weekly_q25.csv`
  - `soiling_kit_sr_corrected_weekly_q25.csv`
  - Datos PVStand (generados desde archivos principales)

### Outputs
- **Gráfico**: `graficos_analisis_integrado_py/consolidados/consolidated_weekly_q25_no_trend.png`
  - Gráfico consolidado con todas las series normalizadas
  - Comparación visual de diferentes métodos de análisis

---

## Estructura de Directorios

```
SOILING/
├── datos/                          # Datos de entrada
│   ├── soiling_kit_raw_data.csv    ✅ CORREGIDO
│   ├── raw_dustiq_data.csv         ✅ CORREGIDO
│   ├── refcells_data.csv           ✅ CORREGIDO
│   ├── raw_pvstand_iv_data.csv     ✅ CORREGIDO
│   ├── data_temp.csv               ✅ CORREGIDO
│   ├── raw_pv_glasses_data.csv     ✅ CORREGIDO
│   ├── calendario_muestras_seleccionado.csv ✅ CORREGIDO
│   └── raw_iv600_data.csv          ✅ CORREGIDO
├── datos_procesados_analisis_integrado_py/  # Datos de salida
│   ├── soiling_kit/
│   ├── dustiq/
│   ├── ref_cells/
│   ├── pv_stand/
│   ├── pv_glasses/
│   ├── calendario/
│   └── iv600/
├── graficos_analisis_integrado_py/  # Gráficos generados
│   ├── soiling_kit/
│   ├── dustiq/
│   ├── ref_cells/
│   ├── pvstand/
│   ├── pvstand_solar_noon/
│   ├── pv_glasses/
│   ├── calendario/
│   ├── iv600/
│   └── consolidados/
└── analysis/                       # Código de los analizadores
    ├── soiling_kit_analyzer.py
    ├── dustiq_analyzer.py
    ├── ref_cells_analyzer.py
    ├── pvstand_analyzer.py
    ├── pvstand_analyzer_solar_noon.py
    ├── pv_glasses_analyzer.py
    ├── calendar_analyzer.py
    ├── analisis_iv600_fixed.py
    └── consolidated_weekly_q25_plot.py
```

---

## Configuración y Parámetros

### Archivos de Configuración
- `config/settings.py` - Configuración general del sistema
- `config/paths.py` - Definición de rutas de archivos
- `config/logging_config.py` - Configuración de logging

### Parámetros Importantes
- **Fechas de análisis**: Definidas en `settings.py`
- **Filtros horarios**: Configurables por analizador
- **Umbrales de filtrado**: Definidos para cada tipo de sensor
- **Correcciones de temperatura**: Coeficientes específicos por analizador

---

## Requisitos del Sistema

### Dependencias Python
- pandas
- numpy
- matplotlib
- scipy
- pytz
- openpyxl

### Estructura de Datos
- Todos los archivos de entrada deben estar en formato CSV
- Timestamps deben estar en formato UTC
- Columnas numéricas deben ser convertibles a float
- Datos faltantes deben estar marcados como NaN

---

## Notas de Uso

1. **Preprocesamiento**: Algunos análisis requieren preprocesamiento previo (ej: datos de temperatura para PVStand)
2. **Orden de ejecución**: Se recomienda ejecutar análisis individuales antes del consolidado
3. **Gestión de errores**: El sistema maneja errores de archivos faltantes y datos corruptos
4. **Logging**: Todos los procesos generan logs detallados para debugging

---

## Correcciones Realizadas

### Nombres de Archivos Corregidos:
- ✅ `raw_soiling_kit_data.csv` → `soiling_kit_raw_data.csv`
- ✅ `raw_ref_cells_data.csv` → `refcells_data.csv`
- ✅ Todos los demás nombres de archivos verificados y corregidos

### Columnas Corregidas:
- ✅ **Soiling Kit**: `Isc(e)`, `Isc(p)`, `Te(C)`, `Tp(C)`
- ✅ **DustIQ**: `SR_C11_Avg`, `SR_C12_Avg`
- ✅ **RefCells**: `1RC410(w.m-2)`, `1RC411(w.m-2)`, `1RC412(w.m-2)`
- ✅ **PVStand**: `PERC1_fixed_1MD43420160719_*`, `PERC2_fixed_1MD43920160719_*`
- ✅ **PVStand Temp**: `1TE416(C)`, `1TE418(C)`

---

## Contacto y Soporte

Para consultas sobre el sistema de análisis de soiling, contactar al equipo de desarrollo. 