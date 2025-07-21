# Analizador de Desviaciones Estadísticas

## 📊 Descripción General

El **Analizador de Desviaciones Estadísticas** es un módulo avanzado del sistema SOILING que detecta automáticamente anomalías y desviaciones en los datos de sensores utilizando múltiples métodos estadísticos y de machine learning.

## 🎯 Objetivos

- **Detección automática de anomalías** en datos de sensores de soiling
- **Identificación de sensores defectuosos** o descalibrados
- **Análisis de calidad de datos** para validar mediciones
- **Alertas tempranas** de problemas en equipos
- **Análisis comparativo** entre diferentes métodos de detección

## 🔬 Métodos de Detección Implementados

### 1. **Z-Score**
- **Descripción**: Detecta valores que se desvían significativamente de la media
- **Parámetro**: Umbral de desviación estándar (default: 3.0)
- **Uso**: Detección de outliers extremos
- **Ventajas**: Simple, rápido, interpretable
- **Limitaciones**: Asume distribución normal

### 2. **IQR (Interquartile Range)**
- **Descripción**: Usa el rango intercuartílico para detectar outliers
- **Parámetro**: Factor multiplicador (default: 1.5)
- **Uso**: Detección robusta sin asumir distribución
- **Ventajas**: No paramétrico, robusto
- **Limitaciones**: Puede ser menos sensible

### 3. **Isolation Forest**
- **Descripción**: Algoritmo de ML que aísla anomalías
- **Parámetro**: Proporción de contaminación (default: 0.1)
- **Uso**: Detección de anomalías complejas
- **Ventajas**: Maneja datos multidimensionales
- **Limitaciones**: Requiere más datos

### 4. **DBSCAN Clustering**
- **Descripción**: Identifica puntos que no pertenecen a clusters
- **Parámetros**: eps (distancia), min_samples (mínimo de puntos)
- **Uso**: Detección basada en densidad
- **Ventajas**: Encuentra patrones complejos
- **Limitaciones**: Sensible a parámetros

### 5. **Rolling Deviation**
- **Descripción**: Detecta desviaciones de la media móvil
- **Parámetros**: Ventana temporal, umbral de desviación
- **Uso**: Detección de cambios graduales
- **Ventajas**: Adaptativo a tendencias
- **Limitaciones**: Requiere datos temporales

### 6. **Seasonal Anomalies**
- **Descripción**: Considera patrones estacionales/diarios
- **Parámetro**: Período estacional (default: 24 horas)
- **Uso**: Detección respetando ciclos naturales
- **Ventajas**: Reduce falsos positivos
- **Limitaciones**: Requiere patrones claros

## 🏗️ Estructura del Módulo

```
analysis/statistical_deviation_analyzer.py
├── StatisticalDeviationAnalyzer (clase principal)
├── detect_z_score_anomalies()
├── detect_iqr_anomalies()
├── detect_isolation_forest_anomalies()
├── detect_dbscan_anomalies()
├── detect_rolling_deviation_anomalies()
├── detect_seasonal_anomalies()
├── analyze_sensor_data()
├── create_anomaly_visualization()
├── create_summary_report()
└── analyze_all_sensors_deviations()
```

## 📈 Outputs Generados

### 1. **Gráficos Individuales por Sensor**
- **Ubicación**: `graficos_analisis_integrado_py/statistical_deviations/`
- **Formato**: PNG de alta resolución
- **Contenido**:
  - Serie temporal con anomalías marcadas por método
  - Histograma de distribución de datos
  - Comparación de métodos de detección
  - Estadísticas descriptivas

### 2. **Gráfico Resumen Consolidado**
- **Archivo**: `anomaly_detection_summary.png`
- **Contenido**:
  - Número de anomalías por sensor y método
  - Porcentajes de anomalías
  - Estadísticas descriptivas normalizadas
  - Calidad de datos por sensor

### 3. **Reporte CSV**
- **Ubicación**: `datos_procesados_analisis_integrado_py/statistical_deviations/`
- **Archivo**: `anomaly_detection_summary.csv`
- **Contenido**:
  - Estadísticas por sensor
  - Conteo de anomalías por método
  - Porcentajes de anomalías
  - Métricas de calidad de datos

## ⚙️ Configuración

### Parámetros en `config/settings.py`

```python
# Umbrales de detección
STATISTICAL_DEVIATION_Z_SCORE_THRESHOLD = 3.0
STATISTICAL_DEVIATION_IQR_FACTOR = 1.5
STATISTICAL_DEVIATION_ISOLATION_CONTAMINATION = 0.1

# Parámetros temporales
STATISTICAL_DEVIATION_ROLLING_WINDOW = 24
STATISTICAL_DEVIATION_ROLLING_THRESHOLD = 3.0
STATISTICAL_DEVIATION_SEASONAL_PERIOD = 24

# Criterios de calidad
STATISTICAL_DEVIATION_MIN_DATA_POINTS = 100
STATISTICAL_DEVIATION_DBSCAN_EPS = 0.5
STATISTICAL_DEVIATION_DBSCAN_MIN_SAMPLES = 5
```

## 🚀 Uso del Analizador

### 1. **Desde el Menú Principal**
```bash
python main.py
# Seleccionar opción 12: "Análisis de Desviaciones Estadísticas"
```

### 2. **Uso Programático**
```python
from analysis.statistical_deviation_analyzer import StatisticalDeviationAnalyzer

# Crear analizador
analyzer = StatisticalDeviationAnalyzer()

# Analizar un sensor específico
results = analyzer.analyze_sensor_data(
    data=sensor_series,
    sensor_name="DustIQ_C11",
    methods=['z_score', 'iqr', 'isolation_forest']
)

# Crear visualización
analyzer.create_anomaly_visualization(sensor_series, results['anomalies'], "DustIQ_C11")
```

### 3. **Análisis Completo**
```python
from analysis.statistical_deviation_analyzer import analyze_all_sensors_deviations

# Ejecutar análisis completo
results = analyze_all_sensors_deviations()
```

## 📊 Sensores Analizados Automáticamente

El analizador procesa automáticamente los siguientes sensores:

| Sensor | Archivo de Datos | Columna | Descripción |
|--------|------------------|---------|-------------|
| DustIQ_C11 | raw_dustiq_data.csv | SR_C11_Avg | Sensor óptico C11 |
| DustIQ_C12 | raw_dustiq_data.csv | SR_C12_Avg | Sensor óptico C12 |
| RefCell_410 | refcells_data.csv | 1RC410(w.m-2) | Celda de referencia 410 |
| RefCell_411 | refcells_data.csv | 1RC411(w.m-2) | Celda de referencia 411 |
| RefCell_412 | refcells_data.csv | 1RC412(w.m-2) | Celda de referencia 412 |
| SoilingKit_Isc_Exposed | soiling_kit_raw_data.csv | Isc(e) | Corriente módulo expuesto |
| SoilingKit_Isc_Protected | soiling_kit_raw_data.csv | Isc(p) | Corriente módulo protegido |

## 🔍 Interpretación de Resultados

### 1. **Tipos de Anomalías Detectadas**

#### **Picos Extremos**
- **Causa**: Interferencias, errores de medición
- **Detección**: Z-Score, IQR
- **Acción**: Verificar calibración del sensor

#### **Deriva Gradual**
- **Causa**: Descalibración, envejecimiento del sensor
- **Detección**: Rolling Deviation, Seasonal
- **Acción**: Recalibración necesaria

#### **Valores Constantes**
- **Causa**: Sensor bloqueado o defectuoso
- **Detección**: Todos los métodos
- **Acción**: Reemplazo del sensor

#### **Patrones Anómalos**
- **Causa**: Condiciones ambientales extremas
- **Detección**: Isolation Forest, DBSCAN
- **Acción**: Investigar causas ambientales

### 2. **Umbrales de Alerta**

| Porcentaje de Anomalías | Estado | Acción Recomendada |
|-------------------------|--------|-------------------|
| < 5% | Normal | Monitoreo rutinario |
| 5-10% | Precaución | Revisión semanal |
| 10-20% | Alerta | Investigación inmediata |
| > 20% | Crítico | Reemplazo/recalibración urgente |

## 🛠️ Mantenimiento y Calibración

### 1. **Monitoreo Continuo**
- Ejecutar análisis semanalmente
- Establecer alertas automáticas
- Mantener histórico de anomalías

### 2. **Calibración Preventiva**
- Recalibrar sensores con >10% anomalías
- Verificar condiciones ambientales extremas
- Documentar acciones correctivas

### 3. **Validación Cruzada**
- Comparar sensores del mismo tipo
- Verificar con mediciones de referencia
- Analizar correlaciones temporales

## 📝 Logs y Debugging

### Información Registrada
- Número de anomalías por método
- Estadísticas descriptivas
- Errores de procesamiento
- Archivos procesados y omitidos

### Niveles de Log
```python
logger.info("Análisis completado para sensor X")
logger.warning("Datos insuficientes para sensor Y")
logger.error("Error procesando archivo Z")
```

## 🔧 Personalización Avanzada

### 1. **Añadir Nuevos Métodos**
```python
def detect_custom_anomalies(self, data: pd.Series) -> pd.Series:
    # Implementar método personalizado
    anomalies = custom_algorithm(data)
    return anomalies
```

### 2. **Filtros Específicos por Sensor**
```python
# Configurar umbrales específicos por tipo de sensor
sensor_configs = {
    'DustIQ': {'z_threshold': 2.5},
    'RefCell': {'z_threshold': 3.5},
    'SoilingKit': {'z_threshold': 3.0}
}
```

### 3. **Integración con Alertas**
```python
def send_anomaly_alert(sensor_name, anomaly_count, percentage):
    if percentage > 15:  # Umbral crítico
        send_email_alert(f"Sensor {sensor_name}: {anomaly_count} anomalías")
```

## 🚨 Solución de Problemas

### Problemas Comunes

1. **"Datos insuficientes"**
   - **Causa**: Menos de 100 puntos de datos
   - **Solución**: Verificar archivos de entrada

2. **"Archivo no encontrado"**
   - **Causa**: Ruta incorrecta o archivo faltante
   - **Solución**: Verificar estructura de directorios

3. **"Error en método X"**
   - **Causa**: Datos incompatibles con el método
   - **Solución**: Revisar calidad de datos de entrada

4. **Demasiadas anomalías detectadas**
   - **Causa**: Umbrales muy restrictivos
   - **Solución**: Ajustar parámetros en settings.py

## 📚 Referencias

- [Isolation Forest Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [DBSCAN Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Statistical Outlier Detection Methods](https://en.wikipedia.org/wiki/Outlier)

## 🤝 Contribuciones

Para añadir nuevos métodos de detección o mejorar los existentes:

1. Implementar el método en la clase `StatisticalDeviationAnalyzer`
2. Añadir configuración en `settings.py`
3. Actualizar la documentación
4. Crear tests unitarios

## 📞 Soporte

Para reportar bugs o solicitar nuevas funcionalidades, contactar al equipo de desarrollo del sistema SOILING. 