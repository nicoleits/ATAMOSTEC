# 🌫️ Dashboard DustIQ - Análisis de Soiling Ratio

Dashboard dedicado y especializado para el análisis de datos de DustIQ, enfocado en el análisis de soiling ratio y pérdidas de eficiencia en sistemas fotovoltaicos.

## 🚀 Características Principales

### 📊 **Análisis Completo de Soiling Ratio**
- **Vista General**: Métricas principales, evolución temporal y distribución de valores
- **Franjas Horarias Fijas**: Análisis por períodos específicos del día
- **Mediodía Solar**: Análisis especializado en el período de máxima irradiación
- **Comparación Temporal**: Comparación entre diferentes períodos
- **Estadísticas Detalladas**: Análisis estadístico completo

### 🎛️ **Filtros y Configuración**
- **Filtro de Fechas**: Selección de rango temporal personalizable
- **Umbral SR**: Filtro por valor mínimo de Soiling Ratio
- **Configuración de Análisis**: Parámetros específicos por tipo de análisis

### 📈 **Visualizaciones Interactivas**
- Gráficos de serie temporal con Plotly
- Histogramas y box plots
- Gráficos de comparación temporal
- Tablas estadísticas interactivas

## 🛠️ Instalación y Configuración

### Requisitos Previos
```bash
# Activar entorno virtual
cd /home/nicole/SR/SOILING
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements_dashboard.txt
```

### Archivos Requeridos
- `dashboard_dustiq_dedicated.py` - Dashboard principal
- `dustiq_dashboard_utils.py` - Funciones auxiliares
- `datos/raw_dustiq_data.csv` - Datos de DustIQ (fallback)

## 🚀 Ejecución

### Ejecución Local
```bash
cd /home/nicole/SR/SOILING
source .venv/bin/activate
streamlit run dashboard_dustiq_dedicated.py
```

### Ejecución en Modo Headless
```bash
cd /home/nicole/SR/SOILING
source .venv/bin/activate
streamlit run dashboard_dustiq_dedicated.py --server.headless true --server.port 8501
```

## 📊 Tipos de Análisis Disponibles

### 1. 📈 Vista General
- **Métricas Principales**: Promedio, mediana, desviación estándar, pérdida promedio
- **Evolución Temporal**: Gráfico de serie temporal del Soiling Ratio
- **Distribución de Valores**: Histograma y box plot
- **Línea de Referencia**: Indicador visual del 100% de eficiencia

### 2. 🕐 Franjas Horarias Fijas
- **Franjas Disponibles**: 10:00-11:00, 12:00-13:00, 14:00-15:00, 15:00-16:00
- **Análisis por Franja**: Estadísticas específicas por período horario
- **Tendencias**: Análisis de tendencias temporales por franja
- **Comparación**: Comparación entre diferentes franjas horarias

### 3. ☀️ Mediodía Solar
- **Análisis Semanal/Diario**: Opciones de agregación temporal
- **Ventana Configurable**: Duración de la ventana de análisis (30-120 min)
- **Tendencia Solar**: Análisis específico del período de máxima irradiación
- **Estadísticas Especializadas**: Métricas optimizadas para análisis solar

### 4. 📅 Comparación Temporal
- **Períodos Mensuales**: Comparación entre meses
- **Franjas de Comparación**: Selección de franja horaria para comparación
- **Box Plots**: Visualización de distribución por período
- **Análisis Estacional**: Comparación entre estaciones

### 5. 📊 Estadísticas Detalladas
- **Estadísticas Generales**: Tabla completa de métricas
- **Análisis por Hora**: Estadísticas desglosadas por hora del día
- **Gráfico de Barras**: Promedio de SR por hora
- **Métricas Avanzadas**: R², tendencias, percentiles

## 🔧 Configuración de Base de Datos

### ClickHouse (Modo Online)
```python
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}
```

### Archivo Local (Modo Offline)
- **Ubicación**: `datos/raw_dustiq_data.csv`
- **Formato**: CSV con columnas `timestamp`, `SR_C11_Avg`, `SR_C12_Avg`
- **Fallback Automático**: Si ClickHouse no está disponible

## 📁 Estructura de Datos

### Columnas Requeridas
- `timestamp`: Fecha y hora de la medición
- `SR_C11_Avg`: Soiling Ratio promedio del canal 11
- `SR_C12_Avg`: Soiling Ratio promedio del canal 12

### Filtros Aplicados
- **Rango de Fechas**: 2024-06-24 a 2025-07-31
- **Umbral SR**: Solo valores > 0
- **Ordenamiento**: Por timestamp ascendente

## 🎯 Casos de Uso

### Análisis de Rendimiento
1. Seleccionar "📈 Vista General"
2. Ajustar rango de fechas
3. Revisar métricas principales
4. Analizar evolución temporal

### Análisis por Horarios
1. Seleccionar "🕐 Franjas Horarias Fijas"
2. Elegir franjas de interés
3. Activar tendencias
4. Comparar rendimiento por período

### Análisis Solar Especializado
1. Seleccionar "☀️ Mediodía Solar"
2. Configurar duración de ventana
3. Elegir agregación temporal
4. Analizar tendencias solares

### Comparación Estacional
1. Seleccionar "📅 Comparación Temporal"
2. Elegir tipo de período
3. Seleccionar meses a comparar
4. Analizar diferencias estacionales

## 🔍 Validación de Datos

El dashboard incluye validación automática de datos:
- **Verificación de Columnas**: Existencia de columnas requeridas
- **Verificación de Tipos**: Tipos de datos correctos
- **Verificación de Rango**: Valores dentro de rangos esperados
- **Verificación de Completitud**: Datos faltantes y duplicados

## 📊 Exportación de Datos

### Funcionalidades de Exportación
- **CSV**: Exportación de datos filtrados
- **Gráficos**: Descarga de visualizaciones
- **Estadísticas**: Exportación de métricas calculadas

## 🛠️ Personalización

### Modificación de Configuración
- **Franjas Horarias**: Editar `franjas_disponibles` en el código
- **Rangos de Fechas**: Modificar consultas SQL
- **Métricas**: Agregar nuevas métricas en las funciones de análisis

### Extensión de Funcionalidades
- **Nuevos Tipos de Análisis**: Agregar nuevas secciones
- **Visualizaciones**: Implementar nuevos gráficos
- **Filtros**: Agregar filtros adicionales

## 🐛 Solución de Problemas

### Error de Conexión a ClickHouse
- Verificar configuración de red
- Comprobar credenciales
- El dashboard cambiará automáticamente a modo offline

### Error de Carga de Datos
- Verificar existencia de archivo CSV
- Comprobar formato de datos
- Revisar permisos de archivo

### Error de Visualización
- Verificar instalación de Plotly
- Comprobar datos de entrada
- Revisar configuración de Streamlit

## 📞 Soporte

Para soporte técnico o consultas sobre el dashboard:
- **Desarrollador**: Equipo ATAMOSTEC
- **Documentación**: README_Dashboard_DustIQ.md
- **Código Fuente**: Repositorio Git

---

**🌫️ Dashboard DustIQ** - Análisis de Soiling Ratio | Desarrollado para ATAMOSTEC 