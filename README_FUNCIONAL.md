# 🌫️ Dashboard DustIQ - Aplicación Funcional

## 📋 Descripción
Esta es la aplicación Streamlit que funciona correctamente para el análisis de datos de DustIQ (Soiling Ratio).

## 🚀 Cómo Ejecutar

### 1. Activar el entorno virtual
```bash
cd /home/nicole/SR/SOILING
source .venv/bin/activate
```

### 2. Ejecutar la aplicación
```bash
streamlit run streamlit_app.py
```

### 3. Acceder a la aplicación
- **URL**: `http://localhost:8501`
- **Navegador**: Abrir el enlace que aparece en la terminal

## 🔧 Funcionalidades

### ✅ Lo que funciona:
- **Conexión a ClickHouse**: Carga datos desde la base de datos
- **Filtros de fecha**: Funcionan correctamente sin errores
- **Gráficos interactivos**: Visualización con Plotly
- **Análisis de Soiling Ratio**: Cálculos y métricas
- **Franjas horarias**: Filtrado por horarios específicos
- **Exportación de datos**: Descarga de resultados

### 📊 Datos disponibles:
- **DustIQ**: Datos de Soiling Ratio (SR_C11_Avg, SR_C12_Avg)
- **Período**: Desde 2024-06-24 hasta 2025-07-31
- **Frecuencia**: Datos horarios

## 🛠️ Configuración

### Base de datos ClickHouse:
- **Host**: 146.83.153.212
- **Puerto**: 30091
- **Base de datos**: PSDA
- **Tabla**: dustiq

### Dependencias:
- streamlit
- pandas
- plotly
- numpy
- clickhouse-connect

## 📁 Archivos importantes:
- `streamlit_app.py` - Aplicación principal (ÚNICA que funciona)
- `requirements.txt` - Dependencias de Python
- `.streamlit/config.toml` - Configuración de Streamlit

## 🔍 Solución de problemas:

### Si la aplicación no inicia:
1. Verificar que el entorno virtual esté activado
2. Verificar que todas las dependencias estén instaladas: `pip install -r requirements.txt`
3. Verificar conectividad a la base de datos

### Si no hay datos:
1. Verificar conexión a internet
2. Verificar que la base de datos esté disponible
3. Usar el botón "Recargar Datos" en la aplicación

## 📝 Notas importantes:
- Esta es la ÚNICA aplicación que funciona correctamente
- Se han eliminado todas las versiones que causaban problemas
- Los filtros de fecha funcionan sin errores
- La aplicación es estable y confiable

## 🎯 Uso recomendado:
1. Ejecutar la aplicación
2. Configurar fechas en el sidebar
3. Seleccionar frecuencia temporal
4. Elegir franjas horarias
5. Analizar los gráficos y métricas
6. Exportar datos si es necesario 