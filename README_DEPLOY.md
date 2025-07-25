# 🌫️ Dashboard DustIQ - Despliegue en Streamlit Cloud

## 📋 Descripción
Dashboard especializado para el análisis de datos de DustIQ, enfocado en el análisis de soiling ratio y pérdidas de eficiencia en sistemas fotovoltaicos.

## 🚀 Despliegue en Streamlit Cloud

### Requisitos Previos
- Cuenta en [GitHub](https://github.com)
- Cuenta en [Streamlit Cloud](https://streamlit.io/cloud)

### Pasos para Desplegar

1. **Subir código a GitHub**
   ```bash
   git add .
   git commit -m "Preparar dashboard para despliegue"
   git push origin master
   ```

2. **Conectar con Streamlit Cloud**
   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Conecta tu cuenta de GitHub
   - Selecciona el repositorio `SOILING`
   - Configura el archivo principal: `streamlit_app.py`

3. **Configuración**
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.9 o superior
   - **Requirements file**: `requirements_streamlit.txt`

## 🔧 Configuración de Base de Datos

### ClickHouse (Modo Online)
El dashboard intentará conectarse a ClickHouse. Si no está disponible, cambiará automáticamente a modo offline.

### Archivo Local (Modo Offline)
Si ClickHouse no está disponible, el dashboard usará datos locales almacenados en `datos/raw_dustiq_data.csv`.

## 📊 Funcionalidades

- **Vista General**: Métricas principales y evolución temporal
- **Franjas Horarias Fijas**: Análisis por períodos específicos
- **Mediodía Solar**: Análisis especializado en máxima irradiación
- **Comparación Temporal**: Comparación entre diferentes períodos
- **Estadísticas Detalladas**: Análisis estadístico completo

## 🎛️ Configuración de Usuario

- **Frecuencia Temporal**: Diario, Semanal, Mensual
- **Franjas Horarias**: Selección personalizable
- **Filtros de Fechas**: Rango temporal configurable
- **Umbral SR**: Filtro por valor mínimo

## 🔗 Enlaces

- **Dashboard**: [URL del despliegue]
- **Repositorio**: [URL del repositorio GitHub]
- **Documentación**: README_Dashboard_DustIQ.md

---

**Desarrollado para ATAMOSTEC** | **Dashboard DustIQ** - Análisis de Soiling Ratio 