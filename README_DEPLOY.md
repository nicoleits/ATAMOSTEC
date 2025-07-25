# 🔋🌫️ Dashboard Integrado - DustIQ + PVStand

## 📋 Descripción
Sistema de dashboards especializados para el análisis de datos de DustIQ (soiling ratio) y PVStand (curvas IV), enfocado en el análisis de eficiencia y pérdidas en sistemas fotovoltaicos.

## 🚀 Dashboards Disponibles

### 1. 🌫️ Dashboard DustIQ (Solo Soiling Ratio)
- **Archivo**: `streamlit_app.py`
- **Funcionalidad**: Análisis completo de soiling ratio
- **Características**:
  - Vista general con métricas principales
  - Análisis por franjas horarias
  - Mediodía solar especializado
  - Comparación temporal
  - Estadísticas detalladas

### 2. 🔋🌫️ Dashboard Integrado v3 (DustIQ + PVStand)
- **Archivo**: `streamlit_app_integrado_v3.py`
- **Funcionalidad**: Análisis combinado de ambos sistemas
- **Características**:
  - Filtros globales sincronizados
  - Análisis de soiling ratio (DustIQ)
  - Curvas IV interactivas (PVStand)
  - Información del sistema integrada
  - Interfaz unificada y robusta

## 🚀 Despliegue en Streamlit Cloud

### Requisitos Previos
- Cuenta en [GitHub](https://github.com)
- Cuenta en [Streamlit Cloud](https://streamlit.io/cloud)

### Pasos para Desplegar

1. **Subir código a GitHub**
   ```bash
   git add .
   git commit -m "Actualizar dashboards"
   git push origin master
   ```

2. **Conectar con Streamlit Cloud**
   - Ve a [share.streamlit.io](https://share.streamlit.io)
   - Conecta tu cuenta de GitHub
   - Selecciona el repositorio `ATAMOSTEC`

3. **Configuración por Dashboard**

   **Para Dashboard DustIQ:**
   - **Main file path**: `SOILING/streamlit_app.py`
   - **Requirements file**: `SOILING/requirements_streamlit.txt`

   **Para Dashboard Integrado:**
   - **Main file path**: `SOILING/streamlit_app_integrado_v3.py`
   - **Requirements file**: `SOILING/requirements_streamlit.txt`

## 🔧 Configuración de Base de Datos

### ClickHouse (Modo Online)
Los dashboards se conectan automáticamente a ClickHouse:
- **Host**: 146.83.153.212:30091
- **Base de datos**: PSDA.dustiq (DustIQ), ref_data.iv_curves_* (PVStand)

### Fallback Automático
Si ClickHouse no está disponible, los dashboards mostrarán mensajes informativos sobre el estado de conexión.

## 📊 Funcionalidades por Dashboard

### 🌫️ Dashboard DustIQ
- **Métricas Principales**: Promedio, mediana, desviación estándar
- **Evolución Temporal**: Gráficos con frecuencia configurable
- **Filtros Avanzados**: Fechas, umbral SR, franjas horarias
- **Análisis Estadístico**: Pérdidas por soiling, tendencias

### 🔋🌫️ Dashboard Integrado v3
- **Filtros Globales**: Sincronizados entre sistemas
- **DustIQ**: Análisis completo de soiling ratio
- **PVStand**: Curvas IV interactivas por fecha/hora
- **Información del Sistema**: Estado de conexión y configuración

## 🎛️ Configuración de Usuario

### Filtros Comunes
- **Frecuencia Temporal**: Diario, Semanal, Mensual
- **Rango de Fechas**: Configurable por usuario
- **Umbral SR**: Filtro por valor mínimo (DustIQ)
- **Módulos PVStand**: Selección de perc1fixed/perc2fixed

### Características Especiales
- **Carga Inteligente**: Cache de datos para mejor rendimiento
- **Manejo de Errores**: Fallbacks automáticos
- **Interfaz Responsiva**: Optimizada para diferentes dispositivos

## 📁 Estructura del Proyecto

```
SOILING/
├── streamlit_app.py                    # Dashboard DustIQ
├── streamlit_app_integrado_v3.py       # Dashboard Integrado v3
├── dashboard_integrado_v3.py           # Lógica del dashboard integrado
├── requirements_streamlit.txt          # Dependencias para Streamlit Cloud
├── README_DEPLOY.md                    # Este archivo
├── README_Dashboard_DustIQ.md          # Documentación técnica DustIQ
└── README_INI_CORREGIDO.md             # Documentación inicial
```

## 🔗 Enlaces Útiles

- **Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io)
- **GitHub**: [nicoleits/ATAMOSTEC](https://github.com/nicoleits/ATAMOSTEC)
- **Documentación DustIQ**: README_Dashboard_DustIQ.md

## 🛠️ Desarrollo Local

### Instalación
```bash
cd SOILING
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements_streamlit.txt
```

### Ejecución
```bash
# Dashboard DustIQ
streamlit run streamlit_app.py

# Dashboard Integrado
streamlit run streamlit_app_integrado_v3.py
```

## 📈 Estado del Proyecto

- ✅ **Dashboard DustIQ**: Funcional y estable
- ✅ **Dashboard Integrado v3**: Funcional y optimizado
- ✅ **Despliegue Streamlit Cloud**: Configurado
- ✅ **Documentación**: Actualizada
- ✅ **Código Limpio**: Archivos obsoletos eliminados

---

**Desarrollado para ATAMOSTEC** | **Dashboard Integrado** - DustIQ + PVStand 