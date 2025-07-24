# 🔋 Dashboard Curvas IV PVStand

Dashboard interactivo para visualizar las curvas IV (Intensidad-Voltaje) de los módulos PVStand.

## 📋 Características

- **Selección de módulo**: perc1fixed o perc2fixed
- **Selección de fecha**: Visualizar curvas de cualquier fecha disponible
- **Selección de curva**: Múltiples curvas por día (diferentes horas)
- **Visualización interactiva**: Curvas IV con corriente vs voltaje
- **Parámetros eléctricos**: PMP, ISC, VOC, IMP, VMP, Factor de Forma
- **Descarga de datos**: Exportar datos de la curva seleccionada

## 🚀 Instalación y Uso

### 1. Instalar dependencias
```bash
cd /home/nicole/SR/SOILING
source .venv/bin/activate
pip install streamlit plotly
```

### 2. Ejecutar el dashboard
```bash
# Opción 1: Usar el script automático
./run_dashboard.sh

# Opción 2: Ejecutar manualmente
streamlit run dashboard_pvstand_curves.py
```

### 3. Acceder al dashboard
Abrir el navegador en: `http://localhost:8501`

## 📊 Funcionalidades del Dashboard

### Panel de Control (Sidebar)
- **Módulo**: Seleccionar entre perc1fixed y perc2fixed
- **Fecha**: Elegir la fecha de las curvas a visualizar
- **Curva del día**: Seleccionar la curva específica por hora
- **Información de la curva**: Estadísticas y parámetros

### Visualizaciones Principales

#### 1. Curva IV
- **Eje X**: Voltaje (V)
- **Eje Y**: Corriente (A)
- **Punto destacado**: Punto de Máxima Potencia (PMP)
- **Líneas de referencia**: ISC y VOC

#### 2. Curva de Potencia
- **Eje X**: Voltaje (V)
- **Eje Y**: Potencia (W)
- **Punto destacado**: Potencia Máxima (PMP)

### Información Detallada
- **Estadísticas de la curva**: Número de puntos, rangos de valores
- **Parámetros eléctricos**: PMP, ISC, VOC, IMP, VMP
- **Factor de forma**: Indicador de calidad de la curva
- **Tabla de datos**: Todos los puntos de medición
- **Descarga**: Exportar datos en formato CSV

## 📁 Estructura de Archivos

```
SOILING/
├── dashboard_pvstand_curves.py    # Dashboard principal
├── run_dashboard.sh               # Script de ejecución
├── requirements_dashboard.txt     # Dependencias
├── README_Dashboard.md           # Este archivo
└── datos/
    └── raw_pvstand_curves_data.csv  # Datos de curvas IV
```

## 🔧 Requisitos

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- Numpy
- Datos de curvas IV (archivo `raw_pvstand_curves_data.csv`)

## 📈 Interpretación de las Curvas

### Curva IV Típica
- **Forma**: Curva característica de un módulo fotovoltaico
- **ISC**: Corriente de cortocircuito (punto más alto en Y)
- **VOC**: Voltaje de circuito abierto (punto más a la derecha en X)
- **PMP**: Punto de máxima potencia (producto I×V máximo)

### Factor de Forma
- **>80%**: Excelente calidad
- **70-80%**: Buena calidad
- **<70%**: Calidad baja

## 🐛 Solución de Problemas

### Error: "No se encontró el archivo"
- Asegúrate de haber ejecutado `download_pvstand_curves` primero
- Verifica que el archivo `raw_pvstand_curves_data.csv` existe en `/home/nicole/SR/SOILING/datos/`

### Error: "No hay datos disponibles"
- Verifica que la fecha seleccionada tenga datos
- Cambia el módulo o la fecha

### Dashboard no carga
- Verifica que Streamlit esté instalado: `pip install streamlit`
- Revisa los logs en la terminal

## 📞 Soporte

Para problemas o mejoras, revisa:
1. Los logs en la terminal donde ejecutaste el dashboard
2. Que todos los archivos estén en las ubicaciones correctas
3. Que las dependencias estén instaladas correctamente 