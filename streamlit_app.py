#!/usr/bin/env python3
"""
Dashboard DustIQ - Análisis de Soiling Ratio
Archivo principal para despliegue en Streamlit Cloud
"""

import streamlit as st

# Importar el dashboard principal
from dashboard_dustiq_dedicated import *

# Configuración de la página
st.set_page_config(
    page_title="Dashboard DustIQ - Análisis de Soiling",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# El resto del código se ejecuta automáticamente desde el archivo importado 