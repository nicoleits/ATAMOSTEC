#!/usr/bin/env python3
"""
Streamlit App Integrado v4 - Entry Point para Streamlit Cloud
Dashboard Integrado: DustIQ + PVStand + Soiling Kit
"""

import streamlit as st
import sys
import os

# Verificar que plotly esté disponible
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError as e:
    st.error(f"Error: No se pudo importar plotly. {e}")
    st.error("Verifica que plotly esté incluido en requirements_streamlit.txt")
    st.stop()

# Verificar que pandas esté disponible
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    st.error(f"Error: No se pudo importar pandas/numpy. {e}")
    st.stop()

# Verificar que clickhouse-connect esté disponible
try:
    import clickhouse_connect
except ImportError as e:
    st.error(f"Error: No se pudo importar clickhouse-connect. {e}")
    st.stop()

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar el dashboard principal
try:
    from dashboard_integrado_v4 import *
except ImportError as e:
    st.error(f"Error al importar el dashboard: {e}")
    st.error("Verifica que el archivo dashboard_integrado_v4.py esté presente")
    st.stop() 