#!/usr/bin/env python3
"""
Streamlit App Integrado v4 - Entry Point para Streamlit Cloud
Dashboard Integrado: DustIQ + PVStand + Soiling Kit
"""

import streamlit as st
import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar el dashboard principal
try:
    from dashboard_integrado_v4 import *
except ImportError as e:
    st.error(f"Error al importar el dashboard: {e}")
    st.error("Verifica que el archivo dashboard_integrado_v4.py esté presente")
    st.stop() 