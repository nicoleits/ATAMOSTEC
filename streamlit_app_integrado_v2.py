#!/usr/bin/env python3
"""
Dashboard Integrado v2 - DustIQ + PVStand
Archivo principal para despliegue en Streamlit Cloud
Versión con filtros globales sincronizados
"""

import streamlit as st
import sys
import os

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Integrado v2 - DustIQ + PVStand",
    page_icon="🔋🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar el directorio actual al path para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Importar y ejecutar el dashboard integrado v2
    import dashboard_integrado_v2

    # El dashboard se ejecuta automáticamente al importar el módulo

except Exception as e:
    st.error(f"❌ Error al cargar el dashboard integrado v2: {str(e)}")
    st.info("🔄 Intentando cargar en modo de respaldo...")

    # Modo de respaldo: mostrar información básica
    st.title("🔋🌫️ Dashboard Integrado v2 - DustIQ + PVStand")
    st.markdown("---")

    st.warning("⚠️ El dashboard integrado v2 no pudo cargarse completamente.")
    st.info("📋 Información del sistema:")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Python version:** {sys.version}")
        st.write(f"**Streamlit version:** {st.__version__}")

    with col2:
        st.write(f"**Working directory:** {os.getcwd()}")
        st.write(f"**Files in directory:** {len(os.listdir('.'))}")

    st.error("🔧 Por favor, verifica la configuración del despliegue.") 