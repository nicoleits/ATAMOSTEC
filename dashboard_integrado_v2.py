#!/usr/bin/env python3
"""
Dashboard Integrado - DustIQ + PVStand (Versión 2)
Aplicación unificada con filtros globales sincronizados
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os
import clickhouse_connect

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Integrado - DustIQ + PVStand",
    page_icon="🔋🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔋🌫️ Dashboard Integrado - DustIQ + PVStand")
st.markdown("---")

# Configuración de ClickHouse
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}

# ===== FILTROS GLOBALES (SIDEBAR) =====

st.sidebar.header("🎛️ Filtros Globales")

# Filtro de fechas global
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "📅 Fecha Inicio:",
        value=pd.to_datetime('2024-07-01').date(),
        key="global_start_date"
    )

with col2:
    end_date = st.date_input(
        "📅 Fecha Fin:",
        value=pd.to_datetime('2025-07-31').date(),
        key="global_end_date"
    )

# Configuración de análisis global
st.sidebar.subheader("⚙️ Configuración de Análisis")

# Frecuencia temporal
freq_options = {
    "Diario": "1D",
    "Semanal": "1W", 
    "Mensual": "1M"
}
selected_freq = st.sidebar.selectbox(
    "📅 Frecuencia Temporal:",
    list(freq_options.keys()),
    index=1,
    key="global_freq"
)

# Franjas horarias disponibles
franjas_disponibles = {
    "10:00-11:00": ("10:00", "11:00"),
    "12:00-13:00": ("12:00", "13:00"), 
    "14:00-15:00": ("14:00", "15:00"),
    "16:00-17:00": ("16:00", "17:00")
}

# Opciones de franjas
franja_options = ["Todas las franjas", "Mediodía Solar", "Personalizado"]
selected_franja_option = st.sidebar.selectbox(
    "🕐 Franjas Horarias:",
    franja_options,
    index=0,
    key="global_franja_option"
)

# Selección personalizada de franjas
selected_franjas = []
if selected_franja_option == "Personalizado":
    selected_franjas = st.sidebar.multiselect(
        "🕐 Seleccionar franjas específicas:",
        list(franjas_disponibles.keys()),
        default=["12:00-13:00", "14:00-15:00"],
        key="global_franjas"
    )
elif selected_franja_option == "Todas las franjas":
    selected_franjas = list(franjas_disponibles.keys())
elif selected_franja_option == "Mediodía Solar":
    selected_franjas = ["Mediodía Solar"]

# Filtro de umbral SR (solo para DustIQ)
sr_threshold = st.sidebar.slider(
    "🎚️ Umbral SR (%):",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1,
    help="Filtrar datos con Soiling Ratio mayor a este valor",
    key="global_sr_threshold"
)

# Selección de módulos PVStand
modules = ['perc1fixed', 'perc2fixed']
selected_modules = st.sidebar.multiselect(
    "📊 Módulos PVStand:",
    modules,
    default=modules[:1],
    key="global_pvstand_modules"
)

# ===== FUNCIONES DE CARGA DE DATOS =====

@st.cache_data(ttl=3600)
def load_dustiq_data():
    """Carga los datos de DustIQ desde ClickHouse."""
    try:
        with st.spinner("🔄 Conectando a ClickHouse para DustIQ..."):
            client = clickhouse_connect.get_client(
                host=CLICKHOUSE_CONFIG['host'],
                port=CLICKHOUSE_CONFIG['port'],
                user=CLICKHOUSE_CONFIG['user'],
                password=CLICKHOUSE_CONFIG['password'],
                connect_timeout=10,
                send_receive_timeout=30
            )
        
        query_dustiq = """
        SELECT 
            Stamptime,
            Attribute,
            Measure
        FROM PSDA.dustiq 
        WHERE Stamptime >= '2024-06-24' AND Stamptime <= '2025-07-31'
        AND Attribute IN ('SR_C11_Avg', 'SR_C12_Avg')
        ORDER BY Stamptime, Attribute
        """
        
        with st.spinner("📊 Descargando datos de DustIQ..."):
            data_dustiq = client.query(query_dustiq)
        
        df_raw = pd.DataFrame(data_dustiq.result_set,
                             columns=['Stamptime', 'Attribute', 'Measure'])
        
        df_raw['Stamptime'] = pd.to_datetime(df_raw['Stamptime'])
        if df_raw['Stamptime'].dt.tz is not None:
            df_raw['Stamptime'] = df_raw['Stamptime'].dt.tz_localize(None)
        
        df_clean = df_raw.groupby(['Stamptime', 'Attribute'])['Measure'].mean().reset_index()
        df_dustiq = df_clean.pivot(index='Stamptime', columns='Attribute', values='Measure')
        df_dustiq.columns.name = None
        
        if 'SR_C11_Avg' in df_dustiq.columns:
            df_dustiq = df_dustiq[df_dustiq['SR_C11_Avg'] > 0]
        
        df_dustiq = df_dustiq.sort_index()
        client.close()
        
        return df_dustiq
        
    except Exception as e:
        st.warning(f"⚠️ Error de conexión a ClickHouse para DustIQ: {str(e)[:100]}...")
        return None

@st.cache_data(ttl=3600)
def load_pvstand_data():
    """Carga los datos de curvas IV de PVStand desde ClickHouse."""
    try:
        with st.spinner("🔄 Conectando a ClickHouse para PVStand..."):
            client = clickhouse_connect.get_client(
                host=CLICKHOUSE_CONFIG['host'],
                port=CLICKHOUSE_CONFIG['port'],
                user=CLICKHOUSE_CONFIG['user'],
                password=CLICKHOUSE_CONFIG['password'],
                connect_timeout=10,
                send_receive_timeout=30
            )
        
        query_perc1_curves = """
        SELECT
            fecha,
            hora,
            corriente,
            voltaje,
            potencia
        FROM ref_data.iv_curves_perc1_fixed_medio_dia_solar
        WHERE fecha >= '2024-07-01' AND fecha <= '2025-07-31'
        ORDER BY fecha, hora
        """

        query_perc2_curves = """
        SELECT
            fecha,
            hora,
            corriente,
            voltaje,
            potencia
        FROM ref_data.iv_curves_perc2_fixed_medio_dia_solar
        WHERE fecha >= '2024-07-01' AND fecha <= '2025-07-31'
        ORDER BY fecha, hora
        """

        with st.spinner("📊 Descargando datos de PVStand..."):
            data_perc1_curves = client.query(query_perc1_curves)
            data_perc2_curves = client.query(query_perc2_curves)

        df_perc1_curves = pd.DataFrame(data_perc1_curves.result_set,
                                      columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])
        df_perc2_curves = pd.DataFrame(data_perc2_curves.result_set,
                                      columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])

        df_perc1_curves['module'] = 'perc1fixed'
        df_perc2_curves['module'] = 'perc2fixed'

        df_pvstand_curves = pd.concat([df_perc1_curves, df_perc2_curves], ignore_index=True)

        df_pvstand_curves['fecha'] = pd.to_datetime(df_pvstand_curves['fecha'])
        df_pvstand_curves['timestamp'] = pd.to_datetime(
            df_pvstand_curves['fecha'].astype(str) + ' ' + df_pvstand_curves['hora'].astype(str)
        )

        if df_pvstand_curves['timestamp'].dt.tz is None:
            df_pvstand_curves['timestamp'] = df_pvstand_curves['timestamp'].dt.tz_localize('UTC')
        else:
            df_pvstand_curves['timestamp'] = df_pvstand_curves['timestamp'].dt.tz_convert('UTC')

        df_pvstand_curves = df_pvstand_curves.sort_values('timestamp')
        df_pvstand_curves = df_pvstand_curves[['timestamp', 'module', 'corriente', 'voltaje', 'potencia', 'fecha', 'hora']]

        client.close()
        return df_pvstand_curves

    except Exception as e:
        st.warning(f"⚠️ Error de conexión a ClickHouse para PVStand: {str(e)[:100]}...")
        return None

# ===== CARGA DE DATOS =====

with st.spinner("🔄 Cargando datos de ambos sistemas..."):
    df_dustiq = load_dustiq_data()
    df_pvstand = load_pvstand_data()

if df_dustiq is None and df_pvstand is None:
    st.error("❌ No se pudieron cargar los datos de ningún sistema. Verifica la conexión a ClickHouse.")
    st.stop()

# Información de datos en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Información de Datos")

if df_dustiq is not None:
    df_dustiq_filtered = df_dustiq.loc[start_date:end_date]
    st.sidebar.metric("DustIQ - Puntos", f"{len(df_dustiq_filtered):,}")

if df_pvstand is not None:
    df_pvstand_filtered = df_pvstand[
        (df_pvstand['fecha'].dt.date >= start_date) & 
        (df_pvstand['fecha'].dt.date <= end_date)
    ]
    st.sidebar.metric("PVStand - Puntos", f"{len(df_pvstand_filtered):,}")

st.sidebar.metric("Rango de fechas", f"{start_date} a {end_date}")

# ===== INTERFAZ PRINCIPAL =====

# Pestañas principales
tab1, tab2, tab3 = st.tabs([
    "🌫️ DustIQ - Soiling Ratio", 
    "🔋 PVStand - Curvas IV", 
    "ℹ️ Información del Sistema"
])

# ===== PESTAÑA 1: DUSTIQ =====
with tab1:
    if df_dustiq is not None:
        st.subheader("🌫️ Análisis de Soiling Ratio - DustIQ")
        
        # Aplicar filtros globales a DustIQ
        df_dustiq_filtered = df_dustiq.loc[start_date:end_date]
        
        # Aplicar filtro de umbral
        sr_column = 'SR_C11_Avg'
        if sr_column in df_dustiq_filtered.columns:
            df_sr_filtered = df_dustiq_filtered[df_dustiq_filtered[sr_column] > sr_threshold][sr_column].copy()
        else:
            st.error(f"❌ Columna {sr_column} no encontrada en los datos")
            st.stop()
        
        # Mostrar configuración actual
        st.info(f"📅 **Frecuencia**: {selected_freq} | 🕐 **Franjas**: {', '.join(selected_franjas)}")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Promedio SR (%)",
                f"{df_sr_filtered.mean():.2f}",
                delta=f"{df_sr_filtered.mean() - 100:.2f}"
            )
        
        with col2:
            st.metric(
                "Mediana SR (%)",
                f"{df_sr_filtered.median():.2f}",
                delta=f"{df_sr_filtered.median() - 100:.2f}"
            )
        
        with col3:
            st.metric(
                "Desv. Estándar",
                f"{df_sr_filtered.std():.2f}"
            )
        
        with col4:
            st.metric(
                "Pérdida Promedio",
                f"{100 - df_sr_filtered.mean():.2f}%"
            )
        
        # Gráfico de serie temporal
        st.subheader(f"📊 Evolución Temporal del Soiling Ratio ({selected_freq})")
        
        # Resamplear datos según frecuencia seleccionada
        df_resampled = df_sr_filtered.resample(freq_options[selected_freq], origin='start').quantile(0.25)
        
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=df_resampled.index,
            y=df_resampled.values,
            mode='lines+markers',
            name='Soiling Ratio',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            opacity=0.8
        ))
        
        # Línea de referencia al 100%
        fig_timeline.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="Referencia 100%",
            annotation_position="top right"
        )
        
        fig_timeline.update_layout(
            title=f"Evolución del Soiling Ratio - {selected_freq}",
            xaxis_title="Fecha",
            yaxis_title="Soiling Ratio (%)",
            height=500,
            yaxis=dict(range=[90, 110]),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
    else:
        st.error("❌ No se pudieron cargar los datos de DustIQ")

# ===== PESTAÑA 2: PVSTAND =====
with tab2:
    if df_pvstand is not None:
        st.subheader("🔋 Análisis de Curvas IV - PVStand")
        
        # Aplicar filtros globales a PVStand
        df_pvstand_filtered = df_pvstand[
            (df_pvstand['fecha'].dt.date >= start_date) & 
            (df_pvstand['fecha'].dt.date <= end_date) &
            (df_pvstand['module'].isin(selected_modules))
        ]
        
        if not selected_modules:
            st.warning("⚠️ Por favor selecciona al menos un módulo en los filtros globales")
            st.stop()
        
        # Seleccionar fecha para visualización de curvas
        available_dates = sorted(df_pvstand_filtered['fecha'].dt.date.unique())
        if available_dates:
            selected_date = st.selectbox(
                "📅 Seleccionar Fecha para Curvas IV:",
                available_dates,
                index=len(available_dates)-1 if len(available_dates) > 0 else 0,
                key="pvstand_curve_date"
            )
            
            # Filtrar datos por fecha seleccionada
            df_date = df_pvstand_filtered[df_pvstand_filtered['fecha'].dt.date == selected_date]
            
            # Seleccionar curva del día
            available_curves = sorted(df_date['hora'].unique())
            if available_curves:
                selected_curve = st.selectbox(
                    "🕐 Seleccionar Curva:",
                    available_curves,
                    index=len(available_curves)-1 if len(available_curves) > 0 else 0,
                    key="pvstand_curve_time"
                )
                
                # Filtrar datos por curva seleccionada
                df_curve = df_date[df_date['hora'] == selected_curve]
                
                # Mostrar información de la curva
                st.info(f"**Fecha:** {selected_date} | **Hora:** {selected_curve} | **Módulos:** {', '.join(selected_modules)} | **Puntos:** {len(df_curve)}")
        
        # Gráficos de curvas IV
        if 'df_curve' in locals() and not df_curve.empty:
            # Gráfico de Curva IV (Corriente vs Voltaje)
            st.subheader("📈 Curva IV - Corriente vs Voltaje")
            
            fig_iv = go.Figure()
            
            for module in selected_modules:
                df_module_curve = df_curve[df_curve['module'] == module]
                if not df_module_curve.empty:
                    # Ordenar por voltaje para una curva suave
                    df_module_curve = df_module_curve.sort_values('voltaje')
                    
                    fig_iv.add_trace(go.Scatter(
                        x=df_module_curve['voltaje'],
                        y=df_module_curve['corriente'],
                        mode='lines+markers',
                        name=f'{module} - I vs V',
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
            
            fig_iv.update_layout(
                title=f"Curva IV - {selected_date} {selected_curve}",
                xaxis_title="Voltaje (V)",
                yaxis_title="Corriente (A)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_iv, use_container_width=True)
            
            # Gráfico de Potencia vs Voltaje
            st.subheader("⚡ Curva de Potencia")
            
            fig_power = go.Figure()
            
            for module in selected_modules:
                df_module_curve = df_curve[df_curve['module'] == module]
                if not df_module_curve.empty:
                    # Ordenar por voltaje
                    df_module_curve = df_module_curve.sort_values('voltaje')
                    
                    fig_power.add_trace(go.Scatter(
                        x=df_module_curve['voltaje'],
                        y=df_module_curve['potencia'],
                        mode='lines+markers',
                        name=f'{module} - P vs V',
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
            
            fig_power.update_layout(
                title=f"Curva de Potencia - {selected_date} {selected_curve}",
                xaxis_title="Voltaje (V)",
                yaxis_title="Potencia (W)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_power, use_container_width=True)
            
            # Tabla de datos
            st.subheader("📋 Datos de la Curva")
            st.dataframe(df_curve[['module', 'corriente', 'voltaje', 'potencia']], use_container_width=True)
        else:
            st.warning("⚠️ No hay datos disponibles para la fecha y curva seleccionadas")
    else:
        st.error("❌ No se pudieron cargar los datos de PVStand")

# ===== PESTAÑA 3: INFORMACIÓN DEL SISTEMA =====
with tab3:
    st.subheader("ℹ️ Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌫️ DustIQ")
        st.write("**Propósito:** Medición de soiling ratio mediante sensores ópticos")
        st.write("**Datos:** SR_C11_Avg, SR_C12_Avg")
        st.write("**Frecuencia:** Mediciones continuas")
        st.write("**Base de datos:** PSDA.dustiq")
        
        if df_dustiq is not None:
            st.write(f"**Registros disponibles:** {len(df_dustiq):,}")
            st.write(f"**Rango temporal:** {df_dustiq.index.min().date()} a {df_dustiq.index.max().date()}")
    
    with col2:
        st.subheader("🔋 PVStand")
        st.write("**Propósito:** Medición de curvas IV de módulos fotovoltaicos")
        st.write("**Datos:** Corriente, Voltaje, Potencia")
        st.write("**Frecuencia:** Curvas diarias")
        st.write("**Base de datos:** ref_data.iv_curves_*")
        
        if df_pvstand is not None:
            st.write(f"**Registros disponibles:** {len(df_pvstand):,}")
            st.write(f"**Módulos:** {', '.join(df_pvstand['module'].unique())}")
            st.write(f"**Rango temporal:** {df_pvstand['fecha'].min().date()} a {df_pvstand['fecha'].max().date()}")
    
    st.markdown("---")
    
    st.subheader("🔧 Configuración Técnica")
    st.write(f"**Servidor ClickHouse:** {CLICKHOUSE_CONFIG['host']}:{CLICKHOUSE_CONFIG['port']}")
    st.write(f"**Usuario:** {CLICKHOUSE_CONFIG['user']}")
    st.write(f"**Última actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Estado de conexión
    st.subheader("📡 Estado de Conexión")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if df_dustiq is not None:
            st.success("✅ DustIQ - Conectado")
        else:
            st.error("❌ DustIQ - Sin conexión")
    
    with col2:
        if df_pvstand is not None:
            st.success("✅ PVStand - Conectado")
        else:
            st.error("❌ PVStand - Sin conexión") 