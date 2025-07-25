#!/usr/bin/env python3
"""
Dashboard Integrado v3 - DustIQ + PVStand
Versión simplificada y robusta
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import clickhouse_connect

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Integrado v3 - DustIQ + PVStand",
    page_icon="🔋🌫️",
    layout="wide"
)

# Título principal
st.title("🔋🌫️ Dashboard Integrado v3 - DustIQ + PVStand")
st.markdown("---")

# Configuración de ClickHouse
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}

# ===== FILTROS GLOBALES =====

st.sidebar.header("🎛️ Filtros Globales")

# Filtro de fechas
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "📅 Fecha Inicio:",
        value=pd.to_datetime('2024-07-01').date(),
        key="start_date"
    )

with col2:
    end_date = st.date_input(
        "📅 Fecha Fin:",
        value=pd.to_datetime('2025-07-31').date(),
        key="end_date"
    )

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
    key="freq"
)

# Umbral SR
sr_threshold = st.sidebar.slider(
    "🎚️ Umbral SR (%):",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1,
    key="sr_threshold"
)

# Módulos PVStand
selected_modules = st.sidebar.multiselect(
    "📊 Módulos PVStand:",
    ['perc1fixed', 'perc2fixed'],
    default=['perc1fixed'],
    key="modules"
)

# ===== FUNCIONES DE CARGA =====

@st.cache_data(ttl=3600)
def load_dustiq_data():
    """Carga datos de DustIQ."""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password'],
            connect_timeout=10,
            send_receive_timeout=30
        )
        
        query = """
        SELECT Stamptime, Attribute, Measure
        FROM PSDA.dustiq 
        WHERE Stamptime >= '2024-06-24' AND Stamptime <= '2025-07-31'
        AND Attribute IN ('SR_C11_Avg', 'SR_C12_Avg')
        ORDER BY Stamptime, Attribute
        """
        
        data = client.query(query)
        df_raw = pd.DataFrame(data.result_set, columns=['Stamptime', 'Attribute', 'Measure'])
        
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
        st.warning(f"⚠️ Error DustIQ: {str(e)[:100]}...")
        return None

@st.cache_data(ttl=3600)
def load_pvstand_data():
    """Carga datos de PVStand."""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password'],
            connect_timeout=10,
            send_receive_timeout=30
        )
        
        query_perc1 = """
        SELECT fecha, hora, corriente, voltaje, potencia
        FROM ref_data.iv_curves_perc1_fixed_medio_dia_solar
        WHERE fecha >= '2024-07-01' AND fecha <= '2025-07-31'
        ORDER BY fecha, hora
        """

        query_perc2 = """
        SELECT fecha, hora, corriente, voltaje, potencia
        FROM ref_data.iv_curves_perc2_fixed_medio_dia_solar
        WHERE fecha >= '2024-07-01' AND fecha <= '2025-07-31'
        ORDER BY fecha, hora
        """

        data_perc1 = client.query(query_perc1)
        data_perc2 = client.query(query_perc2)

        df_perc1 = pd.DataFrame(data_perc1.result_set, columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])
        df_perc2 = pd.DataFrame(data_perc2.result_set, columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])

        df_perc1['module'] = 'perc1fixed'
        df_perc2['module'] = 'perc2fixed'

        df_pvstand = pd.concat([df_perc1, df_perc2], ignore_index=True)
        df_pvstand['fecha'] = pd.to_datetime(df_pvstand['fecha'])
        df_pvstand['timestamp'] = pd.to_datetime(df_pvstand['fecha'].astype(str) + ' ' + df_pvstand['hora'].astype(str))

        if df_pvstand['timestamp'].dt.tz is None:
            df_pvstand['timestamp'] = df_pvstand['timestamp'].dt.tz_localize('UTC')
        else:
            df_pvstand['timestamp'] = df_pvstand['timestamp'].dt.tz_convert('UTC')

        df_pvstand = df_pvstand.sort_values('timestamp')
        df_pvstand = df_pvstand[['timestamp', 'module', 'corriente', 'voltaje', 'potencia', 'fecha', 'hora']]

        client.close()
        return df_pvstand

    except Exception as e:
        st.warning(f"⚠️ Error PVStand: {str(e)[:100]}...")
        return None

# ===== CARGA DE DATOS =====

with st.spinner("🔄 Cargando datos..."):
    df_dustiq = load_dustiq_data()
    df_pvstand = load_pvstand_data()

if df_dustiq is None and df_pvstand is None:
    st.error("❌ No se pudieron cargar los datos. Verifica la conexión.")
    st.stop()

# Información en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Información")

if df_dustiq is not None:
    df_dustiq_filtered = df_dustiq.loc[start_date:end_date]
    st.sidebar.metric("DustIQ - Puntos", f"{len(df_dustiq_filtered):,}")

if df_pvstand is not None:
    df_pvstand_filtered = df_pvstand[
        (df_pvstand['fecha'].dt.date >= start_date) & 
        (df_pvstand['fecha'].dt.date <= end_date)
    ]
    st.sidebar.metric("PVStand - Puntos", f"{len(df_pvstand_filtered):,}")

# ===== PESTAÑAS =====

tab1, tab2, tab3 = st.tabs([
    "🌫️ DustIQ - Soiling Ratio", 
    "🔋 PVStand - Curvas IV", 
    "ℹ️ Información del Sistema"
])

# ===== PESTAÑA 1: DUSTIQ =====
with tab1:
    if df_dustiq is not None:
        st.subheader("🌫️ Análisis de Soiling Ratio - DustIQ")
        
        # Aplicar filtros
        df_dustiq_filtered = df_dustiq.loc[start_date:end_date]
        
        sr_column = 'SR_C11_Avg'
        if sr_column in df_dustiq_filtered.columns:
            df_sr_filtered = df_dustiq_filtered[df_dustiq_filtered[sr_column] > sr_threshold][sr_column].copy()
        else:
            st.error(f"❌ Columna {sr_column} no encontrada")
            st.stop()
        
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Promedio SR (%)", f"{df_sr_filtered.mean():.2f}")
        
        with col2:
            st.metric("Mediana SR (%)", f"{df_sr_filtered.median():.2f}")
        
        with col3:
            st.metric("Desv. Estándar", f"{df_sr_filtered.std():.2f}")
        
        with col4:
            st.metric("Pérdida Promedio", f"{100 - df_sr_filtered.mean():.2f}%")
        
        # Gráfico
        st.subheader(f"📊 Evolución Temporal ({selected_freq})")
        
        df_resampled = df_sr_filtered.resample(freq_options[selected_freq], origin='start').quantile(0.25)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_resampled.index,
            y=df_resampled.values,
            mode='lines+markers',
            name='Soiling Ratio',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Referencia 100%")
        
        fig.update_layout(
            title=f"Evolución del Soiling Ratio - {selected_freq}",
            xaxis_title="Fecha",
            yaxis_title="Soiling Ratio (%)",
            height=500,
            yaxis=dict(range=[90, 110])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("❌ No se pudieron cargar los datos de DustIQ")

# ===== PESTAÑA 2: PVSTAND =====
with tab2:
    if df_pvstand is not None:
        st.subheader("🔋 Análisis de Curvas IV - PVStand")
        
        # Aplicar filtros
        df_pvstand_filtered = df_pvstand[
            (df_pvstand['fecha'].dt.date >= start_date) & 
            (df_pvstand['fecha'].dt.date <= end_date) &
            (df_pvstand['module'].isin(selected_modules))
        ]
        
        if not selected_modules:
            st.warning("⚠️ Selecciona al menos un módulo")
            st.stop()
        
        # Seleccionar fecha
        available_dates = sorted(df_pvstand_filtered['fecha'].dt.date.unique())
        if available_dates:
            selected_date = st.selectbox(
                "📅 Seleccionar Fecha:",
                available_dates,
                index=len(available_dates)-1 if len(available_dates) > 0 else 0
            )
            
            df_date = df_pvstand_filtered[df_pvstand_filtered['fecha'].dt.date == selected_date]
            
            available_curves = sorted(df_date['hora'].unique())
            if available_curves:
                selected_curve = st.selectbox(
                    "🕐 Seleccionar Curva:",
                    available_curves,
                    index=len(available_curves)-1 if len(available_curves) > 0 else 0
                )
                
                df_curve = df_date[df_date['hora'] == selected_curve]
                
                if not df_curve.empty:
                    st.info(f"**Fecha:** {selected_date} | **Hora:** {selected_curve} | **Puntos:** {len(df_curve)}")
                    
                    # Gráfico IV
                    st.subheader("📈 Curva IV - Corriente vs Voltaje")
                    
                    fig_iv = go.Figure()
                    
                    for module in selected_modules:
                        df_module = df_curve[df_curve['module'] == module]
                        if not df_module.empty:
                            df_module = df_module.sort_values('voltaje')
                            
                            fig_iv.add_trace(go.Scatter(
                                x=df_module['voltaje'],
                                y=df_module['corriente'],
                                mode='lines+markers',
                                name=f'{module}',
                                line=dict(width=2)
                            ))
                    
                    fig_iv.update_layout(
                        title=f"Curva IV - {selected_date} {selected_curve}",
                        xaxis_title="Voltaje (V)",
                        yaxis_title="Corriente (A)",
                        height=500
                    )
                    
                    st.plotly_chart(fig_iv, use_container_width=True)
                    
                    # Gráfico Potencia
                    st.subheader("⚡ Curva de Potencia")
                    
                    fig_power = go.Figure()
                    
                    for module in selected_modules:
                        df_module = df_curve[df_curve['module'] == module]
                        if not df_module.empty:
                            df_module = df_module.sort_values('voltaje')
                            
                            fig_power.add_trace(go.Scatter(
                                x=df_module['voltaje'],
                                y=df_module['potencia'],
                                mode='lines+markers',
                                name=f'{module}',
                                line=dict(width=2)
                            ))
                    
                    fig_power.update_layout(
                        title=f"Curva de Potencia - {selected_date} {selected_curve}",
                        xaxis_title="Voltaje (V)",
                        yaxis_title="Potencia (W)",
                        height=500
                    )
                    
                    st.plotly_chart(fig_power, use_container_width=True)
                    
                    # Tabla de datos
                    st.subheader("📋 Datos de la Curva")
                    st.dataframe(df_curve[['module', 'corriente', 'voltaje', 'potencia']], use_container_width=True)
                else:
                    st.warning("⚠️ No hay datos para la fecha y curva seleccionadas")
            else:
                st.warning("⚠️ No hay curvas disponibles para la fecha seleccionada")
        else:
            st.warning("⚠️ No hay fechas disponibles en el rango seleccionado")
    else:
        st.error("❌ No se pudieron cargar los datos de PVStand")

# ===== PESTAÑA 3: INFORMACIÓN =====
with tab3:
    st.subheader("ℹ️ Información del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌫️ DustIQ")
        st.write("**Propósito:** Medición de soiling ratio")
        st.write("**Datos:** SR_C11_Avg, SR_C12_Avg")
        st.write("**Base de datos:** PSDA.dustiq")
        
        if df_dustiq is not None:
            st.write(f"**Registros:** {len(df_dustiq):,}")
            st.write(f"**Rango:** {df_dustiq.index.min().date()} a {df_dustiq.index.max().date()}")
    
    with col2:
        st.subheader("🔋 PVStand")
        st.write("**Propósito:** Curvas IV de módulos")
        st.write("**Datos:** Corriente, Voltaje, Potencia")
        st.write("**Base de datos:** ref_data.iv_curves_*")
        
        if df_pvstand is not None:
            st.write(f"**Registros:** {len(df_pvstand):,}")
            st.write(f"**Módulos:** {', '.join(df_pvstand['module'].unique())}")
            st.write(f"**Rango:** {df_pvstand['fecha'].min().date()} a {df_pvstand['fecha'].max().date()}")
    
    st.markdown("---")
    
    st.subheader("🔧 Configuración")
    st.write(f"**Servidor:** {CLICKHOUSE_CONFIG['host']}:{CLICKHOUSE_CONFIG['port']}")
    st.write(f"**Usuario:** {CLICKHOUSE_CONFIG['user']}")
    st.write(f"**Actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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