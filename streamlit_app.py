#!/usr/bin/env python3
"""
Dashboard DustIQ - Análisis de Soiling Ratio
Archivo principal para despliegue en Streamlit Cloud
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
    page_title="Dashboard DustIQ - Análisis de Soiling",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🌫️ Dashboard DustIQ - Análisis de Soiling Ratio")
st.markdown("---")

# Configuración de ClickHouse
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}

@st.cache_data(ttl=3600)
def load_dustiq_data_from_clickhouse():
    """Carga los datos de DustIQ desde ClickHouse."""
    try:
        with st.spinner("🔄 Conectando a ClickHouse..."):
            client = clickhouse_connect.get_client(
                host=CLICKHOUSE_CONFIG['host'],
                port=CLICKHOUSE_CONFIG['port'],
                user=CLICKHOUSE_CONFIG['user'],
                password=CLICKHOUSE_CONFIG['password'],
                connect_timeout=10,
                send_receive_timeout=30
            )
        
        # Consulta para datos de DustIQ
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
        
        # Procesar datos
        with st.spinner("🔄 Procesando datos..."):
            df_raw = pd.DataFrame(data_dustiq.result_set,
                                 columns=['Stamptime', 'Attribute', 'Measure'])
            
            # Convertir timestamp
            df_raw['Stamptime'] = pd.to_datetime(df_raw['Stamptime'])
            
            # Asegurar que esté en UTC naive
            if df_raw['Stamptime'].dt.tz is not None:
                df_raw['Stamptime'] = df_raw['Stamptime'].dt.tz_localize(None)
            
            # Manejar duplicados antes del pivot
            df_clean = df_raw.groupby(['Stamptime', 'Attribute'])['Measure'].mean().reset_index()
            
            # Pivotar datos para obtener columnas SR_C11_Avg y SR_C12_Avg
            df_dustiq = df_clean.pivot(index='Stamptime', columns='Attribute', values='Measure')
            
            # Renombrar columnas si es necesario
            df_dustiq.columns.name = None  # Remover nombre de columnas
            
            # Filtrar valores > 0
            if 'SR_C11_Avg' in df_dustiq.columns:
                df_dustiq = df_dustiq[df_dustiq['SR_C11_Avg'] > 0]
            
            df_dustiq = df_dustiq.sort_index()
        
        client.close()
        
        st.success(f"✅ Datos cargados desde ClickHouse: {len(df_dustiq):,} registros")
        return df_dustiq
        
    except Exception as e:
        st.warning(f"⚠️ Error de conexión a ClickHouse: {str(e)[:100]}...")
        return None

@st.cache_data
def load_dustiq_data_from_file():
    """Carga los datos de DustIQ desde archivo local."""
    try:
        with st.spinner("📁 Cargando datos desde archivo local..."):
            # Intentar diferentes rutas posibles
            possible_paths = [
                "datos/raw_dustiq_data.csv",
                "raw_dustiq_data.csv"
            ]
            
            for file_path in possible_paths:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                    
                    # Asegurar que esté en UTC naive
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Filtrar por fechas y umbral
                    df = df[df.index >= pd.to_datetime('2024-06-24')]
                    if 'SR_C11_Avg' in df.columns:
                        df = df[df['SR_C11_Avg'] > 0]
                    
                    st.success(f"✅ Datos cargados desde archivo: {len(df):,} registros")
                    st.info("🔄 **Modo Offline**: Usando datos almacenados localmente")
                    return df
            
            st.error("❌ No se encontró el archivo de datos DustIQ")
            return None
            
    except Exception as e:
        st.error(f"❌ Error al cargar datos desde archivo: {e}")
        return None

def procesar_datos_configuracion(df_sr, selected_freq, selected_franjas, franjas_disponibles):
    """Procesa los datos según la configuración seleccionada."""
    resultados = {}
    
    for franja in selected_franjas:
        if franja == "Mediodía Solar":
            # Procesar mediodía solar (franja de 2 horas centrada en mediodía)
            data_franja = df_sr.between_time('11:30', '13:30')
            if not data_franja.empty:
                data_procesada = data_franja.resample(selected_freq, origin='start').quantile(0.25)
                resultados["Mediodía Solar (11:30-13:30)"] = data_procesada
        else:
            # Procesar franjas horarias fijas
            if franja in franjas_disponibles:
                start_time, end_time = franjas_disponibles[franja]
                data_franja = df_sr.between_time(start_time, end_time)
                if not data_franja.empty:
                    data_procesada = data_franja.resample(selected_freq, origin='start').quantile(0.25)
                    resultados[franja] = data_procesada
    
    return resultados

# Cargar datos con fallback inteligente
with st.spinner("🔄 Cargando datos de DustIQ..."):
    # Intentar cargar desde ClickHouse primero
    df_dustiq = load_dustiq_data_from_clickhouse()
    
    # Si falla, intentar desde archivo local
    if df_dustiq is None:
        st.warning("⚠️ Cambiando a modo offline...")
        df_dustiq = load_dustiq_data_from_file()

if df_dustiq is None:
    st.error("❌ No se pudieron cargar los datos de DustIQ. Verifica la conexión o la disponibilidad del archivo.")
    st.stop()

# Sidebar para filtros
st.sidebar.header("🎛️ Filtros y Configuración")

# Filtro de fechas
min_date = df_dustiq.index.min().date()
max_date = df_dustiq.index.max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "📅 Fecha Inicio:",
        value=max(min_date, pd.to_datetime('2024-07-01').date()),
        min_value=min_date,
        max_value=max_date
    )

with col2:
    end_date = st.date_input(
        "📅 Fecha Fin:",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

# Filtrar datos por rango de fechas
df_dustiq_filtered = df_dustiq.loc[start_date:end_date]

# Filtro de umbral SR
sr_threshold = st.sidebar.slider(
    "🎚️ Umbral SR (%):",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1,
    help="Filtrar datos con Soiling Ratio mayor a este valor"
)

# Aplicar filtro de umbral
sr_column = 'SR_C11_Avg'
if sr_column in df_dustiq_filtered.columns:
    df_sr_filtered = df_dustiq_filtered[df_dustiq_filtered[sr_column] > sr_threshold][sr_column].copy()
else:
    st.error(f"❌ Columna {sr_column} no encontrada en los datos")
    st.stop()

# Configuración de análisis
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
    index=1  # Semanal por defecto
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
    index=0
)

# Selección personalizada de franjas
selected_franjas = []
if selected_franja_option == "Personalizado":
    selected_franjas = st.sidebar.multiselect(
        "🕐 Seleccionar franjas específicas:",
        list(franjas_disponibles.keys()),
        default=["12:00-13:00", "14:00-15:00"]
    )
elif selected_franja_option == "Todas las franjas":
    selected_franjas = list(franjas_disponibles.keys())
elif selected_franja_option == "Mediodía Solar":
    selected_franjas = ["Mediodía Solar"]

# Tipo de análisis
analysis_type = st.sidebar.selectbox(
    "📊 Tipo de Análisis:",
    [
        "📈 Vista General",
        "🕐 Franjas Horarias Fijas"
    ]
)

# Información de datos en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Información de Datos")
st.sidebar.metric("Puntos totales", f"{len(df_dustiq_filtered):,}")
st.sidebar.metric("Puntos filtrados", f"{len(df_sr_filtered):,}")
st.sidebar.metric("Rango de fechas", f"{start_date} a {end_date}")

# === VISTA GENERAL ===
if analysis_type == "📈 Vista General":
    st.subheader("📈 Vista General de Soiling Ratio")
    
    # Mostrar configuración actual
    st.info(f"📅 **Frecuencia**: {selected_freq} | 🕐 **Franjas**: {', '.join(selected_franjas)}")
    
    # Procesar datos según la configuración
    datos_procesados = procesar_datos_configuracion(
        df_sr_filtered, 
        freq_options[selected_freq], 
        selected_franjas, 
        franjas_disponibles
    )
    
    if datos_procesados:
        # Métricas principales (usando datos agregados)
        datos_combinados = pd.concat(datos_procesados.values()).dropna()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Promedio SR (%)",
                f"{datos_combinados.mean():.2f}",
                delta=f"{datos_combinados.mean() - 100:.2f}"
            )
        
        with col2:
            st.metric(
                "Mediana SR (%)",
                f"{datos_combinados.median():.2f}",
                delta=f"{datos_combinados.median() - 100:.2f}"
            )
        
        with col3:
            st.metric(
                "Desv. Estándar",
                f"{datos_combinados.std():.2f}"
            )
        
        with col4:
            st.metric(
                "Pérdida Promedio",
                f"{100 - datos_combinados.mean():.2f}%"
            )
        
        # Gráfico de serie temporal con datos procesados
        st.subheader(f"📊 Evolución Temporal del Soiling Ratio ({selected_freq})")
        
        fig_timeline = go.Figure()
        
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (franja, datos) in enumerate(datos_procesados.items()):
            if not datos.empty:
                color = colores[i % len(colores)]
                
                fig_timeline.add_trace(go.Scatter(
                    x=datos.index,
                    y=datos.values,
                    mode='lines+markers',
                    name=franja,
                    line=dict(color=color, width=2),
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
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.warning("⚠️ No hay datos disponibles para la configuración seleccionada")

# === FRANJAS HORARIAS FIJAS ===
elif analysis_type == "🕐 Franjas Horarias Fijas":
    st.subheader("🕐 Análisis por Franjas Horarias Fijas")
    
    # Mostrar configuración actual
    st.info(f"📅 **Frecuencia**: {selected_freq} | 🕐 **Franjas**: {', '.join(selected_franjas)}")
    
    # Procesar datos según la configuración
    datos_procesados = procesar_datos_configuracion(
        df_sr_filtered, 
        freq_options[selected_freq], 
        selected_franjas, 
        franjas_disponibles
    )
    
    if datos_procesados:
        # Crear gráfico sin tendencias
        fig = go.Figure()
        
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (franja, datos) in enumerate(datos_procesados.items()):
            if not datos.empty:
                color = colores[i % len(colores)]
                
                fig.add_trace(go.Scatter(
                    x=datos.index,
                    y=datos.values,
                    mode='lines+markers',
                    name=franja,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    opacity=0.8
                ))
        
        # Configurar el gráfico
        fig.update_layout(
            title=f"Soiling Ratio - {selected_freq} - {', '.join(selected_franjas)}",
            xaxis_title="Fecha",
            yaxis_title="Soiling Ratio (%)",
            height=500,
            yaxis=dict(range=[90, 110]),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Línea de referencia al 100%
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="Referencia 100%",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas por franja
        st.subheader("📊 Estadísticas por Franja")
        
        stats_franjas = []
        for franja, datos in datos_procesados.items():
            if not datos.empty:
                stats_franjas.append({
                    'Franja': franja,
                    'Promedio (%)': f"{datos.mean():.2f}",
                    'Mediana (%)': f"{datos.median():.2f}",
                    'Desv. Est.': f"{datos.std():.2f}",
                    'Pérdida (%)': f"{100 - datos.mean():.2f}",
                    'Mínimo (%)': f"{datos.min():.2f}",
                    'Máximo (%)': f"{datos.max():.2f}",
                    'Puntos': len(datos)
                })
        
        if stats_franjas:
            df_stats = pd.DataFrame(stats_franjas)
            st.dataframe(df_stats, use_container_width=True)
    else:
        st.warning("⚠️ No hay datos disponibles para las franjas seleccionadas") 