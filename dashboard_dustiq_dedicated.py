import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import clickhouse_connect
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Importar funciones auxiliares para DustIQ
try:
    from dustiq_dashboard_utils import (
        calcular_tendencia_dustiq, 
        calculate_solar_noon_approximate,
        procesar_datos_franjas_horarias,
        procesar_datos_mediodia_solar,
        crear_grafico_franjas_horarias,
        crear_grafico_mediodia_solar,
        generar_estadisticas_dustiq,
        crear_tabla_estadisticas,
        validar_datos_dustiq,
        export_data_to_csv
    )
except ImportError:
    st.error("❌ No se pudieron importar las funciones auxiliares de DustIQ. Asegúrate de que el archivo dustiq_dashboard_utils.py esté disponible.")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Dashboard DustIQ - Análisis de Soiling",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de ClickHouse
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}

# Título principal
st.title("🌫️ Dashboard DustIQ - Análisis de Soiling Ratio")
st.markdown("---")

# Función para cargar datos de DustIQ desde ClickHouse
@st.cache_data(ttl=3600)
def load_dustiq_data_from_clickhouse():
    """Carga los datos de DustIQ directamente desde ClickHouse."""
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
            timestamp,
            SR_C11_Avg,
            SR_C12_Avg
        FROM ref_data.dustiq_processed_data
        WHERE timestamp >= '2024-06-24' AND timestamp <= '2025-07-31'
        AND SR_C11_Avg > 0
        ORDER BY timestamp
        """
        
        with st.spinner("📊 Descargando datos de DustIQ..."):
            data_dustiq = client.query(query_dustiq)
        
        # Procesar datos
        with st.spinner("🔄 Procesando datos..."):
            df_dustiq = pd.DataFrame(data_dustiq.result_set,
                                    columns=['timestamp', 'SR_C11_Avg', 'SR_C12_Avg'])
            
            # Convertir timestamp
            df_dustiq['timestamp'] = pd.to_datetime(df_dustiq['timestamp'])
            
            # Asegurar que esté en UTC naive
            if df_dustiq['timestamp'].dt.tz is not None:
                df_dustiq['timestamp'] = df_dustiq['timestamp'].dt.tz_localize(None)
            
            df_dustiq.set_index('timestamp', inplace=True)
            df_dustiq = df_dustiq.sort_index()
        
        client.close()
        
        st.success(f"✅ Datos cargados desde ClickHouse: {len(df_dustiq):,} registros")
        return df_dustiq
        
    except Exception as e:
        st.warning(f"⚠️ Error de conexión a ClickHouse: {str(e)[:100]}...")
        return None

# Función para cargar datos de DustIQ desde archivo local
@st.cache_data
def load_dustiq_data_from_file():
    """Carga los datos de DustIQ desde archivo local."""
    try:
        with st.spinner("📁 Cargando datos desde archivo local..."):
            # Intentar diferentes rutas posibles
            possible_paths = [
                "datos/raw_dustiq_data.csv",
                "/home/nicole/SR/SOILING/datos/raw_dustiq_data.csv",
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

# Validar datos cargados
es_valido, mensajes_validacion = validar_datos_dustiq(df_dustiq)

with st.expander("🔍 Información de Validación de Datos", expanded=False):
    for mensaje in mensajes_validacion:
        st.write(mensaje)

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

# Tipo de análisis
analysis_type = st.sidebar.selectbox(
    "📊 Tipo de Análisis:",
    [
        "📈 Vista General",
        "🕐 Franjas Horarias Fijas",
        "☀️ Mediodía Solar",
        "📅 Comparación Temporal",
        "📊 Estadísticas Detalladas"
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
    st.subheader("📊 Evolución Temporal del Soiling Ratio")
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=df_sr_filtered.index,
        y=df_sr_filtered.values,
        mode='lines',
        name='Soiling Ratio',
        line=dict(color='blue', width=1),
        opacity=0.7
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
        title="Evolución del Soiling Ratio en el Tiempo",
        xaxis_title="Fecha",
        yaxis_title="Soiling Ratio (%)",
        height=500,
        yaxis=dict(range=[90, 110]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Distribución de valores
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribución de Valores")
        
        fig_hist = px.histogram(
            df_sr_filtered,
            nbins=50,
            title="Distribución del Soiling Ratio",
            labels={'value': 'Soiling Ratio (%)', 'count': 'Frecuencia'}
        )
        
        fig_hist.add_vline(
            x=100,
            line_dash="dash",
            line_color="red",
            annotation_text="Referencia 100%"
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("📈 Box Plot")
        
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=df_sr_filtered.values,
            name='Soiling Ratio',
            boxpoints='outliers',
            marker_color='lightblue'
        ))
        
        fig_box.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="Referencia 100%"
        )
        
        fig_box.update_layout(
            title="Distribución Estadística",
            yaxis_title="Soiling Ratio (%)",
            height=400,
            yaxis=dict(range=[90, 110])
        )
        
        st.plotly_chart(fig_box, use_container_width=True)

# === FRANJAS HORARIAS FIJAS ===
elif analysis_type == "🕐 Franjas Horarias Fijas":
    st.subheader("🕐 Análisis por Franjas Horarias Fijas")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Configuración")
        
        # Seleccionar franjas horarias
        franjas_disponibles = {
            '10:00-11:00': ('10:00', '11:00'),
            '12:00-13:00': ('12:00', '13:00'),
            '14:00-15:00': ('14:00', '15:00'),
            '15:00-16:00': ('15:00', '16:00')
        }
        
        franjas_seleccionadas = st.multiselect(
            "🕐 Seleccionar Franjas:",
            list(franjas_disponibles.keys()),
            default=['12:00-13:00', '14:00-15:00']
        )
        
        mostrar_tendencias = st.checkbox("📈 Mostrar Tendencias", value=True)
        
        if st.button("🔄 Actualizar Análisis"):
            st.rerun()
    
    with col1:
        if franjas_seleccionadas:
            # Procesar datos por franjas
            franjas_dict = {k: franjas_disponibles[k] for k in franjas_seleccionadas}
            datos_procesados = procesar_datos_franjas_horarias(df_sr_filtered, franjas_dict)
            
            # Crear gráfico
            fig_franjas = crear_grafico_franjas_horarias(
                datos_procesados,
                "Análisis por Franjas Horarias",
                mostrar_tendencias=mostrar_tendencias
            )
            
            st.plotly_chart(fig_franjas, use_container_width=True)
            
            # Estadísticas por franja
            st.subheader("📊 Estadísticas por Franja")
            
            stats_franjas = []
            for franja, datos in datos_procesados.items():
                if not datos.empty:
                    slope, intercept, r_squared = calcular_tendencia_dustiq(
                        np.arange(len(datos)), datos.values
                    )
                    
                    stats_franjas.append({
                        'Franja': franja,
                        'Promedio (%)': f"{datos.mean():.2f}",
                        'Mediana (%)': f"{datos.median():.2f}",
                        'Desv. Est.': f"{datos.std():.2f}",
                        'Pérdida (%)': f"{100 - datos.mean():.2f}",
                        'Tendencia (%/día)': f"{slope * 100:.4f}" if slope else "N/A",
                        'R²': f"{r_squared:.3f}" if r_squared else "N/A"
                    })
            
            if stats_franjas:
                df_stats = pd.DataFrame(stats_franjas)
                st.dataframe(df_stats, use_container_width=True)
        else:
            st.warning("⚠️ Por favor selecciona al menos una franja horaria")

# === MEDIODÍA SOLAR ===
elif analysis_type == "☀️ Mediodía Solar":
    st.subheader("☀️ Análisis de Mediodía Solar")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Configuración")
        
        # Opciones de análisis
        tipo_analisis = st.selectbox(
            "📊 Tipo de Análisis:",
            ["Semanal", "Diario"]
        )
        
        duracion_ventana = st.slider(
            "⏱️ Duración Ventana (min):",
            min_value=30,
            max_value=120,
            value=60,
            step=15
        )
        
        mostrar_tendencia = st.checkbox("📈 Mostrar Tendencia", value=True)
        
        if st.button("🔄 Actualizar Análisis"):
            st.rerun()
    
    with col1:
        # Procesar datos de mediodía solar
        datos_procesados = procesar_datos_mediodia_solar(
            df_sr_filtered,
            duracion_ventana_minutos=duracion_ventana,
            freq='1W' if tipo_analisis == "Semanal" else '1D'
        )
        
        if not datos_procesados.empty:
            # Crear gráfico
            fig_mediodia = crear_grafico_mediodia_solar(
                datos_procesados,
                f"Análisis de Mediodía Solar - {tipo_analisis}",
                mostrar_tendencia=mostrar_tendencia
            )
            
            st.plotly_chart(fig_mediodia, use_container_width=True)
            
            # Estadísticas
            st.subheader("📊 Estadísticas de Mediodía Solar")
            
            slope, intercept, r_squared = calcular_tendencia_dustiq(
                np.arange(len(datos_procesados)), datos_procesados.values
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Promedio SR (%)", f"{datos_procesados.mean():.2f}")
            
            with col2:
                st.metric("Mediana SR (%)", f"{datos_procesados.median():.2f}")
            
            with col3:
                st.metric("Tendencia (%/día)", f"{slope * 100:.4f}" if slope else "N/A")
            
            with col4:
                st.metric("R²", f"{r_squared:.3f}" if r_squared else "N/A")
        else:
            st.warning("⚠️ No hay datos suficientes para el análisis de mediodía solar")

# === COMPARACIÓN TEMPORAL ===
elif analysis_type == "📅 Comparación Temporal":
    st.subheader("📅 Análisis de Comparación Temporal")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Configuración")
        
        # Selección de períodos
        periodo_tipo = st.selectbox(
            "📅 Tipo de Período:",
            ["Mensual", "Semanal", "Estacional"]
        )
        
        if periodo_tipo == "Mensual":
            meses_disponibles = []
            for date in pd.date_range(start_date, end_date, freq='MS'):
                meses_disponibles.append(date.strftime('%Y-%m'))
            
            meses_seleccionados = st.multiselect(
                "📅 Seleccionar Meses:",
                meses_disponibles,
                default=meses_disponibles[-3:] if len(meses_disponibles) >= 3 else meses_disponibles
            )
            
            franja_comparacion = st.selectbox(
                "🕐 Franja Horaria:",
                ["12:00-13:00", "14:00-15:00", "Mediodía Solar (±60 min)"]
            )
    
    with col1:
        if periodo_tipo == "Mensual" and meses_seleccionados:
            fig_temporal = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for i, mes in enumerate(meses_seleccionados):
                # Filtrar datos por mes
                mes_inicio = pd.to_datetime(f"{mes}-01")
                if mes == meses_disponibles[-1]:
                    mes_fin = end_date
                else:
                    mes_fin = mes_inicio + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                
                datos_mes = df_sr_filtered[
                    (df_sr_filtered.index >= mes_inicio) & 
                    (df_sr_filtered.index <= mes_fin)
                ]
                
                if not datos_mes.empty:
                    if franja_comparacion == "Mediodía Solar (±60 min)":
                        # Usar análisis de mediodía solar
                        datos_procesados = procesar_datos_mediodia_solar(
                            datos_mes,
                            duracion_ventana_minutos=60,
                            freq='1D'
                        )
                    else:
                        # Usar franja horaria fija
                        start_time, end_time = franja_comparacion.split('-')
                        datos_procesados = datos_mes.between_time(
                            start_time, end_time
                        ).resample('1D').quantile(0.25).dropna()
                    
                    if not datos_procesados.empty:
                        fig_temporal.add_trace(go.Box(
                            y=datos_procesados.values,
                            name=f"{mes}",
                            marker_color=colors[i % len(colors)],
                            boxpoints='outliers'
                        ))
            
            fig_temporal.update_layout(
                title=f"Comparación Temporal por Meses - {franja_comparacion}",
                xaxis_title="Mes",
                yaxis_title="Soiling Ratio (%)",
                height=500,
                yaxis=dict(range=[90, 110])
            )
            
            st.plotly_chart(fig_temporal, use_container_width=True)

# === ESTADÍSTICAS DETALLADAS ===
elif analysis_type == "📊 Estadísticas Detalladas":
    st.subheader("📊 Estadísticas Detalladas")
    
    # Generar estadísticas completas
    stats_completas = generar_estadisticas_dustiq(df_sr_filtered)
    
    # Mostrar tabla de estadísticas
    if stats_completas:
        st.subheader("📈 Estadísticas Generales")
        tabla_stats = crear_tabla_estadisticas(stats_completas)
        st.plotly_chart(tabla_stats, use_container_width=True)
    
    # Estadísticas por hora del día
    st.subheader("🕐 Estadísticas por Hora del Día")
    
    df_sr_filtered_hora = df_sr_filtered.copy()
    df_sr_filtered_hora['hora'] = df_sr_filtered_hora.index.hour
    
    stats_por_hora = df_sr_filtered_hora.groupby('hora').agg({
        'SR_C11_Avg': ['mean', 'std', 'count']
    }).round(2)
    
    stats_por_hora.columns = ['Promedio (%)', 'Desv. Est.', 'Cantidad']
    st.dataframe(stats_por_hora, use_container_width=True)
    
    # Gráfico de estadísticas por hora
    fig_hora = go.Figure()
    
    fig_hora.add_trace(go.Bar(
        x=stats_por_hora.index,
        y=stats_por_hora['Promedio (%)'],
        name='Promedio SR (%)',
        marker_color='lightblue'
    ))
    
    fig_hora.add_hline(
        y=100,
        line_dash="dash",
        line_color="red",
        annotation_text="Referencia 100%"
    )
    
    fig_hora.update_layout(
        title="Promedio de Soiling Ratio por Hora del Día",
        xaxis_title="Hora del Día",
        yaxis_title="Soiling Ratio (%)",
        height=400,
        yaxis=dict(range=[90, 110])
    )
    
    st.plotly_chart(fig_hora, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🌫️ **Dashboard DustIQ** - Análisis de Soiling Ratio | Desarrollado para ATAMOSTEC") 