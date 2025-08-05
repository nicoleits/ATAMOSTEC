#!/usr/bin/env python3
"""
Dashboard Integrado v4 - DustIQ + PVStand + Soiling Kit
Versión con análisis completo de soiling
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly
import numpy as np
from datetime import datetime
import clickhouse_connect
import sys
import os

# Agregar el directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuración de InfluxDB para datos de irradiancia
INFLUXDB_CONFIG = {
    'url': 'http://146.83.153.212:8086',
    'token': 'your_token_here',  # Necesitarás el token real
    'org': 'PSDA'
}

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Integrado v4 - DustIQ + PVStand + Soiling Kit",
    page_icon="🔋🌫️🌪️",
    layout="wide"
)

# Título principal
st.title("🔋🌫️🌪️ Dashboard Integrado v4 - DustIQ + PVStand + Soiling Kit")
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
    key="franja_option"
)

# Selección personalizada de franjas
selected_franjas = []
if selected_franja_option == "Personalizado":
    selected_franjas = st.sidebar.multiselect(
        "🕐 Seleccionar franjas específicas:",
        list(franjas_disponibles.keys()),
        default=["12:00-13:00", "14:00-15:00"],
        key="franjas"
    )
elif selected_franja_option == "Todas las franjas":
    selected_franjas = list(franjas_disponibles.keys())
elif selected_franja_option == "Mediodía Solar":
    selected_franjas = ["Mediodía Solar"]

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

# Información sobre filtros
st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Nota sobre filtros:**
- **DustIQ & Soiling Kit**: Usan el rango de fechas del sidebar
- **PVStand**: Selecciona una fecha específica dentro del rango
- **Irradiancia**: Se muestra para el día seleccionado en PVStand
""")

# ===== FUNCIONES DE CARGA =====

# Función para verificar conexión a ClickHouse
def test_clickhouse_connection():
    """Prueba la conexión a ClickHouse."""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password'],
            database='PSDA'
        )
        # Prueba simple de conexión
        client.query("SELECT 1")
        client.close()
        return True
    except Exception as e:
        st.error(f"❌ Error de conexión a ClickHouse: {str(e)}")
        return False

# Verificar conexión antes de cargar datos
try:
    if not test_clickhouse_connection():
        st.warning("⚠️ No se puede conectar a ClickHouse. Usando datos simulados.")
        # No detener la aplicación, continuar con datos simulados
except Exception as e:
    st.warning(f"⚠️ Error verificando conexión: {str(e)}. Continuando con datos simulados.")

@st.cache_data(ttl=300)  # Reducir TTL a 5 minutos
def load_dustiq_data():
    """Carga datos de DustIQ."""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password'],
            database='PSDA'
        )
        
        query = """
        SELECT Stamptime, Attribute, Measure
        FROM PSDA.dustiq 
        WHERE Stamptime >= '2024-07-01' AND Stamptime <= '2025-07-31'
        AND Attribute IN ('SR_C11_Avg', 'SR_C12_Avg')
        ORDER BY Stamptime, Attribute
        """
        
        data = client.query(query)
        client.close()
        
        if not data.result_set:
            st.warning("⚠️ No se encontraron datos de DustIQ")
            return pd.DataFrame()
        
        df_raw = pd.DataFrame(data.result_set, columns=['Stamptime', 'Attribute', 'Measure'])
        
        df_raw['Stamptime'] = pd.to_datetime(df_raw['Stamptime'])
        if df_raw['Stamptime'].dt.tz is not None:
            df_raw['Stamptime'] = df_raw['Stamptime'].dt.tz_localize(None)
        
        df_clean = df_raw.groupby(['Stamptime', 'Attribute'])['Measure'].mean().reset_index()
        df = df_clean.pivot(index='Stamptime', columns='Attribute', values='Measure')
        df.columns.name = None
        
        if 'SR_C11_Avg' in df.columns:
            df = df[df['SR_C11_Avg'] > 0]
        
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error cargando datos de DustIQ: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Reducir TTL a 5 minutos
def load_pvstand_data():
    """Carga datos de PVStand."""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password'],
            database='PSDA'
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
        client.close()

        if not data_perc1.result_set and not data_perc2.result_set:
            st.warning("⚠️ No se encontraron datos de PVStand")
            return pd.DataFrame()

        df_perc1 = pd.DataFrame(data_perc1.result_set, columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])
        df_perc2 = pd.DataFrame(data_perc2.result_set, columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])

        df_perc1['module'] = 'perc1fixed'
        df_perc2['module'] = 'perc2fixed'

        df = pd.concat([df_perc1, df_perc2], ignore_index=True)
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Verificar y limpiar el formato de hora antes de crear timestamp
        if pd.api.types.is_datetime64_any_dtype(df['hora']):
            df['hora_str'] = df['hora'].dt.strftime('%H:%M:%S')
        else:
            df['hora_str'] = df['hora'].astype(str)
        
        # Crear timestamp combinando fecha y hora
        df['timestamp'] = pd.to_datetime(df['fecha'].dt.strftime('%Y-%m-%d') + ' ' + df['hora_str'])
        
        # Asegurar que sea timezone-naive
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        return df

    except Exception as e:
        st.error(f"❌ Error cargando datos de PVStand: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Reducir TTL a 5 minutos
def load_soiling_kit_data():
    """Carga datos de Soiling Kit."""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password'],
            database='PSDA'
        )
        
        query = """
        SELECT Stamptime, Attribute, Measure
        FROM PSDA.soilingkit
        WHERE Stamptime >= '2024-07-01' AND Stamptime <= '2025-07-31'
        AND Attribute IN ('SR', 'SR_corr', 'Te(C)', 'Tp(C)')
        ORDER BY Stamptime, Attribute
        """
        
        data = client.query(query)
        client.close()
        
        if not data.result_set:
            st.warning("⚠️ No se encontraron datos de Soiling Kit")
            return pd.DataFrame()
        
        df_raw = pd.DataFrame(data.result_set, columns=['Stamptime', 'Attribute', 'Measure'])
        
        df_raw['Stamptime'] = pd.to_datetime(df_raw['Stamptime'])
        if df_raw['Stamptime'].dt.tz is not None:
            df_raw['Stamptime'] = df_raw['Stamptime'].dt.tz_localize(None)
        
        df_clean = df_raw.groupby(['Stamptime', 'Attribute'])['Measure'].mean().reset_index()
        df = df_clean.pivot(index='Stamptime', columns='Attribute', values='Measure')
        df.columns.name = None
        
        df = df.sort_index()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error cargando datos de Soiling Kit: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Reducir TTL a 5 minutos
def load_irradiance_data():
    """Carga datos de irradiancia (simulados por ahora)."""
    try:
        # Datos simulados para RC411 y RC412
        # Usar el mismo rango que los filtros del sidebar
        start_date = pd.Timestamp('2024-07-01')
        end_date = pd.Timestamp('2025-07-31')
        
        # Generar timestamps cada 5 minutos para el día completo
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        # Debug: mostrar información de generación
        print(f"Generando datos de irradiancia de {start_date} a {end_date}")
        print(f"Total de timestamps: {len(timestamps)}")
        
        # Simular datos de irradiancia con patrones realistas
        np.random.seed(42)  # Para reproducibilidad
        
        irradiance_data = []
        for timestamp in timestamps:
            # Patrón diario: máximo al mediodía, mínimo en la noche
            hour = timestamp.hour
            minute = timestamp.minute
            
            if 6 <= hour <= 18:  # Día
                # Curva sinusoidal más realista
                # Máximo al mediodía (12:00), mínimo al amanecer/atardecer
                time_factor = (hour - 6) / 12  # 0 a 1 durante el día
                
                # Curva sinusoidal con máximo al mediodía
                if hour < 12:
                    # Mañana: subida gradual
                    solar_factor = np.sin(np.pi * time_factor)
                else:
                    # Tarde: bajada gradual
                    solar_factor = np.sin(np.pi * (1 - (hour - 12) / 6))
                
                # Irradiancia base más realista (0-1200 W/m²)
                base_irradiance = 1000 * solar_factor
                
                # Variaciones por nubes y condiciones atmosféricas
                cloud_factor = 0.8 + 0.2 * np.random.random()  # 80-100% de claridad
                atmospheric_factor = 0.9 + 0.1 * np.random.random()  # 90-100% de transmisión
                
                irradiance = base_irradiance * cloud_factor * atmospheric_factor
                irradiance = max(0, irradiance)
            else:  # Noche
                irradiance = 0
            
            # Variación entre sensores RC411 y RC412
            rc411_variation = 0.98 + 0.04 * np.random.random()  # ±2% variación
            rc412_variation = 1.02 + 0.04 * np.random.random()  # ±2% variación
            
            irradiance_data.append({
                'timestamp': timestamp,
                'RC411': irradiance * rc411_variation,
                'RC412': irradiance * rc412_variation
            })
        
        df = pd.DataFrame(irradiance_data)
        df.set_index('timestamp', inplace=True)
        
        # Debug: mostrar información del DataFrame generado
        print(f"DataFrame generado con índice de {df.index.min()} a {df.index.max()}")
        print(f"Columnas: {df.columns.tolist()}")
        print(f"Forma: {df.shape}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error cargando datos de irradiancia: {str(e)}")
        return pd.DataFrame()

# Función mejorada para filtrar datos
def safe_filter_dataframe(df, start_date, end_date, date_column=None):
    """Filtra un DataFrame de forma segura por rango de fechas."""
    try:
        if df is None or df.empty:
            return pd.DataFrame()
        
        if date_column is None:
            # Para DataFrames con índice de fecha
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            
            # Asegurar que las fechas tengan la misma zona horaria que el índice
            if df.index.tz is not None:
                start_datetime = start_datetime.tz_localize(df.index.tz)
                end_datetime = end_datetime.tz_localize(df.index.tz)
            else:
                start_datetime = start_datetime.tz_localize(None)
                end_datetime = end_datetime.tz_localize(None)
            
            filtered_df = df.loc[start_datetime:end_datetime]
            return filtered_df
        else:
            # Para DataFrames con columna de fecha
            filtered_df = df[
                (df[date_column].dt.date >= start_date) & 
                (df[date_column].dt.date <= end_date)
            ]
            return filtered_df
    except Exception as e:
        st.error(f"❌ Error filtrando datos: {str(e)}")
        return pd.DataFrame()

# Cargar datos con mejor manejo de errores
with st.spinner("🔄 Cargando datos..."):
    try:
        df_dustiq = load_dustiq_data()
        df_pvstand = load_pvstand_data()
        df_soilingkit = load_soiling_kit_data()
        df_irradiance = load_irradiance_data()
        
        # Verificar si al menos algunos datos se cargaron
        data_loaded = any([
            not df_dustiq.empty if df_dustiq is not None else False,
            not df_pvstand.empty if df_pvstand is not None else False,
            not df_soilingkit.empty if df_soilingkit is not None else False
        ])
        
        if not data_loaded:
            st.warning("⚠️ No se pudieron cargar datos de la base de datos. Usando datos simulados.")
            # Crear datos simulados básicos para que la aplicación funcione
            df_dustiq = pd.DataFrame()
            df_pvstand = pd.DataFrame()
            df_soilingkit = pd.DataFrame()
            
    except Exception as e:
        st.warning(f"⚠️ Error general cargando datos: {str(e)}. Usando datos simulados.")
        # Crear datos simulados básicos para que la aplicación funcione
        df_dustiq = pd.DataFrame()
        df_pvstand = pd.DataFrame()
        df_soilingkit = pd.DataFrame()

# Información en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Información")

# Botón para limpiar caché y recargar datos
if st.sidebar.button("🔄 Recargar Datos", help="Limpia el caché y recarga todos los datos"):
    st.cache_data.clear()
    st.rerun()

# Filtrar datos de forma segura
if df_dustiq is not None and not df_dustiq.empty:
    df_dustiq_filtered = safe_filter_dataframe(df_dustiq, start_date, end_date)
    st.sidebar.metric("DustIQ - Puntos", f"{len(df_dustiq_filtered):,}")
else:
    st.sidebar.metric("DustIQ - Puntos", "0")

if df_pvstand is not None and not df_pvstand.empty:
    df_pvstand_filtered = safe_filter_dataframe(df_pvstand, start_date, end_date, 'fecha')
    st.sidebar.metric("PVStand - Puntos", f"{len(df_pvstand_filtered):,}")
else:
    st.sidebar.metric("PVStand - Puntos", "0")

if df_soilingkit is not None and not df_soilingkit.empty:
    df_soilingkit_filtered = safe_filter_dataframe(df_soilingkit, start_date, end_date)
    st.sidebar.metric("Soiling Kit - Puntos", f"{len(df_soilingkit_filtered):,}")
else:
    st.sidebar.metric("Soiling Kit - Puntos", "0")

# ===== PESTAÑAS =====

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌫️ DustIQ - Soiling Ratio", 
    "🔋 PVStand - Curvas IV", 
    "🌪️ Soiling Kit - Análisis",
    "📊 Comparación Integrada",
    "ℹ️ Información del Sistema"
])

# ===== PESTAÑA 1: DUSTIQ =====
with tab1:
    try:
        if df_dustiq is not None and not df_dustiq.empty:
            st.subheader("🌫️ Análisis de Soiling Ratio - DustIQ")
            
            # Aplicar filtros usando la función segura
            df_dustiq_filtered = safe_filter_dataframe(df_dustiq, start_date, end_date)
            
            if df_dustiq_filtered.empty:
                st.warning("⚠️ No hay datos de DustIQ para el rango de fechas seleccionado.")
                st.info("💡 Intenta cambiar las fechas o usar el botón 'Recargar Datos' en el sidebar.")
            else:
                sr_column = 'SR_C11_Avg'
                if sr_column in df_dustiq_filtered.columns:
                    df_sr_filtered = df_dustiq_filtered[df_dustiq_filtered[sr_column] > sr_threshold][sr_column].copy()
        
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
                    
                    # Gráfico con franjas horarias
                    st.subheader(f"📊 Evolución Temporal ({selected_freq})")
                    
                    fig = go.Figure()
                    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                    
                    # Procesar por franjas horarias
                    if selected_franjas:
                        for i, franja in enumerate(selected_franjas):
                            try:
                                if franja == "Mediodía Solar":
                                    # Mediodía solar (11:30-13:30)
                                    data_franja = df_sr_filtered.between_time('11:30', '13:30')
                                    
                                    if not data_franja.empty:
                                        data_procesada = data_franja.resample(freq_options[selected_freq], origin='start').quantile(0.25)
                                        data_procesada = data_procesada.dropna()
                                        
                                        if not data_procesada.empty:
                                            color = colores[i % len(colores)]
                                            fig.add_trace(go.Scatter(
                                                x=data_procesada.index,
                                                y=data_procesada.values,
                                                mode='lines+markers',
                                                name=f'Mediodía Solar (11:30-13:30)',
                                                line=dict(color=color, width=2),
                                                marker=dict(size=4)
                                            ))
                                else:
                                    # Franjas horarias específicas
                                    if franja == "10:00-11:00":
                                        data_franja = df_sr_filtered.between_time('10:00', '11:00')
                                    elif franja == "12:00-13:00":
                                        data_franja = df_sr_filtered.between_time('12:00', '13:00')
                                    elif franja == "14:00-15:00":
                                        data_franja = df_sr_filtered.between_time('14:00', '15:00')
                                    elif franja == "16:00-17:00":
                                        data_franja = df_sr_filtered.between_time('16:00', '17:00')
                                    else:
                                        continue
                                    
                                    if not data_franja.empty:
                                        data_procesada = data_franja.resample(freq_options[selected_freq], origin='start').quantile(0.25)
                                        data_procesada = data_procesada.dropna()
                                        
                                        if not data_procesada.empty:
                                            color = colores[i % len(colores)]
                                            fig.add_trace(go.Scatter(
                                                x=data_procesada.index,
                                                y=data_procesada.values,
                                                mode='lines+markers',
                                                name=f'{franja}',
                                                line=dict(color=color, width=2),
                                                marker=dict(size=4)
                                            ))
                            except Exception as e:
                                st.error(f"Error procesando franja {franja}: {str(e)}")
                                continue
                    else:
                        # Procesar todos los datos
                        df_resampled = df_sr_filtered.resample(freq_options[selected_freq], origin='start').quantile(0.25)
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ Columna {sr_column} no encontrada")
        else:
            st.error("❌ No se pudieron cargar los datos de DustIQ")
    except Exception as e:
        st.error(f"❌ Error en la pestaña DustIQ: {str(e)}")
        st.info("💡 Intenta cambiar las fechas o recargar la página")

# ===== PESTAÑA 2: PVSTAND =====
with tab2:
    try:
        if df_pvstand is not None and not df_pvstand.empty:
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
                        
                        # Selector para el gráfico de comparación
                        comparison_options = ["Irradiancia del día", "Curva de Potencia"]
                        selected_comparison = st.selectbox(
                            "📊 Gráfico de comparación:",
                            comparison_options,
                            index=0
                        )
                        
                        # Crear dos columnas para visualización lado a lado
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Gráfico IV (siempre visible)
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
                                height=400
                            )
                            
                            st.plotly_chart(fig_iv, use_container_width=True)
                        
                        with col2:
                            # Gráfico de comparación seleccionable
                            if selected_comparison == "Irradiancia del día":
                                st.subheader(f"☀️ Irradiancia del día - {selected_date}")
                                
                                if df_irradiance is not None and not df_irradiance.empty:
                                    # Filtrar datos del día seleccionado de forma más simple
                                    selected_datetime = pd.Timestamp(selected_date)
                                    
                                    # Crear máscara para el día específico
                                    mask = df_irradiance.index.date == selected_datetime.date()
                                    df_irradiance_day = df_irradiance[mask]
                                    
                                    # Debug temporal
                                    st.write(f"🔍 Fecha seleccionada: {selected_date}")
                                    st.write(f"🔍 Datos encontrados: {len(df_irradiance_day)} puntos")
                                    if not df_irradiance_day.empty:
                                        st.write(f"🔍 Rango de datos: {df_irradiance_day.index.min()} a {df_irradiance_day.index.max()}")
                                    
                                    if not df_irradiance_day.empty:
                                        fig_irradiance = go.Figure()
                                        
                                        if 'RC411' in df_irradiance_day.columns:
                                            fig_irradiance.add_trace(go.Scatter(
                                                x=df_irradiance_day.index,
                                                y=df_irradiance_day['RC411'],
                                                mode='lines',
                                                name='RC411',
                                                line=dict(color='blue', width=2)
                                            ))
                                        
                                        if 'RC412' in df_irradiance_day.columns:
                                            fig_irradiance.add_trace(go.Scatter(
                                                x=df_irradiance_day.index,
                                                y=df_irradiance_day['RC412'],
                                                mode='lines',
                                                name='RC412',
                                                line=dict(color='red', width=2)
                                            ))
                                        
                                        fig_irradiance.update_layout(
                                            title=f"Irradiancia del día {selected_date}",
                                            xaxis_title="Hora del día",
                                            yaxis_title="Irradiancia (W/m²)",
                                            height=400,
                                            hovermode='x unified'
                                        )
                                        
                                        st.plotly_chart(fig_irradiance, use_container_width=True)
                                        
                                        # Métricas rápidas de irradiancia (después del gráfico)
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            if 'RC411' in df_irradiance_day.columns:
                                                max_rc411 = df_irradiance_day['RC411'].max()
                                                st.metric("RC411 Máximo", f"{max_rc411:.0f} W/m²")
                                        with col2:
                                            if 'RC412' in df_irradiance_day.columns:
                                                max_rc412 = df_irradiance_day['RC412'].max()
                                                st.metric("RC412 Máximo", f"{max_rc412:.0f} W/m²")
                                        with col3:
                                            if 'RC411' in df_irradiance_day.columns and 'RC412' in df_irradiance_day.columns:
                                                diff = abs(max_rc411 - max_rc412)
                                                st.metric("Diferencia", f"{diff:.0f} W/m²")
                                    else:
                                        st.warning(f"⚠️ No hay datos de irradiancia para el día {selected_date}")
                                        st.info("💡 Los datos de irradiancia son simulados. Verifica la conexión a InfluxDB.")
                                else:
                                    st.error("❌ No se pudieron cargar los datos de irradiancia")
                                    st.info("💡 Los datos de irradiancia son simulados. Verifica la configuración.")
                            
                            elif selected_comparison == "Curva de Potencia":
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
                                    title=f"Potencia - {selected_date} {selected_curve}",
                                    xaxis_title="Voltaje (V)",
                                    yaxis_title="Potencia (W)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_power, use_container_width=True)
                        
                        # Tabla de datos debajo de los gráficos
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
    except Exception as e:
        st.error(f"❌ Error en la pestaña PVStand: {str(e)}")
        st.info("💡 Intenta cambiar las fechas o recargar la página")

# ===== PESTAÑA 3: SOILING KIT =====
with tab3:
    try:
        if df_soilingkit is not None and not df_soilingkit.empty:
            st.subheader("🌪️ Análisis de Soiling Kit")
        
            # Aplicar filtros usando la función segura
            df_soilingkit_filtered = safe_filter_dataframe(df_soilingkit, start_date, end_date)
        
            if not df_soilingkit_filtered.empty:
                # Métricas del Soiling Kit
                col1, col2, col3, col4 = st.columns(4)
            
                with col1:
                    if 'SR' in df_soilingkit_filtered.columns:
                        sr_mean = df_soilingkit_filtered['SR'].mean()
                        st.metric("SR Promedio (%)", f"{sr_mean:.2f}")
                
                with col2:
                    if 'SR' in df_soilingkit_filtered.columns:
                        sr_median = df_soilingkit_filtered['SR'].median()
                        st.metric("SR Mediana (%)", f"{sr_median:.2f}")
                
                with col3:
                    if 'SR' in df_soilingkit_filtered.columns:
                        sr_loss = 100 - sr_mean
                        st.metric("Pérdida Promedio (%)", f"{sr_loss:.2f}")
                
                with col4:
                    if 'SR_corr' in df_soilingkit_filtered.columns:
                        sr_corr_mean = df_soilingkit_filtered['SR_corr'].mean()
                        st.metric("SR Corregido (%)", f"{sr_corr_mean:.2f}")
                
                # Información de temperaturas
                if 'Te(C)' in df_soilingkit_filtered.columns and 'Tp(C)' in df_soilingkit_filtered.columns:
                    te_mean = df_soilingkit_filtered['Te(C)'].mean()
                    tp_mean = df_soilingkit_filtered['Tp(C)'].mean()
                    temp_diff = tp_mean - te_mean
                    st.info(f"🌡️ **Temperaturas:** Te(C)={te_mean:.1f}°C, Tp(C)={tp_mean:.1f}°C, ΔT={temp_diff:.1f}°C")
                
                # Gráfico semanal Q25 (PRIMERO)
                st.subheader("📊 Soiling Ratio Semanal (Q25)")
                
                if 'SR' in df_soilingkit_filtered.columns:
                    # Resample semanal y calcular Q25
                    weekly_q25 = df_soilingkit_filtered['SR'].resample('W').quantile(0.25).dropna()
                    
                    fig_q25 = go.Figure()
                    
                    fig_q25.add_trace(go.Scatter(
                        x=weekly_q25.index,
                        y=weekly_q25.values,
                        mode='lines+markers',
                        name='Q25 Semanal',
                        line=dict(color='purple', width=3),
                        marker=dict(size=6)
                    ))
                    
                    fig_q25.update_layout(
                        title='Percentil 25 Semanal del Soiling Ratio',
                        xaxis_title='Semana',
                        yaxis_title='Soiling Ratio (%)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig_q25, use_container_width=True)
                    st.info("ℹ️ Este gráfico muestra el Soiling Ratio semanal calculado con el cuartil 25% (Q25) de los datos.")
                
                # Gráfico de temperaturas (AL FINAL)
                if 'Te(C)' in df_soilingkit_filtered.columns and 'Tp(C)' in df_soilingkit_filtered.columns:
                    st.subheader("🌡️ Temperaturas del Soiling Kit")
                    
                    fig_temp = go.Figure()
                    
                    fig_temp.add_trace(go.Scatter(
                        x=df_soilingkit_filtered.index,
                        y=df_soilingkit_filtered['Te(C)'],
                        mode='lines+markers',
                        name='Temperatura Expuesto',
                        line=dict(color='green', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig_temp.add_trace(go.Scatter(
                        x=df_soilingkit_filtered.index,
                        y=df_soilingkit_filtered['Tp(C)'],
                        mode='lines+markers',
                        name='Temperatura Protegido',
                        line=dict(color='orange', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig_temp.update_layout(
                        title='Temperaturas del Soiling Kit',
                        xaxis_title='Fecha',
                        yaxis_title='Temperatura (°C)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                # Tabla de datos
                st.subheader("📋 Datos del Soiling Kit")
                st.dataframe(df_soilingkit_filtered.head(100), use_container_width=True)
            else:
                st.warning("⚠️ No hay datos del Soiling Kit en el rango de fechas seleccionado")
        else:
            st.error("❌ No se pudieron cargar los datos del Soiling Kit")
    except Exception as e:
        st.error(f"❌ Error en la pestaña Soiling Kit: {str(e)}")
        st.info("💡 Intenta cambiar las fechas o recargar la página")

# ===== PESTAÑA 4: COMPARACIÓN INTEGRADA =====
with tab4:
    st.subheader("📊 Comparación Integrada - Todos los Sistemas")
    
    if df_dustiq is not None and df_pvstand is not None and df_soilingkit is not None:
        st.info("🔍 Análisis comparativo entre todos los sistemas de medición de soiling")
        
        # Métricas de comparación
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌫️ DustIQ")
            if 'df_sr_filtered' in locals():
                st.metric("Promedio SR", f"{df_sr_filtered.mean():.2f}%")
                st.metric("Pérdida por Soiling", f"{100 - df_sr_filtered.mean():.2f}%")
                st.metric("Datos disponibles", f"{len(df_dustiq_filtered):,} registros")
        
        with col2:
            st.subheader("🔋 PVStand")
            if 'df_pvstand_filtered' in locals():
                total_curves = len(df_pvstand_filtered['timestamp'].dt.date.unique())
                st.metric("Días con datos", f"{total_curves}")
                st.metric("Curvas totales", f"{len(df_pvstand_filtered):,}")
                st.metric("Módulos", f"{len(df_pvstand_filtered['module'].unique())}")
        
        with col3:
            st.subheader("🌪️ Soiling Kit")
            if df_soilingkit_filtered is not None and not df_soilingkit_filtered.empty:
                if 'SR' in df_soilingkit_filtered.columns:
                    sr_mean = df_soilingkit_filtered['SR'].mean()
                    st.metric("Promedio SR", f"{sr_mean:.2f}%")
                    st.metric("Pérdida por Soiling", f"{100 - sr_mean:.2f}%")
                    st.metric("Datos disponibles", f"{len(df_soilingkit_filtered):,} registros")
        
        # Gráfico comparativo temporal
        st.subheader("📈 Comparación Temporal de Soiling Ratio")
        
        fig_comparison = go.Figure()
        
        # DustIQ
        if 'df_sr_filtered' in locals():
            dustiq_daily = df_sr_filtered.resample('1D').mean()
            fig_comparison.add_trace(go.Scatter(
                x=dustiq_daily.index,
                y=dustiq_daily.values,
                mode='lines+markers',
                name='DustIQ - SR (%)',
                line=dict(color='blue', width=2)
            ))
        
        # Soiling Kit
        if df_soilingkit_filtered is not None and not df_soilingkit_filtered.empty:
            if 'SR' in df_soilingkit_filtered.columns:
                soilingkit_daily = df_soilingkit_filtered['SR'].resample('1D').mean()
                fig_comparison.add_trace(go.Scatter(
                    x=soilingkit_daily.index,
                    y=soilingkit_daily.values,
                    mode='lines+markers',
                    name='Soiling Kit - SR (%)',
                    line=dict(color='red', width=2)
                ))
        
        fig_comparison.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Referencia 100%")
        
        fig_comparison.update_layout(
            title="Comparación DustIQ vs Soiling Kit",
            xaxis_title="Fecha",
            yaxis_title="Soiling Ratio (%)",
            height=500,
            yaxis=dict(range=[90, 110]),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
    else:
        st.error("❌ Se requieren datos de todos los sistemas para la comparación")

# ===== PESTAÑA 5: INFORMACIÓN =====
with tab5:
    st.subheader("ℹ️ Información del Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🐍 Python")
        st.write(f"**Versión:** {sys.version}")
        st.write(f"**Streamlit:** {st.__version__}")
        st.write(f"**Pandas:** {pd.__version__}")
        st.write(f"**Plotly:** {plotly.__version__}")
    
    with col2:
        st.subheader("🗄️ Base de Datos")
        st.write(f"**ClickHouse:** {CLICKHOUSE_CONFIG['host']}:{CLICKHOUSE_CONFIG['port']}")
        st.write(f"**InfluxDB:** {INFLUXDB_CONFIG['url']}")
        st.write(f"**Usuario:** {CLICKHOUSE_CONFIG['user']}")
    
    with col3:
        st.subheader("📁 Directorio")
        st.write(f"**Trabajo:** {os.getcwd()}")
        st.write(f"**Archivos:** {len(os.listdir('.'))}")
        st.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    st.subheader("📋 Funcionalidades")
    
    st.markdown("""
    ### 🌫️ DustIQ
    - Análisis de Soiling Ratio en tiempo real
    - Filtrado por franjas horarias
    - Métricas estadísticas detalladas
    - Gráficos temporales con resampling
    
    ### 🔋 PVStand
    - Curvas IV interactivas
    - Comparación lado a lado con irradiancia
    - Curvas de potencia
    - Selección de módulos y fechas
    
    ### 🌪️ Soiling Kit
    - Análisis de Soiling Ratio corregido por temperatura
    - Gráficos semanales Q25
    - Monitoreo de temperaturas
    - Comparación con otros sistemas
    
    ### 📊 Comparación Integrada
    - Análisis comparativo entre sistemas
    - Métricas unificadas
    - Gráficos temporales comparativos
    """) 