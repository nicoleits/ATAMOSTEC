#!/usr/bin/env python3
"""
Dashboard Integrado v4 - DustIQ + PVStand + Soiling Kit
Versión con análisis completo de soiling
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import clickhouse_connect
import sys
import os

# Agregar el directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Las funciones de análisis del Soiling Kit están implementadas directamente en este archivo

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
        
        # Verificar y limpiar el formato de hora antes de crear timestamp
        # Si hora es datetime, extraer solo la parte de tiempo
        if pd.api.types.is_datetime64_any_dtype(df_pvstand['hora']):
            df_pvstand['hora_str'] = df_pvstand['hora'].dt.strftime('%H:%M:%S')
        else:
            df_pvstand['hora_str'] = df_pvstand['hora'].astype(str)
        
        # Crear timestamp combinando fecha y hora
        df_pvstand['timestamp'] = pd.to_datetime(df_pvstand['fecha'].dt.strftime('%Y-%m-%d') + ' ' + df_pvstand['hora_str'])
        
        # Asegurar que sea timezone-naive
        if df_pvstand['timestamp'].dt.tz is not None:
            df_pvstand['timestamp'] = df_pvstand['timestamp'].dt.tz_localize(None)

        df_pvstand = df_pvstand.sort_values('timestamp')
        df_pvstand = df_pvstand[['timestamp', 'module', 'corriente', 'voltaje', 'potencia', 'fecha', 'hora']]

        client.close()
        return df_pvstand

    except Exception as e:
        st.warning(f"⚠️ Error PVStand: {str(e)[:100]}...")
        return None

@st.cache_data(ttl=3600)
def load_soiling_kit_data():
    """Carga datos del Soiling Kit."""
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
        FROM PSDA.soilingkit 
        WHERE Stamptime >= '2024-07-01' AND Stamptime <= '2025-07-31'
        AND Attribute IN ('Isc(e)', 'Isc(p)', 'Te(C)', 'Tp(C)')
        ORDER BY Stamptime, Attribute
        """
        
        data = client.query(query)
        df_raw = pd.DataFrame(data.result_set, columns=['Stamptime', 'Attribute', 'Measure'])
        
        df_raw['Stamptime'] = pd.to_datetime(df_raw['Stamptime'])
        if df_raw['Stamptime'].dt.tz is not None:
            df_raw['Stamptime'] = df_raw['Stamptime'].dt.tz_localize(None)
        
        df_clean = df_raw.groupby(['Stamptime', 'Attribute'])['Measure'].mean().reset_index()
        df_soilingkit = df_clean.pivot(index='Stamptime', columns='Attribute', values='Measure')
        df_soilingkit.columns.name = None
        
        # Calcular Soiling Ratio
        if 'Isc(p)' in df_soilingkit.columns and 'Isc(e)' in df_soilingkit.columns:
            df_soilingkit['SR'] = (df_soilingkit['Isc(p)'] / df_soilingkit['Isc(e)']) * 100
        
        # Calcular corrección de temperatura si hay datos de temperatura
        if 'Tp(C)' in df_soilingkit.columns and 'Te(C)' in df_soilingkit.columns:
            df_soilingkit['Isc(p)_corr'] = df_soilingkit['Isc(p)'] * (1 + 0.0004 * (25 - df_soilingkit['Tp(C)']))
            df_soilingkit['Isc(e)_corr'] = df_soilingkit['Isc(e)'] * (1 + 0.0004 * (25 - df_soilingkit['Te(C)']))
            df_soilingkit['SR_corr'] = (df_soilingkit['Isc(p)_corr'] / df_soilingkit['Isc(e)_corr']) * 100
        
        df_soilingkit = df_soilingkit.sort_index()
        client.close()
        
        return df_soilingkit
        
    except Exception as e:
        st.warning(f"⚠️ Error Soiling Kit: {str(e)[:100]}...")
        return None

@st.cache_data(ttl=3600)
def load_irradiance_data():
    """Carga datos de irradiancia de las fotoceldas desde InfluxDB."""
    try:
        # Por ahora retornamos datos simulados hasta tener el token de InfluxDB
        # TODO: Implementar conexión real a InfluxDB cuando tengas el token
        
        # Crear datos simulados de irradiancia para RC411 y RC412
        dates = pd.date_range(start='2024-07-01', end='2025-07-31', freq='1min')
        
        # Crear patrones diurnos más realistas
        hours = dates.hour
        day_pattern = np.where((hours >= 6) & (hours <= 18), 1, 0)
        
        # Simular irradiancia con patrón sinusoidal durante el día
        solar_noon = 12
        time_factor = np.sin(np.pi * (hours - 6) / 12) * day_pattern
        time_factor = np.where(time_factor < 0, 0, time_factor)
        
        df_irradiance = pd.DataFrame({
            '1RC411(w.m-2)': (800 + 200 * time_factor + np.random.normal(0, 50, len(dates))) * day_pattern,
            '1RC412(w.m-2)': (820 + 200 * time_factor + np.random.normal(0, 50, len(dates))) * day_pattern
        }, index=dates)
        
        # Asegurar valores no negativos
        df_irradiance = df_irradiance.clip(lower=0)
        
        return df_irradiance
        
    except Exception as e:
        st.warning(f"⚠️ Error Irradiancia: {str(e)[:100]}...")
        return None

# ===== CARGA DE DATOS =====

with st.spinner("🔄 Cargando datos..."):
    df_dustiq = load_dustiq_data()
    df_pvstand = load_pvstand_data()
    df_soilingkit = load_soiling_kit_data()
    df_irradiance = load_irradiance_data()

if df_dustiq is None and df_pvstand is None and df_soilingkit is None:
    st.error("❌ No se pudieron cargar los datos. Verifica la conexión.")
    st.stop()

# Información en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Información")

# Función auxiliar para filtrar datos de forma segura
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
            
            return df.loc[start_datetime:end_datetime]
        else:
            # Para DataFrames con columna de fecha
            return df[
                (df[date_column].dt.date >= start_date) & 
                (df[date_column].dt.date <= end_date)
            ]
    except Exception as e:
        st.sidebar.error(f"Error filtrando datos: {str(e)}")
        return pd.DataFrame()

# Filtrar datos de forma segura
if df_dustiq is not None:
    df_dustiq_filtered = safe_filter_dataframe(df_dustiq, start_date, end_date)
    st.sidebar.metric("DustIQ - Puntos", f"{len(df_dustiq_filtered):,}")

if df_pvstand is not None:
    df_pvstand_filtered = safe_filter_dataframe(df_pvstand, start_date, end_date, 'fecha')
    st.sidebar.metric("PVStand - Puntos", f"{len(df_pvstand_filtered):,}")

if df_soilingkit is not None:
    df_soilingkit_filtered = safe_filter_dataframe(df_soilingkit, start_date, end_date)
    st.sidebar.metric("Soiling Kit - Puntos", f"{len(df_soilingkit_filtered):,}")

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
        if df_dustiq is not None:
            st.subheader("🌫️ Análisis de Soiling Ratio - DustIQ")
            
            # Aplicar filtros usando la función segura
            df_dustiq_filtered = safe_filter_dataframe(df_dustiq, start_date, end_date)
            
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
                            st.info(f"Procesando franja: {franja}")
                            data_franja = df_sr_filtered.between_time('11:30', '13:30')
                            st.info(f"Datos encontrados en mediodía solar: {len(data_franja)} puntos")
                            
                            if not data_franja.empty:
                                data_procesada = data_franja.resample(freq_options[selected_freq], origin='start').quantile(0.25)
                                data_procesada = data_procesada.dropna()  # Eliminar valores NaN
                                
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
                                    st.warning(f"No hay datos procesados para {franja}")
                            else:
                                st.warning(f"No hay datos en el rango de {franja}")
                        else:
                            # Franjas horarias específicas
                            if franja == "10:00-11:00":
                                data_franja = df_sr_filtered.between_time('10:00', '11:00')
                            elif franja == "12:00-13:00":
                                data_franja = df_sr_filtered.between_time('12:00', '13:00')
                            elif franja == "14:00-15:00":
                                data_franja = df_sr_filtered.between_time('14:00', '15:00")
                            elif franja == "16:00-17:00":
                                data_franja = df_sr_filtered.between_time('16:00', '17:00")
                            else:
                                continue
                            
                            if not data_franja.empty:
                                data_procesada = data_franja.resample(freq_options[selected_freq], origin='start').quantile(0.25)
                                data_procesada = data_procesada.dropna()  # Eliminar valores NaN
                                
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
                                else:
                                    st.warning(f"No hay datos procesados para {franja}")
                            else:
                                st.warning(f"No hay datos en el rango de {franja}")
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
            st.error("❌ No se pudieron cargar los datos de DustIQ")
    except Exception as e:
        st.error(f"❌ Error en la pestaña DustIQ: {str(e)}")
        st.info("💡 Intenta cambiar las fechas o recargar la página")

# ===== PESTAÑA 2: PVSTAND =====
with tab2:
    try:
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
                            st.subheader("☀️ Irradiancia del día")
                            
                            if df_irradiance is not None:
                                # Filtrar datos del día seleccionado
                                start_time = pd.Timestamp(selected_date)
                                end_time = pd.Timestamp(selected_date) + pd.Timedelta(days=1)
                                
                                # Asegurar que los índices tengan la misma zona horaria
                                if df_irradiance.index.tz is not None:
                                    start_time = start_time.tz_localize(df_irradiance.index.tz)
                                    end_time = end_time.tz_localize(df_irradiance.index.tz)
                                else:
                                    start_time = start_time.tz_localize(None)
                                    end_time = end_time.tz_localize(None)
                                
                                df_irradiance_day = df_irradiance.loc[start_time:end_time]
                                
                                if not df_irradiance_day.empty:
                                    fig_irradiance = go.Figure()
                                    
                                    if '1RC411(w.m-2)' in df_irradiance_day.columns:
                                        fig_irradiance.add_trace(go.Scatter(
                                            x=df_irradiance_day.index,
                                            y=df_irradiance_day['1RC411(w.m-2)'],
                                            mode='lines',
                                            name='RC411',
                                            line=dict(color='blue', width=2)
                                        ))
                                    
                                    if '1RC412(w.m-2)' in df_irradiance_day.columns:
                                        fig_irradiance.add_trace(go.Scatter(
                                            x=df_irradiance_day.index,
                                            y=df_irradiance_day['1RC412(w.m-2)'],
                                            mode='lines',
                                            name='RC412',
                                            line=dict(color='red', width=2)
                                        ))
                                    
                                    fig_irradiance.update_layout(
                                        title=f"Irradiancia - {selected_date}",
                                        xaxis_title="Hora del día",
                                        yaxis_title="Irradiancia (W/m²)",
                                        height=400,
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig_irradiance, use_container_width=True)
                                    
                                    # Métricas rápidas
                                    if '1RC411(w.m-2)' in df_irradiance_day.columns:
                                        # Buscar el valor más cercano a la hora de la curva
                                        curve_time = pd.Timestamp(selected_curve)
                                        if df_irradiance_day.index.tz is not None:
                                            curve_time = curve_time.tz_localize(df_irradiance_day.index.tz)
                                        else:
                                            curve_time = curve_time.tz_localize(None)
                                        
                                        # Encontrar el índice más cercano
                                        closest_idx = df_irradiance_day.index.get_indexer([curve_time], method='nearest')[0]
                                        if closest_idx >= 0:
                                            rc411_at_curve_time = df_irradiance_day.iloc[closest_idx]['1RC411(w.m-2)']
                                            st.metric("RC411 en hora de curva", f"{rc411_at_curve_time:.1f} W/m²")
                                else:
                                    st.warning("⚠️ No hay datos de irradiancia")
                            else:
                                st.error("❌ Error cargando irradiancia")
                        
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
                            
                            # Métricas rápidas de potencia
                            for module in selected_modules:
                                df_module = df_curve[df_curve['module'] == module]
                                if not df_module.empty:
                                    pmp = df_module['potencia'].max()
                                    st.metric(f"{module} - Pmax", f"{pmp:.2f} W")
                    
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
        if df_soilingkit is not None:
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
            
            # Estadísticas detalladas
            st.subheader("📈 Estadísticas Detalladas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Soiling Ratio sin corrección:**")
                if 'SR' in df_soilingkit_filtered.columns:
                    sr_data = df_soilingkit_filtered['SR'].dropna()
                    st.write(f"- Promedio: {sr_data.mean():.2f}%")
                    st.write(f"- Mediana: {sr_data.median():.2f}%")
                    st.write(f"- Desv. Estándar: {sr_data.std():.2f}%")
                    st.write(f"- Mínimo: {sr_data.min():.2f}%")
                    st.write(f"- Máximo: {sr_data.max():.2f}%")
                    st.write(f"- Pérdida promedio: {100 - sr_data.mean():.2f}%")
            
            with col2:
                st.write("**Soiling Ratio con corrección:**")
                if 'SR_corr' in df_soilingkit_filtered.columns:
                    sr_corr_data = df_soilingkit_filtered['SR_corr'].dropna()
                    st.write(f"- Promedio: {sr_corr_data.mean():.2f}%")
                    st.write(f"- Mediana: {sr_corr_data.median():.2f}%")
                    st.write(f"- Desv. Estándar: {sr_corr_data.std():.2f}%")
                    st.write(f"- Mínimo: {sr_corr_data.min():.2f}%")
                    st.write(f"- Máximo: {sr_corr_data.max():.2f}%")
                    st.write(f"- Pérdida promedio: {100 - sr_corr_data.mean():.2f}%")
            
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
    
    with col3:
        st.subheader("🌪️ Soiling Kit")
        st.write("**Propósito:** Medición directa de soiling")
        st.write("**Datos:** Isc(e), Isc(p), Te(C), Tp(C)")
        st.write("**Base de datos:** PSDA.soilingkit")
        
        if df_soilingkit is not None:
            st.write(f"**Registros:** {len(df_soilingkit):,}")
            st.write(f"**Rango:** {df_soilingkit.index.min().date()} a {df_soilingkit.index.max().date()}")
    
    st.markdown("---")
    
    st.subheader("🔧 Configuración")
    st.write(f"**Servidor:** {CLICKHOUSE_CONFIG['host']}:{CLICKHOUSE_CONFIG['port']}")
    st.write(f"**Usuario:** {CLICKHOUSE_CONFIG['user']}")
    st.write(f"**Actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Estado de conexión
    st.subheader("📡 Estado de Conexión")
    
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        if df_soilingkit is not None:
            st.success("✅ Soiling Kit - Conectado")
        else:
            st.error("❌ Soiling Kit - Sin conexión") 