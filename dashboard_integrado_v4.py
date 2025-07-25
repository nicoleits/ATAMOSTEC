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

# Importar funciones de análisis del Soiling Kit
try:
    from soiling_kit_analysis import (
        load_soiling_kit_data_from_clickhouse,
        calculate_soiling_ratio,
        create_soiling_ratio_plots,
        create_weekly_q25_plot,
        calculate_soiling_statistics
    )
except ImportError as e:
    st.error(f"Error al importar módulo de análisis: {e}")
    st.stop()

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
        
        # Usar la función del módulo de análisis
        df_soilingkit = load_soiling_kit_data_from_clickhouse(
            client, 
            pd.to_datetime(start_date), 
            pd.to_datetime(end_date)
        )
        
        client.close()
        return df_soilingkit
        
    except Exception as e:
        st.warning(f"⚠️ Error Soiling Kit: {str(e)[:100]}...")
        return None

# ===== CARGA DE DATOS =====

with st.spinner("🔄 Cargando datos..."):
    df_dustiq = load_dustiq_data()
    df_pvstand = load_pvstand_data()
    df_soilingkit = load_soiling_kit_data()

if df_dustiq is None and df_pvstand is None and df_soilingkit is None:
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

if df_soilingkit is not None:
    df_soilingkit_filtered = df_soilingkit.loc[start_date:end_date]
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
        
        # Gráfico con franjas horarias
        st.subheader(f"📊 Evolución Temporal ({selected_freq})")
        
        fig = go.Figure()
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Procesar por franjas horarias
        if selected_franjas:
            for i, franja in enumerate(selected_franjas):
                if franja == "Mediodía Solar":
                    # Mediodía solar (11:30-13:30)
                    data_franja = df_sr_filtered.between_time('11:30', '13:30')
                    if not data_franja.empty:
                        data_procesada = data_franja.resample(freq_options[selected_freq], origin='start').quantile(0.25)
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

# ===== PESTAÑA 3: SOILING KIT =====
with tab3:
    if df_soilingkit is not None:
        st.subheader("🌪️ Análisis de Soiling Kit")
        
        # Aplicar filtros
        df_soilingkit_filtered = df_soilingkit.loc[start_date:end_date]
        
        # Calcular Soiling Ratio
        df_sr_soilingkit = calculate_soiling_ratio(df_soilingkit_filtered)
        
        if df_sr_soilingkit is not None:
            # Métricas del Soiling Kit
            stats = calculate_soiling_statistics(df_sr_soilingkit)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if stats and 'SR' in stats:
                    st.metric("SR Promedio (%)", f"{stats['SR']['promedio']:.2f}")
            
            with col2:
                if stats and 'SR' in stats:
                    st.metric("SR Mediana (%)", f"{stats['SR']['mediana']:.2f}")
            
            with col3:
                if stats and 'SR' in stats:
                    st.metric("Pérdida Promedio (%)", f"{stats['SR']['perdida_promedio']:.2f}")
            
            with col4:
                if stats and 'SR_corr' in stats:
                    st.metric("SR Corregido (%)", f"{stats['SR_corr']['promedio']:.2f}")
            
            # Información adicional
            if stats and 'temperatura' in stats:
                st.info(f"🌡️ **Temperaturas:** Te(C)={stats['temperatura']['Te_promedio']:.1f}°C, Tp(C)={stats['temperatura']['Tp_promedio']:.1f}°C, ΔT={stats['temperatura']['diferencia_temp']:.1f}°C")
            
            # Gráfico específico semanal Q25 (como sk_sr_q25_semanal.png) - PRIMERO
            st.subheader("📈 Gráfico Semanal Q25 - Soiling Kit")
            fig_q25 = create_weekly_q25_plot(df_sr_soilingkit)
            
            if fig_q25 is not None:
                st.plotly_chart(fig_q25, use_container_width=True)
                st.info("ℹ️ Este gráfico muestra el Soiling Ratio semanal calculado con el cuartil 25% (Q25) de los datos, tanto sin corrección como con corrección de temperatura.")
            else:
                st.warning("⚠️ No se pudo generar el gráfico semanal Q25")
            
            # Gráficos adicionales del Soiling Kit
            st.subheader(f"📊 Gráficos Adicionales del Soiling Kit ({selected_freq})")
            
            # Crear gráficos usando las funciones del módulo
            fig_sr, fig_temp = create_soiling_ratio_plots(
                df_sr_soilingkit, 
                freq_options[selected_freq], 
                selected_franjas
            )
            
            if fig_sr is not None:
                st.plotly_chart(fig_sr, use_container_width=True)
            else:
                st.warning("⚠️ No se pudieron generar los gráficos de Soiling Ratio")
            
            if fig_temp is not None:
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info("ℹ️ No hay datos de temperatura disponibles")
            
            # Estadísticas detalladas
            st.subheader("📈 Estadísticas Detalladas")
            
            if stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Soiling Ratio sin corrección:**")
                    if 'SR' in stats:
                        st.write(f"- Promedio: {stats['SR']['promedio']:.2f}%")
                        st.write(f"- Mediana: {stats['SR']['mediana']:.2f}%")
                        st.write(f"- Desv. Estándar: {stats['SR']['desv_std']:.2f}%")
                        st.write(f"- Mínimo: {stats['SR']['minimo']:.2f}%")
                        st.write(f"- Máximo: {stats['SR']['maximo']:.2f}%")
                        st.write(f"- Pérdida promedio: {stats['SR']['perdida_promedio']:.2f}%")
                
                with col2:
                    st.write("**Soiling Ratio con corrección:**")
                    if 'SR_corr' in stats:
                        st.write(f"- Promedio: {stats['SR_corr']['promedio']:.2f}%")
                        st.write(f"- Mediana: {stats['SR_corr']['mediana']:.2f}%")
                        st.write(f"- Desv. Estándar: {stats['SR_corr']['desv_std']:.2f}%")
                        st.write(f"- Mínimo: {stats['SR_corr']['minimo']:.2f}%")
                        st.write(f"- Máximo: {stats['SR_corr']['maximo']:.2f}%")
                        st.write(f"- Pérdida promedio: {stats['SR_corr']['perdida_promedio']:.2f}%")
            
            # Tabla de datos
            st.subheader("📋 Datos del Soiling Kit")
            st.dataframe(df_sr_soilingkit.head(100), use_container_width=True)
            
        else:
            st.error("❌ No se pudieron calcular los datos del Soiling Ratio")
    else:
        st.error("❌ No se pudieron cargar los datos del Soiling Kit")

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
            if 'df_sr_soilingkit' in locals() and df_sr_soilingkit is not None:
                stats_sk = calculate_soiling_statistics(df_sr_soilingkit)
                if stats_sk and 'SR' in stats_sk:
                    st.metric("Promedio SR", f"{stats_sk['SR']['promedio']:.2f}%")
                    st.metric("Pérdida por Soiling", f"{stats_sk['SR']['perdida_promedio']:.2f}%")
                    st.metric("Datos disponibles", f"{len(df_sr_soilingkit):,} registros")
        
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
        if 'df_sr_soilingkit' in locals() and df_sr_soilingkit is not None:
            soilingkit_daily = df_sr_soilingkit['SR'].resample('1D').mean()
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