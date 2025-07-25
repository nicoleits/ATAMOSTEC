#!/usr/bin/env python3
"""
Dashboard Integrado - DustIQ + PVStand
Aplicación unificada para análisis de soiling y curvas IV
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

# ===== FUNCIONES PARA DUSTIQ =====

@st.cache_data(ttl=3600)
def load_dustiq_data_from_clickhouse():
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
        with st.spinner("🔄 Procesando datos DustIQ..."):
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
        
        st.success(f"✅ Datos DustIQ cargados: {len(df_dustiq):,} registros")
        return df_dustiq
        
    except Exception as e:
        st.warning(f"⚠️ Error de conexión a ClickHouse para DustIQ: {str(e)[:100]}...")
        return None

# ===== FUNCIONES PARA PVSTAND =====

@st.cache_data(ttl=3600)
def load_pvstand_data_from_clickhouse():
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
        
        # Consulta para curvas IV de PVStand
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

        # Procesar datos
        with st.spinner("🔄 Procesando datos PVStand..."):
            df_perc1_curves = pd.DataFrame(data_perc1_curves.result_set,
                                          columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])
            df_perc2_curves = pd.DataFrame(data_perc2_curves.result_set,
                                          columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])

            # Agregar identificador de módulo
            df_perc1_curves['module'] = 'perc1fixed'
            df_perc2_curves['module'] = 'perc2fixed'

            # Combinar datos
            df_pvstand_curves = pd.concat([df_perc1_curves, df_perc2_curves], ignore_index=True)

            # Convertir fecha y hora a datetime
            df_pvstand_curves['fecha'] = pd.to_datetime(df_pvstand_curves['fecha'])
            df_pvstand_curves['timestamp'] = pd.to_datetime(
                df_pvstand_curves['fecha'].astype(str) + ' ' + df_pvstand_curves['hora'].astype(str)
            )

            # Asegurar que esté en UTC
            if df_pvstand_curves['timestamp'].dt.tz is None:
                df_pvstand_curves['timestamp'] = df_pvstand_curves['timestamp'].dt.tz_localize('UTC')
            else:
                df_pvstand_curves['timestamp'] = df_pvstand_curves['timestamp'].dt.tz_convert('UTC')

            # Ordenar por timestamp
            df_pvstand_curves = df_pvstand_curves.sort_values('timestamp')

            # Reorganizar columnas
            df_pvstand_curves = df_pvstand_curves[['timestamp', 'module', 'corriente', 'voltaje', 'potencia', 'fecha', 'hora']]

        client.close()
        
        st.success(f"✅ Datos PVStand cargados: {len(df_pvstand_curves):,} registros")
        return df_pvstand_curves

    except Exception as e:
        st.warning(f"⚠️ Error de conexión a ClickHouse para PVStand: {str(e)[:100]}...")
        return None

# ===== FUNCIONES DE PROCESAMIENTO =====

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

def calcular_parametros_iv(corriente, voltaje, potencia):
    """Calcula parámetros eléctricos de la curva IV."""
    if len(corriente) == 0:
        return None
    
    # Parámetros básicos
    isc = max(corriente)  # Corriente de cortocircuito
    voc = max(voltaje)    # Voltaje de circuito abierto
    pmp = max(potencia)   # Potencia máxima
    
    # Punto de máxima potencia
    idx_pmp = np.argmax(potencia)
    imp = corriente[idx_pmp]  # Corriente en punto de máxima potencia
    vmp = voltaje[idx_pmp]    # Voltaje en punto de máxima potencia
    
    # Factor de forma
    fill_factor = (imp * vmp) / (isc * voc) if (isc * voc) > 0 else 0
    
    return {
        'ISC': isc,
        'VOC': voc,
        'PMP': pmp,
        'IMP': imp,
        'VMP': vmp,
        'FF': fill_factor
    }

# ===== CARGA DE DATOS =====

# Cargar datos con fallback inteligente
with st.spinner("🔄 Cargando datos de ambos sistemas..."):
    # Intentar cargar desde ClickHouse primero
    df_dustiq = load_dustiq_data_from_clickhouse()
    df_pvstand = load_pvstand_data_from_clickhouse()

if df_dustiq is None and df_pvstand is None:
    st.error("❌ No se pudieron cargar los datos de ningún sistema. Verifica la conexión a ClickHouse.")
    st.stop()

# ===== INTERFAZ PRINCIPAL =====

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs([
    "🌫️ DustIQ - Soiling Ratio", 
    "🔋 PVStand - Curvas IV", 
    "📊 Comparación Integrada",
    "ℹ️ Información del Sistema"
])

# ===== PESTAÑA 1: DUSTIQ =====
with tab1:
    if df_dustiq is not None:
        st.subheader("🌫️ Análisis de Soiling Ratio - DustIQ")
        
        # Sidebar para filtros DustIQ
        with st.sidebar:
            st.header("🎛️ Filtros DustIQ")
            
            # Filtro de fechas
            min_date = df_dustiq.index.min().date()
            max_date = df_dustiq.index.max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "📅 Fecha Inicio:",
                    value=max(min_date, pd.to_datetime('2024-07-01').date()),
                    min_value=min_date,
                    max_value=max_date,
                    key="dustiq_start"
                )
            
            with col2:
                end_date = st.date_input(
                    "📅 Fecha Fin:",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="dustiq_end"
                )
            
            # Filtrar datos por rango de fechas
            df_dustiq_filtered = df_dustiq.loc[start_date:end_date]
            
            # Filtro de umbral SR
            sr_threshold = st.slider(
                "🎚️ Umbral SR (%):",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                help="Filtrar datos con Soiling Ratio mayor a este valor",
                key="dustiq_threshold"
            )
            
            # Configuración de análisis
            st.subheader("⚙️ Configuración de Análisis")
            
            # Frecuencia temporal
            freq_options = {
                "Diario": "1D",
                "Semanal": "1W", 
                "Mensual": "1M"
            }
            selected_freq = st.selectbox(
                "📅 Frecuencia Temporal:",
                list(freq_options.keys()),
                index=1,
                key="dustiq_freq"
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
            selected_franja_option = st.selectbox(
                "🕐 Franjas Horarias:",
                franja_options,
                index=0,
                key="dustiq_franja_option"
            )
            
            # Selección personalizada de franjas
            selected_franjas = []
            if selected_franja_option == "Personalizado":
                selected_franjas = st.multiselect(
                    "🕐 Seleccionar franjas específicas:",
                    list(franjas_disponibles.keys()),
                    default=["12:00-13:00", "14:00-15:00"],
                    key="dustiq_franjas"
                )
            elif selected_franja_option == "Todas las franjas":
                selected_franjas = list(franjas_disponibles.keys())
            elif selected_franja_option == "Mediodía Solar":
                selected_franjas = ["Mediodía Solar"]
            
            # Tipo de análisis
            analysis_type = st.selectbox(
                "📊 Tipo de Análisis:",
                [
                    "📈 Vista General",
                    "🕐 Franjas Horarias Fijas"
                ],
                key="dustiq_analysis"
            )
        
        # Aplicar filtro de umbral
        sr_column = 'SR_C11_Avg'
        if sr_column in df_dustiq_filtered.columns:
            df_sr_filtered = df_dustiq_filtered[df_dustiq_filtered[sr_column] > sr_threshold][sr_column].copy()
        else:
            st.error(f"❌ Columna {sr_column} no encontrada en los datos")
            st.stop()
        
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
            # Métricas principales
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
            
            # Gráfico de serie temporal
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
            
            # Estadísticas por franja
            if analysis_type == "🕐 Franjas Horarias Fijas":
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
            st.warning("⚠️ No hay datos disponibles para la configuración seleccionada")
    else:
        st.error("❌ No se pudieron cargar los datos de DustIQ")

# ===== PESTAÑA 2: PVSTAND =====
with tab2:
    if df_pvstand is not None:
        st.subheader("🔋 Análisis de Curvas IV - PVStand")
        
        # Sidebar para filtros PVStand
        with st.sidebar:
            st.header("🎛️ Filtros PVStand")
            
            # Seleccionar módulos
            modules = df_pvstand['module'].unique()
            selected_modules = st.multiselect(
                "📊 Seleccionar Módulos:",
                modules,
                default=modules[:1],
                key="pvstand_modules"
            )
            
            if not selected_modules:
                st.warning("⚠️ Por favor selecciona al menos un módulo")
                st.stop()
            
            # Filtrar datos por módulos seleccionados
            df_module = df_pvstand[df_pvstand['module'].isin(selected_modules)]
            
            # Seleccionar fecha
            available_dates = sorted(df_module['fecha'].dt.date.unique())
            if available_dates:
                selected_date = st.selectbox(
                    "📅 Seleccionar Fecha:",
                    available_dates,
                    index=len(available_dates)-1 if len(available_dates) > 0 else 0,
                    key="pvstand_date"
                )
                
                # Filtrar datos por fecha seleccionada
                df_date = df_module[df_module['fecha'].dt.date == selected_date]
                
                # Seleccionar curva del día
                available_curves = sorted(df_date['hora'].unique())
                if available_curves:
                    selected_curve = st.selectbox(
                        "🕐 Seleccionar Curva:",
                        available_curves,
                        index=len(available_curves)-1 if len(available_curves) > 0 else 0,
                        key="pvstand_curve"
                    )
                    
                    # Filtrar datos por curva seleccionada
                    df_curve = df_date[df_date['hora'] == selected_curve]
                    
                    # Mostrar información de la curva
                    st.subheader("📊 Información de la Curva")
                    st.write(f"**Fecha:** {selected_date}")
                    st.write(f"**Hora:** {selected_curve}")
                    st.write(f"**Módulos:** {', '.join(selected_modules)}")
                    st.write(f"**Puntos de medición:** {len(df_curve)}")
        
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
            
            # Parámetros eléctricos
            st.subheader("🔧 Parámetros Eléctricos")
            
            params_data = []
            for module in selected_modules:
                df_module_curve = df_curve[df_curve['module'] == module]
                if not df_module_curve.empty:
                    params = calcular_parametros_iv(
                        df_module_curve['corriente'].values,
                        df_module_curve['voltaje'].values,
                        df_module_curve['potencia'].values
                    )
                    
                    if params:
                        params_data.append({
                            'Módulo': module,
                            'ISC (A)': f"{params['ISC']:.3f}",
                            'VOC (V)': f"{params['VOC']:.3f}",
                            'PMP (W)': f"{params['PMP']:.3f}",
                            'IMP (A)': f"{params['IMP']:.3f}",
                            'VMP (V)': f"{params['VMP']:.3f}",
                            'FF': f"{params['FF']:.3f}"
                        })
            
            if params_data:
                df_params = pd.DataFrame(params_data)
                st.dataframe(df_params, use_container_width=True)
            
            # Tabla de datos
            st.subheader("📋 Datos de la Curva")
            st.dataframe(df_curve[['module', 'corriente', 'voltaje', 'potencia']], use_container_width=True)
        else:
            st.warning("⚠️ No hay datos disponibles para la fecha y curva seleccionadas")
    else:
        st.error("❌ No se pudieron cargar los datos de PVStand")

# ===== PESTAÑA 3: COMPARACIÓN INTEGRADA =====
with tab3:
    st.subheader("📊 Comparación Integrada - DustIQ vs PVStand")
    
    if df_dustiq is not None and df_pvstand is not None:
        st.info("🔍 Análisis comparativo entre sistemas de medición de soiling")
        
        # Métricas de comparación
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌫️ DustIQ")
            if 'datos_combinados' in locals():
                st.metric("Promedio SR", f"{datos_combinados.mean():.2f}%")
                st.metric("Pérdida por Soiling", f"{100 - datos_combinados.mean():.2f}%")
                st.metric("Datos disponibles", f"{len(df_dustiq):,} registros")
        
        with col2:
            st.subheader("🔋 PVStand")
            if 'df_pvstand' in locals():
                # Calcular métricas de PVStand
                total_curves = len(df_pvstand['timestamp'].dt.date.unique())
                st.metric("Días con datos", f"{total_curves}")
                st.metric("Curvas totales", f"{len(df_pvstand):,}")
                st.metric("Módulos", f"{len(df_pvstand['module'].unique())}")
        
        # Gráfico de comparación temporal
        st.subheader("📈 Comparación Temporal")
        
        # Preparar datos para comparación
        if 'datos_combinados' in locals():
            # Resamplear datos de DustIQ a diario para comparación
            dustiq_daily = datos_combinados.resample('1D').mean()
            
            # Preparar datos de PVStand (promedio diario de PMP)
            pvstand_daily = df_pvstand.groupby([df_pvstand['timestamp'].dt.date, 'module'])['potencia'].max().reset_index()
            pvstand_daily['timestamp'] = pd.to_datetime(pvstand_daily['timestamp'])
            pvstand_daily = pvstand_daily.set_index('timestamp')
            pvstand_daily_avg = pvstand_daily.groupby(pvstand_daily.index)['potencia'].mean()
            
            # Gráfico comparativo
            fig_comparison = go.Figure()
            
            # DustIQ
            fig_comparison.add_trace(go.Scatter(
                x=dustiq_daily.index,
                y=dustiq_daily.values,
                mode='lines+markers',
                name='DustIQ - SR (%)',
                line=dict(color='blue', width=2),
                yaxis='y'
            ))
            
            # PVStand (normalizado)
            if not pvstand_daily_avg.empty:
                # Normalizar potencia a porcentaje
                pvstand_normalized = (pvstand_daily_avg / pvstand_daily_avg.max()) * 100
                
                fig_comparison.add_trace(go.Scatter(
                    x=pvstand_normalized.index,
                    y=pvstand_normalized.values,
                    mode='lines+markers',
                    name='PVStand - Potencia Normalizada (%)',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
            
            fig_comparison.update_layout(
                title="Comparación DustIQ vs PVStand",
                xaxis_title="Fecha",
                yaxis=dict(title="DustIQ - Soiling Ratio (%)", side='left'),
                yaxis2=dict(title="PVStand - Potencia Normalizada (%)", side='right', overlaying='y'),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Análisis de correlación
        st.subheader("🔗 Análisis de Correlación")
        
        if 'dustiq_daily' in locals() and 'pvstand_daily_avg' in locals():
            # Alinear fechas
            common_dates = dustiq_daily.index.intersection(pvstand_daily_avg.index)
            
            if len(common_dates) > 5:
                dustiq_aligned = dustiq_daily.loc[common_dates]
                pvstand_aligned = pvstand_daily_avg.loc[common_dates]
                
                # Normalizar PVStand
                pvstand_normalized = (pvstand_aligned / pvstand_aligned.max()) * 100
                
                # Calcular correlación
                correlation = np.corrcoef(dustiq_aligned.values, pvstand_normalized.values)[0, 1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Correlación", f"{correlation:.3f}")
                
                with col2:
                    st.metric("Puntos de comparación", len(common_dates))
                
                with col3:
                    if correlation > 0.7:
                        st.success("✅ Alta correlación")
                    elif correlation > 0.5:
                        st.warning("⚠️ Correlación moderada")
                    else:
                        st.error("❌ Baja correlación")
                
                # Gráfico de dispersión
                fig_scatter = px.scatter(
                    x=dustiq_aligned.values,
                    y=pvstand_normalized.values,
                    title="Correlación DustIQ vs PVStand",
                    labels={'x': 'DustIQ - Soiling Ratio (%)', 'y': 'PVStand - Potencia Normalizada (%)'}
                )
                
                # Agregar línea de tendencia
                fig_scatter.add_trace(go.Scatter(
                    x=[dustiq_aligned.min(), dustiq_aligned.max()],
                    y=[pvstand_normalized.min(), pvstand_normalized.max()],
                    mode='lines',
                    name='Línea de tendencia',
                    line=dict(dash='dash', color='red')
                ))
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("⚠️ No hay suficientes fechas comunes para análisis de correlación")
    else:
        st.error("❌ Se requieren datos de ambos sistemas para la comparación")

# ===== PESTAÑA 4: INFORMACIÓN DEL SISTEMA =====
with tab4:
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