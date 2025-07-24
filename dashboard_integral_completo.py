import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import clickhouse_connect
from influxdb_client import InfluxDBClient
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Integral PVStand - Todas las Tecnologías",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de bases de datos
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}

INFLUXDB_CONFIG = {
    'url': "http://146.83.153.212:8086",
    'token': "Psda2020",
    'org': "PSDA",
    'bucket': "psda"
}

# Título principal
st.title("🔋 Dashboard Integral PVStand - Todas las Tecnologías")
st.markdown("---")

# Sidebar para navegación
st.sidebar.header("🎛️ Navegación")

# Menú de páginas
page = st.sidebar.selectbox(
    "📱 Seleccionar Análisis:",
    [
        "🏠 Página Principal",
        "📊 Curvas IV PVStand",
        "🔍 Celdas de Referencia",
        "🪟 PV Glasses",
        "🧹 Soiling Kit",
        "🌫️ DustIQ",
        "🌡️ Temperatura Módulos",
        "📈 Comparaciones",
        "📋 Resumen Ejecutivo"
    ]
)

# Función para cargar datos desde ClickHouse
@st.cache_data(ttl=3600)
def load_pvstand_curves():
    """Carga datos de curvas IV de PVStand desde ClickHouse."""
    try:
        client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)
        
        query_perc1 = """
        SELECT fecha, hora, corriente, voltaje, potencia
        FROM ref_data.iv_curves_perc1_fixed_medio_dia_solar
        WHERE fecha >= '2024-07-01' AND fecha <= '2025-07-31'
        ORDER BY fecha, hora
        LIMIT 50000
        """
        
        query_perc2 = """
        SELECT fecha, hora, corriente, voltaje, potencia
        FROM ref_data.iv_curves_perc2_fixed_medio_dia_solar
        WHERE fecha >= '2024-07-01' AND fecha <= '2025-07-31'
        ORDER BY fecha, hora
        LIMIT 50000
        """
        
        data_perc1 = client.query(query_perc1)
        data_perc2 = client.query(query_perc2)
        
        df_perc1 = pd.DataFrame(data_perc1.result_set, columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])
        df_perc2 = pd.DataFrame(data_perc2.result_set, columns=['fecha', 'hora', 'corriente', 'voltaje', 'potencia'])
        
        df_perc1['module'] = 'perc1fixed'
        df_perc2['module'] = 'perc2fixed'
        
        df_combined = pd.concat([df_perc1, df_perc2], ignore_index=True)
        df_combined['fecha'] = pd.to_datetime(df_combined['fecha'])
        df_combined['timestamp'] = pd.to_datetime(df_combined['fecha'].astype(str) + ' ' + df_combined['hora'].astype(str))
        
        client.close()
        return df_combined
        
    except Exception as e:
        st.error(f"Error cargando datos PVStand: {e}")
        return None

# Función para cargar datos desde InfluxDB
@st.cache_data(ttl=3600)
def load_influxdb_data(measurement, start_date, end_date):
    """Carga datos desde InfluxDB."""
    try:
        client = InfluxDBClient(**INFLUXDB_CONFIG)
        query_api = client.query_api()
        
        query = f'''
        from(bucket: "psda")
            |> range(start: {start_date}, stop: {end_date})
            |> filter(fn: (r) => r["_measurement"] == "{measurement}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        result = query_api.query_data_frame(query)
        client.close()
        
        if not result.empty:
            result['_time'] = pd.to_datetime(result['_time'])
            return result
        else:
            return None
            
    except Exception as e:
        st.error(f"Error cargando datos {measurement}: {e}")
        return None

# Página Principal
if page == "🏠 Página Principal":
    st.header("🏠 Bienvenido al Dashboard Integral PVStand")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Tecnologías Disponibles")
        st.markdown("""
        - **🔋 Curvas IV PVStand**: Análisis de curvas intensidad-voltaje
        - **🔍 Celdas de Referencia**: Monitoreo de celdas de calibración
        - **🪟 PV Glasses**: Análisis de módulos con vidrio especial
        - **🧹 Soiling Kit**: Medición de ensuciamiento
        - **🌫️ DustIQ**: Monitoreo de polvo y suciedad
        - **🌡️ Temperatura Módulos**: Análisis térmico
        """)
    
    with col2:
        st.subheader("📈 Funcionalidades")
        st.markdown("""
        - **📊 Visualizaciones interactivas** con Plotly
        - **📅 Filtros por fecha** y tecnología
        - **📋 Comparaciones** entre tecnologías
        - **📥 Descarga de datos** en CSV
        - **📊 Estadísticas** y métricas clave
        """)
    
    st.markdown("---")
    st.subheader("🚀 Inicio Rápido")
    st.info("💡 Usa el menú lateral para navegar entre los diferentes análisis y tecnologías.")

# Página de Curvas IV PVStand
elif page == "📊 Curvas IV PVStand":
    st.header("📊 Curvas IV PVStand")
    
    # Cargar datos
    with st.spinner("🔄 Cargando datos de curvas IV..."):
        df_pvstand = load_pvstand_curves()
    
    if df_pvstand is not None:
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            modules = df_pvstand['module'].unique()
            selected_modules = st.multiselect("Módulos:", modules, default=modules[:1])
        
        with col2:
            available_dates = sorted(df_pvstand['fecha'].dt.date.unique())
            selected_date = st.date_input("Fecha:", value=available_dates[-1] if available_dates else None)
        
        with col3:
            if selected_date:
                df_date = df_pvstand[df_pvstand['fecha'].dt.date == selected_date]
                df_date = df_date[df_date['module'].isin(selected_modules)]
                
                if not df_date.empty:
                    horas = sorted(df_date['hora'].unique())
                    selected_hora = st.selectbox("Hora:", horas)
                    
                    # Filtrar datos
                    df_curve = df_date[df_date['hora'] == selected_hora]
                    
                    if not df_curve.empty:
                        # Gráfico de curvas IV
                        fig_iv = go.Figure()
                        colors = ['blue', 'red', 'green', 'orange']
                        
                        for i, module in enumerate(selected_modules):
                            df_module = df_curve[df_curve['module'] == module]
                            if not df_module.empty:
                                df_sorted = df_module.sort_values('voltaje')
                                
                                fig_iv.add_trace(go.Scatter(
                                    x=df_sorted['voltaje'],
                                    y=df_sorted['corriente'],
                                    mode='lines+markers',
                                    name=f'{module} - IV',
                                    line=dict(color=colors[i % len(colors)], width=2)
                                ))
                        
                        fig_iv.update_layout(
                            title=f"Curva IV - {selected_date} {selected_hora}",
                            xaxis_title="Voltaje (V)",
                            yaxis_title="Corriente (A)",
                            height=500
                        )
                        
                        st.plotly_chart(fig_iv, use_container_width=True)
                        
                        # Estadísticas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.subheader("📊 Estadísticas")
                            for module in selected_modules:
                                df_module = df_curve[df_curve['module'] == module]
                                if not df_module.empty:
                                    pmp = df_module['potencia'].max()
                                    st.write(f"**{module} PMP:** {pmp:.2f} W")
                        
                        with col2:
                            st.subheader("📈 Rango de Valores")
                            st.write(f"**Corriente:** {df_curve['corriente'].min():.4f} - {df_curve['corriente'].max():.4f} A")
                            st.write(f"**Voltaje:** {df_curve['voltaje'].min():.4f} - {df_curve['voltaje'].max():.4f} V")
                        
                        with col3:
                            st.subheader("📥 Descargar")
                            csv = df_curve.to_csv(index=False)
                            st.download_button(
                                label="📥 CSV",
                                data=csv,
                                file_name=f"curvas_iv_{selected_date}.csv",
                                mime="text/csv"
                            )

# Página de Celdas de Referencia
elif page == "🔍 Celdas de Referencia":
    st.header("🔍 Celdas de Referencia")
    
    # Filtros de fecha
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio:", value=datetime(2024, 7, 1))
    with col2:
        end_date = st.date_input("Fecha fin:", value=datetime(2025, 7, 31))
    
    # Cargar datos
    with st.spinner("🔄 Cargando datos de celdas de referencia..."):
        df_refcells = load_influxdb_data("refcells", start_date, end_date)
    
    if df_refcells is not None:
        st.success(f"✅ Datos cargados: {len(df_refcells)} registros")
        
        # Gráfico temporal
        fig = px.line(df_refcells, x='_time', y='value', 
                     title="Evolución Temporal - Celdas de Referencia")
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Estadísticas")
            st.write(f"**Valor medio:** {df_refcells['value'].mean():.4f}")
            st.write(f"**Valor máximo:** {df_refcells['value'].max():.4f}")
            st.write(f"**Valor mínimo:** {df_refcells['value'].min():.4f}")
        
        with col2:
            st.subheader("📥 Descargar")
            csv = df_refcells.to_csv(index=False)
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"refcells_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ No se encontraron datos de celdas de referencia")

# Página de PV Glasses
elif page == "🪟 PV Glasses":
    st.header("🪟 PV Glasses")
    
    # Filtros de fecha
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio:", value=datetime(2024, 7, 1))
    with col2:
        end_date = st.date_input("Fecha fin:", value=datetime(2025, 7, 31))
    
    # Cargar datos
    with st.spinner("🔄 Cargando datos de PV Glasses..."):
        df_pvglasses = load_influxdb_data("pv_glasses", start_date, end_date)
    
    if df_pvglasses is not None:
        st.success(f"✅ Datos cargados: {len(df_pvglasses)} registros")
        
        # Gráfico temporal
        fig = px.line(df_pvglasses, x='_time', y='value', 
                     title="Evolución Temporal - PV Glasses")
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Estadísticas")
            st.write(f"**Valor medio:** {df_pvglasses['value'].mean():.4f}")
            st.write(f"**Valor máximo:** {df_pvglasses['value'].max():.4f}")
            st.write(f"**Valor mínimo:** {df_pvglasses['value'].min():.4f}")
        
        with col2:
            st.subheader("📥 Descargar")
            csv = df_pvglasses.to_csv(index=False)
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"pv_glasses_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ No se encontraron datos de PV Glasses")

# Página de Soiling Kit
elif page == "🧹 Soiling Kit":
    st.header("🧹 Soiling Kit")
    
    # Filtros de fecha
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio:", value=datetime(2024, 7, 1))
    with col2:
        end_date = st.date_input("Fecha fin:", value=datetime(2025, 7, 31))
    
    # Cargar datos
    with st.spinner("🔄 Cargando datos de Soiling Kit..."):
        df_soiling = load_influxdb_data("soiling_kit", start_date, end_date)
    
    if df_soiling is not None:
        st.success(f"✅ Datos cargados: {len(df_soiling)} registros")
        
        # Gráfico temporal
        fig = px.line(df_soiling, x='_time', y='value', 
                     title="Evolución Temporal - Soiling Kit")
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Estadísticas")
            st.write(f"**Valor medio:** {df_soiling['value'].mean():.4f}")
            st.write(f"**Valor máximo:** {df_soiling['value'].max():.4f}")
            st.write(f"**Valor mínimo:** {df_soiling['value'].min():.4f}")
        
        with col2:
            st.subheader("📥 Descargar")
            csv = df_soiling.to_csv(index=False)
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"soiling_kit_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ No se encontraron datos de Soiling Kit")

# Página de DustIQ
elif page == "🌫️ DustIQ":
    st.header("🌫️ DustIQ")
    
    # Filtros de fecha
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio:", value=datetime(2024, 7, 1))
    with col2:
        end_date = st.date_input("Fecha fin:", value=datetime(2025, 7, 31))
    
    # Cargar datos
    with st.spinner("🔄 Cargando datos de DustIQ..."):
        df_dustiq = load_influxdb_data("dustiq", start_date, end_date)
    
    if df_dustiq is not None:
        st.success(f"✅ Datos cargados: {len(df_dustiq)} registros")
        
        # Gráfico temporal
        fig = px.line(df_dustiq, x='_time', y='value', 
                     title="Evolución Temporal - DustIQ")
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Estadísticas")
            st.write(f"**Valor medio:** {df_dustiq['value'].mean():.4f}")
            st.write(f"**Valor máximo:** {df_dustiq['value'].max():.4f}")
            st.write(f"**Valor mínimo:** {df_dustiq['value'].min():.4f}")
        
        with col2:
            st.subheader("📥 Descargar")
            csv = df_dustiq.to_csv(index=False)
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"dustiq_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ No se encontraron datos de DustIQ")

# Página de Temperatura Módulos
elif page == "🌡️ Temperatura Módulos":
    st.header("🌡️ Temperatura Módulos Fijos")
    
    # Filtros de fecha
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio:", value=datetime(2024, 7, 1))
    with col2:
        end_date = st.date_input("Fecha fin:", value=datetime(2025, 7, 31))
    
    # Cargar datos
    with st.spinner("🔄 Cargando datos de temperatura..."):
        df_temp = load_influxdb_data("temp_mod_fixed", start_date, end_date)
    
    if df_temp is not None:
        st.success(f"✅ Datos cargados: {len(df_temp)} registros")
        
        # Gráfico temporal
        fig = px.line(df_temp, x='_time', y='value', 
                     title="Evolución Temporal - Temperatura Módulos")
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Estadísticas")
            st.write(f"**Temperatura media:** {df_temp['value'].mean():.2f}°C")
            st.write(f"**Temperatura máxima:** {df_temp['value'].max():.2f}°C")
            st.write(f"**Temperatura mínima:** {df_temp['value'].min():.2f}°C")
        
        with col2:
            st.subheader("📥 Descargar")
            csv = df_temp.to_csv(index=False)
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"temp_mod_fixed_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ No se encontraron datos de temperatura")

# Página de Comparaciones
elif page == "📈 Comparaciones":
    st.header("📈 Comparaciones entre Tecnologías")
    
    # Filtros de fecha
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio:", value=datetime(2024, 7, 1))
    with col2:
        end_date = st.date_input("Fecha fin:", value=datetime(2025, 7, 31))
    
    # Cargar todos los datos
    with st.spinner("🔄 Cargando datos para comparación..."):
        df_refcells = load_influxdb_data("refcells", start_date, end_date)
        df_pvglasses = load_influxdb_data("pv_glasses", start_date, end_date)
        df_soiling = load_influxdb_data("soiling_kit", start_date, end_date)
        df_dustiq = load_influxdb_data("dustiq", start_date, end_date)
        df_temp = load_influxdb_data("temp_mod_fixed", start_date, end_date)
    
    # Gráfico comparativo
    fig = go.Figure()
    
    if df_refcells is not None:
        fig.add_trace(go.Scatter(x=df_refcells['_time'], y=df_refcells['value'], 
                                name='RefCells', mode='lines'))
    if df_pvglasses is not None:
        fig.add_trace(go.Scatter(x=df_pvglasses['_time'], y=df_pvglasses['value'], 
                                name='PV Glasses', mode='lines'))
    if df_soiling is not None:
        fig.add_trace(go.Scatter(x=df_soiling['_time'], y=df_soiling['value'], 
                                name='Soiling Kit', mode='lines'))
    if df_dustiq is not None:
        fig.add_trace(go.Scatter(x=df_dustiq['_time'], y=df_dustiq['value'], 
                                name='DustIQ', mode='lines'))
    
    fig.update_layout(
        title="Comparación de Tecnologías",
        xaxis_title="Tiempo",
        yaxis_title="Valor",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de estadísticas comparativas
    st.subheader("📊 Estadísticas Comparativas")
    
    stats_data = []
    if df_refcells is not None:
        stats_data.append({
            'Tecnología': 'RefCells',
            'Media': f"{df_refcells['value'].mean():.4f}",
            'Máximo': f"{df_refcells['value'].max():.4f}",
            'Mínimo': f"{df_refcells['value'].min():.4f}",
            'Registros': len(df_refcells)
        })
    
    if df_pvglasses is not None:
        stats_data.append({
            'Tecnología': 'PV Glasses',
            'Media': f"{df_pvglasses['value'].mean():.4f}",
            'Máximo': f"{df_pvglasses['value'].max():.4f}",
            'Mínimo': f"{df_pvglasses['value'].min():.4f}",
            'Registros': len(df_pvglasses)
        })
    
    if df_soiling is not None:
        stats_data.append({
            'Tecnología': 'Soiling Kit',
            'Media': f"{df_soiling['value'].mean():.4f}",
            'Máximo': f"{df_soiling['value'].max():.4f}",
            'Mínimo': f"{df_soiling['value'].min():.4f}",
            'Registros': len(df_soiling)
        })
    
    if df_dustiq is not None:
        stats_data.append({
            'Tecnología': 'DustIQ',
            'Media': f"{df_dustiq['value'].mean():.4f}",
            'Máximo': f"{df_dustiq['value'].max():.4f}",
            'Mínimo': f"{df_dustiq['value'].min():.4f}",
            'Registros': len(df_dustiq)
        })
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True)

# Página de Resumen Ejecutivo
elif page == "📋 Resumen Ejecutivo":
    st.header("📋 Resumen Ejecutivo")
    
    st.subheader("🎯 Objetivo del Proyecto")
    st.markdown("""
    Este dashboard integral permite el análisis completo de todas las tecnologías de monitoreo 
    y análisis de sistemas fotovoltaicos implementadas en el proyecto PVStand.
    """)
    
    st.subheader("🔧 Tecnologías Implementadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Curvas IV PVStand**
        - Análisis de curvas intensidad-voltaje
        - Comparación entre módulos PERC1 y PERC2
        - Cálculo de parámetros eléctricos (PMP, ISC, VOC, FF)
        
        **🔍 Celdas de Referencia**
        - Monitoreo de celdas de calibración
        - Control de calidad de mediciones
        - Referencia para otros sensores
        
        **🪟 PV Glasses**
        - Análisis de módulos con vidrio especial
        - Evaluación de transmisión óptica
        - Impacto en rendimiento
        """)
    
    with col2:
        st.markdown("""
        **🧹 Soiling Kit**
        - Medición de ensuciamiento de módulos
        - Pérdidas por suciedad
        - Programación de limpieza
        
        **🌫️ DustIQ**
        - Monitoreo continuo de polvo
        - Alerta de suciedad
        - Optimización de mantenimiento
        
        **🌡️ Temperatura Módulos**
        - Análisis térmico de módulos
        - Correlación con rendimiento
        - Detección de anomalías
        """)
    
    st.subheader("📈 Beneficios")
    st.markdown("""
    - **📊 Análisis integral** de todas las tecnologías
    - **🔍 Detección temprana** de problemas
    - **📈 Optimización** del rendimiento del sistema
    - **💰 Reducción de costos** de mantenimiento
    - **📋 Reportes automáticos** y descarga de datos
    """)
    
    st.subheader("🚀 Próximos Pasos")
    st.markdown("""
    - Implementación de alertas automáticas
    - Análisis predictivo de fallas
    - Integración con sistemas de control
    - Expansión a otros sitios
    """)

# Footer
st.markdown("---")
st.markdown("""
### 📚 Información del Dashboard

**Dashboard Integral PVStand** - Análisis completo de todas las tecnologías de monitoreo fotovoltaico.

**Desarrollado por:** Equipo PSDA  
**Última actualización:** Julio 2025  
**Versión:** 1.0 - Dashboard Integral Completo
""") 