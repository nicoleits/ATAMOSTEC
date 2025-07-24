import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import clickhouse_connect

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Curvas IV PVStand",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔋 Dashboard Curvas IV PVStand")
st.markdown("---")

# Configuración de ClickHouse
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}

# Función para cargar datos desde ClickHouse
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data_from_clickhouse():
    """Carga los datos de curvas IV de PVStand directamente desde ClickHouse."""
    try:
        st.info("🔄 Conectando a ClickHouse...")
        
        # Conectar a Clickhouse
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password']
        )
        
        # Consultar datos de ambas tablas de curvas IV
        st.info("📊 Consultando datos de curvas IV...")
        
        # Consulta para iv_curves_perc1_fixed_medio_dia_solar
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

        # Consulta para iv_curves_perc2_fixed_medio_dia_solar
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

        # Ejecutar consultas
        st.info("📊 Consultando tabla perc1...")
        data_perc1_curves = client.query(query_perc1_curves)
        st.info("📊 Consultando tabla perc2...")
        data_perc2_curves = client.query(query_perc2_curves)

        # Procesar datos
        st.info("🔄 Procesando datos...")
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
        df_pvstand_curves['timestamp'] = pd.to_datetime(df_pvstand_curves['fecha'].astype(str) + ' ' + df_pvstand_curves['hora'].astype(str))

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
        
        st.success(f"✅ Datos cargados: {len(df_pvstand_curves):,} registros")
        return df_pvstand_curves

    except Exception as e:
        st.error(f"❌ Error al cargar datos desde ClickHouse: {e}")
        return None

# Función para cargar datos desde archivo local (fallback)
@st.cache_data
def load_data_from_file():
    """Carga los datos de curvas IV de PVStand desde archivo local."""
    try:
        # Intentar diferentes rutas posibles
        possible_paths = [
            "/home/nicole/SR/SOILING/datos/raw_pvstand_curves_data.csv",
            "datos/raw_pvstand_curves_data.csv",
            "raw_pvstand_curves_data.csv"
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['fecha'] = pd.to_datetime(df['fecha'])
                st.success(f"✅ Datos cargados desde archivo: {len(df):,} registros")
                return df
        
        st.error("❌ No se encontró el archivo de datos en ninguna ubicación")
        return None
        
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

# Cargar datos
with st.spinner("🔄 Cargando datos de curvas IV..."):
    # Intentar cargar desde ClickHouse primero
    df = load_data_from_clickhouse()
    
    # Si falla, intentar desde archivo local
    if df is None:
        st.warning("⚠️ Intentando cargar desde archivo local...")
        df = load_data_from_file()

if df is None:
    st.error("❌ No se pudieron cargar los datos. Verifica la conexión a ClickHouse o la disponibilidad del archivo.")
    st.stop()

# Asegurar que las columnas de fecha sean datetime
try:
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    st.success("✅ Fechas convertidas correctamente")
except Exception as e:
    st.error(f"❌ Error al convertir fechas: {e}")
    st.stop()

# Sidebar para filtros
st.sidebar.header("🎛️ Filtros")

# 1. Seleccionar módulos (múltiple selección)
modules = df['module'].unique()
selected_modules = st.sidebar.multiselect(
    "📊 Seleccionar Módulos:",
    modules,
    default=modules[:1]  # Por defecto selecciona el primer módulo
)

if not selected_modules:
    st.warning("⚠️ Por favor selecciona al menos un módulo")
    st.stop()

# Filtrar datos por módulos seleccionados
df_module = df[df['module'].isin(selected_modules)]

# 2. Seleccionar fecha
available_dates = sorted(df_module['fecha'].dt.date.unique())
if available_dates:
    selected_date = st.sidebar.date_input(
        "📅 Seleccionar Fecha:",
        value=available_dates[-1],  # Última fecha disponible
        min_value=available_dates[0],
        max_value=available_dates[-1]
    )
    
    # Filtrar datos por fecha
    df_date = df_module[df_module['fecha'].dt.date == selected_date]
    
    # 3. Seleccionar curva del día
    if not df_date.empty:
        # Agrupar por hora para identificar curvas únicas
        curves_info = df_date.groupby('hora').agg({
            'corriente': ['count', 'min', 'max'],
            'voltaje': ['min', 'max'],
            'potencia': ['min', 'max']
        }).round(4)
        
        curves_info.columns = ['puntos', 'I_min', 'I_max', 'V_min', 'V_max', 'P_min', 'P_max']
        curves_info = curves_info.reset_index()
        
        # Crear descripción para cada curva
        curve_descriptions = []
        for idx, row in curves_info.iterrows():
            desc = f"{row['hora']} - {row['puntos']} puntos | I: {row['I_min']}-{row['I_max']}A | V: {row['V_min']}-{row['V_max']}V | P_max: {row['P_max']}W"
            curve_descriptions.append(desc)
        
        selected_curve_idx = st.sidebar.selectbox(
            "⏰ Seleccionar Curva del Día:",
            range(len(curve_descriptions)),
            format_func=lambda x: curve_descriptions[x]
        )
        
        selected_hora = curves_info.iloc[selected_curve_idx]['hora']
        
        # Filtrar datos para la curva seleccionada
        df_curve = df_date[df_date['hora'] == selected_hora]
        
        # Mostrar información de la curva seleccionada
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Información de la Curva")
        st.sidebar.write(f"**Módulos:** {', '.join(selected_modules)}")
        st.sidebar.write(f"**Fecha:** {selected_date}")
        st.sidebar.write(f"**Hora:** {selected_hora}")
        st.sidebar.write(f"**Puntos totales:** {len(df_curve)}")
        
        # Estadísticas de la curva para múltiples módulos
        if not df_curve.empty:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔋 Parámetros por Módulo")
            
            for module in selected_modules:
                df_module_curve = df_curve[df_curve['module'] == module]
                if not df_module_curve.empty:
                    # Calcular parámetros de la curva para este módulo
                    pmp = df_module_curve['potencia'].max()
                    isc = df_module_curve['corriente'].max()
                    voc = df_module_curve['voltaje'].max()
                    imp = df_module_curve.loc[df_module_curve['potencia'].idxmax(), 'corriente']
                    vmp = df_module_curve.loc[df_module_curve['potencia'].idxmax(), 'voltaje']
                    ff = (pmp/(isc*voc)*100) if (isc*voc) > 0 else 0
                    
                    st.sidebar.markdown(f"**{module}:**")
                    st.sidebar.write(f"  PMP: {pmp:.2f} W")
                    st.sidebar.write(f"  ISC: {isc:.4f} A")
                    st.sidebar.write(f"  VOC: {voc:.4f} V")
                    st.sidebar.write(f"  IMP: {imp:.4f} A")
                    st.sidebar.write(f"  VMP: {vmp:.4f} V")
                    st.sidebar.write(f"  FF: {ff:.2f}%")
                    st.sidebar.markdown("---")
        
        # Layout principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 Curva IV - {selected_date} {selected_hora}")
            
            if not df_curve.empty:
                # Crear gráfico de curva IV para múltiples módulos
                fig_iv = go.Figure()
                
                # Colores para diferentes módulos
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                
                for i, module in enumerate(selected_modules):
                    # Filtrar datos para este módulo
                    df_module_curve = df_curve[df_curve['module'] == module]
                    
                    if not df_module_curve.empty:
                        # Ordenar por voltaje para una curva más suave
                        df_module_sorted = df_module_curve.sort_values('voltaje')
                        
                        # Curva IV para este módulo
                        fig_iv.add_trace(go.Scatter(
                            x=df_module_sorted['voltaje'],
                            y=df_module_sorted['corriente'],
                            mode='lines+markers',
                            name=f'{module} - Curva IV',
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4, color=colors[i % len(colors)]),
                            hovertemplate=f'<b>{module}</b><br>' +
                                        '<b>Voltaje:</b> %{x:.4f} V<br>' +
                                        '<b>Corriente:</b> %{y:.4f} A<br>' +
                                        '<b>Potencia:</b> %{text:.4f} W<extra></extra>',
                            text=df_module_sorted['potencia']
                        ))
                        
                        # Punto de máxima potencia para este módulo
                        max_power_idx = df_module_sorted['potencia'].idxmax()
                        max_power_v = df_module_sorted.loc[max_power_idx, 'voltaje']
                        max_power_i = df_module_sorted.loc[max_power_idx, 'corriente']
                        max_power_p = df_module_sorted.loc[max_power_idx, 'potencia']
                        
                        fig_iv.add_trace(go.Scatter(
                            x=[max_power_v],
                            y=[max_power_i],
                            mode='markers',
                            name=f'{module} - PMP',
                            marker=dict(size=10, color=colors[i % len(colors)], symbol='star'),
                            hovertemplate=f'<b>{module} - PMP</b><br>V: {max_power_v:.4f} V<br>I: {max_power_i:.4f} A<br>P: {max_power_p:.4f} W<extra></extra>'
                        ))
                
                # Configurar layout
                fig_iv.update_layout(
                    title=f"Curva IV Comparativa - {selected_date} {selected_hora}",
                    xaxis_title="Voltaje (V)",
                    yaxis_title="Corriente (A)",
                    hovermode='closest',
                    showlegend=True,
                    height=500
                )
                
                st.plotly_chart(fig_iv, use_container_width=True)
        
        with col2:
            st.subheader("📊 Curva de Potencia")
            
            if not df_curve.empty:
                # Gráfico de potencia vs voltaje para múltiples módulos
                fig_power = go.Figure()
                
                for i, module in enumerate(selected_modules):
                    # Filtrar datos para este módulo
                    df_module_curve = df_curve[df_curve['module'] == module]
                    
                    if not df_module_curve.empty:
                        # Ordenar por voltaje
                        df_module_sorted = df_module_curve.sort_values('voltaje')
                        
                        # Curva de potencia para este módulo
                        fig_power.add_trace(go.Scatter(
                            x=df_module_sorted['voltaje'],
                            y=df_module_sorted['potencia'],
                            mode='lines+markers',
                            name=f'{module} - Potencia',
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4, color=colors[i % len(colors)]),
                            hovertemplate=f'<b>{module}</b><br>' +
                                        '<b>Voltaje:</b> %{x:.4f} V<br>' +
                                        '<b>Potencia:</b> %{y:.4f} W<extra></extra>'
                        ))
                        
                        # Punto de máxima potencia para este módulo
                        max_power_idx = df_module_sorted['potencia'].idxmax()
                        max_power_v = df_module_sorted.loc[max_power_idx, 'voltaje']
                        max_power_p = df_module_sorted.loc[max_power_idx, 'potencia']
                        
                        fig_power.add_trace(go.Scatter(
                            x=[max_power_v],
                            y=[max_power_p],
                            mode='markers',
                            name=f'{module} - PMP',
                            marker=dict(size=10, color=colors[i % len(colors)], symbol='star'),
                            hovertemplate=f'<b>{module} - PMP</b><br>V: {max_power_v:.4f} V<br>P: {max_power_p:.4f} W<extra></extra>'
                        ))
                
                fig_power.update_layout(
                    title="Potencia vs Voltaje Comparativa",
                    xaxis_title="Voltaje (V)",
                    yaxis_title="Potencia (W)",
                    hovermode='closest',
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig_power, use_container_width=True)
        
        # Información adicional
        st.markdown("---")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.subheader("📈 Estadísticas Comparativas")
            st.write(f"**Puntos totales:** {len(df_curve)}")
            
            for module in selected_modules:
                df_module_curve = df_curve[df_curve['module'] == module]
                if not df_module_curve.empty:
                    st.write(f"**{module}:** {len(df_module_curve)} puntos")
                    st.write(f"  Corriente: {df_module_curve['corriente'].min():.4f} - {df_module_curve['corriente'].max():.4f} A")
                    st.write(f"  Voltaje: {df_module_curve['voltaje'].min():.4f} - {df_module_curve['voltaje'].max():.4f} V")
        
        with col4:
            st.subheader("🔋 Comparación de PMP")
            pmp_values = {}
            for module in selected_modules:
                df_module_curve = df_curve[df_curve['module'] == module]
                if not df_module_curve.empty:
                    pmp = df_module_curve['potencia'].max()
                    pmp_values[module] = pmp
                    st.write(f"**{module}:** {pmp:.2f} W")
            
            # Mostrar diferencia si hay más de un módulo
            if len(pmp_values) > 1:
                modules_list = list(pmp_values.keys())
                diff = abs(pmp_values[modules_list[0]] - pmp_values[modules_list[1]])
                st.write(f"**Diferencia:** {diff:.2f} W")
        
        with col5:
            st.subheader("📊 Comparación de Factor de Forma")
            for module in selected_modules:
                df_module_curve = df_curve[df_curve['module'] == module]
                if not df_module_curve.empty:
                    pmp = df_module_curve['potencia'].max()
                    isc = df_module_curve['corriente'].max()
                    voc = df_module_curve['voltaje'].max()
                    ff = (pmp/(isc*voc)*100) if (isc*voc) > 0 else 0
                    
                    st.write(f"**{module}:** {ff:.2f}%")
                    
                    # Indicador visual del factor de forma
                    if ff > 80:
                        st.success(f"✅ {module}: Excelente")
                    elif ff > 70:
                        st.info(f"ℹ️ {module}: Bueno")
                    else:
                        st.warning(f"⚠️ {module}: Bajo")
        
        # Tabla de datos
        st.markdown("---")
        st.subheader("📋 Datos de la Curva")
        
        # Mostrar tabla con los datos de la curva
        display_df = df_curve[['timestamp', 'corriente', 'voltaje', 'potencia']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
        display_df = display_df.round(4)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Botón para descargar datos
        csv = display_df.to_csv(index=False)
        modules_str = "_".join(selected_modules)
        st.download_button(
            label="📥 Descargar datos de la curva",
            data=csv,
            file_name=f"curva_iv_{modules_str}_{selected_date}_{selected_hora.replace(':', '-')}.csv",
            mime="text/csv"
        )
        
    else:
        st.warning(f"⚠️ No hay datos disponibles para la fecha {selected_date}")
else:
    st.error("❌ No hay fechas disponibles para el módulo seleccionado")

# Información general en el footer
st.markdown("---")
st.markdown("""
### 📚 Información del Dashboard

Este dashboard permite visualizar las curvas IV (Intensidad-Voltaje) de los módulos PVStand:

- **Curva IV**: Muestra la relación entre corriente (eje Y) y voltaje (eje X)
- **Curva de Potencia**: Muestra la potencia generada vs voltaje
- **Parámetros Eléctricos**: PMP, ISC, VOC, IMP, VMP y Factor de Forma
- **Datos Crudos**: Tabla con todos los puntos de medición de la curva seleccionada

Los datos provienen de las tablas `iv_curves_perc1_fixed_medio_dia_solar` y `iv_curves_perc2_fixed_medio_dia_solar` de ClickHouse.

**🌐 Desplegado en Streamlit Cloud**
""") 