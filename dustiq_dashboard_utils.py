import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats
import streamlit as st

def calcular_tendencia_dustiq(x, y):
    """
    Calcula la línea de tendencia, pendiente y R² para una serie de datos.
    
    Args:
        x: Array de valores x (números ordinales de fechas)
        y: Array de valores y (soiling ratio)
    
    Returns:
        tuple: (slope, intercept, r_squared) o (None, None, None) si falla
    """
    if len(x) < 2 or len(y) < 2:
        return None, None, None
    
    try:
        # Eliminar valores NaN
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if valid_mask.sum() < 2:
            return None, None, None
            
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        r_squared = r_value ** 2
        return slope, intercept, r_squared
    except Exception as e:
        st.error(f"Error calculando tendencia: {e}")
        return None, None, None

def calculate_solar_noon_approximate(date, longitude=-70.6693, latitude=-33.4489):
    """
    Calcula una aproximación del mediodía solar para una fecha dada.
    
    Args:
        date: Fecha (datetime.date)
        longitude: Longitud en grados (default: Santiago, Chile)
        latitude: Latitud en grados (default: Santiago, Chile)
    
    Returns:
        datetime: Hora aproximada del mediodía solar
    """
    # Aproximación simple basada en la ecuación del tiempo
    # Para Chile, el mediodía solar varía entre 12:00 y 12:30 UTC aproximadamente
    
    # Día del año
    day_of_year = date.timetuple().tm_yday
    
    # Corrección por la ecuación del tiempo (aproximada)
    B = 2 * np.pi * (day_of_year - 81) / 365
    E = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    
    # Corrección por longitud
    time_correction = 4 * (longitude - (-75)) + E  # -75° es aproximadamente el meridiano estándar de Chile
    
    # Mediodía solar local
    solar_noon_minutes = 12 * 60 + time_correction
    
    # Convertir a horas y minutos
    hours = int(solar_noon_minutes // 60)
    minutes = int(solar_noon_minutes % 60)
    
    # Crear timestamp
    try:
        solar_noon = pd.Timestamp.combine(date, pd.Timestamp(f'{hours:02d}:{minutes:02d}:00').time())
        return solar_noon
    except:
        # Fallback a 12:15 si hay error
        return pd.Timestamp.combine(date, pd.Timestamp('12:15:00').time())

def procesar_datos_franjas_horarias(df_sr, franjas_dict, freq='1W', stat_func=None):
    """
    Procesa datos de soiling ratio por franjas horarias.
    
    Args:
        df_sr: Serie de pandas con datos de soiling ratio (índice datetime)
        franjas_dict: Diccionario con franjas {'label': ('start_time', 'end_time')}
        freq: Frecuencia de agregación ('1W' para semanal, '1D' para diario)
        stat_func: Función estadística (default: quantile(0.25))
    
    Returns:
        dict: Diccionario con datos procesados por franja
    """
    if stat_func is None:
        stat_func = lambda x: x.quantile(0.25)
    
    resultados = {}
    
    for label, (start_time, end_time) in franjas_dict.items():
        try:
            # Filtrar por franja horaria
            data_franja = df_sr.between_time(start_time, end_time)
            
            if not data_franja.empty:
                # Agregar según frecuencia
                data_agregado = data_franja.resample(freq, origin='start').apply(stat_func)
                data_agregado = data_agregado.dropna()
                
                # Filtrar fechas desde 23 de julio de 2024
                data_agregado = data_agregado[data_agregado.index >= pd.Timestamp('2024-07-23')]
                
                if not data_agregado.empty:
                    resultados[label] = {
                        'data': data_agregado,
                        'stats': {
                            'count': len(data_agregado),
                            'mean': data_agregado.mean(),
                            'std': data_agregado.std(),
                            'min': data_agregado.min(),
                            'max': data_agregado.max()
                        }
                    }
        except Exception as e:
            st.warning(f"Error procesando franja {label}: {e}")
            continue
    
    return resultados

def procesar_datos_mediodia_solar(df_sr, duracion_ventana_minutos=60, freq='1W', stat_func=None):
    """
    Procesa datos de soiling ratio para análisis de mediodía solar.
    
    Args:
        df_sr: Serie de pandas con datos de soiling ratio
        duracion_ventana_minutos: Duración de la ventana alrededor del mediodía solar
        freq: Frecuencia de agregación
        stat_func: Función estadística
    
    Returns:
        pandas.Series: Datos agregados por mediodía solar
    """
    if stat_func is None:
        stat_func = lambda x: x.quantile(0.25)
    
    # Obtener fechas únicas
    fechas_unicas = df_sr.index.date
    fechas_unicas = pd.Series(fechas_unicas).drop_duplicates().sort_values()
    
    datos_mediodia = []
    
    for fecha in fechas_unicas:
        try:
            # Calcular mediodía solar para esta fecha
            solar_noon = calculate_solar_noon_approximate(fecha)
            
            # Definir ventana temporal
            ventana_inicio = solar_noon - timedelta(minutes=duracion_ventana_minutos)
            ventana_fin = solar_noon + timedelta(minutes=duracion_ventana_minutos)
            
            # Filtrar datos del día
            datos_dia = df_sr[df_sr.index.date == fecha]
            
            if not datos_dia.empty:
                # Filtrar datos en la ventana del mediodía solar
                datos_ventana = datos_dia[
                    (datos_dia.index >= ventana_inicio) & 
                    (datos_dia.index <= ventana_fin)
                ]
                
                if not datos_ventana.empty:
                    datos_mediodia.append({
                        'fecha': pd.Timestamp(fecha),
                        'sr_value': stat_func(datos_ventana),
                        'puntos': len(datos_ventana),
                        'solar_noon': solar_noon
                    })
        except Exception as e:
            continue
    
    if not datos_mediodia:
        return pd.Series(dtype=float)
    
    # Convertir a DataFrame y luego a Serie
    df_mediodia = pd.DataFrame(datos_mediodia)
    df_mediodia.set_index('fecha', inplace=True)
    
    # Reagrupar según la frecuencia especificada
    if freq == '1W':
        return df_mediodia['sr_value'].resample('1W', origin='start').mean().dropna()
    elif freq == '1D':
        return df_mediodia['sr_value']
    else:
        return df_mediodia['sr_value'].resample(freq, origin='start').mean().dropna()

def crear_grafico_franjas_horarias(datos_procesados, titulo, mostrar_tendencias=True, height=500):
    """
    Crea un gráfico de Plotly para análisis de franjas horarias.
    
    Args:
        datos_procesados: Diccionario con datos procesados por franja
        titulo: Título del gráfico
        mostrar_tendencias: Si mostrar líneas de tendencia
        height: Altura del gráfico
    
    Returns:
        plotly.graph_objects.Figure: Gráfico de Plotly
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    trend_info = []
    
    for i, (label, datos) in enumerate(datos_procesados.items()):
        data_series = datos['data']
        color = colors[i % len(colors)]
        
        # Agregar serie principal
        fig.add_trace(go.Scatter(
            x=data_series.index,
            y=data_series.values,
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=f'<b>{label}</b><br>Fecha: %{{x}}<br>SR: %{{y:.2f}}%<extra></extra>'
        ))
        
        # Agregar línea de tendencia si está habilitada
        if mostrar_tendencias and len(data_series) >= 2:
            x_num = np.array([d.toordinal() for d in data_series.index])
            y_vals = data_series.values
            
            slope, intercept, r_squared = calcular_tendencia_dustiq(x_num, y_vals)
            
            if slope is not None:
                y_trend = slope * x_num + intercept
                
                fig.add_trace(go.Scatter(
                    x=data_series.index,
                    y=y_trend,
                    mode='lines',
                    name=f'{label} - Tendencia',
                    line=dict(color=color, width=2, dash='dash'),
                    hovertemplate=f'<b>Tendencia {label}</b><br>Pendiente: {slope*7:.4f}%/sem<br>R²: {r_squared:.3f}<extra></extra>',
                    showlegend=False
                ))
                
                trend_info.append({
                    'Franja': label,
                    'Pendiente (%/sem)': f"{slope*7:.4f}",
                    'Pendiente (%/año)': f"{slope*365:.2f}",
                    'R²': f"{r_squared:.3f}",
                    'Puntos': len(data_series)
                })
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Fecha",
        yaxis_title="Soiling Ratio (%)",
        hovermode='closest',
        height=height,
        yaxis=dict(range=[90, 110]),
        showlegend=True
    )
    
    return fig, trend_info

def crear_grafico_mediodia_solar(data_series, titulo, mostrar_tendencia=True, height=500):
    """
    Crea un gráfico para análisis de mediodía solar.
    
    Args:
        data_series: Serie de pandas con datos
        titulo: Título del gráfico
        mostrar_tendencia: Si mostrar línea de tendencia
        height: Altura del gráfico
    
    Returns:
        tuple: (figura de Plotly, información de tendencia)
    """
    fig = go.Figure()
    
    # Agregar datos principales
    fig.add_trace(go.Scatter(
        x=data_series.index,
        y=data_series.values,
        mode='lines+markers',
        name='SR Mediodía Solar',
        line=dict(color='orange', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Mediodía Solar</b><br>Fecha: %{x}<br>SR: %{y:.2f}%<extra></extra>'
    ))
    
    trend_info = None
    
    # Agregar línea de tendencia
    if mostrar_tendencia and len(data_series) >= 2:
        x_num = np.array([d.toordinal() for d in data_series.index])
        y_vals = data_series.values
        
        slope, intercept, r_squared = calcular_tendencia_dustiq(x_num, y_vals)
        
        if slope is not None:
            y_trend = slope * x_num + intercept
            
            fig.add_trace(go.Scatter(
                x=data_series.index,
                y=y_trend,
                mode='lines',
                name='Tendencia',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate=f'<b>Tendencia</b><br>Pendiente: {slope*7:.4f}%/sem<br>R²: {r_squared:.3f}<extra></extra>'
            ))
            
            trend_info = {
                'slope_weekly': slope * 7,
                'slope_annual': slope * 365,
                'r_squared': r_squared,
                'points': len(data_series),
                'slope_daily': slope
            }
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Fecha",
        yaxis_title="Soiling Ratio (%)",
        hovermode='closest',
        height=height,
        yaxis=dict(range=[90, 110])
    )
    
    return fig, trend_info

def generar_estadisticas_dustiq(df_sr, datos_procesados=None):
    """
    Genera estadísticas generales de los datos de DustIQ.
    
    Args:
        df_sr: Serie original de soiling ratio
        datos_procesados: Datos procesados opcionales
    
    Returns:
        dict: Diccionario con estadísticas
    """
    stats = {
        'total_puntos': len(df_sr),
        'rango_fechas': f"{df_sr.index.min().strftime('%Y-%m-%d')} a {df_sr.index.max().strftime('%Y-%m-%d')}",
        'promedio_sr': df_sr.mean(),
        'mediana_sr': df_sr.median(),
        'min_sr': df_sr.min(),
        'max_sr': df_sr.max(),
        'std_sr': df_sr.std(),
        'q25_sr': df_sr.quantile(0.25),
        'q75_sr': df_sr.quantile(0.75)
    }
    
    if datos_procesados:
        stats['franjas_analizadas'] = len(datos_procesados)
        stats['franjas_con_datos'] = sum(1 for datos in datos_procesados.values() if len(datos['data']) > 0)
    
    return stats

def crear_tabla_estadisticas(trend_info_list):
    """
    Crea una tabla de estadísticas para mostrar en Streamlit.
    
    Args:
        trend_info_list: Lista de información de tendencias
    
    Returns:
        pandas.DataFrame: DataFrame para mostrar como tabla
    """
    if not trend_info_list:
        return pd.DataFrame()
    
    return pd.DataFrame(trend_info_list)

def validar_datos_dustiq(df):
    """
    Valida los datos de DustIQ cargados.
    
    Args:
        df: DataFrame de DustIQ
    
    Returns:
        tuple: (es_valido, mensajes_validacion)
    """
    mensajes = []
    es_valido = True
    
    if df is None or df.empty:
        mensajes.append("❌ DataFrame está vacío o es None")
        return False, mensajes
    
    # Verificar índice datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        mensajes.append("⚠️ El índice no es DatetimeIndex")
        es_valido = False
    
    # Verificar columnas esperadas
    columnas_esperadas = ['SR_C11_Avg', 'SR_C12_Avg']
    for col in columnas_esperadas:
        if col not in df.columns:
            mensajes.append(f"⚠️ Columna '{col}' no encontrada")
        else:
            # Verificar valores válidos
            valores_validos = df[col].notna().sum()
            total_valores = len(df[col])
            porcentaje_validos = (valores_validos / total_valores) * 100
            
            mensajes.append(f"✅ {col}: {valores_validos:,}/{total_valores:,} valores válidos ({porcentaje_validos:.1f}%)")
    
    # Verificar rango de fechas
    if not df.empty:
        fecha_min = df.index.min()
        fecha_max = df.index.max()
        mensajes.append(f"📅 Rango de fechas: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
    
    return es_valido, mensajes

def export_data_to_csv(data, filename):
    """
    Prepara datos para exportar a CSV.
    
    Args:
        data: Datos a exportar (DataFrame o Series)
        filename: Nombre del archivo
    
    Returns:
        str: Datos en formato CSV
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    return data.to_csv() 