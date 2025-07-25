#!/usr/bin/env python3
"""
Análisis de Soiling Kit - Cálculo de Soiling Ratio y Visualizaciones
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging

# Configurar logging
logger = logging.getLogger(__name__)

def load_soiling_kit_data_from_clickhouse(client, start_date, end_date):
    """Carga datos del Soiling Kit desde ClickHouse."""
    try:
        # Convertir fechas al formato correcto para ClickHouse
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Consultar datos del Soiling Kit
        query = f"""
        SELECT 
            Stamptime,
            Attribute,
            Measure
        FROM PSDA.soilingkit 
        WHERE Stamptime >= '{start_str}' AND Stamptime <= '{end_str}'
        AND Attribute IN ('Isc(e)', 'Isc(p)', 'Te(C)', 'Tp(C)')
        ORDER BY Stamptime, Attribute
        """
        
        logger.info("Consultando datos del Soiling Kit desde ClickHouse...")
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("No se encontraron datos del Soiling Kit en ClickHouse")
            return None
            
        logger.info(f"Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame
        df_soilingkit = pd.DataFrame(result.result_set, columns=['Stamptime', 'Attribute', 'Measure'])
        
        # Convertir Stamptime a datetime (timezone-naive para evitar problemas)
        df_soilingkit['Stamptime'] = pd.to_datetime(df_soilingkit['Stamptime'])
        if df_soilingkit['Stamptime'].dt.tz is not None:
            df_soilingkit['Stamptime'] = df_soilingkit['Stamptime'].dt.tz_localize(None)

        # Pivotar los datos para convertir de long format a wide format
        logger.info("Pivotando datos de long format a wide format...")

        # Manejar duplicados agregando por promedio
        df_soilingkit_grouped = df_soilingkit.groupby(['Stamptime', 'Attribute'])['Measure'].mean().reset_index()

        # Hacer el pivot sin duplicados
        df_soilingkit_pivot = df_soilingkit_grouped.pivot(index='Stamptime', columns='Attribute', values='Measure')
        df_soilingkit_pivot.index.name = 'timestamp'
        
        logger.info(f"Procesamiento completado: {len(df_soilingkit_pivot)} registros únicos")
        return df_soilingkit_pivot
        
    except Exception as e:
        logger.error(f"Error en la carga de datos del Soiling Kit: {e}")
        return None

def calculate_soiling_ratio(df_soilingkit):
    """Calcula el Soiling Ratio a partir de los datos del Soiling Kit."""
    try:
        # Verificar que tenemos las columnas necesarias
        required_columns = ['Isc(e)', 'Isc(p)']
        if not all(col in df_soilingkit.columns for col in required_columns):
            logger.error(f"Columnas requeridas no encontradas. Disponibles: {list(df_soilingkit.columns)}")
            return None
        
        # Filtrar datos válidos (ambos valores > 0)
        df_valid = df_soilingkit[
            (df_soilingkit['Isc(e)'] > 0) & 
            (df_soilingkit['Isc(p)'] > 0)
        ].copy()
        
        if len(df_valid) == 0:
            logger.warning("No hay datos válidos para calcular Soiling Ratio")
            return None
        
        # Calcular Soiling Ratio (SR = Isc(p) / Isc(e) * 100)
        df_valid['SR'] = (df_valid['Isc(p)'] / df_valid['Isc(e)']) * 100
        
        # Calcular SR con corrección de temperatura si están disponibles
        if 'Te(C)' in df_valid.columns and 'Tp(C)' in df_valid.columns:
            # Coeficiente de temperatura típico para celdas de referencia (~0.05%/°C)
            temp_coeff = 0.0005  # 0.05% por grado Celsius
            
            # Corrección de temperatura: SR_corr = SR * (1 + temp_coeff * (Tp - Te))
            df_valid['SR_corr'] = df_valid['SR'] * (
                1 + temp_coeff * (df_valid['Tp(C)'] - df_valid['Te(C)'])
            )
            
            logger.info("Soiling Ratio calculado con y sin corrección de temperatura")
        else:
            logger.info("Soiling Ratio calculado sin corrección de temperatura")
        
        return df_valid
        
    except Exception as e:
        logger.error(f"Error en el cálculo del Soiling Ratio: {e}")
        return None

def process_soiling_data_by_frequency(df_sr, frequency='1D', franjas=None):
    """Procesa los datos de Soiling Ratio por frecuencia y franjas horarias."""
    try:
        if df_sr is None or len(df_sr) == 0:
            return None
        
        resultados = {}
        
        # Procesar por franjas horarias si se especifican
        if franjas:
            for franja in franjas:
                if franja == "Mediodía Solar":
                    # Mediodía solar (11:30-13:30)
                    data_franja = df_sr.between_time('11:30', '13:30')
                    if not data_franja.empty:
                        data_procesada = data_franja.resample(frequency, origin='start').quantile(0.25)
                        resultados["Mediodía Solar (11:30-13:30)"] = data_procesada
                else:
                    # Franjas horarias específicas
                    if franja == "10:00-11:00":
                        data_franja = df_sr.between_time('10:00', '11:00')
                    elif franja == "12:00-13:00":
                        data_franja = df_sr.between_time('12:00', '13:00')
                    elif franja == "14:00-15:00":
                        data_franja = df_sr.between_time('14:00', '15:00')
                    elif franja == "16:00-17:00":
                        data_franja = df_sr.between_time('16:00', '17:00')
                    else:
                        continue
                    
                    if not data_franja.empty:
                        data_procesada = data_franja.resample(frequency, origin='start').quantile(0.25)
                        resultados[franja] = data_procesada
        else:
            # Procesar todos los datos
            data_procesada = df_sr.resample(frequency, origin='start').quantile(0.25)
            resultados["Todos los datos"] = data_procesada
        
        return resultados
        
    except Exception as e:
        logger.error(f"Error en el procesamiento por frecuencia: {e}")
        return None

def create_soiling_ratio_plots(df_sr, frequency='1D', franjas=None):
    """Crea gráficos de Soiling Ratio con y sin corrección."""
    try:
        if df_sr is None or len(df_sr) == 0:
            return None, None
        
        # Procesar datos por frecuencia y franjas
        datos_procesados = process_soiling_data_by_frequency(df_sr, frequency, franjas)
        
        if not datos_procesados:
            return None, None
        
        # Gráfico 1: Soiling Ratio con y sin corrección
        fig_sr = go.Figure()
        
        colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (franja, datos) in enumerate(datos_procesados.items()):
            if not datos.empty:
                color = colores[i % len(colores)]
                
                # SR sin corrección
                if 'SR' in datos.columns:
                    fig_sr.add_trace(go.Scatter(
                        x=datos.index,
                        y=datos['SR'],
                        mode='lines+markers',
                        name=f'{franja} - SR',
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        opacity=0.8
                    ))
                
                # SR con corrección (si está disponible)
                if 'SR_corr' in datos.columns:
                    fig_sr.add_trace(go.Scatter(
                        x=datos.index,
                        y=datos['SR_corr'],
                        mode='lines+markers',
                        name=f'{franja} - SR Corregido',
                        line=dict(color=color, width=2, dash='dash'),
                        marker=dict(size=4),
                        opacity=0.8
                    ))
        
        # Línea de referencia al 100%
        fig_sr.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="Referencia 100%",
            annotation_position="top right"
        )
        
        freq_name = "Diario" if frequency == "1D" else "Semanal" if frequency == "1W" else "Mensual"
        fig_sr.update_layout(
            title=f"Soiling Ratio - {freq_name}",
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
        
        # Gráfico 2: Temperaturas
        fig_temp = go.Figure()
        
        # Verificar si hay datos de temperatura
        if 'Te(C)' in df_sr.columns and 'Tp(C)' in df_sr.columns:
            # Procesar temperaturas por frecuencia
            temp_data = df_sr[['Te(C)', 'Tp(C)']].resample(frequency, origin='start').mean()
            
            fig_temp.add_trace(go.Scatter(
                x=temp_data.index,
                y=temp_data['Te(C)'],
                mode='lines+markers',
                name='Te(C) - Celda Limpia',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig_temp.add_trace(go.Scatter(
                x=temp_data.index,
                y=temp_data['Tp(C)'],
                mode='lines+markers',
                name='Tp(C) - Celda Sucia',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ))
            
            fig_temp.update_layout(
                title=f"Temperaturas - {freq_name}",
                xaxis_title="Fecha",
                yaxis_title="Temperatura (°C)",
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        else:
            fig_temp = None
        
        return fig_sr, fig_temp
        
    except Exception as e:
        logger.error(f"Error en la creación de gráficos: {e}")
        return None, None

def create_weekly_q25_plot(df_sr):
    """Crea el gráfico semanal Q25 específico del Soiling Kit (como sk_sr_q25_semanal.png)."""
    try:
        if df_sr is None or len(df_sr) == 0:
            return None
        
        # Calcular SR semanal Q25
        sr_weekly_raw_q25 = None
        sr_weekly_corrected_q25 = None
        
        if 'SR' in df_sr.columns:
            sr_weekly_raw_q25 = df_sr['SR'].resample('W').quantile(0.25).dropna()
        
        if 'SR_corr' in df_sr.columns:
            sr_weekly_corrected_q25 = df_sr['SR_corr'].resample('W').quantile(0.25).dropna()
        
        if sr_weekly_raw_q25 is None and sr_weekly_corrected_q25 is None:
            logger.warning("No hay datos para generar gráfico semanal Q25")
            return None
        
        # Crear gráfico
        fig = go.Figure()
        
        # Colores como en el código original
        color_raw = '#1f77b4'  # Azul
        color_corr = '#ff7f0e'  # Naranja
        
        # Agregar SR Raw
        if sr_weekly_raw_q25 is not None and not sr_weekly_raw_q25.empty:
            fig.add_trace(go.Scatter(
                x=sr_weekly_raw_q25.index,
                y=sr_weekly_raw_q25.values,
                mode='lines+markers',
                name='SR Raw (Q25 Semanal)',
                line=dict(color=color_raw, width=2),
                marker=dict(size=6, color=color_raw),
                opacity=0.8
            ))
        
        # Agregar SR Corregido
        if sr_weekly_corrected_q25 is not None and not sr_weekly_corrected_q25.empty:
            fig.add_trace(go.Scatter(
                x=sr_weekly_corrected_q25.index,
                y=sr_weekly_corrected_q25.values,
                mode='lines+markers',
                name='SR Corregido por Temp. (Q25 Semanal)',
                line=dict(color=color_corr, width=2),
                marker=dict(size=6, color=color_corr),
                opacity=0.8
            ))
        
        # Configurar layout como en el código original
        fig.update_layout(
            title='Soiling Kit - SR Semanal Q25',
            xaxis_title='Fecha',
            yaxis_title='Soiling Ratio [%]',
            height=500,
            yaxis=dict(range=[90, 110]),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            )
        )
        
        # Agregar grid a los ejes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Formatear eje X para mostrar fechas claramente
        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            tickangle=30,
            tickmode='auto',
            nticks=10
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error en la creación del gráfico semanal Q25: {e}")
        return None

def calculate_soiling_statistics(df_sr):
    """Calcula estadísticas del Soiling Ratio."""
    try:
        if df_sr is None or len(df_sr) == 0:
            return None
        
        stats = {}
        
        # Estadísticas generales
        if 'SR' in df_sr.columns:
            stats['SR'] = {
                'promedio': df_sr['SR'].mean(),
                'mediana': df_sr['SR'].median(),
                'desv_std': df_sr['SR'].std(),
                'minimo': df_sr['SR'].min(),
                'maximo': df_sr['SR'].max(),
                'perdida_promedio': 100 - df_sr['SR'].mean()
            }
        
        if 'SR_corr' in df_sr.columns:
            stats['SR_corr'] = {
                'promedio': df_sr['SR_corr'].mean(),
                'mediana': df_sr['SR_corr'].median(),
                'desv_std': df_sr['SR_corr'].std(),
                'minimo': df_sr['SR_corr'].min(),
                'maximo': df_sr['SR_corr'].max(),
                'perdida_promedio': 100 - df_sr['SR_corr'].mean()
            }
        
        # Estadísticas de temperatura si están disponibles
        if 'Te(C)' in df_sr.columns and 'Tp(C)' in df_sr.columns:
            stats['temperatura'] = {
                'Te_promedio': df_sr['Te(C)'].mean(),
                'Tp_promedio': df_sr['Tp(C)'].mean(),
                'diferencia_temp': df_sr['Tp(C)'].mean() - df_sr['Te(C)'].mean()
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error en el cálculo de estadísticas: {e}")
        return None 