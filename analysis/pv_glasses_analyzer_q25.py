"""
Análisis de PV Glasses usando Cuantil 25 (Q25)
==============================================

Este script implementa un análisis alternativo de los datos de PV Glasses
usando el cuantil 25 en lugar de promedios, lo que lo hace más robusto
ante outliers y datos anómalos.

Autor: Sistema de Análisis de Soiling
Fecha: 2025-01-13
"""

import os
import sys
import logging
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import re

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_plot_matplotlib(fig, filename: str, output_dir: str, subfolder: str = ""):
    """Guarda un gráfico de matplotlib en el directorio especificado."""
    if subfolder:
        full_output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(full_output_dir, exist_ok=True)
        filepath = os.path.join(full_output_dir, filename)
    else:
        filepath = os.path.join(output_dir, filename)
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico guardado en: {filepath}")

def load_and_process_pv_glasses_data_q25(
    raw_data_path: str,
    usar_mediodia_solar_real: bool = True,
    intervalo_minutos_mediodia: int = 60,
    filtrar_outliers_iqr: bool = False  # Desactivado por defecto para Q25
) -> pl.DataFrame:
    """
    Carga y procesa los datos de PV Glasses usando cuantil 25.
    
    Args:
        raw_data_path: Ruta al archivo CSV raw
        usar_mediodia_solar_real: Si usar filtro de mediodía solar
        intervalo_minutos_mediodia: Ventana en minutos para mediodía solar
        filtrar_outliers_iqr: Si aplicar filtro IQR (recomendado False para Q25)
    
    Returns:
        DataFrame de Polars con datos procesados
    """
    logger.info("Iniciando carga y procesamiento de datos PV Glasses con Q25...")
    
    if not os.path.exists(raw_data_path):
        logger.error(f"Archivo no encontrado: {raw_data_path}")
        return pl.DataFrame()
    
    try:
        # Cargar datos
        logger.info(f"Cargando datos desde {raw_data_path}...")
        df = pl.read_csv(raw_data_path, try_parse_dates=True)
        
        if df.is_empty():
            logger.warning("DataFrame vacío después de cargar.")
            return df
        
        logger.info(f"Datos cargados. Filas totales: {len(df)}")
        
        # Identificar columnas numéricas objetivo
        cols_numericas = ['R_FC1_Avg', 'R_FC2_Avg', 'R_FC3_Avg', 'R_FC4_Avg', 'R_FC5_Avg']
        cols_disponibles = [col for col in cols_numericas if col in df.columns]
        logger.info(f"Columnas numéricas objetivo identificadas: {cols_disponibles}")
        
        # Convertir columna de tiempo
        if '_time' in df.columns:
            logger.info("Convirtiendo columna '_time' a datetime naive...")
            df = df.with_columns(
                pl.col('_time').dt.replace_time_zone(None).alias('_time')
            )
            logger.info("Columna '_time' convertida exitosamente a Datetime naive.")
        
        # Filtro por mediodía solar (opcional)
        if usar_mediodia_solar_real:
            logger.info(f"Filtrando datos usando mediodía solar real +/- {intervalo_minutos_mediodia} minutos.")
            df = aplicar_filtro_mediodia_solar(df, intervalo_minutos_mediodia)
            if df.is_empty():
                logger.warning("DataFrame vacío después del filtro por mediodía solar.")
                return df
            logger.info(f"Filas restantes después del filtrado por mediodía solar: {len(df)}")
        
        # Limpiar datos numéricos básico
        logger.info("Limpiando datos numéricos básicos...")
        for col in cols_disponibles:
            df = df.with_columns(
                pl.col(col).fill_null(0).clip(lower_bound=0).alias(col)
            )
        
        # Calcular REF como promedio de FC1 y FC2
        if 'R_FC1_Avg' in cols_disponibles and 'R_FC2_Avg' in cols_disponibles:
            logger.info("Calculando columna REF como promedio de R_FC1_Avg y R_FC2_Avg...")
            df = df.with_columns(
                ((pl.col('R_FC1_Avg') + pl.col('R_FC2_Avg')) / 2).alias('REF')
            )
            cols_disponibles.append('REF')
        
        # Filtro IQR (opcional, no recomendado para Q25)
        if filtrar_outliers_iqr:
            logger.info("Aplicando filtro IQR...")
            df = aplicar_filtro_iqr(df, cols_disponibles)
            if df.is_empty():
                logger.warning("DataFrame vacío después del filtro IQR.")
                return df
            logger.info(f"Filas restantes después del filtro IQR: {len(df)}")
        
        # Remuestreo diario usando Q25
        logger.info("Remuestreando datos a frecuencia diaria usando Q25...")
        df_daily = remuestrear_diario_q25(df, cols_disponibles)
        logger.info(f"Remuestreo diario Q25 completo. Forma: {df_daily.shape}")
        
        return df_daily
        
    except Exception as e:
        logger.error(f"Error en el procesamiento de datos: {e}", exc_info=True)
        return pl.DataFrame()

def aplicar_filtro_mediodia_solar(df: pl.DataFrame, intervalo_minutos: int) -> pl.DataFrame:
    """Aplica filtro de mediodía solar usando la clase existente."""
    try:
        from classes_codes import medio_dia_solar
        
        if df.is_empty():
            logger.warning("DataFrame vacío antes de calcular mediodía solar.")
            return df
        
        # Obtener rango de fechas
        fechas_df = df.select('_time').to_pandas()
        fecha_min = fechas_df['_time'].min().date()
        fecha_max = fechas_df['_time'].max().date()
        
        logger.info(f"Rango de fechas de datos PV: {fecha_min} a {fecha_max}")
        
        # Calcular intervalos de mediodía solar
        calculador_mediodia = medio_dia_solar(
            datei=fecha_min.strftime('%Y-%m-%d'),
            datef=fecha_max.strftime('%Y-%m-%d'),
            freq="1d",
            inter=intervalo_minutos
        )
        df_intervalos_pd = calculador_mediodia.msd()
        
        logger.info(f"DataFrame de intervalos calculado con {len(df_intervalos_pd)} días")
        
        # Convertir a Polars para filtrado eficiente
        df_intervalos = pl.from_pandas(df_intervalos_pd)
        df_intervalos = df_intervalos.rename({df_intervalos.columns[0]: "inicio", df_intervalos.columns[1]: "fin"})
        
        # Aplicar filtro usando join
        data_df_filtered = None
        for row in df_intervalos.iter_rows(named=True):
            inicio = row['inicio']
            fin = row['fin']
            df_dia = df.filter((pl.col('_time') >= inicio) & (pl.col('_time') <= fin))
            if data_df_filtered is None:
                data_df_filtered = df_dia
            else:
                data_df_filtered = pl.concat([data_df_filtered, df_dia])
        
        return data_df_filtered if data_df_filtered is not None else pl.DataFrame()
        
    except Exception as e:
        logger.error(f"Error al aplicar filtro de mediodía solar: {e}")
        return df

def aplicar_filtro_iqr(df: pl.DataFrame, columnas: list) -> pl.DataFrame:
    """Aplica filtro IQR para eliminar outliers."""
    try:
        # Convertir a pandas para cálculos IQR más fáciles
        df_pd = df.to_pandas()
        
        outliers_indices = set()
        
        for col in columnas:
            if col in df_pd.columns:
                Q1 = df_pd[col].quantile(0.25)
                Q3 = df_pd[col].quantile(0.75)
                IQR = Q3 - Q1
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                
                outliers_col = df_pd[(df_pd[col] < limite_inferior) | (df_pd[col] > limite_superior)].index
                outliers_indices.update(outliers_col)
                
                logger.info(f"Columna {col}: {len(outliers_col)} outliers detectados")
        
        logger.info(f"Total outliers únicos: {len(outliers_indices)}")
        
        # Eliminar outliers
        df_limpio = df_pd.drop(index=outliers_indices)
        
        return pl.from_pandas(df_limpio)
        
    except Exception as e:
        logger.error(f"Error al aplicar filtro IQR: {e}")
        return df

def remuestrear_diario_q25(df: pl.DataFrame, columnas: list) -> pl.DataFrame:
    """Remuestrea los datos a frecuencia diaria usando Q25."""
    try:
        if df.is_empty():
            return df
        
        # Agregar columna de fecha para agrupación
        df_with_date = df.with_columns(
            pl.col('_time').dt.date().alias('fecha')
        )
        
        # Calcular Q25 por día para cada columna
        agg_expressions = [pl.col('_time').first().alias('_time')]  # Mantener una referencia temporal
        
        for col in columnas:
            if col in df.columns:
                agg_expressions.append(
                    pl.col(col).quantile(0.25).alias(col)
                )
        
        df_daily = df_with_date.group_by('fecha').agg(agg_expressions)
        
        # Ordenar por fecha y limpiar
        df_daily = df_daily.sort('_time').drop('fecha')
        
        # Eliminar filas con todos los valores nulos
        df_daily = df_daily.filter(
            ~pl.all_horizontal([pl.col(col).is_null() for col in columnas if col in df_daily.columns])
        )
        
        return df_daily
        
    except Exception as e:
        logger.error(f"Error en remuestreo diario Q25: {e}")
        return df

def analizar_calendario_muestras_q25(
    file_path: str,
    output_csv_dir: str,
    sheet_name: str = "Hoja1"
) -> pd.DataFrame:
    """
    Analiza el calendario de muestras y lo prepara para el análisis Q25.
    Reutiliza la lógica existente pero adaptada.
    """
    logger.info("--- Iniciando Análisis de Calendario de Muestras (Q25) ---")
    
    try:
        # Importar y usar la función existente
        from . import pv_glasses_analyzer
        return pv_glasses_analyzer.analizar_calendario_muestras(file_path, output_csv_dir, sheet_name)
    except Exception as e:
        logger.error(f"Error en análisis de calendario: {e}")
        return pd.DataFrame()

def seleccionar_irradiancia_post_exposicion_q25(
    path_irradiancia_csv: str,
    path_calendario_csv: str,
    output_csv: str
) -> None:
    """
    Selecciona datos de irradiancia post-exposición para análisis Q25.
    """
    logger.info("--- Iniciando Selección de Irradiancia Post-Exposición (Q25) ---")
    
    try:
        # Cargar calendario
        df_calendario = pd.read_csv(path_calendario_csv, parse_dates=['Inicio Exposición', 'Fin Exposicion'])
        df_calendario = df_calendario[df_calendario['Estructura'].fillna('').str.strip() == 'Fija a RC']
        df_calendario.dropna(subset=['Fin Exposicion', 'Periodo', 'Masa A', 'Masa B', 'Masa C'], inplace=True)
        
        logger.info(f"Calendario cargado: {len(df_calendario)} eventos")
        
        # Cargar datos de irradiancia Q25
        df_irradiancia = pl.read_csv(path_irradiancia_csv, try_parse_dates=True)
        if '_time' in df_irradiancia.columns and df_irradiancia['_time'].dtype == pl.Datetime:
            df_irradiancia = df_irradiancia.with_columns(pl.col('_time').dt.date().alias('_time'))
        
        logger.info(f"Datos de irradiancia Q25 cargados: {len(df_irradiancia)} días")
        
        # Seleccionar datos post-exposición
        datos_seleccionados = []
        
        for index, row in df_calendario.iterrows():
            periodo = row['Periodo']
            fecha_fin_evento = row['Fin Exposicion'].date()
            masa_a, masa_b, masa_c = row['Masa A'], row['Masa B'], row['Masa C']
            
            # Lógica específica para diferentes periodos y fechas
            if periodo == '1 año':
                # Grupo anual: 6 días post-exposición
                fecha_inicio_seleccion = fecha_fin_evento + timedelta(days=1)
                num_dias_seleccion = 6
                fecha_fin_seleccion = fecha_inicio_seleccion + timedelta(days=num_dias_seleccion - 1)
                logger.info(f"Grupo anual detectado: seleccionando {num_dias_seleccion} días post-exposición")
            elif periodo == 'Semestral':
                # Lógica específica para grupo Semestral según fecha
                if fecha_fin_evento == datetime(2025, 1, 16).date():
                    # Caso especial: incluir el día de fin de exposición + 2 días posteriores
                    fecha_inicio_seleccion = fecha_fin_evento  # Incluir el mismo día
                    num_dias_seleccion = 3  # 16, 17, 18 (solo 3 días consecutivos)
                    fecha_fin_seleccion = fecha_inicio_seleccion + timedelta(days=num_dias_seleccion - 1)
                    logger.info(f"Grupo Semestral (2025-01-16): seleccionando {num_dias_seleccion} días consecutivos incluyendo día de fin de exposición (2025-01-16 a {fecha_fin_seleccion})")
                elif fecha_fin_evento == datetime(2025, 3, 4).date():
                    # Caso normal: 6 días post-exposición
                    fecha_inicio_seleccion = fecha_fin_evento + timedelta(days=1)
                    num_dias_seleccion = 6
                    fecha_fin_seleccion = fecha_inicio_seleccion + timedelta(days=num_dias_seleccion - 1)
                    logger.info(f"Grupo Semestral (2025-03-04): seleccionando {num_dias_seleccion} días post-exposición")
                else:
                    # Valor por defecto para otras fechas semestrales
                    fecha_inicio_seleccion = fecha_fin_evento + timedelta(days=1)
                    num_dias_seleccion = 5
                    fecha_fin_seleccion = fecha_inicio_seleccion + timedelta(days=num_dias_seleccion - 1)
                    logger.info(f"Grupo Semestral (fecha no específica): seleccionando {num_dias_seleccion} días post-exposición")
            elif periodo == 'Cuatrimestral':
                # Grupo Cuatrimestral: lógica específica según fecha para manejar outliers
                if fecha_fin_evento == datetime(2025, 5, 13).date():
                    # Caso especial para 2025-05-13: excluir días problemáticos (14, 15 y 20)
                    # Seleccionar solo 5 días: 13, 16, 17, 18, 19 (excluyendo 14, 15 y 20)
                    fechas_especificas = [
                        fecha_fin_evento,  # 2025-05-13
                        fecha_fin_evento + timedelta(days=3),  # 2025-05-16 (saltear 14 y 15)
                        fecha_fin_evento + timedelta(days=4),  # 2025-05-17
                        fecha_fin_evento + timedelta(days=5),  # 2025-05-18
                        fecha_fin_evento + timedelta(days=6)   # 2025-05-19 (excluir 20)
                    ]
                    logger.info(f"Grupo Cuatrimestral (2025-05-13): seleccionando 5 días específicos excluyendo outliers (14, 15 y 20 mayo)")
                    logger.info(f"Fechas seleccionadas: {[fecha.strftime('%Y-%m-%d') for fecha in fechas_especificas]}")
                    
                    # Filtrar datos para fechas específicas
                    datos_evento = pl.DataFrame()
                    for fecha_especifica in fechas_especificas:
                        datos_fecha = df_irradiancia.filter(pl.col('_time') == fecha_especifica)
                        if not datos_fecha.is_empty():
                            datos_evento = pl.concat([datos_evento, datos_fecha]) if not datos_evento.is_empty() else datos_fecha
                else:
                    # Otros cuatrimestrales: configuración específica según fecha
                    if fecha_fin_evento == datetime(2025, 2, 25).date():
                        # Caso específico para 2025-02-25: solo 5 días
                        fecha_inicio_seleccion = fecha_fin_evento  # Incluir el día de fin de exposición
                        num_dias_seleccion = 5
                        fecha_fin_seleccion = fecha_inicio_seleccion + timedelta(days=num_dias_seleccion - 1)
                        logger.info(f"Grupo Cuatrimestral (2025-02-25): seleccionando {num_dias_seleccion} días incluyendo día de fin de exposición ({fecha_fin_evento} a {fecha_fin_seleccion})")
                    else:
                        # Otros cuatrimestrales: 6 días incluyendo día de fin de exposición
                        fecha_inicio_seleccion = fecha_fin_evento  # Incluir el día de fin de exposición
                        num_dias_seleccion = 6
                        fecha_fin_seleccion = fecha_inicio_seleccion + timedelta(days=num_dias_seleccion - 1)
                        logger.info(f"Grupo Cuatrimestral detectado: seleccionando {num_dias_seleccion} días incluyendo día de fin de exposición ({fecha_fin_evento} a {fecha_fin_seleccion})")
                    
                    datos_evento = df_irradiancia.filter(
                        (pl.col('_time') >= fecha_inicio_seleccion) & 
                        (pl.col('_time') <= fecha_fin_seleccion)
                    )
            else:
                # Otros periodos: 5 días post-exposición
                fecha_inicio_seleccion = fecha_fin_evento + timedelta(days=1)
                num_dias_seleccion = 5
                fecha_fin_seleccion = fecha_inicio_seleccion + timedelta(days=num_dias_seleccion - 1)
            
            # Para casos que no son el cuatrimestral especial, aplicar filtro normal
            if periodo != 'Cuatrimestral' or fecha_fin_evento != datetime(2025, 5, 13).date():
                datos_evento = df_irradiancia.filter(
                    (pl.col('_time') >= fecha_inicio_seleccion) & 
                    (pl.col('_time') <= fecha_fin_seleccion)
                )
            
            if not datos_evento.is_empty():
                datos_evento = datos_evento.with_columns([
                    pl.lit(periodo).alias('Periodo_Referencia'),
                    pl.lit(fecha_fin_evento.strftime('%Y-%m-%d')).alias('Fecha_Fin_Exposicion_Referencia'),
                    pl.lit(masa_a).alias('Masa_A_Referencia'),
                    pl.lit(masa_b).alias('Masa_B_Referencia'),
                    pl.lit(masa_c).alias('Masa_C_Referencia')
                ])
                datos_seleccionados.append(datos_evento)
            else:
                logger.info(f"No hay datos Q25 para {periodo} con fin en {fecha_fin_evento}")
        
        if datos_seleccionados:
            df_final = pl.concat(datos_seleccionados)
            df_final.write_csv(output_csv)
            logger.info(f"Datos de irradiancia Q25 seleccionados guardados: {output_csv}")
        else:
            logger.warning("No se seleccionaron datos de irradiancia Q25")
            
    except Exception as e:
        logger.error(f"Error en selección de irradiancia Q25: {e}")

def calcular_soiling_ratios_q25(
    path_seleccion_csv: str,
    output_csv: str,
    umbral_irradiancia_ref: int = 300
) -> None:
    """
    Calcula Soiling Ratios usando datos Q25.
    """
    logger.info("--- Iniciando Cálculo de Soiling Ratios Q25 ---")
    
    try:
        df = pl.read_csv(path_seleccion_csv, try_parse_dates=True)
        
        if df.is_empty():
            logger.warning("No hay datos para calcular SR Q25")
            return
        
        logger.info(f"Datos cargados para SR Q25: {len(df)} filas")
        
        # Calcular SR para FC1-FC5
        sr_cols = []
        for i in range(1, 6):
            fc_col = f'R_FC{i}_Avg'
            sr_col = f'SR_R_FC{i}'
            
            if fc_col in df.columns and 'REF' in df.columns:
                df = df.with_columns(
                    (pl.col(fc_col) / pl.col('REF')).alias(sr_col)
                )
                sr_cols.append(sr_col)
                logger.info(f"Calculado {sr_col}")
        
        # Aplicar filtro de irradiancia REF >= umbral
        if 'REF' in df.columns:
            df_filtrado = df.filter(pl.col('REF') >= umbral_irradiancia_ref)
            logger.info(f"Filtro REF >= {umbral_irradiancia_ref}: {len(df_filtrado)} filas restantes")
        else:
            df_filtrado = df
        
        # Guardar resultados
        df_filtrado.write_csv(output_csv)
        logger.info(f"Soiling Ratios Q25 guardados: {output_csv}")
        
        # Estadísticas
        for sr_col in sr_cols:
            if sr_col in df_filtrado.columns:
                valores = df_filtrado[sr_col].drop_nulls()
                if len(valores) > 0:
                    logger.info(f"{sr_col}: media={valores.mean():.3f}, mediana={valores.median():.3f}, std={valores.std():.3f}")
        
    except Exception as e:
        logger.error(f"Error en cálculo de SR Q25: {e}")

def generar_graficos_sr_q25(
    path_sr_csv: str,
    output_graph_dir: str,
    subfolder: str = "pv_glasses_q25"
) -> None:
    """
    Genera gráficos de Soiling Ratios Q25 por periodo.
    """
    logger.info("--- Iniciando Generación de Gráficos SR Q25 ---")
    
    try:
        df = pl.read_csv(path_sr_csv, try_parse_dates=True)
        
        if df.is_empty():
            logger.warning("No hay datos para graficar SR Q25")
            return
        
        # Crear directorio de salida
        full_output_path = os.path.join(output_graph_dir, subfolder)
        os.makedirs(full_output_path, exist_ok=True)
        
        # Mapeo SR-Masa
        correspondencia_sr_masa = {
            'SR_R_FC3': 'Masa_C_Referencia',
            'SR_R_FC4': 'Masa_B_Referencia', 
            'SR_R_FC5': 'Masa_A_Referencia'
        }
        
        # Obtener periodos únicos
        periodos_unicos = df['Periodo_Referencia'].unique().sort()
        logger.info(f"Periodos únicos encontrados: {periodos_unicos}")
        
        # Datos para gráfico de barras
        datos_grafico_barras = []
        
        # Generar gráfico por periodo
        for periodo in periodos_unicos:
            logger.info(f"--- Generando gráfico Q25 para periodo: {periodo} ---")
            
            df_periodo = df.filter(pl.col('Periodo_Referencia') == periodo)
            
            if df_periodo.is_empty():
                logger.warning(f"No hay datos Q25 para periodo {periodo}")
                continue
            
            # Calcular estadísticas Q25 para el gráfico de barras
            promedios_periodo = {'Periodo': periodo}
            
            # Crear gráfico individual del periodo
            fig, ax = plt.subplots(figsize=(15, 8))
            
            for sr_col, masa_col in correspondencia_sr_masa.items():
                if sr_col in df_periodo.columns and masa_col in df_periodo.columns:
                    df_filtrado = df_periodo.filter(pl.col(masa_col) > 0)
                    
                    if not df_filtrado.is_empty():
                        fechas = df_filtrado['_time'].to_pandas()
                        valores_sr = df_filtrado[sr_col].to_pandas()
                        
                        ax.plot(fechas, valores_sr, 'o-', label=f"{sr_col} (Masa {masa_col[-1]} > 0)", markersize=6)
                        
                        # Calcular Q25 para el gráfico de barras
                        q25_valor = valores_sr.quantile(0.25)
                        promedios_periodo[f'Q25_{sr_col}'] = q25_valor
                        
                        logger.info(f"{sr_col}: {len(valores_sr)} puntos, Q25={q25_valor:.3f}")
            
            datos_grafico_barras.append(promedios_periodo)
            
            # Configurar gráfico del periodo
            ax.set_title(f"Soiling Ratios Q25 - Periodo: {periodo}", fontsize=14)
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Soiling Ratio (SR)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            fig.autofmt_xdate()
            
            # Guardar gráfico del periodo
            nombre_archivo = f"SR_Q25_Periodo_{periodo.replace(' ', '_').lower()}_MasasCorregidas.png"
            save_plot_matplotlib(fig, nombre_archivo, output_graph_dir, subfolder=subfolder)
            plt.close(fig)
        
        # Generar gráfico de barras Q25
        if datos_grafico_barras:
            generar_grafico_barras_q25(datos_grafico_barras, output_graph_dir, subfolder)
        
    except Exception as e:
        logger.error(f"Error generando gráficos SR Q25: {e}")

def generar_grafico_barras_q25(datos_barras: list, output_graph_dir: str, subfolder: str):
    """Genera gráfico de barras con valores Q25 por periodo."""
    try:
        df_barras = pd.DataFrame(datos_barras)
        
        if df_barras.empty:
            logger.warning("No hay datos para gráfico de barras Q25")
            return
        
        # Orden de periodos
        orden_periodos = ['semanal', '2 semanas', 'Mensual', 'Trimestral', 'Cuatrimestral', 'Semestral', '1 año']
        df_filtrado = df_barras[df_barras['Periodo'].isin(orden_periodos)].copy()
        
        if df_filtrado.empty:
            logger.warning("No hay periodos válidos para gráfico de barras Q25")
            return
        
        df_filtrado['Periodo'] = pd.Categorical(df_filtrado['Periodo'], categories=orden_periodos, ordered=True)
        df_filtrado = df_filtrado.sort_values('Periodo').set_index('Periodo')
        
        # Columnas Q25 para graficar
        cols_q25 = [col for col in df_filtrado.columns if col.startswith('Q25_SR_R_FC')]
        
        if not cols_q25:
            logger.warning("No se encontraron columnas Q25 para graficar")
            return
        
        # Crear gráfico de barras
        fig, ax = plt.subplots(figsize=(14, 8))
        df_filtrado[cols_q25].plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Cuantil 25 (Q25) de Soiling Ratios por Periodo de Exposición')
        ax.set_ylabel('Q25 Soiling Ratio (SR)')
        ax.set_xlabel('Periodo de Referencia')
        
        # Etiquetas de leyenda
        legend_labels = [col.replace('Q25_', '').replace('SR_R_', 'SR ') for col in cols_q25]
        ax.legend(title='Tipos de SR', labels=legend_labels)
        
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        # Añadir valores en las barras
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if pd.notna(height) and height != 0:
                    ax.annotate(f'{height:.3f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Guardar
        filename = "SR_Q25_por_Periodo_Barras.png"
        save_plot_matplotlib(fig, filename, output_graph_dir, subfolder=subfolder)
        plt.close(fig)
        
        logger.info("Gráfico de barras Q25 generado exitosamente")
        
    except Exception as e:
        logger.error(f"Error generando gráfico de barras Q25: {e}")

def ejecutar_analisis_pv_glasses_q25(
    raw_data_path: str = "datos/raw_pv_glasses_data.csv",
    calendario_path: str = "datos/20241114 Calendario toma de muestras soiling.xlsx",
    output_csv_dir: str = "datos_procesados_analisis_integrado_py",
    output_graph_dir: str = "graficos_analisis_integrado_py"
) -> None:
    """
    Ejecuta el análisis completo de PV Glasses usando Q25.
    """
    logger.info("=== INICIANDO ANÁLISIS PV GLASSES Q25 ===")
    
    try:
        # 1. Procesar datos raw con Q25
        logger.info("Paso 1: Procesando datos raw con Q25...")
        df_q25 = load_and_process_pv_glasses_data_q25(
            raw_data_path=raw_data_path,
            usar_mediodia_solar_real=True,
            intervalo_minutos_mediodia=60,
            filtrar_outliers_iqr=False  # No usar IQR con Q25
        )
        
        if df_q25.is_empty():
            logger.error("No se pudieron procesar los datos con Q25")
            return
        
        # Guardar datos Q25 procesados
        q25_data_path = os.path.join(output_csv_dir, "pv_glasses_q25", "datos_q25_diarios.csv")
        os.makedirs(os.path.dirname(q25_data_path), exist_ok=True)
        df_q25.write_csv(q25_data_path)
        logger.info(f"Datos Q25 guardados: {q25_data_path}")
        
        # 2. Analizar calendario
        logger.info("Paso 2: Analizando calendario de muestras...")
        calendario_csv = os.path.join(output_csv_dir, "calendario_muestras_seleccionado.csv")
        
        if not os.path.exists(calendario_csv):
            analizar_calendario_muestras_q25(calendario_path, output_csv_dir)
        
        # 3. Seleccionar irradiancia post-exposición
        logger.info("Paso 3: Seleccionando irradiancia post-exposición Q25...")
        seleccion_q25_csv = os.path.join(output_csv_dir, "pv_glasses_q25", "seleccion_irradiancia_q25.csv")
        seleccionar_irradiancia_post_exposicion_q25(q25_data_path, calendario_csv, seleccion_q25_csv)
        
        # 4. Calcular Soiling Ratios Q25
        logger.info("Paso 4: Calculando Soiling Ratios Q25...")
        sr_q25_csv = os.path.join(output_csv_dir, "pv_glasses_q25", "soiling_ratios_q25.csv")
        calcular_soiling_ratios_q25(seleccion_q25_csv, sr_q25_csv)
        
        # 5. Generar gráficos
        logger.info("Paso 5: Generando gráficos Q25...")
        generar_graficos_sr_q25(sr_q25_csv, output_graph_dir)
        
        logger.info("=== ANÁLISIS PV GLASSES Q25 COMPLETADO ===")
        
    except Exception as e:
        logger.error(f"Error en análisis PV Glasses Q25: {e}", exc_info=True)

if __name__ == "__main__":
    # Ejecutar análisis Q25
    ejecutar_analisis_pv_glasses_q25()
