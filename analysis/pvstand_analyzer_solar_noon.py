# analysis/pvstand_analyzer_solar_noon.py
# Versión del analizador PVStand que usa medio día solar dinámico

import os
import sys
import matplotlib.pyplot as plt
import logging

# Agregar el directorio raíz del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import timezone, timedelta

from config.logging_config import logger
from config import paths, settings
from utils.helpers import normalize_series_from_date_pd
from utils.solar_time import UtilsMedioDiaSolar

def save_plot_matplotlib(fig, filename_base, output_dir, subfolder=None, dpi=300):
    """
    Guarda una figura de Matplotlib en el directorio especificado, opcionalmente en un subdirectorio.
    """
    full_output_dir = output_dir
    if subfolder:
        full_output_dir = os.path.join(output_dir, subfolder)
    os.makedirs(full_output_dir, exist_ok=True)

    filepath = os.path.join(full_output_dir, filename_base)
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logging.info(f"Gráfico guardado en: {filepath}")
    except Exception as e:
        logging.error(f"Error al guardar el gráfico en {filepath}: {e}")
    finally:
        plt.close(fig)

def filter_by_solar_noon(df, hours_window=2.5):
    """
    Filtra un DataFrame por rangos de medio día solar dinámicos.
    
    Args:
        df: DataFrame con índice DatetimeIndex en UTC
        hours_window: Ventana en horas alrededor del medio día solar (total = 2 * hours_window)
    
    Returns:
        DataFrame filtrado por medio día solar
    """
    if df.empty:
        return df
        
    logger.info(f"Aplicando filtro de medio día solar (±{hours_window} horas alrededor del medio día solar real)")
    
    # Obtener rango de fechas del DataFrame
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    
    # Inicializar utilidades de medio día solar con parámetros correctos
    solar_utils = UtilsMedioDiaSolar(
        datei=start_date,
        datef=end_date,
        freq='D',
        inter=int(hours_window * 2 * 60),  # Convertir horas a minutos
        tz_local_str=settings.DUSTIQ_LOCAL_TIMEZONE_STR,
        lat=settings.SITE_LATITUDE,
        lon=settings.SITE_LONGITUDE,
        alt=settings.SITE_ALTITUDE
    )
    
    # Obtener intervalos de medio día solar
    solar_intervals_df = solar_utils.msd()
    
    if solar_intervals_df.empty:
        logger.warning("No se pudieron calcular intervalos de medio día solar. Retornando DataFrame vacío.")
        return pd.DataFrame()
    
    # Crear máscara para filtrar por medio día solar
    mask = pd.Series(False, index=df.index)
    
    # Aplicar cada intervalo de medio día solar
    for _, row in solar_intervals_df.iterrows():
        start_time = pd.Timestamp(row[0], tz='UTC')
        end_time = pd.Timestamp(row[1], tz='UTC')
        
        # Aplicar máscara para este intervalo
        interval_mask = (df.index >= start_time) & (df.index <= end_time)
        mask = mask | interval_mask
        
        logger.debug(f"Intervalo medio día solar: {start_time} - {end_time}")
    
    filtered_df = df[mask]
    logger.info(f"Filtro de medio día solar aplicado: {len(df)} -> {len(filtered_df)} puntos ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df

def analyze_pvstand_data_solar_noon(
    pv_iv_data_filepath: str, 
    temperature_data_filepath: str,
    solar_noon_window_hours: float = 2.5
) -> bool:
    """
    Analiza los datos de PVStand IV usando filtrado por medio día solar dinámico.
    
    Args:
        pv_iv_data_filepath: Ruta al archivo CSV de datos IV de PVStand 
        temperature_data_filepath: Ruta al archivo CSV de datos de temperatura procesados
        solar_noon_window_hours: Ventana en horas alrededor del medio día solar (±)
        
    Returns:
        True si el análisis fue exitoso, False en caso contrario.
    """
    logger.info("=== INICIO DEL ANÁLISIS PVSTAND CON MEDIO DÍA SOLAR ===")
    logger.info(f"Archivo de datos IV: {pv_iv_data_filepath}")
    logger.info(f"Archivo de datos de temperatura: {temperature_data_filepath}")
    logger.info(f"Ventana de medio día solar: ±{solar_noon_window_hours} horas")
    logger.info(f"Coordenadas del sitio: Lat={settings.SITE_LATITUDE}°, Lon={settings.SITE_LONGITUDE}°")
    
    # Verificar que los archivos existan
    if not os.path.exists(pv_iv_data_filepath):
        logger.error(f"El archivo de datos IV no existe: {pv_iv_data_filepath}")
        return False
        
    if not os.path.exists(temperature_data_filepath):
        logger.error(f"El archivo de datos de temperatura no existe: {temperature_data_filepath}")
        return False

    # Parámetros de configuración
    output_csv_dir = paths.PVSTAND_OUTPUT_SUBDIR_CSV
    output_graph_dir = paths.BASE_OUTPUT_GRAPH_DIR

    filter_start_date_str = settings.PVSTAND_ANALYSIS_START_DATE_STR
    filter_end_date_str = settings.PVSTAND_ANALYSIS_END_DATE_STR
    
    pv_module_soiled_id = settings.PVSTAND_MODULE_SOILED_ID
    pv_module_reference_id = settings.PVSTAND_MODULE_REFERENCE_ID
    temp_sensor_soiled_col = settings.PVSTAND_TEMP_SENSOR_SOILED_COL
    temp_sensor_reference_col = settings.PVSTAND_TEMP_SENSOR_REFERENCE_COL
    
    alpha_isc_corr = settings.PVSTAND_ALPHA_ISC_CORR
    beta_pmax_corr = settings.PVSTAND_BETA_PMAX_CORR
    temp_ref_correction_c = settings.PVSTAND_TEMP_REF_CORRECTION_C
    
    normalize_sr_flag = settings.PVSTAND_NORMALIZE_SR_FLAG
    normalize_sr_ref_date_str = settings.PVSTAND_NORMALIZE_SR_REF_DATE_STR
    pmax_sr_offset = settings.PVSTAND_PMAX_SR_OFFSET
    
    sr_min_filter_threshold = settings.PVSTAND_SR_MIN_FILTER_THRESHOLD
    sr_max_filter_threshold = settings.PVSTAND_SR_MAX_FILTER_THRESHOLD
    
    save_figures_setting = settings.SAVE_FIGURES
    show_figures_setting = settings.SHOW_FIGURES
    
    resample_freq_minutes = settings.PVSTAND_RESAMPLE_FREQ_MINUTES
    graph_quantile = settings.PVSTAND_GRAPH_QUANTILE
    
    # Crear directorio de salida específico para análisis solar noon
    output_csv_dir_solar = os.path.join(output_csv_dir, "solar_noon")
    os.makedirs(output_csv_dir_solar, exist_ok=True)

    try:
        start_date_dt = pd.Timestamp(filter_start_date_str, tz='UTC')
        end_date_dt = pd.Timestamp(filter_end_date_str, tz='UTC')
        logger.info(f"Periodo de análisis: {start_date_dt.date()} a {end_date_dt.date()}")
    except Exception as e:
        logger.error(f"Error al parsear fechas desde settings: {e}. Abortando análisis.")
        return False

    # --- Cargar datos PVStand IV ---
    df_pvstand_raw_data = pd.DataFrame()
    if os.path.exists(pv_iv_data_filepath):
        logger.info("Cargando datos PVStand IV...")
        try:
            use_cols_iv = ['_time', '_measurement', 'Imax', 'Pmax', 'Umax']
            df_pvstand_raw_data = pd.read_csv(
                pv_iv_data_filepath, 
                usecols=lambda c: c in use_cols_iv or c.startswith('_time')
            )
            
            time_col_actual = [col for col in df_pvstand_raw_data.columns if col.startswith('_time')]
            if not time_col_actual:
                logger.error("No se encontró la columna '_time' en los datos IV de PVStand.")
                return False
            if time_col_actual[0] != '_time':
                df_pvstand_raw_data.rename(columns={time_col_actual[0]: '_time'}, inplace=True)

            df_pvstand_raw_data['_time'] = pd.to_datetime(
                df_pvstand_raw_data['_time'], 
                errors='coerce', 
                format=settings.PVSTAND_IV_DATA_TIME_FORMAT
            )
            df_pvstand_raw_data.dropna(subset=['_time'], inplace=True)
            df_pvstand_raw_data.set_index('_time', inplace=True)
            logger.info(f"Datos PVStand IV cargados: {len(df_pvstand_raw_data)} filas iniciales.")

            # Asegurar zona horaria UTC
            if df_pvstand_raw_data.index.tz is None:
                df_pvstand_raw_data.index = df_pvstand_raw_data.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            elif df_pvstand_raw_data.index.tz != timezone.utc:
                df_pvstand_raw_data.index = df_pvstand_raw_data.index.tz_convert('UTC')
            
            logger.info(f"Zona horaria del índice PVStand asegurada a UTC: {df_pvstand_raw_data.index.tz}")

            # Pivotar datos si es necesario
            if '_measurement' in df_pvstand_raw_data.columns:
                logger.info("Pivotando datos PVStand IV por '_measurement'...")
                values_to_pivot = [col for col in ['Imax', 'Pmax', 'Umax'] if col in df_pvstand_raw_data.columns]
                if not values_to_pivot:
                    logger.error("Ninguna de las columnas de valores (Imax, Pmax, Umax) encontradas para pivotar.")
                    return False
                
                df_pvstand_pivot = df_pvstand_raw_data.pivot_table(
                    index=df_pvstand_raw_data.index, 
                    columns='_measurement', 
                    values=values_to_pivot
                )
                df_pvstand_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pvstand_pivot.columns]
                df_pvstand_raw_data = df_pvstand_pivot
                logger.info(f"Datos PVStand IV pivotados. {len(df_pvstand_raw_data)} filas. Columnas ejemplo: {df_pvstand_raw_data.columns[:5].tolist()}...")

            # Filtrar por rango de fechas
            start_date_only = pd.Timestamp(start_date_dt.date(), tz='UTC')
            end_date_only = pd.Timestamp(end_date_dt.date(), tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_pvstand_raw_data = df_pvstand_raw_data[(df_pvstand_raw_data.index >= start_date_only) & (df_pvstand_raw_data.index <= end_date_only)]
            
            # *** APLICAR FILTRO DE MEDIO DÍA SOLAR EN LUGAR DE HORARIO FIJO ***
            df_pvstand_raw_data = filter_by_solar_noon(df_pvstand_raw_data, solar_noon_window_hours)
            
            logger.info(f"Datos PVStand IV filtrados por medio día solar: {len(df_pvstand_raw_data)} puntos.")
            if df_pvstand_raw_data.empty:
                logger.warning("PVStand IV vacío después de filtro de medio día solar. No se puede continuar.")
                return False
            
        except Exception as e:
            logger.error(f"Error cargando/preprocesando PVStand IV: {e}", exc_info=True)
            return False
    else:
        logger.error(f"Archivo PVStand IV no encontrado: {pv_iv_data_filepath}")
        return False

    # --- Cargar datos de Temperatura ---
    logger.info(f"Cargando datos de Temperatura preprocesados desde: {temperature_data_filepath}")
    df_temp_processed = pd.DataFrame()
    if temperature_data_filepath and os.path.exists(temperature_data_filepath):
        try:
            df_temp_intermediate = pd.read_csv(temperature_data_filepath)
            
            time_col = df_temp_intermediate.columns[0]
            df_temp_intermediate[time_col] = pd.to_datetime(df_temp_intermediate[time_col], errors='coerce')
            df_temp_intermediate.set_index(time_col, inplace=True)
            
            # Eliminar columnas innecesarias
            cols_to_remove = ['_start', '_stop', '_measurement']
            cols_found_to_remove = [col for col in cols_to_remove if col in df_temp_intermediate.columns]
            if cols_found_to_remove:
                df_temp_intermediate.drop(columns=cols_found_to_remove, inplace=True)
                logger.info(f"Eliminadas columnas innecesarias: {cols_found_to_remove}")
            
            logger.info(f"Archivo de temperatura cargado con {len(df_temp_intermediate)} filas.")
            
            if df_temp_intermediate.index.isna().any():
                logger.warning(f"Eliminando {df_temp_intermediate.index.isna().sum()} filas con timestamps inválidos.")
                df_temp_intermediate = df_temp_intermediate[~df_temp_intermediate.index.isna()]

            if not df_temp_intermediate.empty:
                # Convertir columnas de temperatura a numérico
                sensor_cols = [col for col in df_temp_intermediate.columns if col.startswith('1TE') and col.endswith('(C)')]
                if sensor_cols:
                    logger.info(f"Convirtiendo columnas de sensores de temperatura a numérico: {sensor_cols}")
                    for col in sensor_cols:
                        df_temp_intermediate[col] = pd.to_numeric(df_temp_intermediate[col], errors='coerce')
                
                df_temp_processed = df_temp_intermediate
                
                # Asegurar zona horaria UTC
                if isinstance(df_temp_processed.index, pd.DatetimeIndex):
                    if df_temp_processed.index.tz is None:
                        df_temp_processed.index = df_temp_processed.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
                    elif df_temp_processed.index.tz != timezone.utc:
                        df_temp_processed.index = df_temp_processed.index.tz_convert('UTC')
                
                # Filtrar por rango de fechas
                df_temp_processed = df_temp_processed.sort_index()
                df_temp_processed = df_temp_processed[(df_temp_processed.index >= start_date_only) & (df_temp_processed.index <= end_date_only)]
                
                # *** APLICAR FILTRO DE MEDIO DÍA SOLAR A TEMPERATURA ***
                df_temp_processed = filter_by_solar_noon(df_temp_processed, solar_noon_window_hours)
                
                logger.info(f"Datos de Temperatura filtrados por medio día solar: {len(df_temp_processed)} filas.")
                if not df_temp_processed.empty:
                    logger.info(f"Rango de tiempo datos de Temperatura: {df_temp_processed.index.min()} a {df_temp_processed.index.max()}")
                    logger.info(f"Columnas de temperatura disponibles: {sensor_cols}")
            
        except Exception as e:
            logger.error(f"Error cargando/preprocesando datos de Temperatura: {e}", exc_info=True)
    
    # --- Remuestreo ---
    df_pvstand_resampled = pd.DataFrame()
    if not df_pvstand_raw_data.empty:
        logger.info(f"Remuestreando datos PVStand IV a {resample_freq_minutes} minuto(s) (promedio)...")
        df_pvstand_resampled = df_pvstand_raw_data.resample(f'{resample_freq_minutes}min').mean()
        logger.info(f"PVStand IV remuestreado: {len(df_pvstand_resampled)} puntos.")
        if not df_pvstand_resampled.empty:
            logger.info(f"Rango de tiempo PVStand IV (remuestreado): {df_pvstand_resampled.index.min()} a {df_pvstand_resampled.index.max()}")
    else:
        logger.warning("Datos PVStand IV (filtrados por medio día solar) están vacíos. No se puede remuestrear.")
        return False

    df_temp_resampled = pd.DataFrame()
    if not df_temp_processed.empty:
        logger.info(f"Remuestreando datos de Temperatura a {resample_freq_minutes} minuto(s) (promedio)...")
        df_temp_resampled = df_temp_processed.resample(f'{resample_freq_minutes}min').mean()
        logger.info(f"Temperatura remuestreada: {len(df_temp_resampled)} puntos.")
        if not df_temp_resampled.empty:
            logger.info(f"Rango de tiempo Temperatura (remuestreada): {df_temp_resampled.index.min()} a {df_temp_resampled.index.max()}")
    else:
        logger.warning("Datos de Temperatura (filtrados por medio día solar) están vacíos. No se puede remuestrear.")

    # --- Definición de Columnas para SR ---
    col_isc_soiled = f'{pv_module_soiled_id}_Imax'
    col_pmax_soiled = f'{pv_module_soiled_id}_Pmax'
    col_isc_reference = f'{pv_module_reference_id}_Imax'
    col_pmax_reference = f'{pv_module_reference_id}_Pmax'

    logger.info(f"Columnas para SR: Soiled Isc='{col_isc_soiled}', Pmax='{col_pmax_soiled}'")
    logger.info(f"Columnas para SR: Reference Isc='{col_isc_reference}', Pmax='{col_pmax_reference}'")
    logger.info(f"Columnas de temp para corrección: Soiled='{temp_sensor_soiled_col}', Reference='{temp_sensor_reference_col}'")

    # Inicializar Series para SRs
    sr_isc_pvstand_raw_no_offset = pd.Series(dtype=float, name="SR_Isc_Uncorrected_Raw_NoOffset")
    sr_pmax_pvstand_raw_no_offset = pd.Series(dtype=float, name="SR_Pmax_Uncorrected_Raw_NoOffset")
    sr_isc_pvstand_corrected_raw_no_offset = pd.Series(dtype=float, name="SR_Isc_Corrected_Raw_NoOffset")
    sr_pmax_pvstand_corrected_raw_no_offset = pd.Series(dtype=float, name="SR_Pmax_Corrected_Raw_NoOffset")

    # --- Cálculo de SR SIN corrección de temperatura ---
    if not df_pvstand_resampled.empty:
        logger.info("Calculando SRs SIN corrección de temperatura (datos filtrados por medio día solar)...")
        if col_isc_soiled in df_pvstand_resampled.columns and col_isc_reference in df_pvstand_resampled.columns:
            # Filtrar valores anormalmente bajos en referencia que causan SRs extremos
            isc_reference_filtered = df_pvstand_resampled[col_isc_reference].copy()
            isc_soiled_filtered = df_pvstand_resampled[col_isc_soiled].copy()
            
            # Filtrar también por Pmax para mantener consistencia de datos
            if col_pmax_soiled in df_pvstand_resampled.columns and col_pmax_reference in df_pvstand_resampled.columns:
                pmax_outlier_threshold = 170.0  # Watts
                pmax_consistency_mask = (df_pvstand_resampled[col_pmax_soiled] >= pmax_outlier_threshold) & (df_pvstand_resampled[col_pmax_reference] >= pmax_outlier_threshold)
                isc_soiled_filtered = isc_soiled_filtered[pmax_consistency_mask]
                isc_reference_filtered = isc_reference_filtered[pmax_consistency_mask]
                logger.info(f"Aplicado filtro de consistencia Pmax (>={pmax_outlier_threshold}W) a datos Isc. Datos restantes: {len(isc_soiled_filtered)}")
            
            # Excluir casos donde la referencia es < 0.5A (anormalmente baja para Isc)
            valid_isc_reference_mask = isc_reference_filtered >= 0.5
            isc_reference_filtered = isc_reference_filtered[valid_isc_reference_mask]
            isc_soiled_filtered = isc_soiled_filtered[valid_isc_reference_mask]
            
            denom_isc = isc_reference_filtered.replace(0, np.nan)
            if denom_isc.count() > 0:
                sr_isc_calc_raw = isc_soiled_filtered.div(denom_isc)
                
                # Aplicar filtro variable por fecha: 102% para ago-sep, 100% para oct en adelante
                early_months_cutoff = pd.Timestamp('2024-10-01', tz='UTC')
                early_mask = sr_isc_calc_raw.index < early_months_cutoff
                late_mask = sr_isc_calc_raw.index >= early_months_cutoff
                
                # Filtro para primeros meses (ago-sep): hasta 102%
                sr_isc_early = sr_isc_calc_raw[early_mask]
                sr_isc_early_filtered = sr_isc_early[(sr_isc_early.notna()) & (sr_isc_early >= sr_min_filter_threshold) & (sr_isc_early <= 1.02)]
                
                # Filtro para meses posteriores (oct en adelante): hasta 100%
                sr_isc_late = sr_isc_calc_raw[late_mask]
                sr_isc_late_filtered = sr_isc_late[(sr_isc_late.notna()) & (sr_isc_late >= sr_min_filter_threshold) & (sr_isc_late <= 1.0)]
                
                # Combinar ambos filtros
                sr_isc_filtered = pd.concat([sr_isc_early_filtered, sr_isc_late_filtered]).sort_index()
                
                sr_isc_pvstand_raw_no_offset = (100 * sr_isc_filtered).rename("SR_Isc_Uncorrected_Raw_NoOffset")
                logger.info(f"SR Isc (sin corregir, raw) con MEDIO DÍA SOLAR calculado: {len(sr_isc_pvstand_raw_no_offset)} puntos válidos.")
        else: 
            logger.warning(f"Columnas faltantes para SR Isc sin corregir: {col_isc_soiled} o {col_isc_reference}")

        if col_pmax_soiled in df_pvstand_resampled.columns and col_pmax_reference in df_pvstand_resampled.columns:
            # Filtrar valores anormalmente bajos en referencia que causan SRs extremos
            pmax_reference_filtered = df_pvstand_resampled[col_pmax_reference].copy()
            pmax_soiled_filtered = df_pvstand_resampled[col_pmax_soiled].copy()
            
            # Filtrar valores de Pmax menores a 170W (considerados outliers) - ANTES de otros filtros
            pmax_outlier_threshold = 170.0  # Watts
            pmax_outlier_mask = (pmax_soiled_filtered >= pmax_outlier_threshold) & (pmax_reference_filtered >= pmax_outlier_threshold)
            pmax_soiled_filtered = pmax_soiled_filtered[pmax_outlier_mask]
            pmax_reference_filtered = pmax_reference_filtered[pmax_outlier_mask]
            logger.info(f"Filtrados outliers de Pmax (<{pmax_outlier_threshold}W) en cálculo SIN corrección con MEDIO DÍA SOLAR. Datos restantes: {len(pmax_soiled_filtered)}")
            
            # Excluir casos donde la referencia es < 200W (anormalmente baja para este sistema)
            valid_reference_mask = pmax_reference_filtered >= 200.0
            pmax_reference_filtered = pmax_reference_filtered[valid_reference_mask]
            pmax_soiled_filtered = pmax_soiled_filtered[valid_reference_mask]
            
            denom_pmax = pmax_reference_filtered.replace(0, np.nan)
            if denom_pmax.count() > 0:
                sr_pmax_calc_raw = pmax_soiled_filtered.div(denom_pmax)
                
                # Aplicar filtro variable por fecha: 102% para ago-sep, 100% para oct en adelante
                early_months_cutoff = pd.Timestamp('2024-10-01', tz='UTC')
                early_mask = sr_pmax_calc_raw.index < early_months_cutoff
                late_mask = sr_pmax_calc_raw.index >= early_months_cutoff
                
                # Filtro para primeros meses (ago-sep): hasta 102%
                sr_pmax_early = sr_pmax_calc_raw[early_mask]
                sr_pmax_early_filtered = sr_pmax_early[(sr_pmax_early.notna()) & (sr_pmax_early >= sr_min_filter_threshold) & (sr_pmax_early <= 1.02)]
                
                # Filtro para meses posteriores (oct en adelante): hasta 100%
                sr_pmax_late = sr_pmax_calc_raw[late_mask]
                sr_pmax_late_filtered = sr_pmax_late[(sr_pmax_late.notna()) & (sr_pmax_late >= sr_min_filter_threshold) & (sr_pmax_late <= 1.0)]
                
                # Combinar ambos filtros
                sr_pmax_filtered = pd.concat([sr_pmax_early_filtered, sr_pmax_late_filtered]).sort_index()
                
                sr_pmax_pvstand_raw_no_offset = (100 * sr_pmax_filtered).rename("SR_Pmax_Uncorrected_Raw_NoOffset")
                logger.info(f"SR Pmax (sin corregir, raw, ANTES de offset) con MEDIO DÍA SOLAR calculado: {len(sr_pmax_pvstand_raw_no_offset)} puntos válidos.")
        else: 
            logger.warning(f"Columnas faltantes para SR Pmax sin corregir: {col_pmax_soiled} o {col_pmax_reference}")
    else:
        logger.warning("PVStand IV remuestreado está vacío. No se pueden calcular SRs sin corrección.")
        return False

    # --- Merge y Cálculo de SR CON corrección de temperatura ---
    df_merged_for_corr = pd.DataFrame()
    if not df_pvstand_resampled.empty and not df_temp_resampled.empty:
        logger.info("Alineando datos PVStand (filtrados por medio día solar) con Temperatura usando merge_asof (nearest)...")
        df_merged_for_corr = pd.merge_asof(
            df_pvstand_resampled.sort_index(), 
            df_temp_resampled.sort_index(), 
            left_index=True, 
            right_index=True, 
            direction='nearest', 
            tolerance=pd.Timedelta(minutes=resample_freq_minutes)
        )
        logger.info(f"Datos PVStand y Temperatura alineados (df_merged_for_corr ANTES dropna temp cols): {len(df_merged_for_corr)} puntos.")

        if temp_sensor_soiled_col in df_merged_for_corr.columns and temp_sensor_reference_col in df_merged_for_corr.columns:
             # Filtrar valores nulos Y valores cero anómalos en temperatura
             df_merged_for_corr.dropna(subset=[temp_sensor_soiled_col, temp_sensor_reference_col], how='any', inplace=True)
             
             # Filtrar temperaturas cero o anormalmente bajas (< 5°C) que causan correcciones extremas
             temp_valid_mask = (df_merged_for_corr[temp_sensor_soiled_col] > 5.0) & (df_merged_for_corr[temp_sensor_reference_col] > 5.0)
             df_merged_for_corr = df_merged_for_corr[temp_valid_mask]
             logger.info(f"Filtrados valores de temperatura anómalos (≤5°C). Datos restantes: {len(df_merged_for_corr)}")
             
             # Filtrar temperaturas extremadamente altas (>80°C) que pueden ser errores de sensor
             temp_high_valid_mask = (df_merged_for_corr[temp_sensor_soiled_col] <= 80.0) & (df_merged_for_corr[temp_sensor_reference_col] <= 80.0)
             df_merged_for_corr = df_merged_for_corr[temp_high_valid_mask]
             logger.info(f"Filtradas temperaturas extremadamente altas (>80°C). Datos restantes: {len(df_merged_for_corr)}")
             
             # Filtrar valores de Pmax menores a 170W (considerados outliers)
             pmax_outlier_threshold = 170.0  # Watts
             pmax_valid_mask = (df_merged_for_corr[col_pmax_soiled] >= pmax_outlier_threshold) & (df_merged_for_corr[col_pmax_reference] >= pmax_outlier_threshold)
             df_merged_for_corr = df_merged_for_corr[pmax_valid_mask]
             logger.info(f"Filtrados outliers de Pmax (<{pmax_outlier_threshold}W) en datos con MEDIO DÍA SOLAR. Datos restantes: {len(df_merged_for_corr)}")
        else:
            logger.warning(f"Columnas de temperatura ({temp_sensor_soiled_col}, {temp_sensor_reference_col}) no encontradas en df_merged_for_corr.")
        
        logger.info(f"Datos PVStand y Temperatura alineados (df_merged_for_corr DESPUÉS dropna temp cols): {len(df_merged_for_corr)} puntos.")
        
        if df_merged_for_corr.empty: 
            logger.warning("El DataFrame fusionado para corrección de temperatura está vacío después de dropna. No se calcularán SRs corregidos.")
        else:
            logger.info("Calculando SRs CON corrección de temperatura (datos filtrados por medio día solar)...")
            required_cols_corr = [col_isc_soiled, col_isc_reference, col_pmax_soiled, col_pmax_reference, temp_sensor_soiled_col, temp_sensor_reference_col]
            if all(c in df_merged_for_corr.columns for c in required_cols_corr):
                # Corrección de temperatura para Isc
                isc_soiled_corr_val = df_merged_for_corr[col_isc_soiled] / (1 + alpha_isc_corr * (df_merged_for_corr[temp_sensor_soiled_col] - temp_ref_correction_c))
                isc_ref_corr_val = df_merged_for_corr[col_isc_reference] / (1 + alpha_isc_corr * (df_merged_for_corr[temp_sensor_reference_col] - temp_ref_correction_c))
                
                # Filtrar valores anormalmente bajos en referencia corregida que causan SRs extremos
                valid_isc_reference_corr_mask = isc_ref_corr_val >= 0.5
                isc_soiled_corr_filtered = isc_soiled_corr_val[valid_isc_reference_corr_mask]
                isc_ref_corr_filtered = isc_ref_corr_val[valid_isc_reference_corr_mask]
                
                denom_isc_c = isc_ref_corr_filtered.replace(0, np.nan)
                if denom_isc_c.count() > 0:
                    sr_isc_c_calc_raw = isc_soiled_corr_filtered.div(denom_isc_c)
                    # Aplicar filtro variable por fecha: 102% para ago-sep, 100% para oct en adelante
                    early_months_cutoff = pd.Timestamp('2024-10-01', tz='UTC')
                    early_mask = sr_isc_c_calc_raw.index < early_months_cutoff
                    late_mask = sr_isc_c_calc_raw.index >= early_months_cutoff
                    
                    # Filtro para primeros meses (ago-sep): hasta 102%
                    sr_isc_c_early = sr_isc_c_calc_raw[early_mask]
                    sr_isc_c_early_filtered = sr_isc_c_early[(sr_isc_c_early.notna()) & (sr_isc_c_early >= sr_min_filter_threshold) & (sr_isc_c_early <= 1.02)]
                    
                    # Filtro para meses posteriores (oct en adelante): hasta 100%
                    sr_isc_c_late = sr_isc_c_calc_raw[late_mask]
                    sr_isc_c_late_filtered = sr_isc_c_late[(sr_isc_c_late.notna()) & (sr_isc_c_late >= sr_min_filter_threshold) & (sr_isc_c_late <= 1.0)]
                    
                    # Combinar ambos filtros
                    sr_isc_c_filtered = pd.concat([sr_isc_c_early_filtered, sr_isc_c_late_filtered]).sort_index()
                    
                    sr_isc_pvstand_corrected_raw_no_offset = (100 * sr_isc_c_filtered).rename("SR_Isc_Corrected_Raw_NoOffset")
                    logger.info(f"SR Isc (corregido, raw) con MEDIO DÍA SOLAR calculado: {len(sr_isc_pvstand_corrected_raw_no_offset)} puntos válidos.")

                # Corrección de temperatura para Pmax
                pmax_soiled_corr_val = df_merged_for_corr[col_pmax_soiled] / (1 + beta_pmax_corr * (df_merged_for_corr[temp_sensor_soiled_col] - temp_ref_correction_c))
                pmax_ref_corr_val = df_merged_for_corr[col_pmax_reference] / (1 + beta_pmax_corr * (df_merged_for_corr[temp_sensor_reference_col] - temp_ref_correction_c))
                
                # Filtrar valores anormalmente bajos en referencia corregida que causan SRs extremos
                valid_reference_corr_mask = pmax_ref_corr_val >= 200.0
                pmax_soiled_corr_filtered = pmax_soiled_corr_val[valid_reference_corr_mask]
                pmax_ref_corr_filtered = pmax_ref_corr_val[valid_reference_corr_mask]
                
                denom_pmax_c = pmax_ref_corr_filtered.replace(0, np.nan)
                if denom_pmax_c.count() > 0:
                    sr_pmax_c_calc_raw = pmax_soiled_corr_filtered.div(denom_pmax_c)
                    # Aplicar filtro variable por fecha: 102% para ago-sep, 100% para oct en adelante
                    early_months_cutoff = pd.Timestamp('2024-10-01', tz='UTC')
                    early_mask = sr_pmax_c_calc_raw.index < early_months_cutoff
                    late_mask = sr_pmax_c_calc_raw.index >= early_months_cutoff
                    
                    # Filtro para primeros meses (ago-sep): hasta 102%
                    sr_pmax_c_early = sr_pmax_c_calc_raw[early_mask]
                    sr_pmax_c_early_filtered = sr_pmax_c_early[(sr_pmax_c_early.notna()) & (sr_pmax_c_early >= sr_min_filter_threshold) & (sr_pmax_c_early <= 1.02)]
                    
                    # Filtro para meses posteriores (oct en adelante): hasta 100%
                    sr_pmax_c_late = sr_pmax_c_calc_raw[late_mask]
                    sr_pmax_c_late_filtered = sr_pmax_c_late[(sr_pmax_c_late.notna()) & (sr_pmax_c_late >= sr_min_filter_threshold) & (sr_pmax_c_late <= 1.0)]
                    
                    # Combinar ambos filtros
                    sr_pmax_c_filtered = pd.concat([sr_pmax_c_early_filtered, sr_pmax_c_late_filtered]).sort_index()
                    
                    sr_pmax_pvstand_corrected_raw_no_offset = (100 * sr_pmax_c_filtered).rename("SR_Pmax_Corrected_Raw_NoOffset")
                    logger.info(f"SR Pmax (corregido, raw, ANTES de offset) con MEDIO DÍA SOLAR calculado: {len(sr_pmax_pvstand_corrected_raw_no_offset)} puntos válidos.")
            else:
                missing_actual = [c for c in required_cols_corr if c not in df_merged_for_corr.columns]
                logger.warning(f"Columnas faltantes en df_merged_for_corr para SR corregido: {missing_actual}. No se calcularán SRs corregidos.")
    else:
        logger.warning("Uno o ambos DataFrames (PVStand IV remuestreado, Temperatura remuestreada) están vacíos. No se puede realizar merge ni calcular SRs corregidos.")

    # --- Preparación de SRs para guardado y graficado ---
    # Copiar _raw_no_offset a las series base que podrían recibir offset
    sr_isc_pvstand = sr_isc_pvstand_raw_no_offset.copy()
    sr_pmax_pvstand = sr_pmax_pvstand_raw_no_offset.copy()
    sr_isc_pvstand_corrected = sr_isc_pvstand_corrected_raw_no_offset.copy()
    sr_pmax_pvstand_corrected = sr_pmax_pvstand_corrected_raw_no_offset.copy()

    if pmax_sr_offset != 0:
        logger.info(f"Aplicando offset de {pmax_sr_offset}% a SR Pmax (no corregido y corregido si existen).")
        if not sr_pmax_pvstand.empty: sr_pmax_pvstand += pmax_sr_offset
        if not sr_pmax_pvstand_corrected.empty: sr_pmax_pvstand_corrected += pmax_sr_offset
    
    sr_isc_pvstand.name = "SR_Isc_Uncorrected"
    sr_pmax_pvstand.name = "SR_Pmax_Uncorrected_Offset"
    sr_isc_pvstand_corrected.name = "SR_Isc_Corrected"
    sr_pmax_pvstand_corrected.name = "SR_Pmax_Corrected_Offset"

    sr_isc_pvstand_no_norm = sr_isc_pvstand.copy().rename("SR_Isc_Uncorrected_NoNorm")
    sr_pmax_pvstand_no_norm = sr_pmax_pvstand.copy().rename("SR_Pmax_Uncorrected_Offset_NoNorm")
    sr_isc_pvstand_corrected_no_norm = sr_isc_pvstand_corrected.copy().rename("SR_Isc_Corrected_NoNorm")
    sr_pmax_pvstand_corrected_no_norm = sr_pmax_pvstand_corrected.copy().rename("SR_Pmax_Corrected_Offset_NoNorm")
    
    if normalize_sr_flag:
        logger.info(f"Normalizando SRs principales (con offset si aplica) usando fecha de referencia: {normalize_sr_ref_date_str}...")
        if not sr_isc_pvstand.empty: sr_isc_pvstand = normalize_series_from_date_pd(sr_isc_pvstand, normalize_sr_ref_date_str, sr_isc_pvstand.name, target_ref_value=100.0)
        if not sr_pmax_pvstand.empty: sr_pmax_pvstand = normalize_series_from_date_pd(sr_pmax_pvstand, normalize_sr_ref_date_str, sr_pmax_pvstand.name, target_ref_value=100.0)
        if not sr_isc_pvstand_corrected.empty: sr_isc_pvstand_corrected = normalize_series_from_date_pd(sr_isc_pvstand_corrected, normalize_sr_ref_date_str, sr_isc_pvstand_corrected.name, target_ref_value=100.0)
        if not sr_pmax_pvstand_corrected.empty: sr_pmax_pvstand_corrected = normalize_series_from_date_pd(sr_pmax_pvstand_corrected, normalize_sr_ref_date_str, sr_pmax_pvstand_corrected.name, target_ref_value=100.0)

    # --- Guardar CSVs ---
    pvstand_graph_subdir_name = "pvstand_solar_noon" # Subdirectorio específico para gráficos de medio día solar
    graph_base_plus_subdir = os.path.join(output_graph_dir, pvstand_graph_subdir_name)
    os.makedirs(graph_base_plus_subdir, exist_ok=True)

    # --- Guardar datos utilizados para cálculos de SR (Temperatura + Potencias) ---
    if not df_merged_for_corr.empty:
        # Extraer datos de temperatura y potencias utilizados para los cálculos
        df_data_used_for_sr = df_merged_for_corr[[
            col_pmax_soiled, col_pmax_reference, col_isc_soiled, col_isc_reference,
            temp_sensor_soiled_col, temp_sensor_reference_col
        ]].copy()
        
        # Renombrar columnas para mayor claridad
        df_data_used_for_sr.rename(columns={
            col_pmax_soiled: 'Pmax_Soiled_Original_W',
            col_pmax_reference: 'Pmax_Reference_Original_W',
            col_isc_soiled: 'Isc_Soiled_Original_A',
            col_isc_reference: 'Isc_Reference_Original_A',
            temp_sensor_soiled_col: 'Temp_Modulo_Soiled_C',
            temp_sensor_reference_col: 'Temp_Modulo_Reference_C'
        }, inplace=True)
        
        # Calcular diferencia de temperatura
        df_data_used_for_sr['Diferencia_Temperatura_C'] = abs(
            df_data_used_for_sr['Temp_Modulo_Soiled_C'] - 
            df_data_used_for_sr['Temp_Modulo_Reference_C']
        )
        
        # Calcular factores de corrección de temperatura
        df_data_used_for_sr['Factor_Corr_Isc_Soiled'] = 1 + alpha_isc_corr * (df_data_used_for_sr['Temp_Modulo_Soiled_C'] - temp_ref_correction_c)
        df_data_used_for_sr['Factor_Corr_Isc_Reference'] = 1 + alpha_isc_corr * (df_data_used_for_sr['Temp_Modulo_Reference_C'] - temp_ref_correction_c)
        df_data_used_for_sr['Factor_Corr_Pmax_Soiled'] = 1 + beta_pmax_corr * (df_data_used_for_sr['Temp_Modulo_Soiled_C'] - temp_ref_correction_c)
        df_data_used_for_sr['Factor_Corr_Pmax_Reference'] = 1 + beta_pmax_corr * (df_data_used_for_sr['Temp_Modulo_Reference_C'] - temp_ref_correction_c)
        
        # Calcular potencias e corrientes corregidas por temperatura
        df_data_used_for_sr['Pmax_Soiled_Temp_Corrected_W'] = df_data_used_for_sr['Pmax_Soiled_Original_W'] / df_data_used_for_sr['Factor_Corr_Pmax_Soiled']
        df_data_used_for_sr['Pmax_Reference_Temp_Corrected_W'] = df_data_used_for_sr['Pmax_Reference_Original_W'] / df_data_used_for_sr['Factor_Corr_Pmax_Reference']
        df_data_used_for_sr['Isc_Soiled_Temp_Corrected_A'] = df_data_used_for_sr['Isc_Soiled_Original_A'] / df_data_used_for_sr['Factor_Corr_Isc_Soiled']
        df_data_used_for_sr['Isc_Reference_Temp_Corrected_A'] = df_data_used_for_sr['Isc_Reference_Original_A'] / df_data_used_for_sr['Factor_Corr_Isc_Reference']
        
        # Calcular SRs instantáneos para verificación
        df_data_used_for_sr['SR_Pmax_Original_Percent'] = 100 * (df_data_used_for_sr['Pmax_Soiled_Original_W'] / df_data_used_for_sr['Pmax_Reference_Original_W'])
        df_data_used_for_sr['SR_Pmax_Temp_Corrected_Percent'] = 100 * (df_data_used_for_sr['Pmax_Soiled_Temp_Corrected_W'] / df_data_used_for_sr['Pmax_Reference_Temp_Corrected_W'])
        df_data_used_for_sr['SR_Isc_Original_Percent'] = 100 * (df_data_used_for_sr['Isc_Soiled_Original_A'] / df_data_used_for_sr['Isc_Reference_Original_A'])
        df_data_used_for_sr['SR_Isc_Temp_Corrected_Percent'] = 100 * (df_data_used_for_sr['Isc_Soiled_Temp_Corrected_A'] / df_data_used_for_sr['Isc_Reference_Temp_Corrected_A'])
        
        csv_filename_sr_data = os.path.join(output_csv_dir_solar, "pvstand_datos_completos_calculos_sr_solar_noon.csv")
        df_data_used_for_sr.to_csv(csv_filename_sr_data)
        logger.info(f"Datos completos utilizados para cálculos de SR (MEDIO DÍA SOLAR) guardados en: {csv_filename_sr_data}")

    df_sr_to_save_main = pd.DataFrame({
        sr_isc_pvstand.name: sr_isc_pvstand,
        sr_pmax_pvstand.name: sr_pmax_pvstand,
        sr_isc_pvstand_corrected.name: sr_isc_pvstand_corrected,
        sr_pmax_pvstand_corrected.name: sr_pmax_pvstand_corrected
    }).sort_index().dropna(how='all')
    
    if not df_sr_to_save_main.empty:
        norm_suffix = 'norm' if normalize_sr_flag else 'abs'
        csv_filename_main = os.path.join(output_csv_dir_solar, f"pvstand_sr_main_{norm_suffix}_solar_noon.csv")
        df_sr_to_save_main.to_csv(csv_filename_main)
        logger.info(f"PVStand SRs (main) con MEDIO DÍA SOLAR guardados en: {csv_filename_main}")

    df_sr_to_save_no_norm = pd.DataFrame({
        sr_isc_pvstand_no_norm.name: sr_isc_pvstand_no_norm,
        sr_pmax_pvstand_no_norm.name: sr_pmax_pvstand_no_norm,
        sr_isc_pvstand_corrected_no_norm.name: sr_isc_pvstand_corrected_no_norm,
        sr_pmax_pvstand_corrected_no_norm.name: sr_pmax_pvstand_corrected_no_norm
    }).sort_index().dropna(how='all')
    if not df_sr_to_save_no_norm.empty:
        csv_filename_no_norm = os.path.join(output_csv_dir_solar, "pvstand_sr_no_norm_with_offset_solar_noon.csv")
        df_sr_to_save_no_norm.to_csv(csv_filename_no_norm)
        logger.info(f"PVStand SRs (no normalizados, con offset) con MEDIO DÍA SOLAR guardados en: {csv_filename_no_norm}")

    df_sr_to_save_raw = pd.DataFrame({
        sr_isc_pvstand_raw_no_offset.name: sr_isc_pvstand_raw_no_offset,
        sr_pmax_pvstand_raw_no_offset.name: sr_pmax_pvstand_raw_no_offset,
        sr_isc_pvstand_corrected_raw_no_offset.name: sr_isc_pvstand_corrected_raw_no_offset,
        sr_pmax_pvstand_corrected_raw_no_offset.name: sr_pmax_pvstand_corrected_raw_no_offset
    }).sort_index().dropna(how='all')
    if not df_sr_to_save_raw.empty:
        csv_filename_raw = os.path.join(output_csv_dir_solar, "pvstand_sr_raw_no_offset_solar_noon.csv")
        df_sr_to_save_raw.to_csv(csv_filename_raw)
        logger.info(f"PVStand SRs (raw, sin offset) con MEDIO DÍA SOLAR guardados en: {csv_filename_raw}")

    # --- Graficado ---
    # Colores fijos por posición para todos los gráficos
    colores_fijos = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    def limpiar_leyenda(nombre):
        nombre = nombre.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').replace('SR ', 'SR ').strip()
        return nombre

    def _plot_sr_section_internal(df_to_plot, title_prefix, filename_suffix, is_normalized_section_flag_param):
        if df_to_plot.empty:
            logger.info(f"No hay datos para graficar en la sección: {title_prefix}")
            return

        logger.info(f"Generando gráficos con MEDIO DÍA SOLAR para: {title_prefix}")
        num_series = len(df_to_plot.columns)
        if num_series == 0: return
            
        line_styles = ['-', '--', '-.', ':'] * (num_series // 4 + 1)
        base_markersize = 4 if 'Media Móvil' not in filename_suffix else 2
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_'] * (num_series // 16 + 1)

        plotting_configs_local = [
            ('1W', f'Semanal Q({graph_quantile*100:.0f}%)'), 
            ('3D', f'3 Días Q({graph_quantile*100:.0f}%)')
        ]
        
        for resample_rule_str, plot_desc_agg_str in plotting_configs_local:
            fig, ax = plt.subplots(figsize=(15, 8))
            has_data_for_this_agg_plot = False
            
            for i, col_name_plot in enumerate(df_to_plot.columns):
                series_plot = df_to_plot[col_name_plot]
                if not isinstance(series_plot, pd.Series) or series_plot.dropna().empty or not isinstance(series_plot.index, pd.DatetimeIndex):
                    continue
                try:
                    data_agg = pd.Series(dtype=float)
                    if 'Media Móvil' in plot_desc_agg_str:
                        min_periods_val = 1
                        if len(series_plot.dropna()) >= 1:
                            data_agg = series_plot.dropna().rolling(window=resample_rule_str, center=True, min_periods=min_periods_val).quantile(graph_quantile).dropna()
                        else: 
                            data_agg = series_plot.resample('D').quantile(graph_quantile).dropna() 
                    else:
                        data_agg = series_plot.resample(resample_rule_str).quantile(graph_quantile).dropna()
                    if not data_agg.empty:
                        ax.plot(data_agg.index, data_agg.values, 
                                linestyle=line_styles[i % len(line_styles)], 
                                marker=markers[i % len(markers)] if 'Media Móvil' not in plot_desc_agg_str else None, 
                                markersize=base_markersize, alpha=0.8, label=limpiar_leyenda(col_name_plot), color=colores_fijos[i % len(colores_fijos)])
                        has_data_for_this_agg_plot = True
                        
                        # --- Línea de tendencia global, solo para el gráfico semanal normalizado ---
                        if filename_suffix == 'norm' and 'semanal' in plot_desc_agg_str.lower():
                            import numpy as np
                            from sklearn.linear_model import LinearRegression
                            from sklearn.metrics import r2_score
                            x = np.arange(len(data_agg)).reshape(-1, 1)
                            y = data_agg.values.reshape(-1, 1)
                            model = LinearRegression().fit(x, y)
                            y_pred = model.predict(x)
                            pendiente = model.coef_[0][0]
                            r2 = r2_score(y, y_pred)
                            pendiente_semana = pendiente
                            # Línea de tendencia: recta, continua, opaca, sin punteo, mismo color
                            ax.plot(data_agg.index, y_pred.flatten(), '-', color=colores_fijos[i % len(colores_fijos)], alpha=0.5, linewidth=2, label=f"Tendencia SR: {pendiente_semana:.3f} [%/semana], R2={r2:.2f}")
                except Exception as e_plot:
                    logger.error(f"Error graficando '{col_name_plot}' para '{plot_desc_agg_str}' en '{title_prefix}': {e_plot}", exc_info=True)
            
            if has_data_for_this_agg_plot:
                # Título especial para el gráfico semanal normalizado
                if filename_suffix == 'norm' and 'semanal' in plot_desc_agg_str.lower():
                    current_plot_title = 'Soiling Ratio PVStand (Medio Día Solar)'
                else:
                    current_plot_title = f'{title_prefix} ({plot_desc_agg_str}) - Medio Día Solar'

                ax.set_title(current_plot_title, fontsize=20)
                ax.set_ylabel('Soiling Ratio Normalizado [%]' if is_normalized_section_flag_param and normalize_sr_flag else 'Soiling Ratio [%]', fontsize=18)
                ax.set_xlabel('Fecha', fontsize=18)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.legend(loc='best', fontsize=14)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='both', labelsize=14)
                plt.xticks(rotation=30, ha='right', fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                
                file_suffix_cleaned = filename_suffix.lower().replace(" ", "_").replace("(", "").replace(")", "")
                agg_cleaned = plot_desc_agg_str.lower().replace(' ', '_').replace('%', '').replace('(', '').replace(')', '').replace('.', '')
                plot_filename = f"pvstand_sr_{file_suffix_cleaned}_{agg_cleaned}_solar_noon.png"
                
                if save_figures_setting: 
                    save_plot_matplotlib(fig, plot_filename, graph_base_plus_subdir, subfolder=None)
                    logger.info(f"Gráfico (MEDIO DÍA SOLAR) guardado en: {os.path.join(graph_base_plus_subdir, plot_filename)}")
                
                if show_figures_setting: 
                    plt.show(block=True)
                    logger.info("Gráfico (MEDIO DÍA SOLAR) mostrado")
                else:
                    plt.close(fig)
            else:
                logger.info(f"No hay datos para graficar para {title_prefix} ({plot_desc_agg_str}) con MEDIO DÍA SOLAR. Plot omitido.")
                if 'fig' in locals() and fig is not None and plt.fignum_exists(fig.number): plt.close(fig)

    _plot_sr_section_internal(df_sr_to_save_main, 
                     f"PVStand SRs ({'Normalizado' if normalize_sr_flag else 'Absoluto'}) - Medio Día Solar", 
                     f"{'norm' if normalize_sr_flag else 'abs'}", 
                     is_normalized_section_flag_param=True) 

    _plot_sr_section_internal(df_sr_to_save_no_norm, 
                     "PVStand SRs (No Normalizado, Con Offset Pmax) - Medio Día Solar", 
                     "no_norm_with_offset", 
                     is_normalized_section_flag_param=False)

    _plot_sr_section_internal(df_sr_to_save_raw, 
                     "PVStand SRs (Raw, Sin Offset) - Medio Día Solar", 
                     "raw_no_offset", 
                     is_normalized_section_flag_param=False)
    
    # --- Gráfico de Potencias Diarias Promedio en Mediodía Solar ---
    if not df_merged_for_corr.empty:
        _plot_daily_power_averages_solar_noon(df_merged_for_corr, col_pmax_soiled, col_pmax_reference, 
                                            graph_base_plus_subdir, save_figures_setting, show_figures_setting)
        
        # --- Gráfico de Corrientes de Cortocircuito Diarias Promedio en Mediodía Solar ---
        col_imax_soiled = f"{pv_module_soiled_id}_Imax"
        col_imax_reference = f"{pv_module_reference_id}_Imax"
        
        if col_imax_soiled in df_merged_for_corr.columns and col_imax_reference in df_merged_for_corr.columns:
            _plot_daily_isc_averages_solar_noon(df_merged_for_corr, col_imax_soiled, col_imax_reference, 
                                               graph_base_plus_subdir, save_figures_setting, show_figures_setting)
        else:
            logger.warning(f"No se encontraron columnas de corrientes: {col_imax_soiled}, {col_imax_reference}")
            logger.info(f"Columnas disponibles: {list(df_merged_for_corr.columns)}")
        
        # --- Análisis Estadístico de Potencias en Mediodía Solar ---
        power_stats_df, power_monthly_stats, power_weekly_stats = _generate_power_statistical_analysis_solar_noon(
            df_merged_for_corr, col_pmax_soiled, col_pmax_reference,
                                                       output_csv_dir_solar, graph_base_plus_subdir, 
                                                       save_figures_setting, show_figures_setting)

    # --- Gráfico de SR Diarios (Q25, Sin Offset) ---
    if not df_sr_to_save_raw.empty:
        _plot_daily_sr_q25_no_offset_solar_noon(df_sr_to_save_raw, graph_base_plus_subdir, 
                                               save_figures_setting, show_figures_setting)

    # --- Gráfico de SR Semanal Normalizado con Tendencia ---
    if not df_sr_to_save_raw.empty:
        _plot_weekly_sr_normalized_with_trend_solar_noon(df_sr_to_save_raw, graph_base_plus_subdir, 
                                                        save_figures_setting, show_figures_setting)

    # --- Gráfico de SR Semanal Normalizado con Tendencia (Sin 2 Primeras Semanas) ---
    if not df_sr_to_save_raw.empty:
        _plot_weekly_sr_normalized_trend_no_first_2weeks_solar_noon(df_sr_to_save_raw, graph_base_plus_subdir, 
                                                                   save_figures_setting, show_figures_setting)

    # --- Generar Excel consolidado con datos procesados (mismo procesamiento que gráficos) ---
    logger.info("Generando Excel consolidado con datos agregados (MEDIO DÍA SOLAR)...")
    
    # --- Generar Excel consolidado con todas las tablas ---
    try:
        import openpyxl
        excel_filename = f"pvstand_datos_completos_agregados_solar_noon_Q{int(graph_quantile*100)}.xlsx"
        excel_filepath = os.path.join(output_csv_dir_solar, excel_filename)
        
        with pd.ExcelWriter(excel_filepath, engine='openpyxl',
                           date_format='YYYY-MM-DD', 
                           datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
            # Hoja con datos principales (normalizados o absolutos) - Mediodía Solar
            if not df_sr_to_save_main.empty:
                # Datos semanales
                df_main_weekly = pd.DataFrame()
                for col_name in df_sr_to_save_main.columns:
                    series_data = df_sr_to_save_main[col_name]
                    if not series_data.dropna().empty:
                        data_agg = series_data.resample('1W').quantile(graph_quantile).dropna()
                        if not data_agg.empty:
                            clean_col_name = col_name.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').strip()
                            df_main_weekly[clean_col_name] = data_agg
                
                if not df_main_weekly.empty:
                    sheet_name = f"SR_{'Norm' if normalize_sr_flag else 'Abs'}_Semanal_SN"
                    df_main_weekly.to_excel(writer, sheet_name=sheet_name)
                
                # Datos cada 3 días
                df_main_3d = pd.DataFrame()
                for col_name in df_sr_to_save_main.columns:
                    series_data = df_sr_to_save_main[col_name]
                    if not series_data.dropna().empty:
                        data_agg = series_data.resample('3D').quantile(graph_quantile).dropna()
                        if not data_agg.empty:
                            clean_col_name = col_name.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').strip()
                            df_main_3d[clean_col_name] = data_agg
                
                if not df_main_3d.empty:
                    sheet_name = f"SR_{'Norm' if normalize_sr_flag else 'Abs'}_3Dias_SN"
                    df_main_3d.to_excel(writer, sheet_name=sheet_name)
            
            # Hoja con datos raw - Mediodía Solar
            if not df_sr_to_save_raw.empty:
                df_raw_weekly = pd.DataFrame()
                for col_name in df_sr_to_save_raw.columns:
                    series_data = df_sr_to_save_raw[col_name]
                    if not series_data.dropna().empty:
                        data_agg = series_data.resample('1W').quantile(graph_quantile).dropna()
                        if not data_agg.empty:
                            clean_col_name = col_name.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').strip()
                            df_raw_weekly[clean_col_name] = data_agg
                
                if not df_raw_weekly.empty:
                    df_raw_weekly.to_excel(writer, sheet_name="SR_Raw_Semanal_SN")
            
            # Hoja con estadísticas consolidadas - Mediodía Solar
            if not df_sr_to_save_main.empty:
                df_main_for_stats = df_sr_to_save_main.copy()
                stats_consolidado = pd.DataFrame({
                    'Serie': df_main_for_stats.columns,
                    'Cantidad_Puntos': [df_main_for_stats[col].count() for col in df_main_for_stats.columns],
                    'Promedio': [df_main_for_stats[col].mean() for col in df_main_for_stats.columns],
                    'Mediana': [df_main_for_stats[col].median() for col in df_main_for_stats.columns],
                    'Desv_Std': [df_main_for_stats[col].std() for col in df_main_for_stats.columns],
                    'Valor_Min': [df_main_for_stats[col].min() for col in df_main_for_stats.columns],
                    'Valor_Max': [df_main_for_stats[col].max() for col in df_main_for_stats.columns],
                    'Rango_Fechas': [f"{df_main_for_stats[col].dropna().index.min().strftime('%Y-%m-%d')} a {df_main_for_stats[col].dropna().index.max().strftime('%Y-%m-%d')}" if df_main_for_stats[col].count() > 0 else "N/A" for col in df_main_for_stats.columns]
                })
                stats_consolidado.to_excel(writer, sheet_name="Estadisticas_SN", index=False)
                
            # Hoja con estadísticas de validez de resamples semanales - Mediodía Solar
            if not df_sr_to_save_main.empty:
                logger.info("Generando estadísticas de validez de resamples semanales (MEDIO DÍA SOLAR)...")
                
                # Crear DataFrame para estadísticas de validez semanal
                validez_data = []
                
                for col_name in df_sr_to_save_main.columns:
                    series_original = df_sr_to_save_main[col_name].dropna()
                    if series_original.empty:
                        continue
                        
                    # Agrupar por semanas
                    semanas_grouped = series_original.groupby(pd.Grouper(freq='1W'))
                    
                    for semana_inicio, datos_semana in semanas_grouped:
                        if datos_semana.empty:
                            continue
                            
                        # Calcular estadísticas de validez para esta semana
                        num_puntos = len(datos_semana)
                        
                        # Para mediodía solar, los días teóricos son menores debido a la ventana de tiempo
                        semana_fin = semana_inicio + pd.Timedelta(days=6)
                        dias_teoricos = min(7, (min(semana_fin, series_original.index.max()) - max(semana_inicio, series_original.index.min())).days + 1)
                        
                        # Calcular días con datos reales
                        if hasattr(datos_semana.index, 'date'):
                            dias_con_datos = len(np.unique(datos_semana.index.date))
                        else:
                            dias_con_datos = datos_semana.resample('D').count().astype(bool).sum()
                        
                        cobertura_dias = (dias_con_datos / dias_teoricos * 100) if dias_teoricos > 0 else 0
                        
                        # Estadísticas de dispersión
                        promedio = datos_semana.mean()
                        mediana = datos_semana.median()
                        desv_std = datos_semana.std()
                        coef_variacion = (desv_std / promedio * 100) if promedio != 0 else 0
                        rango = datos_semana.max() - datos_semana.min()
                        
                        # Cuantil usado para el resample
                        valor_resample = datos_semana.quantile(graph_quantile)
                        
                        # Diferencia entre cuantil y promedio (para evaluar sesgo)
                        sesgo_cuantil = abs(valor_resample - promedio)
                        sesgo_cuantil_pct = (sesgo_cuantil / promedio * 100) if promedio != 0 else 0
                        
                        # Evaluación de representatividad adaptada para mediodía solar (umbrales más bajos)
                        if cobertura_dias >= 70 and num_puntos >= 30 and coef_variacion <= 15:
                            representatividad = "Excelente"
                        elif cobertura_dias >= 50 and num_puntos >= 20 and coef_variacion <= 25:
                            representatividad = "Buena"
                        elif cobertura_dias >= 30 and num_puntos >= 10:
                            representatividad = "Aceptable"
                        else:
                            representatividad = "Limitada"
                        
                        validez_data.append({
                            'Serie': col_name.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').strip(),
                            'Semana_Inicio': semana_inicio.strftime('%Y-%m-%d'),
                            'Num_Puntos_Originales': num_puntos,
                            'Dias_Con_Datos': dias_con_datos,
                            'Dias_Teoricos': dias_teoricos,
                            'Cobertura_Dias_Pct': round(cobertura_dias, 1),
                            'Promedio_Original': round(promedio, 2),
                            'Mediana_Original': round(mediana, 2),
                            'Valor_Resample_Q' + str(int(graph_quantile*100)): round(valor_resample, 2),
                            'Desv_Std': round(desv_std, 2),
                            'Coef_Variacion_Pct': round(coef_variacion, 1),
                            'Rango_Min_Max': round(rango, 2),
                            'Sesgo_Cuantil_vs_Promedio': round(sesgo_cuantil, 2),
                            'Sesgo_Cuantil_Pct': round(sesgo_cuantil_pct, 1),
                            'Representatividad': representatividad,
                            'Metodo': f'Mediodía Solar (±{solar_noon_window_hours}h)'
                        })
                
                if validez_data:
                    df_validez = pd.DataFrame(validez_data)
                    df_validez.to_excel(writer, sheet_name="Validez_Resamples_Semanales_SN", index=False)
                    logger.info("Hoja de validez de resamples semanales (MEDIO DÍA SOLAR) agregada al Excel")
                    
                    # Crear resumen de representatividad por serie
                    resumen_repr = df_validez.groupby('Serie')['Representatividad'].value_counts().unstack(fill_value=0)
                    resumen_repr['Total_Semanas'] = resumen_repr.sum(axis=1)
                    
                    # Calcular porcentajes
                    for col in ['Excelente', 'Buena', 'Aceptable', 'Limitada']:
                        if col in resumen_repr.columns:
                            resumen_repr[f'Pct_{col}'] = round(resumen_repr[col] / resumen_repr['Total_Semanas'] * 100, 1)
                    
                    resumen_repr.to_excel(writer, sheet_name="Resumen_Representatividad_SN")
                    logger.info("Hoja de resumen de representatividad (MEDIO DÍA SOLAR) agregada al Excel")
                    
                    # Generar gráficos de validez después de calcular estadísticas
                    _plot_validity_statistics_solar_noon(validez_data, graph_base_plus_subdir, 
                                                       save_figures_setting, show_figures_setting)
                
            # Hoja adicional con información del método de mediodía solar
            info_solar_noon = pd.DataFrame({
                'Parametro': [
                    'Metodo_Analisis',
                    'Ventana_Mediodia_Solar_Horas',
                    'Latitud_Sitio',
                    'Longitud_Sitio',
                    'Zona_Horaria_Local',
                    'Cuantil_Agregacion',
                    'Fecha_Inicio_Analisis',
                    'Fecha_Fin_Analisis'
                ],
                'Valor': [
                    'Medio Día Solar Dinámico',
                    f'±{solar_noon_window_hours}',
                    settings.SITE_LATITUDE,
                    settings.SITE_LONGITUDE,
                    settings.DUSTIQ_LOCAL_TIMEZONE_STR,
                    f'{graph_quantile*100:.0f}%',
                    filter_start_date_str,
                    filter_end_date_str
                ]
            })
            info_solar_noon.to_excel(writer, sheet_name="Info_Metodo_SN", index=False)
            
            # Hoja con análisis estadístico de potencias (formato 3 columnas)
            if power_stats_df is not None and not power_stats_df.empty:
                power_stats_df.to_excel(writer, sheet_name="Analisis_Potencias_SN", index=False)
                logger.info("Hoja de análisis estadístico de potencias agregada al Excel")
            
            # Hoja con estadísticas mensuales de potencias
            if power_monthly_stats is not None and not power_monthly_stats.empty:
                power_monthly_stats.to_excel(writer, sheet_name="Estadisticas_Mensuales_SN")
                logger.info("Hoja de estadísticas mensuales agregada al Excel")
            
            # Hoja con estadísticas semanales de potencias
            if power_weekly_stats is not None and not power_weekly_stats.empty:
                power_weekly_stats.to_excel(writer, sheet_name="Estadisticas_Semanales_SN")
                logger.info("Hoja de estadísticas semanales agregada al Excel")
        
        logger.info(f"Excel consolidado con datos agregados (MEDIO DÍA SOLAR) guardado: {excel_filepath}")
        
    except ImportError:
        logger.warning("openpyxl no está disponible. No se pudo generar el archivo Excel consolidado para mediodía solar.")
    except Exception as e:
        logger.error(f"Error generando Excel consolidado para mediodía solar: {e}")

    # --- Generar archivos CSV semanales Q25 para consolidación ---
    logger.info("Generando archivos CSV semanales Q25 para consolidación...")
    
    try:
        # Crear directorio CSV si no existe
        csv_output_dir = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "pv_stand")
        os.makedirs(csv_output_dir, exist_ok=True)
        
        # CSV: Datos semanales Q25 de SR
        if 'df_sr_raw_no_offset' in locals() and not df_sr_raw_no_offset.empty:
            weekly_q25_df = pd.DataFrame()
            
            for col in df_sr_raw_no_offset.columns:
                series = df_sr_raw_no_offset[col].dropna()
                if not series.empty:
                    # Aplicar filtro: eliminar SR menores a 80%
                    series_filtered = series[series >= 80.0]
                    if not series_filtered.empty:
                        # Calcular quintil 0.25 semanal
                        weekly_q25 = series_filtered.resample('1W').quantile(0.25).dropna()
                        if not weekly_q25.empty:
                            weekly_q25_df[col] = weekly_q25
            
            if not weekly_q25_df.empty:
                csv_filename = os.path.join(csv_output_dir, 'pvstand_sr_semanal_q25_solar_noon.csv')
                weekly_q25_df.to_csv(csv_filename)
                logger.info(f"CSV semanal Q25 de SR guardado: {csv_filename}")
        
    except Exception as e:
        logger.error(f"Error generando archivos CSV semanales Q25: {e}")
    
    logger.info("=== FIN DEL ANÁLISIS PVSTAND CON MEDIO DÍA SOLAR ===")
    return True

def _plot_daily_isc_averages_solar_noon(df_merged, col_imax_soiled, col_imax_reference, 
                                        output_dir, save_figures, show_figures):
    """
    Genera un gráfico de los promedios diarios de corrientes de cortocircuito durante el mediodía solar.
    
    Args:
        df_merged: DataFrame con datos de corrientes filtrados por mediodía solar
        col_imax_soiled: Nombre de la columna de corriente del módulo sucio
        col_imax_reference: Nombre de la columna de corriente del módulo de referencia
        output_dir: Directorio de salida para los gráficos
        save_figures: Si guardar las figuras
        show_figures: Si mostrar las figuras
    """
    logger.info("Generando gráfico de corrientes de cortocircuito diarias promedio en mediodía solar...")
    
    try:
        # Calcular promedios diarios
        daily_isc_soiled = df_merged[col_imax_soiled].resample('D').mean().dropna()
        daily_isc_reference = df_merged[col_imax_reference].resample('D').mean().dropna()
        
        if daily_isc_soiled.empty or daily_isc_reference.empty:
            logger.warning("No hay datos suficientes para generar gráfico de corrientes diarias.")
            return
        
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plotear las series
        ax.plot(daily_isc_soiled.index, daily_isc_soiled.values, 
               '-o', color='#e74c3c', alpha=0.8, markersize=4, 
               label='Módulo Sucio (Isc)', linewidth=2)
        
        ax.plot(daily_isc_reference.index, daily_isc_reference.values, 
               '-o', color='#3498db', alpha=0.8, markersize=4, 
               label='Módulo Referencia (Isc)', linewidth=2)
        
        # Configurar el gráfico
        ax.set_title('Corrientes de Cortocircuito Diarias Promedio - Mediodía Solar', fontsize=20)
        ax.set_ylabel('Corriente de Cortocircuito Promedio [A]', fontsize=18)
        ax.set_xlabel('Fecha', fontsize=18)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', fontsize=14)
        
        # Configurar límites del eje Y (ajustado para corrientes)
        current_ylim = ax.get_ylim()
        ax.set_ylim(bottom=0, top=current_ylim[1] * 1.05)
        
        # Formatear fechas en el eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        # Estadísticas rápidas para mostrar en el gráfico
        mean_soiled = daily_isc_soiled.mean()
        mean_reference = daily_isc_reference.mean()
        ratio_mean = mean_soiled / mean_reference if mean_reference > 0 else 0
        
        # Agregar texto con estadísticas
        stats_text = f'Promedio Sucio: {mean_soiled:.3f} A\n'
        stats_text += f'Promedio Referencia: {mean_reference:.3f} A\n'
        stats_text += f'Ratio promedio: {ratio_mean:.3f} ({ratio_mean*100:.1f}%)'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=12)
        
        # Guardar y/o mostrar
        plot_filename = "pvstand_corrientes_cortocircuito_diarias_promedio_solar_noon.png"
        
        if save_figures:
            save_plot_matplotlib(fig, plot_filename, output_dir)
            logger.info(f"Gráfico de corrientes diarias guardado en: {os.path.join(output_dir, plot_filename)}")
        
        if show_figures:
            plt.show(block=True)
            logger.info("Gráfico de corrientes diarias mostrado")
        else:
            plt.close(fig)
            
        logger.info("Gráfico de corrientes diarias promedio generado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error generando gráfico de corrientes diarias: {e}", exc_info=True)

def _plot_daily_sr_q25_no_offset_solar_noon(df_sr_raw_no_offset, output_dir, save_figures, show_figures):
    """
    Genera un gráfico de los Soiling Ratios diarios usando quintil 0.25, sin offset.
    
    Args:
        df_sr_raw_no_offset: DataFrame con los SR raw sin offset
        output_dir: Directorio de salida para los gráficos
        save_figures: Si guardar las figuras
        show_figures: Si mostrar las figuras
    """
    logger.info("Generando gráfico de Soiling Ratios diarios (Q25, sin offset) en mediodía solar...")
    
    try:
        if df_sr_raw_no_offset.empty:
            logger.warning("No hay datos de SR para generar gráfico diario.")
            return
            
        # Filtrar solo las columnas que existen y tienen datos
        available_columns = []
        for col in df_sr_raw_no_offset.columns:
            if not df_sr_raw_no_offset[col].dropna().empty:
                available_columns.append(col)
        
        if not available_columns:
            logger.warning("No hay columnas de SR con datos válidos.")
            return
            
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Colores específicos para cada tipo de SR
        colors = {
            'SR_Isc_Uncorrected_Raw_NoOffset': '#1f77b4',      # Azul
            'SR_Pmax_Uncorrected_Raw_NoOffset': '#ff7f0e',     # Naranja  
            'SR_Isc_Corrected_Raw_NoOffset': '#2ca02c',        # Verde
            'SR_Pmax_Corrected_Raw_NoOffset': '#d62728'        # Rojo
        }
        
        # Etiquetas más legibles
        labels = {
            'SR_Isc_Uncorrected_Raw_NoOffset': 'SR Isc (Sin corrección)',
            'SR_Pmax_Uncorrected_Raw_NoOffset': 'SR Pmax (Sin corrección)',
            'SR_Isc_Corrected_Raw_NoOffset': 'SR Isc (Corregido temp.)',
            'SR_Pmax_Corrected_Raw_NoOffset': 'SR Pmax (Corregido temp.)'
        }
        
        has_data = False
        
        for i, col in enumerate(available_columns):
            series = df_sr_raw_no_offset[col].dropna()
            if series.empty:
                continue
                
            # Aplicar filtro: eliminar SR menores a 80%
            series_filtered = series[series >= 80.0]
            if series_filtered.empty:
                logger.warning(f"No hay datos válidos para {col} después del filtro SR >= 80%")
                continue
                
            # Calcular quintil 0.25 diario
            daily_q25 = series_filtered.resample('D').quantile(0.25).dropna()
            
            if daily_q25.empty:
                continue
                
            # Obtener color y etiqueta
            color = colors.get(col, f'C{i}')
            label = labels.get(col, col.replace('_', ' '))
            
            # Plotear
            ax.plot(daily_q25.index, daily_q25.values, 
                   '-o', color=color, alpha=0.8, markersize=4, 
                   label=label, linewidth=2)
            
            has_data = True
        
        if not has_data:
            logger.warning("No se pudo generar gráfico: ninguna serie tiene datos válidos después del remuestreo.")
            plt.close(fig)
            return
        
        # Configurar el gráfico
        ax.set_title('Soiling Ratios Diarios - Mediodía Solar', 
                    fontsize=20, fontweight='bold', pad=20) #(Quintil 25%, Sin Offset)
        ax.set_ylabel('Soiling Ratio [%]', fontsize=18, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=18, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', fontsize=14)
        
        # Configurar límites del eje Y basándose en los datos
        all_values = []
        for col in available_columns:
            series = df_sr_raw_no_offset[col].dropna()
            if not series.empty:
                # Aplicar el mismo filtro para cálculo de límites
                series_filtered = series[series >= 80.0]
                if not series_filtered.empty:
                    daily_q25 = series_filtered.resample('D').quantile(0.25).dropna()
                    if not daily_q25.empty:
                        all_values.extend(daily_q25.values)
        
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            # Agregar un margen del 2% arriba y abajo
            margin = (max_val - min_val) * 0.02
            ax.set_ylim(bottom=max(80, min_val - margin), 
                       top=min(102, max_val + margin))
        else:
            # Límites por defecto si no hay datos
            ax.set_ylim(bottom=85, top=100)
        
        # Formatear fechas en el eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        # Agregar línea de referencia en 100%
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.7, linewidth=1, label='Referencia (100%)')
        
        # Estadísticas rápidas para mostrar en el gráfico
        if len(available_columns) > 0:
            # Tomar la primera serie disponible para estadísticas generales
            main_series = df_sr_raw_no_offset[available_columns[0]].dropna()
            if not main_series.empty:
                # Aplicar filtro para estadísticas
                main_series_filtered = main_series[main_series >= 80.0]
                if not main_series_filtered.empty:
                    daily_stats = main_series_filtered.resample('D').agg(['mean', 'min', 'max', lambda x: x.quantile(0.25)]).dropna()
                    
                    if not daily_stats.empty:
                        mean_q25 = daily_stats.iloc[:, 3].mean()  # Quintil 0.25 promedio
                        min_q25 = daily_stats.iloc[:, 3].min()   # Mínimo del quintil 0.25
                        max_q25 = daily_stats.iloc[:, 3].max()   # Máximo del quintil 0.25
                        
                        # Agregar texto con estadísticas
                        stats_text = f'Estadísticas Q25 (Serie principal, SR≥80%):\n'
                        stats_text += f'Promedio: {mean_q25:.2f}%\n'
                        stats_text += f'Mínimo: {min_q25:.2f}%\n'
                        stats_text += f'Máximo: {max_q25:.2f}%'
                        
                        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                               verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=12)
        
        # Guardar y/o mostrar
        plot_filename = "pvstand_sr_diarios_q25_sin_offset_solar_noon.png"
        
        if save_figures:
            save_plot_matplotlib(fig, plot_filename, output_dir)
            logger.info(f"Gráfico de SR diarios (Q25) guardado en: {os.path.join(output_dir, plot_filename)}")
        
        if show_figures:
            plt.show(block=True)
            logger.info("Gráfico de SR diarios (Q25) mostrado")
        else:
            plt.close(fig)
            
        logger.info("Gráfico de SR diarios (Q25, sin offset) generado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error generando gráfico de SR diarios Q25: {e}", exc_info=True)

def _plot_weekly_sr_normalized_with_trend_solar_noon(df_sr_raw_no_offset, output_dir, save_figures, show_figures):
    """
    Genera un gráfico semanal de SR normalizado al primer valor (100%) con línea de tendencia.
    
    Args:
        df_sr_raw_no_offset: DataFrame con los SR raw sin offset
        output_dir: Directorio de salida para los gráficos
        save_figures: Si guardar las figuras
        show_figures: Si mostrar las figuras
    """
    logger.info("Generando gráfico semanal de SR normalizado con tendencia en mediodía solar...")
    
    try:
        if df_sr_raw_no_offset.empty:
            logger.warning("No hay datos de SR para generar gráfico semanal normalizado.")
            return
            
        # Filtrar solo las columnas que existen y tienen datos
        available_columns = []
        for col in df_sr_raw_no_offset.columns:
            if not df_sr_raw_no_offset[col].dropna().empty:
                available_columns.append(col)
        
        if not available_columns:
            logger.warning("No hay columnas de SR con datos válidos.")
            return
            
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Colores específicos para cada tipo de SR
        colors = {
            'SR_Isc_Uncorrected_Raw_NoOffset': '#1f77b4',      # Azul
            'SR_Pmax_Uncorrected_Raw_NoOffset': '#ff7f0e',     # Naranja  
            'SR_Isc_Corrected_Raw_NoOffset': '#2ca02c',        # Verde
            'SR_Pmax_Corrected_Raw_NoOffset': '#d62728'        # Rojo
        }
        
        # Etiquetas más legibles
        labels = {
            'SR_Isc_Uncorrected_Raw_NoOffset': 'SR Isc (Sin corrección)',
            'SR_Pmax_Uncorrected_Raw_NoOffset': 'SR Pmax (Sin corrección)',
            'SR_Isc_Corrected_Raw_NoOffset': 'SR Isc (Corregido temp.)',
            'SR_Pmax_Corrected_Raw_NoOffset': 'SR Pmax (Corregido temp.)'
        }
        
        has_data = False
        
        for i, col in enumerate(available_columns):
            series = df_sr_raw_no_offset[col].dropna()
            if series.empty:
                continue
                
            # Aplicar filtro: eliminar SR menores a 80%
            series_filtered = series[series >= 80.0]
            if series_filtered.empty:
                logger.warning(f"No hay datos válidos para {col} después del filtro SR >= 80%")
                continue
                
            # Calcular quintil 0.25 semanal
            weekly_q25 = series_filtered.resample('1W').quantile(0.25).dropna()
            
            if weekly_q25.empty:
                continue
                
            # Normalizar al primer valor (ajustar a 100%)
            if len(weekly_q25) > 0:
                first_value = weekly_q25.iloc[0]
                if first_value > 0:
                    weekly_normalized = (weekly_q25 / first_value) * 100.0
                else:
                    continue
            else:
                continue
                
            # Obtener color y etiqueta
            color = colors.get(col, f'C{i}')
            label = labels.get(col, col.replace('_', ' '))
            
            # Plotear serie principal
            ax.plot(weekly_normalized.index, weekly_normalized.values, 
                   '-o', color=color, alpha=0.8, markersize=6, 
                   label=label, linewidth=2)
            
            # Calcular y plotear línea de tendencia
            if len(weekly_normalized) >= 2:
                import numpy as np
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                # Preparar datos para regresión
                x = np.arange(len(weekly_normalized)).reshape(-1, 1)
                y = weekly_normalized.values.reshape(-1, 1)
                
                # Ajustar modelo de regresión lineal
                model = LinearRegression().fit(x, y)
                y_pred = model.predict(x)
                
                # Calcular métricas
                pendiente_semanal = model.coef_[0][0]  # Pendiente por semana
                r2 = r2_score(y, y_pred)
                
                # Plotear línea de tendencia
                ax.plot(weekly_normalized.index, y_pred.flatten(), 
                       '--', color=color, alpha=0.6, linewidth=2,
                       label=f'Tendencia {label}: {pendiente_semanal:.3f}%/sem, R²={r2:.3f}')
            
            has_data = True
        
        if not has_data:
            logger.warning("No se pudo generar gráfico: ninguna serie tiene datos válidos después del procesamiento.")
            plt.close(fig)
            return
        
        # Configurar el gráfico
        ax.set_title('Soiling Ratios Semanales Normalizados (Q25) - Mediodía Solar', 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('Soiling Ratio Normalizado [%]', fontsize=18, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=18, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', fontsize=12)
        
        # Configurar límites del eje Y
        ax.set_ylim(bottom=85, top=105)
        
        # Agregar línea de referencia en 100%
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        
        # Formatear fechas en el eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        # Guardar y/o mostrar
        plot_filename = "pvstand_sr_semanal_normalizado_con_tendencia_solar_noon.png"
        
        if save_figures:
            save_plot_matplotlib(fig, plot_filename, output_dir)
            logger.info(f"Gráfico de SR semanal normalizado con tendencia guardado en: {os.path.join(output_dir, plot_filename)}")
        
        if show_figures:
            plt.show(block=True)
            logger.info("Gráfico de SR semanal normalizado con tendencia mostrado")
        else:
            plt.close(fig)
            
        logger.info("Gráfico de SR semanal normalizado con tendencia generado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error generando gráfico de SR semanal normalizado con tendencia: {e}", exc_info=True)

def _plot_weekly_sr_normalized_trend_no_first_2weeks_solar_noon(df_sr_raw_no_offset, output_dir, save_figures, show_figures):
    """
    Genera un gráfico semanal de SR normalizado al primer valor (100%) con línea de tendencia,
    eliminando las dos primeras semanas del estudio.
    
    Args:
        df_sr_raw_no_offset: DataFrame con los SR raw sin offset
        output_dir: Directorio de salida para los gráficos
        save_figures: Si guardar las figuras
        show_figures: Si mostrar las figuras
    """
    logger.info("Generando gráfico semanal de SR normalizado con tendencia (sin 2 primeras semanas) en mediodía solar...")
    
    try:
        if df_sr_raw_no_offset.empty:
            logger.warning("No hay datos de SR para generar gráfico semanal normalizado (sin 2 primeras semanas).")
            return
            
        # Filtrar solo las columnas que existen y tienen datos
        available_columns = []
        for col in df_sr_raw_no_offset.columns:
            if not df_sr_raw_no_offset[col].dropna().empty:
                available_columns.append(col)
        
        if not available_columns:
            logger.warning("No hay columnas de SR con datos válidos.")
            return
            
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Colores específicos para cada tipo de SR
        colors = {
            'SR_Isc_Uncorrected_Raw_NoOffset': '#1f77b4',      # Azul
            'SR_Pmax_Uncorrected_Raw_NoOffset': '#ff7f0e',     # Naranja  
            'SR_Isc_Corrected_Raw_NoOffset': '#2ca02c',        # Verde
            'SR_Pmax_Corrected_Raw_NoOffset': '#d62728'        # Rojo
        }
        
        # Etiquetas más legibles
        labels = {
            'SR_Isc_Uncorrected_Raw_NoOffset': 'SR Isc (Sin corrección)',
            'SR_Pmax_Uncorrected_Raw_NoOffset': 'SR Pmax (Sin corrección)',
            'SR_Isc_Corrected_Raw_NoOffset': 'SR Isc (Corregido temp.)',
            'SR_Pmax_Corrected_Raw_NoOffset': 'SR Pmax (Corregido temp.)'
        }
        
        has_data = False
        
        for i, col in enumerate(available_columns):
            series = df_sr_raw_no_offset[col].dropna()
            if series.empty:
                continue
                
            # Aplicar filtro: eliminar SR menores a 80%
            series_filtered = series[series >= 80.0]
            if series_filtered.empty:
                logger.warning(f"No hay datos válidos para {col} después del filtro SR >= 80%")
                continue
                
            # Calcular quintil 0.25 semanal
            weekly_q25 = series_filtered.resample('1W').quantile(0.25).dropna()
            
            if weekly_q25.empty or len(weekly_q25) <= 2:
                logger.warning(f"No hay suficientes datos semanales para {col} (necesarios >2 puntos)")
                continue
                
            # Eliminar las dos primeras semanas
            weekly_q25_trimmed = weekly_q25.iloc[2:].copy()
            
            if weekly_q25_trimmed.empty:
                logger.warning(f"No hay datos para {col} después de eliminar las 2 primeras semanas")
                continue
                
            # Normalizar al primer valor después del trimming (ajustar a 100%)
            if len(weekly_q25_trimmed) > 0:
                first_value = weekly_q25_trimmed.iloc[0]
                if first_value > 0:
                    weekly_normalized = (weekly_q25_trimmed / first_value) * 100.0
                else:
                    continue
            else:
                continue
                
            # Obtener color y etiqueta
            color = colors.get(col, f'C{i}')
            label = labels.get(col, col.replace('_', ' '))
            
            # Plotear serie principal
            ax.plot(weekly_normalized.index, weekly_normalized.values, 
                   '-o', color=color, alpha=0.8, markersize=6, 
                   label=label, linewidth=2)
            
            # Calcular y plotear línea de tendencia
            if len(weekly_normalized) >= 2:
                import numpy as np
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                # Preparar datos para regresión
                x = np.arange(len(weekly_normalized)).reshape(-1, 1)
                y = weekly_normalized.values.reshape(-1, 1)
                
                # Ajustar modelo de regresión lineal
                model = LinearRegression().fit(x, y)
                y_pred = model.predict(x)
                
                # Calcular métricas
                pendiente_semanal = model.coef_[0][0]  # Pendiente por semana
                r2 = r2_score(y, y_pred)
                
                # Plotear línea de tendencia
                ax.plot(weekly_normalized.index, y_pred.flatten(), 
                       '--', color=color, alpha=0.6, linewidth=2,
                       label=f'Tendencia {label}: {pendiente_semanal:.3f}%/sem, R²={r2:.3f}')
            
            has_data = True
        
        if not has_data:
            logger.warning("No se pudo generar gráfico: ninguna serie tiene datos válidos después del procesamiento.")
            plt.close(fig)
            return
        
        # Configurar el gráfico
        ax.set_title('Soiling Ratios Semanales Normalizados Q25 - Mediodía Solar', 
                    fontsize=20, fontweight='bold', pad=20) # sin 2 primeras semanas    
        ax.set_ylabel('Soiling Ratio Normalizado [%]', fontsize=18, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=18, fontweight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', fontsize=12)
        
        # Configurar límites del eje Y
        ax.set_ylim(bottom=80, top=110)
        
        # Agregar línea de referencia en 100%
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        
        # Formatear fechas en el eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        # Guardar y/o mostrar
        plot_filename = "pvstand_sr_semanal_normalizado_tendencia_sin_2sem_iniciales_solar_noon.png"
        
        if save_figures:
            save_plot_matplotlib(fig, plot_filename, output_dir)
            logger.info(f"Gráfico de SR semanal normalizado con tendencia (sin 2 sem. iniciales) guardado en: {os.path.join(output_dir, plot_filename)}")
        
        if show_figures:
            plt.show(block=True)
            logger.info("Gráfico de SR semanal normalizado con tendencia (sin 2 sem. iniciales) mostrado")
        else:
            plt.close(fig)
            
        logger.info("Gráfico de SR semanal normalizado con tendencia (sin 2 primeras semanas) generado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error generando gráfico de SR semanal normalizado con tendencia (sin 2 sem. iniciales): {e}", exc_info=True)

def _plot_daily_power_averages_solar_noon(df_merged, col_pmax_soiled, col_pmax_reference, 
                                         output_dir, save_figures, show_figures):
    """
    Genera un gráfico de los promedios diarios de potencias durante el mediodía solar.
    
    Args:
        df_merged: DataFrame con datos de potencias filtrados por mediodía solar
        col_pmax_soiled: Nombre de la columna de potencia del módulo sucio
        col_pmax_reference: Nombre de la columna de potencia del módulo de referencia
        output_dir: Directorio de salida para los gráficos
        save_figures: Si guardar las figuras
        show_figures: Si mostrar las figuras
    """
    logger.info("Generando gráfico de potencias diarias promedio en mediodía solar...")
    
    try:
        # Calcular promedios diarios
        daily_power_soiled = df_merged[col_pmax_soiled].resample('D').mean().dropna()
        daily_power_reference = df_merged[col_pmax_reference].resample('D').mean().dropna()
        
        if daily_power_soiled.empty or daily_power_reference.empty:
            logger.warning("No hay datos suficientes para generar gráfico de potencias diarias.")
            return
        
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plotear las series
        ax.plot(daily_power_soiled.index, daily_power_soiled.values, 
               '-o', color='#ff7f0e', alpha=0.8, markersize=4, 
               label='Módulo Sucio', linewidth=2)
        
        ax.plot(daily_power_reference.index, daily_power_reference.values, 
               '-o', color='#1f77b4', alpha=0.8, markersize=4, 
               label='Módulo Referencia', linewidth=2)
        
        # Configurar el gráfico
        ax.set_title('Potencias Diarias Promedio - Mediodía Solar', fontsize=20)
        ax.set_ylabel('Potencia Promedio [W]', fontsize=18)
        ax.set_xlabel('Fecha', fontsize=18)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', fontsize=14)
        
        # Configurar límites del eje Y
        current_ylim = ax.get_ylim()
        ax.set_ylim(bottom=100, top=current_ylim[1])
        
        # Formatear fechas en el eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='both', labelsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        # Guardar y/o mostrar
        plot_filename = "pvstand_potencias_diarias_promedio_solar_noon.png"
        
        if save_figures:
            save_plot_matplotlib(fig, plot_filename, output_dir)
            logger.info(f"Gráfico de potencias diarias guardado en: {os.path.join(output_dir, plot_filename)}")
        
        if show_figures:
            plt.show(block=True)
            logger.info("Gráfico de potencias diarias mostrado")
        else:
            plt.close(fig)
            
        logger.info("Gráfico de potencias diarias promedio generado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error generando gráfico de potencias diarias: {e}", exc_info=True)

def _generate_power_statistical_analysis_solar_noon(df_merged, col_pmax_soiled, col_pmax_reference,
                                                  output_csv_dir, output_graph_dir, save_figures, show_figures):
    """
    Genera un análisis estadístico completo de las potencias durante el mediodía solar.
    
    Args:
        df_merged: DataFrame con datos de potencias filtrados por mediodía solar
        col_pmax_soiled: Nombre de la columna de potencia del módulo sucio
        col_pmax_reference: Nombre de la columna de potencia del módulo de referencia
        output_csv_dir: Directorio de salida para archivos CSV
        output_graph_dir: Directorio de salida para gráficos
        save_figures: Si guardar las figuras
        show_figures: Si mostrar las figuras
        
    Returns:
        tuple: (stats_df, monthly_stats, weekly_stats) - DataFrames para Excel consolidado
    """
    logger.info("Generando análisis estadístico de potencias en mediodía solar...")
    
    try:
        import numpy as np
        from scipy import stats
        
        # Extraer datos de potencias
        power_soiled = df_merged[col_pmax_soiled].dropna()
        power_reference = df_merged[col_pmax_reference].dropna()
        
        if power_soiled.empty or power_reference.empty:
            logger.warning("No hay datos suficientes para análisis estadístico de potencias.")
            return None, None, None
        
        # --- 1. GENERAR INFORME ESTADÍSTICO ---
        stats_report = {}
        
        # Estadísticas básicas
        for name, data in [("Modulo_Sucio", power_soiled), ("Modulo_Referencia", power_reference)]:
            stats_report[f"{name}_Count"] = len(data)
            stats_report[f"{name}_Mean_W"] = data.mean()
            stats_report[f"{name}_Median_W"] = data.median()
            stats_report[f"{name}_Std_W"] = data.std()
            stats_report[f"{name}_Min_W"] = data.min()
            stats_report[f"{name}_Max_W"] = data.max()
            stats_report[f"{name}_Q25_W"] = data.quantile(0.25)
            stats_report[f"{name}_Q75_W"] = data.quantile(0.75)
            stats_report[f"{name}_IQR_W"] = data.quantile(0.75) - data.quantile(0.25)
            stats_report[f"{name}_CV_Percent"] = (data.std() / data.mean()) * 100
            stats_report[f"{name}_Skewness"] = stats.skew(data)
            stats_report[f"{name}_Kurtosis"] = stats.kurtosis(data)
        
        # Estadísticas comparativas
        power_diff = power_soiled - power_reference
        power_ratio = power_soiled / power_reference
        
        stats_report["Diferencia_Mean_W"] = power_diff.mean()
        stats_report["Diferencia_Std_W"] = power_diff.std()
        stats_report["Diferencia_Min_W"] = power_diff.min()
        stats_report["Diferencia_Max_W"] = power_diff.max()
        stats_report["Ratio_Mean"] = power_ratio.mean()
        stats_report["Ratio_Std"] = power_ratio.std()
        stats_report["SR_Mean_Percent"] = power_ratio.mean() * 100
        stats_report["SR_Std_Percent"] = power_ratio.std() * 100
        
        # Test estadísticos
        t_stat, p_value = stats.ttest_rel(power_soiled, power_reference)
        stats_report["TTest_Statistic"] = t_stat
        stats_report["TTest_PValue"] = p_value
        stats_report["TTest_Significant_005"] = p_value < 0.05
        
        # Test de normalidad
        _, p_norm_soiled = stats.shapiro(power_soiled.sample(min(5000, len(power_soiled))))
        _, p_norm_ref = stats.shapiro(power_reference.sample(min(5000, len(power_reference))))
        stats_report["Shapiro_PValue_Soiled"] = p_norm_soiled
        stats_report["Shapiro_PValue_Reference"] = p_norm_ref
        stats_report["Normal_Distribution_Soiled"] = p_norm_soiled > 0.05
        stats_report["Normal_Distribution_Reference"] = p_norm_ref > 0.05
        
        # Generar informe estadístico en formato de 3 columnas
        parametros_comparativos = [
            ("Cantidad de datos", "Count", stats_report["Modulo_Sucio_Count"], stats_report["Modulo_Referencia_Count"]),
            ("Media [W]", "Mean_W", stats_report["Modulo_Sucio_Mean_W"], stats_report["Modulo_Referencia_Mean_W"]),
            ("Mediana [W]", "Median_W", stats_report["Modulo_Sucio_Median_W"], stats_report["Modulo_Referencia_Median_W"]),
            ("Desviación Estándar [W]", "Std_W", stats_report["Modulo_Sucio_Std_W"], stats_report["Modulo_Referencia_Std_W"]),
            ("Mínimo [W]", "Min_W", stats_report["Modulo_Sucio_Min_W"], stats_report["Modulo_Referencia_Min_W"]),
            ("Máximo [W]", "Max_W", stats_report["Modulo_Sucio_Max_W"], stats_report["Modulo_Referencia_Max_W"]),
            ("Cuartil 25% [W]", "Q25_W", stats_report["Modulo_Sucio_Q25_W"], stats_report["Modulo_Referencia_Q25_W"]),
            ("Cuartil 75% [W]", "Q75_W", stats_report["Modulo_Sucio_Q75_W"], stats_report["Modulo_Referencia_Q75_W"]),
            ("Rango Intercuartílico [W]", "IQR_W", stats_report["Modulo_Sucio_IQR_W"], stats_report["Modulo_Referencia_IQR_W"]),
            ("Coeficiente de Variación [%]", "CV_Percent", stats_report["Modulo_Sucio_CV_Percent"], stats_report["Modulo_Referencia_CV_Percent"]),
            ("Asimetría (Skewness)", "Skewness", stats_report["Modulo_Sucio_Skewness"], stats_report["Modulo_Referencia_Skewness"]),
            ("Curtosis (Kurtosis)", "Kurtosis", stats_report["Modulo_Sucio_Kurtosis"], stats_report["Modulo_Referencia_Kurtosis"]),
            ("Distribución Normal (Shapiro p-value)", "Shapiro_PValue", stats_report["Shapiro_PValue_Soiled"], stats_report["Shapiro_PValue_Reference"]),
            ("Distribución es Normal (p>0.05)", "Normal_Distribution", stats_report["Normal_Distribution_Soiled"], stats_report["Normal_Distribution_Reference"])
        ]
        
        # Crear DataFrame con formato de 3 columnas
        stats_3col_data = []
        for param_name, _, val_sucio, val_ref in parametros_comparativos:
            stats_3col_data.append({
                'Parametro': param_name,
                'Modulo_Sucio': val_sucio,
                'Modulo_Referencia': val_ref
            })
        
        # Agregar estadísticas adicionales (no comparativas)
        stats_3col_data.extend([
            {'Parametro': 'Diferencia Media [W]', 'Modulo_Sucio': stats_report["Diferencia_Mean_W"], 'Modulo_Referencia': '-'},
            {'Parametro': 'Diferencia Desv. Estándar [W]', 'Modulo_Sucio': stats_report["Diferencia_Std_W"], 'Modulo_Referencia': '-'},
            {'Parametro': 'Diferencia Mínima [W]', 'Modulo_Sucio': stats_report["Diferencia_Min_W"], 'Modulo_Referencia': '-'},
            {'Parametro': 'Diferencia Máxima [W]', 'Modulo_Sucio': stats_report["Diferencia_Max_W"], 'Modulo_Referencia': '-'},
            {'Parametro': 'Ratio Promedio', 'Modulo_Sucio': stats_report["Ratio_Mean"], 'Modulo_Referencia': '-'},
            {'Parametro': 'Ratio Desv. Estándar', 'Modulo_Sucio': stats_report["Ratio_Std"], 'Modulo_Referencia': '-'},
            {'Parametro': 'SR Promedio [%]', 'Modulo_Sucio': stats_report["SR_Mean_Percent"], 'Modulo_Referencia': '-'},
            {'Parametro': 'SR Desv. Estándar [%]', 'Modulo_Sucio': stats_report["SR_Std_Percent"], 'Modulo_Referencia': '-'},
            {'Parametro': 'Test-t Estadístico', 'Modulo_Sucio': stats_report["TTest_Statistic"], 'Modulo_Referencia': '-'},
            {'Parametro': 'Test-t P-value', 'Modulo_Sucio': stats_report["TTest_PValue"], 'Modulo_Referencia': '-'},
            {'Parametro': 'Test-t Significativo (p<0.05)', 'Modulo_Sucio': stats_report["TTest_Significant_005"], 'Modulo_Referencia': '-'}
        ])
        
        stats_df = pd.DataFrame(stats_3col_data)
        stats_filename = os.path.join(output_csv_dir, "analisis_estadistico_potencias_solar_noon.csv")
        stats_df.to_csv(stats_filename, index=False)
        logger.info(f"Informe estadístico guardado en: {stats_filename}")
        
        # --- 2. GRÁFICO DE DISTRIBUCIONES (HISTOGRAMAS) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma módulo sucio
        ax1.hist(power_soiled, bins=50, alpha=0.7, color='#ff7f0e', density=True, label='Módulo Sucio')
        ax1.axvline(power_soiled.mean(), color='#ff7f0e', linestyle='--', linewidth=2, 
                   label=f'Media: {power_soiled.mean():.1f}W')
        ax1.axvline(power_soiled.median(), color='#ff7f0e', linestyle=':', linewidth=2, 
                   label=f'Mediana: {power_soiled.median():.1f}W')
        ax1.set_title('Distribución - Módulo Sucio', fontsize=14)
        ax1.set_xlabel('Potencia [W]', fontsize=12)
        ax1.set_ylabel('Densidad', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histograma módulo referencia
        ax2.hist(power_reference, bins=50, alpha=0.7, color='#1f77b4', density=True, label='Módulo Referencia')
        ax2.axvline(power_reference.mean(), color='#1f77b4', linestyle='--', linewidth=2, 
                   label=f'Media: {power_reference.mean():.1f}W')
        ax2.axvline(power_reference.median(), color='#1f77b4', linestyle=':', linewidth=2, 
                   label=f'Mediana: {power_reference.median():.1f}W')
        ax2.set_title('Distribución - Módulo Referencia', fontsize=14)
        ax2.set_xlabel('Potencia [W]', fontsize=12)
        ax2.set_ylabel('Densidad', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Distribuciones de Potencia - Mediodía Solar', fontsize=16)
        plt.tight_layout()
        
        if save_figures:
            save_plot_matplotlib(fig, "pvstand_distribuciones_potencia_solar_noon.png", output_graph_dir)
        if show_figures:
            plt.show(block=True)
        else:
            plt.close(fig)
        
        # --- 3. GRÁFICO DE BOXPLOTS COMPARATIVOS ---
        fig, ax = plt.subplots(figsize=(10, 8))
        
        box_data = [power_soiled.values, power_reference.values]
        box_labels = ['Módulo Sucio', 'Módulo Referencia']
        colors = ['#ff7f0e', '#1f77b4']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       notch=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Personalizar elementos del boxplot
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        plt.setp(bp['means'], marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6)
        
        ax.set_title('Comparación Estadística de Potencias - Mediodía Solar', fontsize=16)
        ax.set_ylabel('Potencia [W]', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=100)
        
        # Agregar estadísticas como texto
        stats_text = f"""Estadísticas Comparativas:
Media Sucio: {power_soiled.mean():.1f}W
Media Referencia: {power_reference.mean():.1f}W
Diferencia Media: {(power_soiled.mean() - power_reference.mean()):.1f}W
SR Promedio: {(power_soiled.mean() / power_reference.mean() * 100):.1f}%
P-value (t-test): {p_value:.4f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_figures:
            save_plot_matplotlib(fig, "pvstand_boxplot_comparativo_solar_noon.png", output_graph_dir)
        if show_figures:
            plt.show(block=True)
        else:
            plt.close(fig)
        
        # --- 4. ESTADÍSTICAS MENSUALES ---
        df_monthly = df_merged[[col_pmax_soiled, col_pmax_reference]].copy()
        df_monthly['Month'] = df_monthly.index.to_period('M')
        
        monthly_stats = df_monthly.groupby('Month').agg({
            col_pmax_soiled: ['mean', 'std', 'count'],
            col_pmax_reference: ['mean', 'std', 'count']
        }).round(2)
        
        monthly_stats.columns = ['Sucio_Mean', 'Sucio_Std', 'Sucio_Count', 
                               'Ref_Mean', 'Ref_Std', 'Ref_Count']
        monthly_stats['SR_Mean_Percent'] = (monthly_stats['Sucio_Mean'] / monthly_stats['Ref_Mean'] * 100).round(1)
        monthly_stats['Diferencia_Mean'] = (monthly_stats['Sucio_Mean'] - monthly_stats['Ref_Mean']).round(1)
        
        # Guardar estadísticas mensuales
        monthly_filename = os.path.join(output_csv_dir, "estadisticas_mensuales_potencias_solar_noon.csv")
        monthly_stats.to_csv(monthly_filename)
        logger.info(f"Estadísticas mensuales guardadas en: {monthly_filename}")
        
        # Gráfico de estadísticas mensuales
        if len(monthly_stats) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Potencias promedio mensuales
            months_str = [str(m) for m in monthly_stats.index]
            ax1.plot(months_str, monthly_stats['Sucio_Mean'], '-o', color='#ff7f0e', 
                    label='Módulo Sucio', linewidth=2, markersize=6)
            ax1.plot(months_str, monthly_stats['Ref_Mean'], '-o', color='#1f77b4', 
                    label='Módulo Referencia', linewidth=2, markersize=6)
            ax1.set_title('Potencias Promedio Mensuales - Mediodía Solar', fontsize=14)
            ax1.set_ylabel('Potencia Promedio [W]', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=100)
            
            # SR promedio mensual
            ax2.plot(months_str, monthly_stats['SR_Mean_Percent'], '-o', color='#2ca02c', 
                    linewidth=2, markersize=6)
            ax2.set_title('Soiling Ratio Promedio Mensual', fontsize=14)
            ax2.set_xlabel('Mes', fontsize=12)
            ax2.set_ylabel('SR [%]', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_figures:
                save_plot_matplotlib(fig, "pvstand_estadisticas_mensuales_solar_noon.png", output_graph_dir)
            if show_figures:
                plt.show(block=True)
            else:
                plt.close(fig)
        
        # --- 5. ESTADÍSTICAS SEMANALES ---
        df_weekly = df_merged[[col_pmax_soiled, col_pmax_reference]].copy()
        df_weekly['Week'] = df_weekly.index.to_period('W')
        
        weekly_stats = df_weekly.groupby('Week').agg({
            col_pmax_soiled: ['mean', 'std', 'count'],
            col_pmax_reference: ['mean', 'std', 'count']
        }).round(2)
        
        weekly_stats.columns = ['Sucio_Mean', 'Sucio_Std', 'Sucio_Count', 
                               'Ref_Mean', 'Ref_Std', 'Ref_Count']
        weekly_stats['SR_Mean_Percent'] = (weekly_stats['Sucio_Mean'] / weekly_stats['Ref_Mean'] * 100).round(1)
        weekly_stats['Diferencia_Mean'] = (weekly_stats['Sucio_Mean'] - weekly_stats['Ref_Mean']).round(1)
        
        # Guardar estadísticas semanales
        weekly_filename = os.path.join(output_csv_dir, "estadisticas_semanales_potencias_solar_noon.csv")
        weekly_stats.to_csv(weekly_filename)
        logger.info(f"Estadísticas semanales guardadas en: {weekly_filename}")
        
        logger.info("Análisis estadístico de potencias completado exitosamente.")
        
        # Retornar DataFrames para Excel consolidado
        return stats_df, monthly_stats, weekly_stats
        
    except Exception as e:
        logger.error(f"Error en análisis estadístico de potencias: {e}", exc_info=True)
        return None, None, None

def _plot_validity_statistics_solar_noon(validez_data, output_dir, save_figures, show_figures):
    """
    Genera gráficos de estadísticas de validez de resamples semanales para mediodía solar.
    
    Args:
        validez_data: Lista de diccionarios con datos de validez por semana y serie
        output_dir: Directorio de salida para los gráficos
        save_figures: Si guardar las figuras
        show_figures: Si mostrar las figuras
    """
    logger.info("Generando gráficos de estadísticas de validez de resamples semanales (MEDIO DÍA SOLAR)...")
    
    try:
        if not validez_data:
            logger.warning("No hay datos de validez para generar gráficos.")
            return
            
        # Convertir a DataFrame para facilitar manipulación
        df_validez = pd.DataFrame(validez_data)
        
        # --- 1. GRÁFICO DE EVOLUCIÓN TEMPORAL DE REPRESENTATIVIDAD POR SERIE ---
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Colores para cada categoría de representatividad
        colors_repr = {
            'Excelente': '#2ca02c',    # Verde
            'Buena': '#1f77b4',        # Azul
            'Aceptable': '#ff7f0e',    # Naranja
            'Limitada': '#d62728'      # Rojo
        }
        
        # Obtener series únicas
        series_unicas = df_validez['Serie'].unique()
        
        for i, serie in enumerate(series_unicas):
            df_serie = df_validez[df_validez['Serie'] == serie].copy()
            df_serie['Semana_Inicio'] = pd.to_datetime(df_serie['Semana_Inicio'])
            df_serie = df_serie.sort_values('Semana_Inicio')
            
            # Crear serie numérica para representatividad
            repr_mapping = {'Excelente': 4, 'Buena': 3, 'Aceptable': 2, 'Limitada': 1}
            df_serie['Repr_Numeric'] = df_serie['Representatividad'].map(repr_mapping)
            
            # Plotear línea base
            ax.plot(df_serie['Semana_Inicio'], df_serie['Repr_Numeric'], 
                   '-', alpha=0.6, linewidth=2, label=f'{serie}')
            
            # Agregar puntos coloreados por representatividad
            for repr_cat, color in colors_repr.items():
                mask = df_serie['Representatividad'] == repr_cat
                if mask.any():
                    ax.scatter(df_serie.loc[mask, 'Semana_Inicio'], 
                             df_serie.loc[mask, 'Repr_Numeric'], 
                             c=color, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Configurar gráfico
        ax.set_title('Evolución Temporal de Representatividad de Resamples Semanales - Mediodía Solar', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Fecha', fontsize=14)
        ax.set_ylabel('Nivel de Representatividad', fontsize=14)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['Limitada', 'Aceptable', 'Buena', 'Excelente'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', title='Series')
        
        # Formatear fechas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Crear leyenda adicional para colores de representatividad
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=categoria) 
                          for categoria, color in colors_repr.items()]
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.legend(handles=legend_elements, loc='upper right', title='Representatividad', 
                  bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_figures:
            save_plot_matplotlib(fig, "validez_representatividad_temporal_solar_noon.png", output_dir)
        if show_figures:
            plt.show(block=True)
        else:
            plt.close(fig)
        
        # --- 2. GRÁFICO DE DISTRIBUCIÓN DE REPRESENTATIVIDAD POR SERIE ---
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Contar representatividad por serie
        repr_counts = df_validez.groupby(['Serie', 'Representatividad']).size().unstack(fill_value=0)
        
        # Crear gráfico de barras apiladas
        repr_counts.plot(kind='bar', stacked=True, ax=ax, 
                        color=[colors_repr.get(col, 'gray') for col in repr_counts.columns])
        
        ax.set_title('Distribución de Representatividad por Serie - Mediodía Solar', fontsize=16, pad=20)
        ax.set_xlabel('Serie', fontsize=14)
        ax.set_ylabel('Número de Semanas', fontsize=14)
        ax.legend(title='Representatividad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_figures:
            save_plot_matplotlib(fig, "validez_distribucion_representatividad_solar_noon.png", output_dir)
        if show_figures:
            plt.show(block=True)
        else:
            plt.close(fig)
        
        # --- 3. GRÁFICO DE MÉTRICAS DE VALIDEZ PROMEDIO ---
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Métricas promedio por serie
        metricas_promedio = df_validez.groupby('Serie').agg({
            'Cobertura_Dias_Pct': 'mean',
            'Num_Puntos_Originales': 'mean', 
            'Coef_Variacion_Pct': 'mean',
            'Sesgo_Cuantil_Pct': 'mean'
        }).round(1)
        
        # Gráfico 1: Cobertura de días
        metricas_promedio['Cobertura_Dias_Pct'].plot(kind='bar', ax=ax1, color='#1f77b4')
        ax1.set_title('Cobertura Promedio de Días por Serie', fontsize=12)
        ax1.set_ylabel('Cobertura [%]', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Número de puntos
        metricas_promedio['Num_Puntos_Originales'].plot(kind='bar', ax=ax2, color='#ff7f0e')
        ax2.set_title('Número Promedio de Puntos por Serie', fontsize=12)
        ax2.set_ylabel('Puntos Originales', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Coeficiente de variación
        metricas_promedio['Coef_Variacion_Pct'].plot(kind='bar', ax=ax3, color='#2ca02c')
        ax3.set_title('Coeficiente de Variación Promedio por Serie', fontsize=12)
        ax3.set_ylabel('Coef. Variación [%]', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Sesgo del cuantil
        metricas_promedio['Sesgo_Cuantil_Pct'].plot(kind='bar', ax=ax4, color='#d62728')
        ax4.set_title('Sesgo Cuantil vs Promedio por Serie', fontsize=12)
        ax4.set_ylabel('Sesgo [%]', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Métricas de Validez Promedio - Mediodía Solar', fontsize=16)
        plt.tight_layout()
        
        if save_figures:
            save_plot_matplotlib(fig, "validez_metricas_promedio_solar_noon.png", output_dir)
        if show_figures:
            plt.show(block=True)
        else:
            plt.close(fig)
        
        # --- 4. GRÁFICO DE EVOLUCIÓN TEMPORAL DE MÉTRICAS ---
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convertir fechas para plotting temporal
        df_validez['Semana_Inicio_dt'] = pd.to_datetime(df_validez['Semana_Inicio'])
        
        # Colores para las series
        colores_series = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, serie in enumerate(series_unicas):
            df_serie = df_validez[df_validez['Serie'] == serie].copy()
            df_serie = df_serie.sort_values('Semana_Inicio_dt')
            color = colores_series[i % len(colores_series)]
            
            # Cobertura de días
            ax1.plot(df_serie['Semana_Inicio_dt'], df_serie['Cobertura_Dias_Pct'], 
                    '-o', color=color, label=serie, markersize=4, alpha=0.8)
            
            # Número de puntos
            ax2.plot(df_serie['Semana_Inicio_dt'], df_serie['Num_Puntos_Originales'], 
                    '-o', color=color, label=serie, markersize=4, alpha=0.8)
            
            # Coeficiente de variación
            ax3.plot(df_serie['Semana_Inicio_dt'], df_serie['Coef_Variacion_Pct'], 
                    '-o', color=color, label=serie, markersize=4, alpha=0.8)
            
            # Sesgo del cuantil
            ax4.plot(df_serie['Semana_Inicio_dt'], df_serie['Sesgo_Cuantil_Pct'], 
                    '-o', color=color, label=serie, markersize=4, alpha=0.8)
        
        # Configurar subgráficos
        ax1.set_title('Evolución Cobertura de Días', fontsize=12)
        ax1.set_ylabel('Cobertura [%]', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        ax2.set_title('Evolución Número de Puntos', fontsize=12)
        ax2.set_ylabel('Puntos Originales', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Evolución Coeficiente de Variación', fontsize=12)
        ax3.set_ylabel('Coef. Variación [%]', fontsize=10)
        ax3.set_xlabel('Fecha', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Evolución Sesgo Cuantil', fontsize=12)
        ax4.set_ylabel('Sesgo [%]', fontsize=10)
        ax4.set_xlabel('Fecha', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Formatear fechas en todos los subgráficos
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        plt.suptitle('Evolución Temporal de Métricas de Validez - Mediodía Solar', fontsize=16)
        plt.tight_layout()
        
        if save_figures:
            save_plot_matplotlib(fig, "validez_evolucion_metricas_solar_noon.png", output_dir)
        if show_figures:
            plt.show(block=True)
        else:
            plt.close(fig)
        
        logger.info("Gráficos de estadísticas de validez generados exitosamente.")
        
    except Exception as e:
        logger.error(f"Error generando gráficos de estadísticas de validez: {e}", exc_info=True)

def run_analysis_solar_noon(solar_window_hours: float = 2.5):
    """
    Función para ejecutar el análisis de PVStand con medio día solar.
    
    Args:
        solar_window_hours: Ventana en horas alrededor del medio día solar (±)
    """
    pv_iv_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_IV_DATA_FILENAME)
    temperature_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_TEMP_DATA_FILENAME)
    return analyze_pvstand_data_solar_noon(pv_iv_data_filepath, temperature_data_filepath, solar_window_hours)

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando análisis de PVStand - Mediodía Solar...")
    pv_iv_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_IV_DATA_FILENAME)
    temperature_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_TEMP_DATA_FILENAME)
    run_analysis_solar_noon(pv_iv_data_filepath, temperature_data_filepath) 