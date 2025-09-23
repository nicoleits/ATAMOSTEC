# --- Importación de Librerías ---
import pandas as pd
import polars as pl
from polars.exceptions import NoDataError, ComputeError
import numpy as np
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
import gc
import re

# Añadir el directorio 'analysis' al PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classes_codes import medio_dia_solar

# --- Matplotlib para gráficos ---
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Tuple, Optional, List, Dict, Any

# --- Configuración de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuración de Rutas Principales ---
BASE_INPUT_DIR = "datos"
BASE_OUTPUT_GRAPH_DIR = "graficos_analisis_integrado_py"
BASE_OUTPUT_CSV_DIR = "datos_procesados_analisis_integrado_py"

# Crear directorios de salida si no existen
os.makedirs(BASE_OUTPUT_GRAPH_DIR, exist_ok=True)
os.makedirs(BASE_OUTPUT_CSV_DIR, exist_ok=True)

# --- Configuración de Polars ---
pl.Config.set_float_precision(4)
pl.Config.set_tbl_rows(15)
pl.Config.set_tbl_cols(15)

# --- Configuración de Pandas ---
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 150)

logger.info(f"Librerías importadas. Directorio de datos de entrada: '{BASE_INPUT_DIR}'")
logger.info(f"Directorio de salida para gráficos: '{BASE_OUTPUT_GRAPH_DIR}'")
logger.info(f"Directorio de salida para CSVs procesados: '{BASE_OUTPUT_CSV_DIR}'")

# --- Definición de Periodo de Análisis General ---
ANALYSIS_START_DATE_GENERAL = pd.to_datetime('2024-07-23', dayfirst=False)
ANALYSIS_END_DATE_GENERAL = pd.to_datetime('2025-12-31', dayfirst=False)

logger.info(f"Periodo de análisis general definido: {ANALYSIS_START_DATE_GENERAL.strftime('%Y-%m-%d')} a {ANALYSIS_END_DATE_GENERAL.strftime('%Y-%m-%d')}")

def save_plot_matplotlib(fig, filename_base, output_dir, subfolder=None, dpi=300):
    """
    Guarda una figura de Matplotlib en el directorio especificado, opcionalmente en un subdirectorio.
    Cierra la figura después de guardarla.
    """
    full_output_dir = output_dir
    if subfolder:
        full_output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(full_output_dir, exist_ok=True)

    filepath = os.path.join(full_output_dir, filename_base)
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Gráfico guardado en: {filepath}")
    except Exception as e:
        logger.error(f"Error al guardar el gráfico en {filepath}: {e}")
    finally:
        plt.close(fig)

def calculate_stats_polars(df: pl.DataFrame, column_name: str, total_rows_before_stats: int) -> dict:
    """
    Calcula estadísticas detalladas para una columna numérica específica de un DataFrame de Polars.
    """
    if column_name not in df.columns:
        logger.error(f"Error interno: Columna '{column_name}' no encontrada para calcular stats (Polars).")
        return {
            'column_name': column_name, 'total_data_rows_processed': total_rows_before_stats,
            'rows_in_final_df': 0, 'valid_numbers': 0, 'nan_count': 0, 'negative_count': 0,
            'q1': None, 'q3': None, 'iqr': None,
            'iqr_lower_bound': None, 'iqr_upper_bound': None,
            'outlier_count': 0, 'outlier_percentage': 0.0
        }

    col_series = df.get_column(column_name)
    rows_in_current_df = len(df)

    if not col_series.dtype.is_numeric():
        logger.warning(f"Columna '{column_name}' no es numérica. Saltando cálculo de stats (Polars).")
        return {
            'column_name': column_name, 'total_data_rows_processed': total_rows_before_stats,
            'rows_in_final_df': rows_in_current_df,
            'valid_numbers': 0, 'nan_count': rows_in_current_df, 'negative_count': 0,
            'q1': None, 'q3': None, 'iqr': None,
            'iqr_lower_bound': None, 'iqr_upper_bound': None,
            'outlier_count': 0, 'outlier_percentage': 0.0
        }

    stats = {
        'column_name': column_name,
        'total_data_rows_processed': total_rows_before_stats,
        'rows_in_final_df': rows_in_current_df,
        'valid_numbers': 0, 'nan_count': 0, 'negative_count': 0,
        'q1': None, 'q3': None, 'iqr': None,
        'iqr_lower_bound': None, 'iqr_upper_bound': None,
        'outlier_count': 0,
        'outlier_percentage': 0.0
    }

    stats['nan_count'] = col_series.is_null().sum()
    stats['valid_numbers'] = rows_in_current_df - stats['nan_count']

    if stats['valid_numbers'] > 0:
        non_null_series = col_series.filter(col_series.is_not_null())
        stats['negative_count'] = (non_null_series < 0).sum()
        
        try:
            stats['q1'] = non_null_series.quantile(0.25, interpolation='linear')
            stats['q3'] = non_null_series.quantile(0.75, interpolation='linear')
        except ComputeError as e:
            logger.warning(f"No se pudieron calcular los cuantiles para '{column_name}': {e}")
            stats['q1'] = None
            stats['q3'] = None

        if stats['q1'] is not None and stats['q3'] is not None:
            stats['iqr'] = stats['q3'] - stats['q1']
            if stats['iqr'] is not None:
                lower_bound = stats['q1'] - 1.5 * stats['iqr']
                upper_bound = stats['q3'] + 1.5 * stats['iqr']
                stats['iqr_lower_bound'] = lower_bound
                stats['iqr_upper_bound'] = upper_bound
                
                if lower_bound is not None and upper_bound is not None:
                    outliers = non_null_series.filter(
                        (non_null_series < lower_bound) | (non_null_series > upper_bound)
                    )
                    stats['outlier_count'] = len(outliers)
                    stats['outlier_percentage'] = (stats['outlier_count'] / stats['valid_numbers']) * 100 if stats['valid_numbers'] > 0 else 0.0
                else:
                    logger.warning(f"Límites IQR son None para '{column_name}', no se calculan outliers.")
            else:
                logger.warning(f"IQR es None para '{column_name}' (Q1: {stats['q1']}, Q3: {stats['q3']}). No se calculan límites ni outliers.")
        else:
            logger.warning(f"No se pudo calcular IQR/Outliers para '{column_name}' (Q1: {stats['q1']}, Q3: {stats['q3']}).")
    else:
        logger.info(f"No hay números válidos en la columna '{column_name}' para calcular estadísticas detalladas.")
        
    return stats

def plot_transmitancia_polars(
    df: pl.DataFrame, 
    time_col: str, 
    value_cols: list[str], 
    output_filename_base: str, 
    output_graph_dir: str,
    title_suffix:str = "", 
    is_daily:bool = False,
    subfolder:str = "transmitancia"
):
    """
    Genera y guarda un gráfico de serie temporal para datos de Polars DataFrame.
    """
    if df is None or df.is_empty():
        logger.warning(f"DataFrame para graficar ('{output_filename_base}') está vacío o no existe.")
        return

    if time_col not in df.columns or not isinstance(df[time_col].dtype, pl.Datetime):
        logger.warning(f"La columna de tiempo '{time_col}' no es Datetime o no existe. No se generará gráfico para {output_filename_base}.")
        return
    
    time_data_pd = df[time_col].to_pandas()
    xlabel = f"Tiempo ({time_col}) - UTC"

    plot_cols = [c for c in value_cols if c in df.columns and df[c].dtype.is_numeric()]
    if not plot_cols:
        logger.warning(f"No se encontraron columnas de valores numéricas válidas para graficar en {output_filename_base}.")
        return

    logger.info(f"Generando gráfico {'diario' if is_daily else 'minuto a minuto'} para columnas: {plot_cols}...")
    
    fig, ax = plt.subplots(figsize=(15, 8))

    marker = 'o-' if is_daily else '-'
    linewidth = 1.5 if is_daily else 1.0
    markersize = 4 if is_daily else 2

    for col in plot_cols:
        value_data_pd = df[col].to_pandas()
        ax.plot(time_data_pd, value_data_pd, marker, label=col, linewidth=linewidth, markersize=markersize)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Average Value (Avg)", fontsize=12)
    plot_type_desc = "Daily (Aggregated Average)" if is_daily else "Minute by Minute"
    ax.set_title(f"Temporal Transmittance Series {plot_type_desc}{title_suffix}", fontsize=14)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    
    fig.autofmt_xdate()
    
    if is_daily:
        date_format = '%Y-%m-%d'
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        if len(time_data_pd) > 1:
            days_span = (time_data_pd.iloc[-1] - time_data_pd.iloc[0]).days
            if days_span <= 35:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                ax.xaxis.set_minor_locator(mdates.DayLocator())
            elif days_span <= 180:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
            else:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, days_span // 365 * 2)))
    else:
        date_format = '%Y-%m-%d %H:%M'
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

    ax.tick_params(axis='x', rotation=30, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_png_file = f"{output_filename_base}.png"
    save_plot_matplotlib(fig, output_png_file, output_graph_dir, subfolder=subfolder) 

def analizar_transmitancia_pv_glasses(
    file_path: str,
    output_csv_dir: str,
    output_graph_dir: str,
    column_pattern: str = r"R_FC([1-5])_Avg",
    time_column: str = '_time',
    filter_solar_noon: bool = True,
    solar_noon_start_hour: int = 14,
    solar_noon_end_hour: int = 16,
    usar_mediodia_solar_real: bool = True,
    intervalo_minutos_mediodia: int = 60,
    remove_outliers_iqr: bool = True,
    ref_col1_name: str = "R_FC1_Avg",
    ref_col2_name: str = "R_FC2_Avg"
):
    """
    Analiza los datos de transmitancia de PV Glasses.
    Filtra opcionalmente por mediodía solar (fijo o real),
    realiza limpieza, filtra outliers, calcula REF,
    remuestrea a media diaria y guarda resultados y gráficos.
    """
    filter_desc = ""
    if filter_solar_noon:
        if usar_mediodia_solar_real:
            filter_desc = f" (filtrado MediodiaSolarReal +/- {intervalo_minutos_mediodia}min)"
        else:
            filter_desc = f" (filtrado {solar_noon_start_hour:02d}-{solar_noon_end_hour:02d} UTC)"

    outlier_desc = " (outliers IQR removidos)" if remove_outliers_iqr else ""
    temp_processing_id = f"transmitancia{filter_desc}{outlier_desc}"
    temp_processing_id = temp_processing_id.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '').replace('/', '').replace('\\', '')
    temp_processing_id = re.sub(r'[^a-zA-Z0-9_]', '', temp_processing_id)
    processing_id = temp_processing_id

    SUBFOLDER_TRANSMITANCIA = "transmitancia_pv_glasses"
    os.makedirs(os.path.join(output_csv_dir, SUBFOLDER_TRANSMITANCIA), exist_ok=True)
    os.makedirs(os.path.join(output_graph_dir, SUBFOLDER_TRANSMITANCIA), exist_ok=True)

    logger.info(f"Iniciando análisis de transmitancia para: {file_path}{filter_desc}{outlier_desc} (ID: {processing_id})")

    summary_df_original = None
    summary_df_cleaned_final = None
    cleaned_final_df = None
    daily_resampled_final_df = None

    try:
        logger.info(f"Cargando datos desde {file_path}...")
        try:
            lf = pl.scan_csv(file_path, separator=',', has_header=True, try_parse_dates=False)
        except Exception as e:
            logger.error(f"Error inicial al escanear CSV {file_path}: {e}")
            return None, None, None, None

        original_headers = lf.columns
        if time_column not in original_headers:
            logger.error(f"Error: Columna de tiempo '{time_column}' no encontrada en {original_headers}.")
            return None, None, None, None

        initial_target_numeric_columns = [
            col for col in original_headers if re.match(column_pattern, col)
        ]

        if not initial_target_numeric_columns:
            logger.error(f"Error: No se encontraron columnas numéricas que coincidan con el patrón '{column_pattern}' en {original_headers}")
            return None, None, None, None

        can_calculate_ref = ref_col1_name in initial_target_numeric_columns and ref_col2_name in initial_target_numeric_columns
        if not can_calculate_ref:
            logger.warning(f"No se puede calcular la columna REF porque '{ref_col1_name}' o '{ref_col2_name}' no están en las columnas objetivo.")
        logger.info(f"Columnas numéricas objetivo identificadas: {initial_target_numeric_columns}")

        data_df = lf.collect()

        logger.info(f"Convirtiendo columna '{time_column}' a datetime naive...")
        try:
            if data_df[time_column].dtype == pl.Object or data_df[time_column].dtype == pl.Utf8:
                try:
                    data_df = data_df.with_columns(pl.col(time_column).str.to_datetime("%Y-%m-%d %H:%M:%S%.f%z", strict=False, time_zone="UTC").dt.replace_time_zone(None))
                except pl.exceptions.ComputeError:
                    try:
                        data_df = data_df.with_columns(pl.col(time_column).str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False).dt.replace_time_zone(None))
                    except pl.exceptions.ComputeError:
                        data_df = data_df.with_columns(pl.col(time_column).str.to_datetime(strict=False).dt.replace_time_zone(None))
            elif isinstance(data_df[time_column].dtype, pl.Datetime):
                if data_df[time_column].dt.time_zone() is not None:
                    data_df = data_df.with_columns(pl.col(time_column).dt.replace_time_zone(None))
            else:
                raise ValueError(f"Tipo de columna de tiempo no esperado: {data_df[time_column].dtype}")
            data_df = data_df.with_columns(pl.col(time_column).cast(pl.Datetime).dt.cast_time_unit("us"))
            logger.info(f"Columna '{time_column}' convertida exitosamente a Datetime naive.")
        except Exception as e:
            logger.error(f"Fallo crítico al convertir la columna '{time_column}' a Datetime naive: {e}")
            return None, None, None, None

        for col_c in initial_target_numeric_columns:
            data_df = data_df.with_columns(pl.col(col_c).cast(pl.Float64, strict=False))

        total_rows_loaded = len(data_df)
        logger.info(f"Datos cargados y casteados. Filas totales: {total_rows_loaded}")
        if data_df.is_empty():
            logger.warning("El DataFrame está vacío después de la carga y casteo.")
            return None, None, None, None

        time_col_is_datetime = isinstance(data_df[time_column].dtype, pl.Datetime)
        data_df_processed = data_df

        if filter_solar_noon and time_col_is_datetime:
            if usar_mediodia_solar_real:
                logger.info(f"Filtrando datos usando mediodía solar real +/- {intervalo_minutos_mediodia} minutos.")
                if data_df_processed.is_empty():
                    logger.warning("DataFrame vacío antes de calcular mediodía solar real. Saltando filtro.")
                else:
                    min_date_pv_dt = data_df_processed[time_column].min()
                    max_date_pv_dt = data_df_processed[time_column].max()

                    if min_date_pv_dt is None or max_date_pv_dt is None:
                        logger.warning("No se pudieron obtener fechas min/max de los datos PV. Saltando filtro de mediodía solar real.")
                    else:
                        min_date_pv_str = min_date_pv_dt.strftime('%Y-%m-%d')
                        max_date_pv_str = max_date_pv_dt.strftime('%Y-%m-%d')
                        logger.info(f"Rango de fechas de datos PV: {min_date_pv_str} a {max_date_pv_str}")
                        try:
                            calculador_mediodia = medio_dia_solar(
                                datei=min_date_pv_str,
                                datef=max_date_pv_str,
                                freq="1d",
                                inter=intervalo_minutos_mediodia
                            )
                            df_intervalos_pd = calculador_mediodia.msd()
                            logger.info(f"DataFrame de intervalos de Pandas (df_intervalos_pd) head:\n{df_intervalos_pd.head().to_string()}")

                            df_intervalos_pd = df_intervalos_pd.rename(columns={0: 'SolarNoon_Time_i', 1: 'SolarNoon_Time_f'})
                            
                            df_intervalos_pd['SolarNoon_Time_i'] = pd.to_datetime(df_intervalos_pd['SolarNoon_Time_i'])
                            df_intervalos_pd['fecha_para_join'] = df_intervalos_pd['SolarNoon_Time_i'].dt.date

                            df_intervalos_pl = pl.from_pandas(df_intervalos_pd)

                            df_intervalos_pl = df_intervalos_pl.with_columns([
                                pl.col('SolarNoon_Time_i').cast(pl.Datetime).dt.time().alias('SolarNoon_Time_i_time'),
                                pl.col('SolarNoon_Time_f').cast(pl.Datetime).dt.time().alias('SolarNoon_Time_f_time'),
                                pl.col('fecha_para_join').cast(pl.Date).alias('fecha_para_join')
                            ])

                            data_df_processed_with_date = data_df_processed.with_columns(
                                pl.col(time_column).dt.date().alias('fecha_para_join')
                            )

                            data_df_joined = data_df_processed_with_date.join(
                                df_intervalos_pl.select(['fecha_para_join', 'SolarNoon_Time_i_time', 'SolarNoon_Time_f_time']),
                                on='fecha_para_join',
                                how='left'
                            )

                            data_df_filtered = data_df_joined.filter(
                                pl.col(time_column).dt.time().is_between(
                                    pl.col('SolarNoon_Time_i_time'),
                                    pl.col('SolarNoon_Time_f_time'),
                                    closed='both'
                                ) & pl.col('SolarNoon_Time_i_time').is_not_null()
                            ).drop(['fecha_para_join', 'SolarNoon_Time_i_time', 'SolarNoon_Time_f_time'])

                            logger.info(f"Filas restantes después del filtrado por mediodía solar real: {len(data_df_filtered)}")
                            data_df_processed = data_df_filtered
                            if data_df_processed.is_empty():
                                logger.warning("DataFrame vacío después del filtro por mediodía solar real.")
                        except NameError:
                            logger.error("La clase medio_dia_solar no está definida. Asegúrate de que esté importada correctamente. Usando filtro horario fijo si está configurado.")
                            if solar_noon_start_hour is not None and solar_noon_end_hour is not None:
                                logger.info(f"Fallback: Filtrando datos entre {solar_noon_start_hour:02d}:00 y {solar_noon_end_hour:02d}:59 (horas del día, naive)...")
                                data_df_filtered = data_df_processed.filter(
                                    pl.col(time_column).dt.hour().is_between(solar_noon_start_hour, solar_noon_end_hour, closed='both')
                                )
                                data_df_processed = data_df_filtered
                        except Exception as e_msd:
                            logger.error(f"Error al calcular o aplicar el filtro de mediodía solar real: {e_msd}. Usando filtro horario fijo si está configurado.", exc_info=True)
                            if solar_noon_start_hour is not None and solar_noon_end_hour is not None:
                                logger.info(f"Fallback: Filtrando datos entre {solar_noon_start_hour:02d}:00 y {solar_noon_end_hour:02d}:59 (horas del día, naive)...")
                                data_df_filtered = data_df_processed.filter(
                                    pl.col(time_column).dt.hour().is_between(solar_noon_start_hour, solar_noon_end_hour, closed='both')
                                )
                                data_df_processed = data_df_filtered
            else:
                logger.info(f"Filtrando datos entre {solar_noon_start_hour:02d}:00 y {solar_noon_end_hour:02d}:59 (horas del día, naive)...")
                data_df_filtered = data_df_processed.filter(
                    pl.col(time_column).dt.hour().is_between(solar_noon_start_hour, solar_noon_end_hour, closed='both')
                )
                logger.info(f"Filas restantes después del filtrado por tiempo: {len(data_df_filtered)}")
                data_df_processed = data_df_filtered
                if data_df_processed.is_empty():
                    logger.warning("DataFrame vacío después del filtro horario.")
        elif filter_solar_noon and not time_col_is_datetime:
            logger.warning("Filtrado por tiempo solicitado pero columna de tiempo no es Datetime.")

        total_rows_after_time_filter = len(data_df_processed)
        logger.info("Calculando estadísticas sobre datos originales (potencialmente filtrados por tiempo)...")
        results_list_original = []
        if not data_df_processed.is_empty():
            for col_name in initial_target_numeric_columns:
                stats_orig = calculate_stats_polars(data_df_processed.clone(), col_name, total_rows_after_time_filter)
                results_list_original.append(stats_orig)
        summary_df_original = pl.DataFrame(results_list_original)

        cleaned_data_df_after_basic_clean = data_df_processed.clone()
        if not cleaned_data_df_after_basic_clean.is_empty():
            logger.info("Limpiando datos numéricos (Interpolar, Rellenar NaNs con 0, Cortar Negativos)...")
            cleaned_data_df_after_basic_clean = cleaned_data_df_after_basic_clean.with_columns([
                pl.col(c).interpolate().fill_null(0).clip(lower_bound=0).alias(c) for c in initial_target_numeric_columns
            ])

        cleaned_final_df = cleaned_data_df_after_basic_clean.clone()
        if remove_outliers_iqr and not cleaned_final_df.is_empty():
            logger.info("Calculando límites IQR y filtrando outliers...")
            outlier_bounds = {}
            for col_name in initial_target_numeric_columns:
                col_series = cleaned_final_df.get_column(col_name)
                if col_series.is_not_null().sum() > 0:
                    q1 = col_series.quantile(0.25, interpolation='linear')
                    q3 = col_series.quantile(0.75, interpolation='linear')
                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        if iqr is not None:
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outlier_bounds[col_name] = (lower_bound, upper_bound)

            temp_df_for_outlier_removal = cleaned_final_df.clone()
            for col_name, bounds_tuple in outlier_bounds.items():
                lower, upper = bounds_tuple
                if lower is not None and upper is not None:
                    temp_df_for_outlier_removal = temp_df_for_outlier_removal.filter(
                        pl.col(col_name).is_between(lower, upper, closed='both')
                    )
            cleaned_final_df = temp_df_for_outlier_removal
            logger.info(f"Filas restantes después de la eliminación de outliers: {len(cleaned_final_df)}")
            if cleaned_final_df.is_empty():
                logger.warning("DataFrame vacío después de la eliminación de outliers.")

        final_target_numeric_columns_for_stats = list(initial_target_numeric_columns)
        if can_calculate_ref and cleaned_final_df is not None and not cleaned_final_df.is_empty():
            logger.info(f"Calculando columna REF como promedio de {ref_col1_name} y {ref_col2_name}...")
            cleaned_final_df = cleaned_final_df.with_columns(
                (((pl.col(ref_col1_name) + pl.col(ref_col2_name)) / 2.0)).alias("REF")
            )
            if "REF" not in final_target_numeric_columns_for_stats:
                final_target_numeric_columns_for_stats.append("REF")

        if cleaned_final_df is not None and not cleaned_final_df.is_empty():
            logger.info("Calculando estadísticas sobre datos finales limpios...")
            results_list_cleaned_final = []
            for col_name in final_target_numeric_columns_for_stats:
                if col_name in cleaned_final_df.columns:
                    stats_clean = calculate_stats_polars(cleaned_final_df.clone(), col_name, total_rows_after_time_filter)
                    results_list_cleaned_final.append(stats_clean)
            summary_df_cleaned_final = pl.DataFrame(results_list_cleaned_final)

        daily_resampled_final_df = None
        if cleaned_final_df is not None and not cleaned_final_df.is_empty() and time_col_is_datetime:
            logger.info("Remuestreando datos finales limpios a frecuencia diaria (agregación promedio)...")
            try:
                cols_to_resample = [c for c in final_target_numeric_columns_for_stats if c in cleaned_final_df.columns]
                if cols_to_resample:
                    daily_resampled_final_df = cleaned_final_df.group_by_dynamic(
                        index_column=time_column, every="1d", period="1d", offset="0h", closed='left'
                    ).agg([pl.mean(c).alias(c) for c in cols_to_resample]).sort(time_column)
                    logger.info(f"Remuestreo diario completo. Forma: {daily_resampled_final_df.shape if daily_resampled_final_df is not None else 'N/A'}")
            except Exception as e:
                logger.error(f"Error durante el remuestreo diario: {e}. Saltando remuestreo.")

        base_col_order = ["column_name", "total_data_rows_processed", "rows_in_final_df", "valid_numbers", "nan_count", "negative_count", "q1", "q3", "iqr", "iqr_lower_bound", "iqr_upper_bound", "outlier_count", "outlier_percentage"]

        if summary_df_original is not None and not summary_df_original.is_empty():
            cols_to_select_orig = [c for c in base_col_order if c in summary_df_original.columns]
            summary_df_original_ordered = summary_df_original.select(cols_to_select_orig)
            summary_orig_filename = os.path.join(output_csv_dir, SUBFOLDER_TRANSMITANCIA, f"{processing_id}_summary_original.csv")
            summary_df_original_ordered.write_csv(summary_orig_filename)
            logger.info(f"Resumen de estadísticas originales guardado en: {summary_orig_filename}")

        if summary_df_cleaned_final is not None and not summary_df_cleaned_final.is_empty():
            cols_to_select_clean = [c for c in base_col_order if c in summary_df_cleaned_final.columns]
            summary_df_cleaned_final_ordered = summary_df_cleaned_final.select(cols_to_select_clean)
            summary_clean_filename = os.path.join(output_csv_dir, SUBFOLDER_TRANSMITANCIA, f"{processing_id}_summary_cleaned_final.csv")
            summary_df_cleaned_final_ordered.write_csv(summary_clean_filename)
            logger.info(f"Resumen de estadísticas limpias finales guardado en: {summary_clean_filename}")

        if cleaned_final_df is not None and not cleaned_final_df.is_empty():
            clean_data_filename = os.path.join(output_csv_dir, SUBFOLDER_TRANSMITANCIA, f"{processing_id}_data_cleaned_final.csv")
            cleaned_final_df.write_csv(clean_data_filename, datetime_format="%Y-%m-%dT%H:%M:%S")
            logger.info(f"Datos finales limpios (minuto a minuto) guardados en: {clean_data_filename}")

        if daily_resampled_final_df is not None and not daily_resampled_final_df.is_empty():
            daily_data_filename = os.path.join(output_csv_dir, SUBFOLDER_TRANSMITANCIA, f"{processing_id}_data_daily_resampled_final.csv")
            daily_resampled_final_df.write_csv(daily_data_filename, datetime_format="%Y-%m-%d")
            logger.info(f"Datos diarios remuestreados guardados en: {daily_data_filename}")

        logger.info(f"--- Iniciando graficado general de transmitancia de PV Glasses ---")
        current_plot_title_suffix = ""
        if filter_solar_noon:
            if usar_mediodia_solar_real:
                current_plot_title_suffix += f" (MediodiaSolarReal +/- {intervalo_minutos_mediodia}min)"
            else:
                current_plot_title_suffix += f" ({solar_noon_start_hour:02d}-{solar_noon_end_hour:02d} UTC)"
        if remove_outliers_iqr:
            current_plot_title_suffix += " (Outliers IQR Removidos)"

        if cleaned_final_df is not None and not cleaned_final_df.is_empty():
            plot_cols_min_general = [c for c in final_target_numeric_columns_for_stats if c in cleaned_final_df.columns]
            if plot_cols_min_general:
                plot_transmitancia_polars(
                    cleaned_final_df, time_column, plot_cols_min_general,
                    f"{processing_id}_minuto_a_minuto_GENERAL", output_graph_dir,
                    title_suffix=current_plot_title_suffix + " (General)", is_daily=False, subfolder=SUBFOLDER_TRANSMITANCIA
                )

        if daily_resampled_final_df is not None and not daily_resampled_final_df.is_empty():
            plot_cols_daily_general = [c for c in final_target_numeric_columns_for_stats if c in daily_resampled_final_df.columns]
            if plot_cols_daily_general:
                plot_transmitancia_polars(
                    daily_resampled_final_df, time_column, plot_cols_daily_general,
                    f"{processing_id}_diario_promedio_GENERAL", output_graph_dir,
                    title_suffix=current_plot_title_suffix + " (General Diario)", is_daily=True, subfolder=SUBFOLDER_TRANSMITANCIA
                )

        logger.info(f"--- Análisis de Transmitancia para PV Glasses Finalizado ({processing_id}) ---")
        return summary_df_original, summary_df_cleaned_final, cleaned_final_df, daily_resampled_final_df

    except FileNotFoundError:
        logger.error(f"Error: Archivo no encontrado en {file_path}")
        return None, None, None, None
    except (pl.exceptions.NoDataError, pl.exceptions.SchemaError, ValueError, pl.exceptions.ComputeError) as e:
        logger.error(f"Error específico de Polars o de datos procesando {file_path}: {e}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Un error inesperado ocurrió durante el análisis de transmitancia: {e}", exc_info=True)
        return None, None, None, None 

def analizar_calendario_muestras(
    file_path: str,
    output_csv_dir: str,
    sheet_name: str = "Hoja1"
):
    """
    Analiza el archivo Excel que contiene el calendario de toma de muestras de soiling.
    Genera archivos CSV con los datos procesados y agrupados.
    """
    logger.info("--- Iniciando Sección 3: Análisis de Calendario de Muestras y Generación de CSV de Periodos ---")
    logger.info(f"Cargando datos del calendario de muestras desde: {file_path}, hoja: '{sheet_name}'")

    try:
        df_calendario = pd.read_excel(file_path, sheet_name=sheet_name)
        logger.info(f"Datos del calendario cargados exitosamente desde la hoja '{sheet_name}'.")

        # Limpieza de nombres de columna
        df_calendario.columns = df_calendario.columns.str.strip()

        # Convertir columnas de fecha a datetime
        cols_fechas = ['Inicio Exposición', 'Fecha medición']
        for col in cols_fechas:
            if col in df_calendario.columns:
                df_calendario[col] = pd.to_datetime(df_calendario[col], errors='coerce')
            else:
                logger.warning(f"Columna de fecha esperada '{col}' no encontrada en la hoja '{sheet_name}'.")
        
        # Renombrar 'Fecha medición' a 'Fin Exposicion'
        if 'Fecha medición' in df_calendario.columns:
            df_calendario.rename(columns={'Fecha medición': 'Fin Exposicion'}, inplace=True)
            logger.info("Columna 'Fecha medición' renombrada a 'Fin Exposicion'.")
        else:
            if 'Fin Exposicion' not in df_calendario.columns:
                logger.warning("No se encontró la columna 'Fecha medición' para renombrar, ni una columna 'Fin Exposicion' ya existente.")

        # Guardado del DataFrame de Calendario con Columnas Seleccionadas
        cols_originales_a_mantener = [
            'Inicio Exposición', 'Fin Exposicion', 'Estructura', 'Exposición', 
            'Periodo', 'Masa A', 'Masa B', 'Masa C', 'Estado'
        ]
        cols_existentes_seleccionadas = [col for col in cols_originales_a_mantener if col in df_calendario.columns]

        if cols_existentes_seleccionadas:
            df_calendario_seleccionado = df_calendario[cols_existentes_seleccionadas].copy()
            path_csv_seleccionado = os.path.join(output_csv_dir, 'calendario_muestras_seleccionado.csv')
            logger.info(f"Procesando para CSV (columnas seleccionadas): {cols_existentes_seleccionadas}")
            try:
                df_calendario_seleccionado.to_csv(path_csv_seleccionado, index=False, date_format='%Y-%m-%d')
                logger.info(f"DataFrame con columnas seleccionadas guardado en: {path_csv_seleccionado}")
            except Exception as e:
                logger.error(f"Error al guardar el CSV de calendario seleccionado: {e}")
        else:
            logger.warning("No se pudieron seleccionar columnas para el CSV ya que ninguna de las especificadas existe.")

        # Análisis de Calendario: Agrupar por Periodo para Estructura 'Fija a RC'
        logger.info("Iniciando análisis de calendario: Agrupar por Periodo para Estructura 'Fija a RC'")

        required_cols_for_new_analysis = ['Estructura', 'Periodo', 'Fin Exposicion']
        missing_cols = [col for col in required_cols_for_new_analysis if col not in df_calendario.columns]

        if missing_cols:
            logger.error(f"Faltan columnas necesarias para el nuevo análisis en el DataFrame: {missing_cols}. Abortando esta parte del análisis.")
            print(f"Error: Faltan columnas necesarias para el nuevo análisis en el DataFrame: {missing_cols}.")
            df_resultado_fija_rc = pd.DataFrame()
        else:
            df_fija_rc = df_calendario[
                df_calendario['Estructura'].fillna('').astype(str).str.strip() == 'Fija a RC'
            ].copy()

            if df_fija_rc.empty:
                logger.warning("No se encontraron datos con Estructura 'Fija a RC'.")
                print("No se encontraron datos con Estructura 'Fija a RC'.")
                df_resultado_fija_rc = pd.DataFrame()
            else:
                df_fija_rc.dropna(subset=['Fin Exposicion', 'Periodo'], inplace=True)

                if df_fija_rc.empty:
                    logger.warning("No se encontraron datos con Estructura 'Fija a RC' que tuvieran 'Periodo' y 'Fin Exposicion' válidos después de la limpieza.")
                    print("No se encontraron datos con Estructura 'Fija a RC' que tuvieran 'Periodo' y 'Fin Exposicion' válidos.")
                    df_resultado_fija_rc = pd.DataFrame()
                else:
                    df_resultado_fija_rc = df_fija_rc.groupby('Periodo')['Fin Exposicion'].apply(
                        lambda dates: sorted(list(dates.dropna().unique()))
                    ).reset_index()
                    
                    df_resultado_fija_rc.rename(columns={'Fin Exposicion': 'Fechas Fin Exposicion (Fija a RC)'}, inplace=True)

                    logger.info("Resultado del análisis de calendario para 'Fija a RC':")
                    print("\nResultado del análisis de calendario para 'Fija a RC':")
                    print(df_resultado_fija_rc.to_string())

                    path_csv_fija_rc_periodos = os.path.join(output_csv_dir, 'calendario_fija_rc_por_periodo.csv')
                    try:
                        df_to_save = df_resultado_fija_rc.copy()
                        
                        def format_dates_list(date_list):
                            if isinstance(date_list, list) and all(isinstance(d, pd.Timestamp) for d in date_list):
                                return [d.strftime('%Y-%m-%d') for d in date_list]
                            return date_list

                        df_to_save['Fechas Fin Exposicion (Fija a RC)'] = df_to_save['Fechas Fin Exposicion (Fija a RC)'].apply(format_dates_list)
                        df_to_save['Fechas Fin Exposicion (Fija a RC)'] = df_to_save['Fechas Fin Exposicion (Fija a RC)'].astype(str)
                        
                        df_to_save.to_csv(path_csv_fija_rc_periodos, index=False)
                        logger.info(f"DataFrame agrupado por periodo para 'Fija a RC' guardado en: {path_csv_fija_rc_periodos}")
                    except Exception as e:
                        logger.error(f"Error al guardar el CSV de 'Fija a RC' por periodo: {e}")
                        print(f"Error al guardar el CSV '{path_csv_fija_rc_periodos}': {e}")

        logger.info("--- Fin Sección 3: Análisis de Calendario de Muestras y Generación de CSV de Periodos ---")
        return df_calendario, df_resultado_fija_rc

    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {file_path}")
        print(f"Error: No se encontró el archivo del calendario en la ruta: {file_path}")
        return None, None
    except pd.errors.EmptyDataError:
        logger.error(f"La hoja '{sheet_name}' en el archivo Excel {file_path} está vacía o no se pudo leer.")
        print(f"Error: La hoja '{sheet_name}' en el archivo Excel {file_path} está vacía o no se pudo leer.")
        return None, None
    except KeyError as e:
        logger.error(f"Error de clave al procesar la hoja '{sheet_name}' del archivo Excel: {e}. Verifica los nombres de las columnas.")
        print(f"Error de clave: {e}. Asegúrate de que las columnas esperadas existan en la hoja '{sheet_name}'.")
        return None, None
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado al procesar el calendario (hoja '{sheet_name}'): {e}", exc_info=True)
        print(f"Error inesperado: {e}")
        return None, None 

def plot_soiling_ratios_por_periodo(
    df: pl.DataFrame,
    output_graph_dir: str,
    subfolder: str = "pv_glasses"
):
    """
    Genera gráficos de Soiling Ratios por periodo y un gráfico de barras de promedios.
    """
    if df is None or df.is_empty():
        logger.warning("DataFrame para graficar Soiling Ratios está vacío o no existe.")
        return

    # Crear subdirectorio para los gráficos
    full_output_graph_path = os.path.join(output_graph_dir, subfolder)
    os.makedirs(full_output_graph_path, exist_ok=True)

    # --- Columnas a graficar y mapeo a columnas de Masa (CORREGIDO, igual que el notebook) ---
    correspondencia_sr_masa = {
        'SR_R_FC3': 'Masa_C_Referencia',  # SR_R_FC3 se asocia con Masa C
        'SR_R_FC4': 'Masa_B_Referencia',  # SR_R_FC4 se asocia con Masa B
        'SR_R_FC5': 'Masa_A_Referencia'   # SR_R_FC5 se asocia con Masa A
    }

    # Verificar que las columnas necesarias existan
    columnas_requeridas = ['_time', 'Periodo_Referencia', 'Fecha_Fin_Exposicion_Referencia'] + \
                         list(correspondencia_sr_masa.keys()) + list(correspondencia_sr_masa.values())
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if columnas_faltantes:
        logger.error(f"Faltan columnas necesarias para graficar Soiling Ratios: {columnas_faltantes}")
        return

    # Obtener periodos únicos
    periodos_unicos = df['Periodo_Referencia'].unique().sort()
    logger.info(f"Periodos únicos encontrados: {periodos_unicos}")

    # Diccionario para traducir períodos al inglés
    traduccion_periodos = {
        'semanal': 'weekly',
        '2 semanas': '2 weeks', 
        'Mensual': 'monthly',
        'Trimestral': 'quarterly',
        'Cuatrimestral': '4-monthly',
        'Semestral': 'semiannual',
        '1 año': '1 year'
    }

    # Lista para almacenar datos para el gráfico de barras
    datos_para_grafico_barras = []

    # Generar gráfico para cada periodo
    for periodo in periodos_unicos:
        logger.info(f"\n--- Generando gráfico para el periodo: {periodo} ---")
        
        # Filtrar datos para el periodo actual
        df_periodo = df.filter(pl.col('Periodo_Referencia') == periodo)
        
        if df_periodo.is_empty():
            logger.warning(f"No hay datos para el periodo {periodo}")
            continue

        # Calcular promedios para el gráfico de barras
        periodo_traducido = traduccion_periodos.get(periodo, periodo)  # Usar traducción o mantener original si no está en el diccionario
        promedios_periodo = {
            'Periodo': periodo_traducido
        }

        # Crear figura para el periodo
        fig, ax = plt.subplots(figsize=(15, 8))

        # Graficar cada SR con su masa correspondiente (filtrado por fila, solo para esa curva)
        for sr_col, masa_col in correspondencia_sr_masa.items():
            if sr_col in df_periodo.columns and masa_col in df_periodo.columns:
                df_filtrado = df_periodo.filter(pl.col(masa_col) > 0)
                fechas = df_filtrado['_time']
                valores_sr = df_filtrado[sr_col]

                logger.info(f"Graficando {sr_col} (solo donde {masa_col} > 0): {len(valores_sr)} puntos")

                if len(valores_sr) > 0:
                    valores_sr_porcentaje = valores_sr * 100  # Convertir a porcentaje
                    ax.plot(fechas, valores_sr_porcentaje, 'o-', label=f"{sr_col} (Masa {masa_col[-1]} > 0)")
                    promedio_original = valores_sr.mean() * 100  # Convertir a porcentaje
                    promedio_corregido = promedio_original + 7.5  # Aplicar corrección de +7.5%
                    promedios_periodo[f'Promedio_{sr_col}'] = promedio_corregido
                    logger.info(f"{sr_col}: promedio original = {promedio_original:.2f}%, corregido (+7.5%) = {promedio_corregido:.2f}%")
                else:
                    logger.warning(f"No se graficó {sr_col} porque no hay datos con {masa_col} > 0 en este periodo")

        # Calcular promedio general de SR para este periodo (promedio de FC3, FC4, FC5)
        promedios_sr = []
        for sr_col in correspondencia_sr_masa.keys():
            if f'Promedio_{sr_col}' in promedios_periodo:
                promedios_sr.append(promedios_periodo[f'Promedio_{sr_col}'])
        
        if promedios_sr:
            promedio_general = sum(promedios_sr) / len(promedios_sr)
            promedios_periodo['Promedio_General_SR'] = promedio_general
            logger.info(f"Promedio general de SR para periodo {periodo}: {promedio_general:.2f}% (ya corregido +7.5%, basado en {len(promedios_sr)} tipos de SR)")

        datos_para_grafico_barras.append(promedios_periodo)

        ax.set_title(f"Soiling Ratios for Period: {periodo_traducido} (Filtered by Corresponding Masses > 0)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Soiling Ratio [%]")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45, fontsize=12)
        plt.tight_layout()

        # Guardar gráfico del periodo
        nombre_base = str(periodo).lower()
        nombre_base = re.sub(r'[^\w.\- ]', '_', nombre_base)
        nombre_base = nombre_base.replace(' ', '_')
        nombre_base = re.sub(r'_{2,}', '_', nombre_base)
        nombre_base = nombre_base.strip('_')
        nombre_archivo_periodo = nombre_base if nombre_base else 'default_periodo'
        nombre_grafico = f"SR_Periodo_{nombre_archivo_periodo}_MasasCorregidas.png"

        try:
            save_plot_matplotlib(fig, nombre_grafico, output_graph_dir, subfolder=subfolder)
            logger.info(f"Gráfico guardado: {nombre_grafico}")
        except Exception as e:
            logger.error(f"Error al guardar el gráfico para el periodo {periodo}: {e}")
        finally:
            plt.close(fig)

    # Generar gráfico de barras de promedios
    if datos_para_grafico_barras:
        df_grafico_barras = pd.DataFrame(datos_para_grafico_barras)
        
        if not df_grafico_barras.empty and 'Periodo' in df_grafico_barras.columns:
            # Orden personalizado de periodos
            orden_periodos_deseado = [
                'weekly', '2 weeks', 'monthly',
                'quarterly', '4-monthly', 'semiannual', '1 year'
            ]
            
            # Filtrar y ordenar periodos
            df_grafico_barras_filtrado = df_grafico_barras[df_grafico_barras['Periodo'].isin(orden_periodos_deseado)].copy()
            
            if not df_grafico_barras_filtrado.empty:
                df_grafico_barras_filtrado['Periodo'] = pd.Categorical(
                    df_grafico_barras_filtrado['Periodo'],
                    categories=orden_periodos_deseado,
                    ordered=True
                )
                df_grafico_barras_filtrado = df_grafico_barras_filtrado.sort_values('Periodo')
                df_grafico_barras_plot = df_grafico_barras_filtrado.set_index('Periodo')

                # Columnas de promedio para graficar (solo individuales: FC3, FC4, FC5)
                cols_promedio_para_plot = [f'Promedio_{col_sr}' for col_sr in correspondencia_sr_masa.keys() 
                                         if f'Promedio_{col_sr}' in df_grafico_barras_plot.columns]

                if cols_promedio_para_plot:
                    try:
                        fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
                        df_grafico_barras_plot[cols_promedio_para_plot].plot(kind='bar', ax=ax_bar, width=0.8)

                        ax_bar.set_title('Average Soiling Ratios by Exposure Period', fontsize=16)
                        ax_bar.set_ylabel('Average Soiling Ratio [%]', fontsize=14)
                        ax_bar.set_xlabel('Period', fontsize=14)
                        legend_labels = [col.replace('Promedio_', '').replace('SR_R_', 'SR ') 
                                       for col in cols_promedio_para_plot]
                        ax_bar.legend(title='SR Types', labels=legend_labels)
                        ax_bar.grid(True, linestyle='--', alpha=0.7, axis='y')
                        
                        # Las etiquetas del eje X ya están en inglés desde el DataFrame
                        ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha='right', fontsize=12)

                        # Añadir etiquetas de valor
                        for c in ax_bar.containers:
                            for bar in c:
                                height = bar.get_height()
                                if pd.notna(height) and height != 0:
                                    ax_bar.annotate(f'{height:.1f}',
                                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                                   xytext=(0, 3),
                                                   textcoords="offset points",
                                                   ha='center', va='bottom', fontsize=12)

                        plt.tight_layout()

                        nombre_grafico_barras = "SR_Promedios_por_Periodo_Barras.png"
                        save_plot_matplotlib(fig_bar, nombre_grafico_barras, output_graph_dir, subfolder=subfolder)
                    except Exception as e:
                        logger.error(f"Error al generar el gráfico de barras: {e}")
                    finally:
                        plt.close(fig_bar)
                
                # Generar gráfico separado de promedios generales
                if 'Promedio_General_SR' in df_grafico_barras_plot.columns:
                    try:
                        fig_general, ax_general = plt.subplots(figsize=(12, 6))
                        
                        # Crear gráfico de barras solo con promedios generales
                        promedios_generales = df_grafico_barras_plot['Promedio_General_SR']
                        promedios_generales.plot(kind='bar', ax=ax_general, width=0.6, color='#2E8B57')
                        
                        ax_general.set_title('General Average of Soiling Ratios by Exposure Period')
                        # Ajustar límite superior para dar espacio a las etiquetas
                        max_valor = promedios_generales.max()
                        ax_general.set_ylim(0, max(110, max_valor + 10))  # Mínimo 110% o valor máximo + 10%
                        ax_general.set_ylabel('General Average SR [%]')
                        ax_general.set_xlabel('Period')
                        ax_general.grid(True, linestyle='--', alpha=0.7, axis='y')
                        
                        # Las etiquetas del eje X ya están en inglés desde el DataFrame
                        ax_general.set_xticklabels(ax_general.get_xticklabels(), rotation=45, ha='right', fontsize=12)
                        
                        # Añadir etiquetas de valor con mejor posicionamiento
                        for i, (periodo, valor) in enumerate(promedios_generales.items()):
                            if pd.notna(valor) and valor != 0:
                                # Ajustar posición vertical basada en el valor
                                y_offset = 5 if valor > 95 else 3
                                ax_general.annotate(f'{valor:.1f}',
                                                  xy=(i, valor),
                                                  xytext=(0, y_offset),
                                                  textcoords="offset points",
                                                  ha='center', va='bottom', fontsize=10)
                        
                        plt.tight_layout()
                        
                        nombre_grafico_general = "SR_Promedios_Generales_por_Periodo.png"
                        save_plot_matplotlib(fig_general, nombre_grafico_general, output_graph_dir, subfolder=subfolder)
                        logger.info(f"Gráfico de promedios generales guardado: {nombre_grafico_general}")
                    except Exception as e:
                        logger.error(f"Error al generar el gráfico de promedios generales: {e}")
                    finally:
                        plt.close(fig_general)

def analizar_seleccion_irradiancia_post_exposicion(
    path_irradiancia_csv: str,
    path_calendario_con_masas_csv: str,
    output_seleccion_csv: str
):
    """
    Analiza y selecciona datos de irradiancia post-exposición, generando el archivo seleccion_irradiancia_post_exposicion.csv.
    """
    logger.info("--- Iniciando Sección: Selección de Irradiancia Post-Exposición ---")
    df_calendario_eventos = None
    if not os.path.exists(path_calendario_con_masas_csv):
        logger.error(f"Archivo de calendario con masas {path_calendario_con_masas_csv} no encontrado. Asegúrate que la Sección 3 (Análisis de Calendario) se haya ejecutado y generado 'calendario_muestras_seleccionado.csv'.")
    else:
        try:
            df_calendario_eventos = pd.read_csv(path_calendario_con_masas_csv, parse_dates=['Inicio Exposición', 'Fin Exposicion'])
            logger.info(f"Calendario con masas cargado desde {path_calendario_con_masas_csv}. Filas: {len(df_calendario_eventos)}")
            df_calendario_eventos = df_calendario_eventos[
                df_calendario_eventos['Estructura'].fillna('').astype(str).str.strip() == 'Fija a RC'
            ]
            df_calendario_eventos.dropna(subset=['Fin Exposicion', 'Periodo', 'Masa A', 'Masa B', 'Masa C'], inplace=True)
            logger.info(f"Filas en calendario después de filtrar por 'Fija a RC' y NaNs en columnas clave: {len(df_calendario_eventos)}")
        except Exception as e:
            logger.error(f"Error al cargar o procesar {path_calendario_con_masas_csv}: {e}")
            df_calendario_eventos = pd.DataFrame()

    df_irradiancia = None
    if not os.path.exists(path_irradiancia_csv):
        logger.error(f"Archivo de irradiancia {path_irradiancia_csv} no encontrado.")
    else:
        try:
            df_irradiancia = pl.read_csv(path_irradiancia_csv, try_parse_dates=True)
            if '_time' in df_irradiancia.columns and df_irradiancia['_time'].dtype == pl.Datetime:
                df_irradiancia = df_irradiancia.with_columns(pl.col('_time').dt.date().alias('_time'))
            elif '_time' not in df_irradiancia.columns or df_irradiancia['_time'].dtype != pl.Date:
                logger.error(f"La columna '_time' en {path_irradiancia_csv} no es de tipo Date o no existe después de la conversión inicial.")
                df_irradiancia = pl.DataFrame()
            logger.info(f"Datos de irradiancia cargados desde {path_irradiancia_csv}. Filas: {len(df_irradiancia) if df_irradiancia is not None else 0}")
        except Exception as e:
            logger.error(f"Error al cargar el archivo de irradiancia {path_irradiancia_csv}: {e}")
            df_irradiancia = pl.DataFrame()

    datos_seleccionados_total = []
    if df_calendario_eventos is not None and not df_calendario_eventos.empty and \
       df_irradiancia is not None and not df_irradiancia.is_empty():
        for index, row_evento in df_calendario_eventos.iterrows():
            periodo = row_evento['Periodo']
            fecha_fin_evento_dt = row_evento['Fin Exposicion']
            masa_a = row_evento['Masa A']
            masa_b = row_evento['Masa B']
            masa_c = row_evento['Masa C']
            if pd.isna(fecha_fin_evento_dt):
                logger.warning(f"Fecha 'Fin Exposicion' es NaT para el evento con índice {index}. Saltando.")
                continue
            fecha_fin_evento_date = fecha_fin_evento_dt.date()
            num_dias_irradiancia_a_seleccionar = 5
            fecha_excepcion_semestral = datetime(2025, 1, 16).date()
            if periodo == 'Semestral' and fecha_fin_evento_date == fecha_excepcion_semestral:
                num_dias_irradiancia_a_seleccionar = 4
                logger.info(f"EXCEPCIÓN APLICADA: Para el periodo '{periodo}' con fecha de fin de evento '{fecha_fin_evento_date}', se seleccionarán {num_dias_irradiancia_a_seleccionar} días de irradiancia.")
            fecha_inicio_seleccion = fecha_fin_evento_date + timedelta(days=1)
            fecha_fin_seleccion = fecha_inicio_seleccion + timedelta(days=num_dias_irradiancia_a_seleccionar - 1)
            
            # Filtrar datos anómalos del 9-12 enero 2025 (valores FC5 muy altos)
            fechas_anomalas = [
                datetime(2025, 1, 9).date(),
                datetime(2025, 1, 10).date(),
                datetime(2025, 1, 11).date(),
                datetime(2025, 1, 12).date()
            ]
            
            datos_evento_actual = df_irradiancia.filter(
                (pl.col('_time') >= fecha_inicio_seleccion) & 
                (pl.col('_time') <= fecha_fin_seleccion) &
                (~pl.col('_time').is_in(fechas_anomalas))
            )
            
            # Verificar si se filtraron fechas anómalas
            datos_sin_filtro = df_irradiancia.filter(
                (pl.col('_time') >= fecha_inicio_seleccion) & 
                (pl.col('_time') <= fecha_fin_seleccion)
            )
            if len(datos_sin_filtro) > len(datos_evento_actual):
                fechas_filtradas = len(datos_sin_filtro) - len(datos_evento_actual)
                logger.info(f"FILTRO APLICADO: Se excluyeron {fechas_filtradas} fechas anómalas del 9-12 enero 2025 para el periodo '{periodo}' con fecha fin '{fecha_fin_evento_date}'.")
            if not datos_evento_actual.is_empty():
                datos_evento_actual = datos_evento_actual.with_columns([
                    pl.lit(periodo).alias('Periodo_Referencia'),
                    pl.lit(fecha_fin_evento_date.strftime('%Y-%m-%d')).alias('Fecha_Fin_Exposicion_Referencia'),
                    pl.lit(masa_a).alias('Masa_A_Referencia'),
                    pl.lit(masa_b).alias('Masa_B_Referencia'),
                    pl.lit(masa_c).alias('Masa_C_Referencia')
                ])
                datos_seleccionados_total.append(datos_evento_actual)
            else:
                logger.info(f"No se encontraron datos de irradiancia entre {fecha_inicio_seleccion} y {fecha_fin_seleccion} para el evento con 'Fin Exposicion' {fecha_fin_evento_date} (Periodo: '{periodo}').")
        if datos_seleccionados_total:
            df_final_seleccion = pl.concat(datos_seleccionados_total)
            logger.info(f"Procesamiento completado. Total de filas seleccionadas: {len(df_final_seleccion)}")
            df_final_seleccion.write_csv(output_seleccion_csv)
            logger.info(f"Archivo CSV con datos de irradiancia seleccionados (y masas) guardado en: {output_seleccion_csv}")
        else:
            logger.warning("No se seleccionaron datos de irradiancia. El archivo CSV de salida no se generará.")
    else:
        if df_calendario_eventos is None or df_calendario_eventos.empty:
            logger.error("El DataFrame de calendario de eventos (desde CSV o procesado) está vacío. No se puede continuar.")
        if df_irradiancia is None or df_irradiancia.is_empty():
            logger.error(f"El DataFrame de irradiancia ({path_irradiancia_csv}) está vacío o no se pudo cargar. No se puede continuar.")
    logger.info("--- Fin Sección: Selección de Irradiancia Post-Exposición (con Masas) ---")

def analizar_calculo_soiling_ratios(
    path_seleccion_irradiancia_csv: str,
    output_sr_csv: str,
    umbral_irradiancia_ref: int = 300
):
    """
    Calcula los Soiling Ratios (SR) a partir de los datos de irradiancia seleccionados y genera el archivo seleccion_irradiancia_con_sr.csv.
    """
    logger.info("--- Iniciando Sección: Cálculo de Soiling Ratios (SR) - Para FC1 a FC5 (con Filtro de Irradiancia REF >= 300 W/m2) ---")
    if not os.path.exists(path_seleccion_irradiancia_csv):
        logger.error(f"Archivo {path_seleccion_irradiancia_csv} no encontrado. Asegúrate que la celda de selección de irradiancia (con masas) se haya ejecutado correctamente.")
    else:
        try:
            df_seleccion = pl.read_csv(path_seleccion_irradiancia_csv, try_parse_dates=True)
            logger.info(f"Datos cargados desde {path_seleccion_irradiancia_csv}. Filas: {len(df_seleccion)}")
            columnas_fc_para_sr = ['R_FC1_Avg', 'R_FC2_Avg', 'R_FC3_Avg', 'R_FC4_Avg', 'R_FC5_Avg']
            columna_ref = 'REF'
            if columna_ref not in df_seleccion.columns:
                logger.error(f"La columna de referencia '{columna_ref}' no se encuentra en el DataFrame. No se pueden calcular/filtrar los SR.")
            else:
                df_con_sr = df_seleccion.clone()
                columnas_sr_generadas = []
                for col_fc in columnas_fc_para_sr:
                    if col_fc in df_con_sr.columns:
                        nueva_col_sr = f"SR_{col_fc.replace('_Avg','')}"
                        df_con_sr = df_con_sr.with_columns(
                            (pl.col(col_fc) / pl.col(columna_ref)).alias(nueva_col_sr)
                        )
                        df_con_sr = df_con_sr.with_columns(
                            pl.when(pl.col(nueva_col_sr).is_infinite() | pl.col(nueva_col_sr).is_nan())
                              .then(None)
                              .otherwise(pl.col(nueva_col_sr))
                              .alias(nueva_col_sr)
                        )
                        logger.info(f"Calculado SR inicial para: {nueva_col_sr}")
                        df_con_sr = df_con_sr.with_columns(
                            pl.when(pl.col(columna_ref) < umbral_irradiancia_ref)
                              .then(None)
                              .otherwise(pl.col(nueva_col_sr))
                              .alias(nueva_col_sr)
                        )
                        logger.info(f"Aplicado filtro REF < {umbral_irradiancia_ref} W/m2 a: {nueva_col_sr}")
                        columnas_sr_generadas.append(nueva_col_sr)
                    else:
                        logger.warning(f"Columna {col_fc} no encontrada. No se calculará SR para esta columna.")
                if columnas_sr_generadas:
                    try:
                        df_con_sr.write_csv(output_sr_csv)
                        logger.info(f"Archivo CSV con Soiling Ratios (FC1-FC5, filtrado por REF) guardado en: {output_sr_csv}")
                        logger.info(f"Columnas finales en el df guardado: {df_con_sr.columns}")
                        for col_sr_verif in columnas_sr_generadas:
                            if col_sr_verif in df_con_sr.columns:
                                nulos_con_ref_baja = df_con_sr.filter(
                                    (pl.col(columna_ref) < umbral_irradiancia_ref) & (df_con_sr[col_sr_verif].is_null())
                                ).height
                                total_nulos = df_con_sr[col_sr_verif].is_null().sum()
                                logger.info(f"Columna '{col_sr_verif}': {total_nulos} nulos en total. Filas con REF < {umbral_irradiancia_ref} y SR nulo: {nulos_con_ref_baja}")
                    except Exception as e:
                        logger.error(f"Error al guardar el CSV con Soiling Ratios (FC1-FC5, filtrado por REF): {e}")
                else:
                    logger.warning("No se generaron columnas de Soiling Ratio. El archivo de salida no se creará.")
        except Exception as e:
            logger.error(f"Error al procesar el archivo {path_seleccion_irradiancia_csv} para calcular SR (FC1-FC5, filtrado por REF): {e}")
    logger.info("--- Fin Sección: Cálculo de Soiling Ratios (SR) - Para FC1 a FC5 (con Filtro de Irradiancia REF >= 300 W/m2) ---")

def run_analysis():
    """
    Función principal para ejecutar el análisis de PV Glasses.
    """
    # Crear directorio de logs si no existe
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'analisis_soiling.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Inicializar directorios
    data_dir = 'datos'
    output_dir = 'graficos_analisis_integrado_py'
    output_csv_dir = 'datos_procesados_analisis_integrado_py'
    output_graph_dir = 'graficos_analisis_integrado_py'

    # Crear directorios si no existen
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_graph_dir, exist_ok=True)

    # Definir rutas de archivos
    transmitancia_file = os.path.join(data_dir, 'raw_pv_glasses_data.csv')
    calendario_file = os.path.join(data_dir, '20241114 Calendario toma de muestras soiling.xlsx')

    # Verificar existencia de archivos
    if not os.path.exists(transmitancia_file):
        logger.error(f"Archivo de transmitancia no encontrado: {transmitancia_file}")
        print(f"Error: No se encontró el archivo de transmitancia en {transmitancia_file}")
        return

    if not os.path.exists(calendario_file):
        logger.error(f"Archivo de calendario no encontrado: {calendario_file}")
        print(f"Error: No se encontró el archivo de calendario en {calendario_file}")
        return

    try:
        # Análisis de transmitancia
        logger.info("Iniciando análisis de transmitancia...")
        transmitancia_result = analizar_transmitancia_pv_glasses(
            file_path=transmitancia_file,
            output_csv_dir=output_csv_dir,
            output_graph_dir=output_graph_dir
        )
        if transmitancia_result is None:
            logger.error("El análisis de transmitancia falló")
            return

        # Análisis de calendario
        logger.info("Iniciando análisis de calendario...")
        calendario_result = analizar_calendario_muestras(
            file_path=calendario_file,
            output_csv_dir=output_csv_dir
        )
        if calendario_result is None:
            logger.error("El análisis de calendario falló")
            return

        # --- Sección: Selección de Irradiancia Post-Exposición ---
        path_irradiancia_csv = os.path.join(output_csv_dir, "transmitancia_pv_glasses", "transmitancia_filtrado_MediodiaSolarReal__60min_outliers_IQR_removidos_data_daily_resampled_final.csv")
        path_calendario_con_masas_csv = os.path.join(output_csv_dir, "calendario_muestras_seleccionado.csv")
        output_seleccion_csv = os.path.join(output_csv_dir, "seleccion_irradiancia_post_exposicion.csv")
        analizar_seleccion_irradiancia_post_exposicion(path_irradiancia_csv, path_calendario_con_masas_csv, output_seleccion_csv)

        # --- Sección: Cálculo de Soiling Ratios (SR) ---
        path_seleccion_irradiancia_csv = os.path.join(output_csv_dir, "seleccion_irradiancia_post_exposicion.csv")
        output_sr_csv = os.path.join(output_csv_dir, "seleccion_irradiancia_con_sr.csv")
        analizar_calculo_soiling_ratios(path_seleccion_irradiancia_csv, output_sr_csv)

        # Generar gráficos de Soiling Ratios
        logger.info("Iniciando generación de gráficos de Soiling Ratios...")
        path_sr_csv = os.path.join(output_csv_dir, "seleccion_irradiancia_con_sr.csv")
        if os.path.exists(path_sr_csv):
            df_sr = pl.read_csv(path_sr_csv, try_parse_dates=True)
            plot_soiling_ratios_por_periodo(df_sr, output_graph_dir)
        else:
            logger.error(f"Archivo de Soiling Ratios no encontrado: {path_sr_csv}")

    except Exception as e:
        logger.error(f"Error en la ejecución principal: {e}", exc_info=True)
        print(f"Error: {e}")

    return "Análisis de PV Glasses completado con éxito."

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando análisis de PV Glasses...")
    raw_data_filepath = os.path.join(BASE_INPUT_DIR, 'raw_pv_glasses_data.csv')
    run_analysis()
