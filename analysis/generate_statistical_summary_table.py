"""
Genera una tabla resumen con SR, incertidumbres y estadísticas por metodología y período.
Incluye: semanal, mensual, semestral y anual.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from scipy import stats
import config.paths as paths
import logging

logger = logging.getLogger(__name__)


def calculate_trend(series):
    """
    Calcula la tendencia de una serie temporal.
    
    Returns:
        dict con: slope (%/día), slope_monthly (%/mes), r2, p_value, intercept
    """
    if series.empty or len(series.dropna()) < 2:
        return {
            'slope_daily': np.nan,
            'slope_monthly': np.nan,
            'r2': np.nan,
            'p_value': np.nan,
            'intercept': np.nan
        }
    
    # Convertir índice a días numéricos desde el primer punto
    series_clean = series.dropna()
    if len(series_clean) < 2:
        return {
            'slope_daily': np.nan,
            'slope_monthly': np.nan,
            'r2': np.nan,
            'p_value': np.nan,
            'intercept': np.nan
        }
    
    dates = pd.to_datetime(series_clean.index)
    first_date = dates.min()
    days_from_start = (dates - first_date).total_seconds() / (24 * 3600)
    
    values = series_clean.values
    
    # Calcular regresión lineal
    slope, intercept, r_value, p_value, std_err = stats.linregress(days_from_start, values)
    r2 = r_value ** 2
    
    # Convertir pendiente a %/mes (asumiendo 30 días por mes)
    slope_monthly = slope * 30
    
    return {
        'slope_daily': slope,
        'slope_monthly': slope_monthly,
        'r2': r2,
        'p_value': p_value,
        'intercept': intercept
    }


def load_uncertainty_data(file_path, time_col=None):
    """Carga datos de incertidumbre desde un archivo CSV"""
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Determinar columna de tiempo
        if time_col is None:
            for col in ['timestamp', '_time', 'Original_Timestamp_Col', 'time', 'date']:
                if col in df.columns:
                    time_col = col
                    break
        
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
        
        return df
    except Exception as e:
        logger.warning(f"Error cargando {file_path}: {e}")
        return None


def aggregate_to_period(df, period='W', sr_col='SR_agg', u_col='U_rel_k2'):
    """
    Agrega datos a un período específico (semanal, mensual, semestral, anual).
    
    Args:
        df: DataFrame con índice temporal
        period: 'W' (semanal), 'ME' (mensual), '6ME' (semestral), 'YE' (anual)
        sr_col: Nombre de la columna de SR
        u_col: Nombre de la columna de incertidumbre
    """
    if df is None or df.empty:
        return None
    
    if sr_col not in df.columns:
        return None
    
    # Agregar SR (promedio)
    sr_agg = df[sr_col].resample(period).mean()
    
    # Agregar incertidumbre (promedio)
    if u_col in df.columns:
        u_agg = df[u_col].resample(period).mean()
    else:
        u_agg = pd.Series(index=sr_agg.index, dtype=float)
    
    result = pd.DataFrame({
        'SR': sr_agg,
        'U_rel_k2': u_agg
    })
    
    return result.dropna(subset=['SR'])


def calculate_statistics_for_methodology(method_name, weekly_file, monthly_file, 
                                        sr_col='SR_agg', u_col='U_rel_k2', time_col=None):
    """
    Calcula estadísticas para una metodología específica.
    
    Returns:
        dict con estadísticas por período
    """
    results = {
        'methodology': method_name,
        'weekly': None,
        'monthly': None,
        'semiannual': None,
        'annual': None
    }
    
    # Cargar datos semanales
    df_weekly = load_uncertainty_data(weekly_file, time_col)
    if df_weekly is not None and sr_col in df_weekly.columns:
        results['weekly'] = calculate_period_statistics(df_weekly, sr_col, u_col, 'Semanal')
    
    # Cargar datos mensuales
    df_monthly = load_uncertainty_data(monthly_file, time_col)
    if df_monthly is not None and sr_col in df_monthly.columns:
        results['monthly'] = calculate_period_statistics(df_monthly, sr_col, u_col, 'Mensual')
        
        # Calcular semestral y anual desde datos mensuales
        if not df_monthly.empty:
            df_semiannual = aggregate_to_period(df_monthly, '6ME', sr_col, u_col)
            if df_semiannual is not None and not df_semiannual.empty:
                results['semiannual'] = calculate_period_statistics(df_semiannual, 'SR', 'U_rel_k2', 'Semestral')
            
            df_annual = aggregate_to_period(df_monthly, 'YE', sr_col, u_col)
            if df_annual is not None and not df_annual.empty:
                results['annual'] = calculate_period_statistics(df_annual, 'SR', 'U_rel_k2', 'Anual')
    
    return results


def calculate_period_statistics(df, sr_col, u_col, period_name):
    """
    Calcula estadísticas para un período específico.
    
    Returns:
        dict con todas las estadísticas
    """
    if df is None or df.empty or sr_col not in df.columns:
        return None
    
    sr_series = df[sr_col].dropna()
    
    if sr_series.empty:
        return None
    
    stats_dict = {
        'period': period_name,
        'n_points': len(sr_series),
        'sr_mean': sr_series.mean(),
        'sr_median': sr_series.median(),
        'sr_std': sr_series.std(),
        'sr_min': sr_series.min(),
        'sr_max': sr_series.max(),
        'sr_q25': sr_series.quantile(0.25),
        'sr_q75': sr_series.quantile(0.75),
        'sr_iqr': sr_series.quantile(0.75) - sr_series.quantile(0.25),
        'sr_cv': (sr_series.std() / sr_series.mean() * 100) if sr_series.mean() != 0 else np.nan
    }
    
    # Incertidumbre
    if u_col in df.columns:
        u_series = df[u_col].dropna()
        if not u_series.empty:
            stats_dict['u_mean'] = u_series.mean()
            stats_dict['u_median'] = u_series.median()
            stats_dict['u_std'] = u_series.std()
            stats_dict['u_min'] = u_series.min()
            stats_dict['u_max'] = u_series.max()
        else:
            stats_dict.update({f'u_{k}': np.nan for k in ['mean', 'median', 'std', 'min', 'max']})
    else:
        stats_dict.update({f'u_{k}': np.nan for k in ['mean', 'median', 'std', 'min', 'max']})
    
    # Tendencias
    trend = calculate_trend(sr_series)
    stats_dict.update({
        'trend_slope_monthly': trend['slope_monthly'],
        'trend_r2': trend['r2'],
        'trend_p_value': trend['p_value']
    })
    
    return stats_dict


def generate_summary_table():
    """
    Genera la tabla resumen completa con todas las metodologías.
    
    Returns:
        pd.DataFrame con la tabla resumen
    """
    methodologies = []
    
    # 1. RefCells
    methodologies.append({
        'name': 'RefCells',
        'weekly_file': paths.SR_WEEKLY_ABS_WITH_U_FILE,
        'monthly_file': paths.SR_MONTHLY_ABS_WITH_U_FILE,
        'sr_col': 'SR_agg',
        'u_col': 'U_rel_k2',
        'time_col': 'timestamp'
    })
    
    # 2. DustIQ
    methodologies.append({
        'name': 'DustIQ',
        'weekly_file': paths.DUSTIQ_SR_WEEKLY_ABS_WITH_U_FILE,
        'monthly_file': paths.DUSTIQ_SR_MONTHLY_ABS_WITH_U_FILE,
        'sr_col': 'SR_agg',
        'u_col': 'U_rel_k2',
        'time_col': 'timestamp'
    })
    
    # 3. Soiling Kit
    methodologies.append({
        'name': 'Soiling Kit',
        'weekly_file': paths.SOILING_KIT_SR_WEEKLY_ABS_WITH_U_FILE,
        'monthly_file': paths.SOILING_KIT_SR_MONTHLY_ABS_WITH_U_FILE,
        'sr_col': 'SR_agg',
        'u_col': 'U_rel_k2',
        'time_col': 'Original_Timestamp_Col'
    })
    
    # 4. PVStand Isc
    methodologies.append({
        'name': 'PVStand Isc',
        'weekly_file': os.path.join(paths.PROPAGACION_ERRORES_PVSTAND_DIR, 'sr_isc_weekly_abs_with_U.csv'),
        'monthly_file': os.path.join(paths.PROPAGACION_ERRORES_PVSTAND_DIR, 'sr_isc_monthly_abs_with_U.csv'),
        'sr_col': 'SR_agg',
        'u_col': 'U_rel_k2',
        'time_col': '_time'
    })
    
    # 5. PVStand Pmax
    methodologies.append({
        'name': 'PVStand Pmax',
        'weekly_file': os.path.join(paths.PROPAGACION_ERRORES_PVSTAND_DIR, 'sr_pmax_weekly_abs_with_U.csv'),
        'monthly_file': os.path.join(paths.PROPAGACION_ERRORES_PVSTAND_DIR, 'sr_pmax_monthly_abs_with_U.csv'),
        'sr_col': 'SR_agg',
        'u_col': 'U_rel_k2',
        'time_col': '_time'
    })
    
    # 6. IV600 Pmax
    methodologies.append({
        'name': 'IV600 Pmax',
        'weekly_file': paths.IV600_SR_WEEKLY_ABS_WITH_U_FILE,
        'monthly_file': paths.IV600_SR_MONTHLY_ABS_WITH_U_FILE,
        'sr_col': 'SR_Pmax_1MD434vs1MD439',
        'u_col': 'U_SR_Pmax_1MD434vs1MD439_k2_rel',
        'time_col': 'timestamp'
    })
    
    # 7. IV600 Isc
    methodologies.append({
        'name': 'IV600 Isc',
        'weekly_file': paths.IV600_SR_WEEKLY_ABS_WITH_U_FILE,
        'monthly_file': paths.IV600_SR_MONTHLY_ABS_WITH_U_FILE,
        'sr_col': 'SR_Isc_1MD434vs1MD439',
        'u_col': 'U_SR_Isc_1MD434vs1MD439_k2_rel',
        'time_col': 'timestamp'
    })
    
    # 8. PV Glasses (promedio general)
    methodologies.append({
        'name': 'PV Glasses',
        'weekly_file': paths.PV_GLASSES_SR_WEEKLY_ABS_WITH_U_FILE,
        'monthly_file': paths.PV_GLASSES_SR_MONTHLY_ABS_WITH_U_FILE,
        'sr_col': 'SR_agg',
        'u_col': 'U_rel_k2',
        'time_col': 'Original_Timestamp_Col'
    })
    
    # Calcular estadísticas para cada metodología
    all_results = []
    
    for method in methodologies:
        logger.info(f"Procesando {method['name']}...")
        results = calculate_statistics_for_methodology(
            method['name'],
            method['weekly_file'],
            method['monthly_file'],
            method['sr_col'],
            method['u_col'],
            method['time_col']
        )
        
        # Convertir a filas de tabla
        for period in ['weekly', 'monthly', 'semiannual', 'annual']:
            period_stats = results[period]
            if period_stats is not None:
                row = {
                    'Metodología': results['methodology'],
                    'Período': period_stats['period'],
                    'N_puntos': period_stats['n_points'],
                    'SR_media': period_stats['sr_mean'],
                    'SR_mediana': period_stats['sr_median'],
                    'SR_std': period_stats['sr_std'],
                    'SR_min': period_stats['sr_min'],
                    'SR_max': period_stats['sr_max'],
                    'SR_Q25': period_stats['sr_q25'],
                    'SR_Q75': period_stats['sr_q75'],
                    'SR_IQR': period_stats['sr_iqr'],
                    'SR_CV_%': period_stats['sr_cv'],
                    'U_rel_k2_media_%': period_stats.get('u_mean', np.nan),
                    'U_rel_k2_mediana_%': period_stats.get('u_median', np.nan),
                    'U_rel_k2_std_%': period_stats.get('u_std', np.nan),
                    'U_rel_k2_min_%': period_stats.get('u_min', np.nan),
                    'U_rel_k2_max_%': period_stats.get('u_max', np.nan),
                    'Tendencia_%/mes': period_stats.get('trend_slope_monthly', np.nan),
                    'Tendencia_R2': period_stats.get('trend_r2', np.nan),
                    'Tendencia_p_value': period_stats.get('trend_p_value', np.nan)
                }
                all_results.append(row)
    
    # Crear DataFrame
    df_summary = pd.DataFrame(all_results)
    
    return df_summary


def save_summary_table(output_dir=None):
    """
    Genera y guarda la tabla resumen.
    
    Args:
        output_dir: Directorio de salida (default: graficos_analisis_integrado_py/consolidados)
    """
    if output_dir is None:
        output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "consolidados")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generando tabla resumen estadística...")
    df_summary = generate_summary_table()
    
    if df_summary.empty:
        logger.warning("No se generaron datos para la tabla resumen")
        return None
    
    # Guardar CSV
    csv_path = os.path.join(output_dir, "tabla_resumen_estadistica_sr.csv")
    df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Tabla resumen guardada en CSV: {csv_path}")
    
    # Guardar Excel
    excel_path = os.path.join(output_dir, "tabla_resumen_estadistica_sr.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Resumen Estadístico', index=False)
        logger.info(f"Tabla resumen guardada en Excel: {excel_path}")
    except Exception as e:
        logger.warning(f"No se pudo guardar Excel: {e}")
    
    return df_summary


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    df = save_summary_table()
    if df is not None:
        print("\n✅ Tabla resumen generada exitosamente")
        print(f"\nResumen de la tabla:")
        print(f"  - Total de filas: {len(df)}")
        print(f"  - Metodologías: {df['Metodología'].nunique()}")
        print(f"  - Períodos: {df['Período'].unique()}")
        print(f"\nPrimeras filas:")
        print(df.head(10).to_string())
    else:
        print("❌ Error al generar tabla resumen")

