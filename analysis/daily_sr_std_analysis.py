#!/usr/bin/env python3
"""
Análisis de Desviación Estándar Diaria de Soiling Ratios
========================================================

Este script calcula la desviación estándar diaria de los Soiling Ratios (SR) 
para cada metodología (DustIQ, RefCells, PVStand, Soiling Kit) a partir de 
datos por minuto, permitiendo estudiar la dispersión y consistencia de cada método.

Autor: Asistente IA
Fecha: 2025-01-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging
from datetime import datetime
import sys

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings, paths

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dustiq_minute_data():
    """Cargar datos de DustIQ por minuto desde archivo raw (optimizado para archivos grandes)"""
    try:
        csv_path = os.path.join(paths.BASE_INPUT_DIR, "raw_dustiq_data.csv")
        if os.path.exists(csv_path):
            logger.info("Cargando datos DustIQ (archivo grande, puede tomar tiempo)...")
            
            # Cargar en chunks para manejar archivos grandes
            chunk_size = 50000
            chunks = []
            
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                chunk.set_index('timestamp', inplace=True)
                
                # Usar el promedio de ambos sensores C11 y C12
                chunk['SR_DustIQ'] = chunk[['SR_C11_Avg', 'SR_C12_Avg']].mean(axis=1)
                chunks.append(chunk[['SR_DustIQ']])
            
            # Combinar todos los chunks
            df = pd.concat(chunks, ignore_index=False)
            df = df.sort_index()  # Ordenar por timestamp
            
            logger.info(f"Datos DustIQ cargados: {df.shape}")
            logger.info(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
            return df
        else:
            logger.warning(f"Archivo DustIQ no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos DustIQ: {e}")
        return pd.DataFrame()

def load_refcells_minute_data():
    """Cargar datos de RefCells por minuto desde archivo filtrado"""
    try:
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "ref_cells", "ref_cells_sr_minutal_filtrado.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Usar el promedio de las celdas de referencia (1RC410 y 1RC411)
            sr_columns = [col for col in df.columns if 'RC' in col]
            if sr_columns:
                df['SR_RefCells'] = df[sr_columns].mean(axis=1)
                logger.info(f"Datos RefCells cargados: {df.shape}")
                logger.info(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
                return df[['SR_RefCells']]
            else:
                logger.warning("No se encontraron columnas de SR en RefCells")
                return pd.DataFrame()
        else:
            logger.warning(f"Archivo RefCells no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos RefCells: {e}")
        return pd.DataFrame()

def load_pvstand_minute_data():
    """Cargar datos de PVStand por minuto desde archivo raw"""
    try:
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "pv_stand", "pvstand_sr_raw_no_offset.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['_time'] = pd.to_datetime(df['_time'])
            df.set_index('_time', inplace=True)
            
            # Usar las columnas corregidas (SR_Isc_Corrected_Raw_NoOffset y SR_Pmax_Corrected_Raw_NoOffset)
            corrected_columns = [col for col in df.columns if 'Corrected_Raw_NoOffset' in col]
            if corrected_columns:
                # Calcular promedio de Isc y Pmax corregidos
                df['SR_PVStand'] = df[corrected_columns].mean(axis=1)
                logger.info(f"Datos PVStand cargados: {df.shape}")
                logger.info(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
                return df[['SR_PVStand']]
            else:
                logger.warning("No se encontraron columnas corregidas en PVStand")
                return pd.DataFrame()
        else:
            logger.warning(f"Archivo PVStand no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos PVStand: {e}")
        return pd.DataFrame()

def load_soiling_kit_minute_data():
    """Cargar datos de Soiling Kit por minuto desde archivo filtrado"""
    try:
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "soiling_kit", "soiling_kit_sr_minutal_filtered.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Original_Timestamp_Col'] = pd.to_datetime(df['Original_Timestamp_Col'])
            df.set_index('Original_Timestamp_Col', inplace=True)
            
            # Usar SR_Raw_Filtered (datos sin corrección de temperatura)
            if 'SR_Raw_Filtered' in df.columns:
                df['SR_SoilingKit'] = df['SR_Raw_Filtered']
                logger.info(f"Datos Soiling Kit cargados: {df.shape}")
                logger.info(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
                return df[['SR_SoilingKit']]
            else:
                logger.warning("No se encontró columna SR_Raw_Filtered en Soiling Kit")
                return pd.DataFrame()
        else:
            logger.warning(f"Archivo Soiling Kit no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos Soiling Kit: {e}")
        return pd.DataFrame()

def load_iv600_minute_data():
    """Cargar y procesar datos de IV600 para obtener SR por minuto (para calcular std diaria)"""
    try:
        # Cargar datos raw de IV600
        csv_path = os.path.join(paths.BASE_INPUT_DIR, "raw_iv600_data.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"Archivo IV600 raw no encontrado: {csv_path}")
            return pd.DataFrame()
        
        logger.info("Cargando datos IV600 (archivo grande, puede tomar tiempo)...")
        df_iv600_raw = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Procesar timezone
        if df_iv600_raw.index.tz is not None:
            df_iv600_raw.index = df_iv600_raw.index.tz_convert(None)
        
        # Mapeo de columnas
        column_mapping = {
            'module': 'Module',
            'pmp': 'Pmax',
            'isc': 'Isc',
            'voc': 'Voc',
            'imp': 'Imp',
            'vmp': 'Vmp'
        }
        df_iv600_raw = df_iv600_raw.rename(columns=column_mapping)
        
        # Crear MultiIndex con (Module, Parameter)
        df_iv600_pivot = df_iv600_raw.pivot_table(
            index=df_iv600_raw.index,
            columns='Module',
            values=['Pmax', 'Isc'],
            aggfunc='mean'
        )
        
        # Remuestreo a 5 minutos (mantener resolución alta para std diaria)
        df_iv600_5min = df_iv600_pivot.resample('5T').mean()
        
        # Aplicar corrección de temperatura (simplificada)
        df_pmax_corrected_5T = df_iv600_5min['Pmax'].copy()
        df_isc_corrected_5T = df_iv600_5min['Isc'].copy()
        
        # Definir columnas de módulos (igual que el analizador)
        test_mod_434_pmax_col = ('1MD434', 'Pmax')
        test_mod_440_pmax_col = ('1MD440', 'Pmax')
        ref_mod_pmax_col = ('1MD439', 'Pmax')
        test_mod_434_isc_col = ('1MD434', 'Isc')
        test_mod_440_isc_col = ('1MD440', 'Isc')
        ref_mod_isc_col = ('1MD439', 'Isc')
        
        # Calcular SR por minuto (5 minutos)
        sr_pmp_iv600_minute = pd.DataFrame(index=df_pmax_corrected_5T.index)
        sr_isc_iv600_minute = pd.DataFrame(index=df_isc_corrected_5T.index)
        
        # SR Pmp 434vs439 por minuto
        if test_mod_434_pmax_col in df_pmax_corrected_5T.columns and ref_mod_pmax_col in df_pmax_corrected_5T.columns:
            sr_raw = 100 * df_pmax_corrected_5T[test_mod_434_pmax_col] / df_pmax_corrected_5T[ref_mod_pmax_col]
            sr_pmp_iv600_minute['SR_Pmp_434vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]
        
        # SR Isc 434vs439 por minuto
        if test_mod_434_isc_col in df_isc_corrected_5T.columns and ref_mod_isc_col in df_isc_corrected_5T.columns:
            sr_raw = 100 * df_isc_corrected_5T[test_mod_434_isc_col] / df_isc_corrected_5T[ref_mod_isc_col]
            sr_isc_iv600_minute['SR_Isc_434vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]
        
        # Combinar Pmax e Isc en una sola serie por minuto
        sr_combined_minute = pd.DataFrame(index=df_pmax_corrected_5T.index)
        
        if not sr_pmp_iv600_minute.empty and 'SR_Pmp_434vs439' in sr_pmp_iv600_minute.columns:
            sr_combined_minute['SR_Pmax_IV600'] = sr_pmp_iv600_minute['SR_Pmp_434vs439']
        
        if not sr_isc_iv600_minute.empty and 'SR_Isc_434vs439' in sr_isc_iv600_minute.columns:
            sr_combined_minute['SR_Isc_IV600'] = sr_isc_iv600_minute['SR_Isc_434vs439']
        
        # Calcular promedio de Pmax e Isc por minuto
        if not sr_combined_minute.empty:
            sr_combined_minute['SR_IV600'] = sr_combined_minute[['SR_Pmax_IV600', 'SR_Isc_IV600']].mean(axis=1)
            logger.info(f"Datos IV600 por minuto procesados: {sr_combined_minute.shape}")
            logger.info(f"Rango de fechas: {sr_combined_minute.index.min()} a {sr_combined_minute.index.max()}")
            return sr_combined_minute[['SR_IV600']]
        else:
            logger.warning("No se pudieron calcular SR por minuto para IV600")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error procesando datos IV600 por minuto: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def calculate_daily_std(df, sr_column, method_name):
    """
    Calcular desviación estándar diaria de los SR
    
    Args:
        df: DataFrame con datos por minuto
        sr_column: Nombre de la columna de SR
        method_name: Nombre del método para logging
    
    Returns:
        DataFrame con desviación estándar diaria
    """
    if df.empty or sr_column not in df.columns:
        logger.warning(f"No hay datos válidos para {method_name}")
        return pd.DataFrame()
    
    # Filtrar valores válidos (no NaN)
    valid_data = df[sr_column].dropna()
    
    if valid_data.empty:
        logger.warning(f"No hay datos válidos (no NaN) para {method_name}")
        return pd.DataFrame()
    
    # Calcular desviación estándar diaria para todos los métodos
    daily_std = valid_data.resample('D').std()
    
    # Filtrar días con al menos 10 mediciones para tener una std confiable
    daily_count = valid_data.resample('D').count()
    reliable_days = daily_count >= 10
    daily_std_filtered = daily_std[reliable_days]
    
    logger.info(f"{method_name}: {len(daily_std_filtered)} días con std confiable (≥10 mediciones)")
    logger.info(f"{method_name}: Std promedio diaria: {daily_std_filtered.mean():.3f}%")
    logger.info(f"{method_name}: Std mediana diaria: {daily_std_filtered.median():.3f}%")
    
    return daily_std_filtered

def create_daily_std_comparison_plot(daily_std_data):
    """
    Crear gráfico comparativo de desviaciones estándar diarias
    
    Args:
        daily_std_data: Diccionario con datos de std diaria por método
    """
    if not daily_std_data:
        logger.warning("No hay datos para crear el gráfico")
        return False
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Colores para cada metodología
    colors = {
        'DustIQ': '#FF8C00',      # Naranja
        'RefCells': '#8B4513',    # Marrón
        'PVStand': '#4169E1',     # Azul real
        'SoilingKit': '#2E8B57',  # Verde mar
        'IV600': '#DC143C'        # Carmesí
    }
    
    # Gráfico 1: Series temporales de desviación estándar diaria
    for method, std_series in daily_std_data.items():
        if not std_series.empty:
            color = colors.get(method, '#000000')
            ax1.plot(std_series.index, std_series.values, 
                    color=color, linewidth=1.5, alpha=0.8, 
                    label=f'{method} (n={len(std_series)})')
    
    ax1.set_title('Desviación Estándar Diaria de Soiling Ratios por Metodología', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Desviación Estándar [%]', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # Formatear eje X
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Gráfico 2: Boxplot comparativo
    std_values = []
    method_labels = []
    
    for method, std_series in daily_std_data.items():
        if not std_series.empty:
            std_values.append(std_series.values)
            method_labels.append(f'{method}\n(n={len(std_series)})')
    
    if std_values:
        bp = ax2.boxplot(std_values, labels=method_labels, patch_artist=True)
        
        # Colorear los boxplots
        for patch, method in zip(bp['boxes'], method_labels):
            method_name = method.split('\n')[0]
            patch.set_facecolor(colors.get(method_name, '#CCCCCC'))
            patch.set_alpha(0.7)
    
    ax2.set_title('Distribución de Desviaciones Estándar Diarias', 
                  fontsize=16, fontweight='bold')
    ax2.set_ylabel('Desviación Estándar [%]', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "analisis_varianza_intraday")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = "daily_std_comparison.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Gráfico de comparación guardado en: {plot_path}")
    
    if settings.SHOW_FIGURES:
        plt.show()
    
    plt.close()
    return True

def create_std_statistics_table(daily_std_data):
    """
    Crear tabla de estadísticas de desviación estándar
    
    Args:
        daily_std_data: Diccionario con datos de std diaria por método
    
    Returns:
        DataFrame con estadísticas
    """
    stats_data = []
    
    for method, std_series in daily_std_data.items():
        if not std_series.empty:
            stats_data.append({
                'Metodología': method,
                'Días_analizados': len(std_series),
                'Std_promedio': std_series.mean(),
                'Std_mediana': std_series.median(),
                'Std_min': std_series.min(),
                'Std_max': std_series.max(),
                'Coef_variacion': (std_series.std() / std_series.mean()) * 100,
                'Percentil_25': std_series.quantile(0.25),
                'Percentil_75': std_series.quantile(0.75)
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Guardar tabla
        output_dir = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "statistical_deviations")
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, "daily_std_statistics.csv")
        stats_df.to_csv(csv_path, index=False)
        logger.info(f"Estadísticas guardadas en: {csv_path}")
        
        return stats_df
    else:
        logger.warning("No se pudieron calcular estadísticas")
        return pd.DataFrame()

def main():
    """Función principal para el análisis de desviación estándar diaria"""
    
    logger.info("=== INICIANDO ANÁLISIS DE DESVIACIÓN ESTÁNDAR DIARIA ===")
    
    # Cargar datos por minuto de cada metodología
    logger.info("Cargando datos por minuto...")
    
    dustiq_data = load_dustiq_minute_data()
    refcells_data = load_refcells_minute_data()
    pvstand_data = load_pvstand_minute_data()
    soiling_kit_data = load_soiling_kit_minute_data()
    iv600_data = load_iv600_minute_data()
    
    # Calcular desviación estándar diaria para cada método
    logger.info("Calculando desviaciones estándar diarias...")
    
    daily_std_data = {}
    
    if not dustiq_data.empty:
        daily_std_data['DustIQ'] = calculate_daily_std(dustiq_data, 'SR_DustIQ', 'DustIQ')
    
    if not refcells_data.empty:
        daily_std_data['RefCells'] = calculate_daily_std(refcells_data, 'SR_RefCells', 'RefCells')
    
    if not pvstand_data.empty:
        daily_std_data['PVStand'] = calculate_daily_std(pvstand_data, 'SR_PVStand', 'PVStand')
    
    if not soiling_kit_data.empty:
        daily_std_data['SoilingKit'] = calculate_daily_std(soiling_kit_data, 'SR_SoilingKit', 'Soiling Kit')
    
    if not iv600_data.empty:
        daily_std_data['IV600'] = calculate_daily_std(iv600_data, 'SR_IV600', 'IV600')
    
    # Verificar si hay datos para analizar
    if not daily_std_data:
        logger.error("No se encontraron datos válidos para ningún método")
        return False
    
    # Crear gráfico comparativo
    logger.info("Generando gráfico comparativo...")
    plot_success = create_daily_std_comparison_plot(daily_std_data)
    
    # Crear tabla de estadísticas
    logger.info("Generando estadísticas...")
    stats_df = create_std_statistics_table(daily_std_data)
    
    # Mostrar resumen
    logger.info("=== RESUMEN DEL ANÁLISIS ===")
    for method, std_series in daily_std_data.items():
        if not std_series.empty:
            logger.info(f"{method}: {len(std_series)} días, std promedio: {std_series.mean():.3f}%")
    
    if not stats_df.empty:
        logger.info("\nEstadísticas detalladas:")
        print(stats_df.to_string(index=False, float_format='%.3f'))
    
    logger.info("=== ANÁLISIS COMPLETADO ===")
    return True

if __name__ == "__main__":
    main()
