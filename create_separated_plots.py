#!/usr/bin/env python3
"""
Crear gráficos separados para el análisis de desviación estándar con PVStand separado
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Agregar el directorio de análisis al path
sys.path.append('/home/nicole/SR/SOILING/analysis')
from daily_sr_std_analysis import load_refcells_minute_data, load_soiling_kit_minute_data, calculate_daily_std
import config.paths as paths

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_dustiq_unified_data():
    """Carga los datos de DustIQ con horario unificado"""
    try:
        csv_path = '/home/nicole/SR/SOILING/datos_procesados_analisis_integrado_py/dustiq/dustiq_sr_unified_hours_oct_mar.csv'
        
        if not os.path.exists(csv_path):
            logger.error(f"Archivo no encontrado: {csv_path}")
            return pd.DataFrame()
        
        logger.info(f"Cargando datos DustIQ con horario unificado desde: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        
        logger.info(f"Datos DustIQ unificados cargados: {len(df)} registros")
        logger.info(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error cargando datos DustIQ unificados: {e}")
        return pd.DataFrame()

def load_pvstand_separated_data():
    """Carga datos de PVStand separando Isc y Pmax"""
    try:
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "pv_stand", "pvstand_sr_raw_no_offset.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"Archivo PVStand no encontrado: {csv_path}")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info("Cargando datos PVStand separados (Isc y Pmax)...")
        df = pd.read_csv(csv_path)
        df['_time'] = pd.to_datetime(df['_time'])
        df.set_index('_time', inplace=True)
        
        # Filtrar por período octubre-marzo
        start_date = '2024-10-01'
        end_date = '2025-03-31'
        df_period = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Aplicar filtro de horario unificado (13:00-18:00)
        df_time = df_period.between_time('13:00', '18:00')
        
        logger.info(f"PVStand período Oct-Mar: {len(df_time)} registros")
        
        # Separar Isc y Pmax
        df_isc = pd.DataFrame()
        df_pmax = pd.DataFrame()
        
        if 'SR_Isc_Corrected_Raw_NoOffset' in df_time.columns:
            df_isc = df_time[['SR_Isc_Corrected_Raw_NoOffset']].copy()
            df_isc.columns = ['SR_PVStand_Isc']
            df_isc = df_isc.dropna()
            logger.info(f"PVStand Isc: {len(df_isc)} registros válidos")
        
        if 'SR_Pmax_Corrected_Raw_NoOffset' in df_time.columns:
            df_pmax = df_time[['SR_Pmax_Corrected_Raw_NoOffset']].copy()
            df_pmax.columns = ['SR_PVStand_Pmax']
            df_pmax = df_pmax.dropna()
            logger.info(f"PVStand Pmax: {len(df_pmax)} registros válidos")
        
        return df_isc, df_pmax
        
    except Exception as e:
        logger.error(f"Error cargando datos PVStand separados: {e}")
        return pd.DataFrame(), pd.DataFrame()

def create_time_series_plot(daily_std_data):
    """Crea el gráfico de series temporales"""
    
    logger.info("Generando gráfico de series temporales...")
    
    # Configurar matplotlib
    plt.switch_backend('Agg')
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams['font.size'] = 12
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Colores para cada metodología
    colors = {
        'RefCells': '#2E8B57',      # Verde mar
        'DustIQ': '#4169E1',        # Azul real
        'Soiling Kit': '#FF6347',   # Tomate
        'PVStand_Isc': '#DC143C',   # Rojo carmesí
        'PVStand_Pmax': '#8B0000',  # Rojo oscuro
    }
    
    # Título
    ax.set_title('Daily Standard Deviation of Soiling Ratios by Methodology', 
                  fontsize=16, fontweight='bold', pad=25)
    
    # Plotear cada metodología
    for method, data in daily_std_data.items():
        if not data.empty:
            ax.plot(data.index, data.values, 
                    label=f'{method}', 
                    color=colors.get(method, '#000000'),
                    linewidth=2.5, alpha=0.8, marker='o', markersize=4)
    
    ax.set_ylabel('Standard Deviation (%)', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = '/home/nicole/SR/SOILING/graficos_analisis_integrado_py/analisis_varianza_intraday'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, 'daily_std_timeseries_PVSTAND_SEPARATED_OCT_MAR.png')
    
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"✅ Gráfico de series temporales guardado en: {plot_path}")
        plt.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error guardando gráfico de series temporales: {e}")
        plt.close()
        return False

def create_boxplot(daily_std_data):
    """Crea el gráfico de boxplot"""
    
    logger.info("Generando gráfico de boxplot...")
    
    # Configurar matplotlib
    plt.switch_backend('Agg')
    plt.rcParams['figure.figsize'] = (17, 9)
    plt.rcParams['font.size'] = 12
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(17, 9))
    
    # Colores para cada metodología
    colors = {
        'RefCells': '#2E8B57',      # Verde mar
        'DustIQ': '#4169E1',        # Azul real
        'Soiling Kit': '#FF6347',   # Tomate
        'PVStand_Isc': '#DC143C',   # Rojo carmesí
        'PVStand_Pmax': '#8B0000',  # Rojo oscuro
    }
    
    # Título
    ax.set_title('Daily Standard Deviation Distribution', 
                  fontsize=16, fontweight='bold', pad=10)
    
    # Preparar datos para boxplot
    box_data = []
    box_labels = []
    box_colors = []
    
    for method, data in daily_std_data.items():
        if not data.empty:
            box_data.append(data.values)
            box_labels.append(f'{method}')
            box_colors.append(colors.get(method, '#000000'))
    
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        
        # Colorear las cajas
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('Standard Deviation (%)', fontsize=14)
    ax.set_xlabel('Methodology', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = '/home/nicole/SR/SOILING/graficos_analisis_integrado_py/analisis_varianza_intraday'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, 'daily_std_boxplot_PVSTAND_SEPARATED_OCT_MAR.png')
    
    try:
        plt.savefig(plot_path, dpi=900, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"✅ Gráfico de boxplot guardado en: {plot_path}")
        plt.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error guardando gráfico de boxplot: {e}")
        plt.close()
        return False

def main():
    """Función principal para crear gráficos separados"""
    
    logger.info("=== CREANDO GRÁFICOS SEPARADOS CON PVSTAND SEPARADO ===")
    logger.info("PERÍODO: OCTUBRE 2024 - MARZO 2025")
    logger.info("HORARIO UNIFICADO: 13:00-18:00")
    logger.info("PVSTAND SEPARADO: Isc vs Pmax")
    
    # Configurar fechas del período
    start_date = '2024-10-01'
    end_date = '2025-03-31'
    
    logger.info(f"Período: {start_date} a {end_date}")
    
    # Cargar datos
    logger.info("\nCargando datos...")
    
    # 1. DustIQ (datos con horario unificado)
    logger.info("- Cargando DustIQ (horario unificado 13:00-18:00)...")
    df_dustiq = load_dustiq_unified_data()
    if not df_dustiq.empty:
        logger.info(f"  DustIQ: {len(df_dustiq)} registros unificados")
    
    # 2. RefCells (ya tiene horario 13:00-18:00)
    logger.info("- Cargando RefCells (ya filtrado 13:00-18:00)...")
    df_refcells = load_refcells_minute_data()
    if not df_refcells.empty:
        df_refcells = df_refcells[(df_refcells.index >= start_date) & (df_refcells.index <= end_date)]
        logger.info(f"  RefCells: {len(df_refcells)} registros para el período")
    
    # 3. PVStand (separado Isc y Pmax)
    logger.info("- Cargando PVStand (separando Isc y Pmax)...")
    df_pvstand_isc, df_pvstand_pmax = load_pvstand_separated_data()
    if not df_pvstand_isc.empty:
        logger.info(f"  PVStand Isc: {len(df_pvstand_isc)} registros para el período")
    if not df_pvstand_pmax.empty:
        logger.info(f"  PVStand Pmax: {len(df_pvstand_pmax)} registros para el período")
    
    # 4. Soiling Kit (filtrar a 13:00-18:00)
    logger.info("- Cargando Soiling Kit (filtrando a 13:00-18:00)...")
    df_soiling = load_soiling_kit_minute_data()
    if not df_soiling.empty:
        df_soiling = df_soiling[(df_soiling.index >= start_date) & (df_soiling.index <= end_date)]
        # Aplicar filtro de horario unificado
        df_soiling = df_soiling.between_time('13:00', '18:00')
        logger.info(f"  Soiling Kit: {len(df_soiling)} registros para el período (filtrado 13:00-18:00)")
    
    # Calcular desviaciones estándar diarias
    logger.info("\nCalculando desviaciones estándar diarias...")
    
    daily_std_data = {}
    
    # DustIQ
    if not df_dustiq.empty:
        daily_std_dustiq = calculate_daily_std(df_dustiq, 'SR_DustIQ', 'DustIQ')
        if not daily_std_dustiq.empty:
            daily_std_data['DustIQ'] = daily_std_dustiq
            logger.info(f"  DustIQ: {len(daily_std_dustiq)} días con std confiable")
    
    # RefCells
    if not df_refcells.empty:
        daily_std_refcells = calculate_daily_std(df_refcells, 'SR_RefCells', 'RefCells')
        if not daily_std_refcells.empty:
            daily_std_data['RefCells'] = daily_std_refcells
            logger.info(f"  RefCells: {len(daily_std_refcells)} días con std confiable")
    
    # PVStand Isc
    if not df_pvstand_isc.empty:
        daily_std_pvstand_isc = calculate_daily_std(df_pvstand_isc, 'SR_PVStand_Isc', 'PVStand_Isc')
        if not daily_std_pvstand_isc.empty:
            daily_std_data['PVStand_Isc'] = daily_std_pvstand_isc
            logger.info(f"  PVStand Isc: {len(daily_std_pvstand_isc)} días con std confiable")
    
    # PVStand Pmax
    if not df_pvstand_pmax.empty:
        daily_std_pvstand_pmax = calculate_daily_std(df_pvstand_pmax, 'SR_PVStand_Pmax', 'PVStand_Pmax')
        if not daily_std_pvstand_pmax.empty:
            daily_std_data['PVStand_Pmax'] = daily_std_pvstand_pmax
            logger.info(f"  PVStand Pmax: {len(daily_std_pvstand_pmax)} días con std confiable")
    
    # Soiling Kit
    if not df_soiling.empty:
        daily_std_soiling = calculate_daily_std(df_soiling, 'SR_SoilingKit', 'Soiling Kit')
        if not daily_std_soiling.empty:
            daily_std_data['Soiling Kit'] = daily_std_soiling
            logger.info(f"  Soiling Kit: {len(daily_std_soiling)} días con std confiable")
    
    if not daily_std_data:
        logger.error("No se pudieron calcular desviaciones estándar diarias")
        return False
    
    # Crear gráficos separados
    logger.info("\nCreando gráficos separados...")
    
    # Gráfico 1: Series temporales
    plot1_success = create_time_series_plot(daily_std_data)
    
    # Gráfico 2: Boxplot
    plot2_success = create_boxplot(daily_std_data)
    
    if plot1_success and plot2_success:
        logger.info("\n🎉 ¡GRÁFICOS SEPARADOS CREADOS EXITOSAMENTE!")
        logger.info("Archivos generados:")
        logger.info("- Series temporales: daily_std_timeseries_PVSTAND_SEPARATED_OCT_MAR.png")
        logger.info("- Boxplot: daily_std_boxplot_PVSTAND_SEPARATED_OCT_MAR.png")
        return True
    else:
        logger.error("Error creando algunos gráficos")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
