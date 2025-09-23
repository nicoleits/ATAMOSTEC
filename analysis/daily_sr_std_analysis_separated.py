#!/usr/bin/env python3
"""
Análisis de desviación estándar diaria con PVStand separado (Isc y Pmax)
Período: Octubre 2024 - Marzo 2025
Horario unificado: 13:00-18:00
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

def create_separated_comparison_plot(daily_std_data):
    """Crea el gráfico de comparación con PVStand separado"""
    
    logger.info("Generando gráfico de comparación con PVStand separado...")
    
    # Configurar matplotlib
    plt.switch_backend('Agg')
    plt.rcParams['figure.figsize'] = (20, 16)
    plt.rcParams['font.size'] = 11
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 18))
    
    # Colores para cada metodología
    colors = {
        'RefCells': '#2E8B57',      # Verde mar
        'DustIQ': '#4169E1',        # Azul real
        'Soiling Kit': '#FF6347',   # Tomate
        'PVStand_Isc': '#DC143C',   # Rojo carmesí
        'PVStand_Pmax': '#8B0000',  # Rojo oscuro
    }
    
    # Subplot 1: Series temporales
    ax1.set_title('Desviación Estándar Diaria de Soiling Ratios por Metodología\nPeríodo: Octubre 2024 - Marzo 2025 (HORARIO UNIFICADO 13:00-18:00)\nPVStand Separado: Isc vs Pmax', 
                  fontsize=16, fontweight='bold', pad=25)
    
    for method, data in daily_std_data.items():
        if not data.empty:
            ax1.plot(data.index, data.values, 
                    label=f'{method} (n={len(data)})', 
                    color=colors.get(method, '#000000'),
                    linewidth=2.5, alpha=0.8, marker='o', markersize=4)
    
    ax1.set_ylabel('Desviación Estándar (%)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Subplot 2: Boxplot
    ax2.set_title('Distribución de Desviaciones Estándar Diarias\nPeríodo: Octubre 2024 - Marzo 2025 (HORARIO UNIFICADO 13:00-18:00)\nPVStand Separado: Isc vs Pmax', 
                  fontsize=16, fontweight='bold', pad=25)
    
    box_data = []
    box_labels = []
    box_colors = []
    
    for method, data in daily_std_data.items():
        if not data.empty:
            box_data.append(data.values)
            box_labels.append(f'{method}\n(n={len(data)})')
            box_colors.append(colors.get(method, '#000000'))
    
    if box_data:
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        
        # Colorear las cajas
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Desviación Estándar (%)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = '/home/nicole/SR/SOILING/graficos_analisis_integrado_py/analisis_varianza_intraday'
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, 'daily_std_comparison_PVSTAND_SEPARATED_OCT_MAR.png')
    
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"✅ Gráfico con PVStand separado guardado en: {plot_path}")
        plt.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error guardando gráfico: {e}")
        plt.close()
        return False

def generate_separated_summary_report(daily_std_data):
    """Genera un reporte resumen con PVStand separado"""
    
    logger.info("Generando reporte resumen con PVStand separado...")
    
    summary_data = []
    
    for method, data in daily_std_data.items():
        if not data.empty:
            summary_data.append({
                'Metodología': method,
                'Días Analizados': len(data),
                'Std Promedio (%)': f"{data.mean():.3f}",
                'Std Mediana (%)': f"{data.median():.3f}",
                'Std Mínima (%)': f"{data.min():.3f}",
                'Std Máxima (%)': f"{data.max():.3f}",
                'Coeficiente Variación': f"{data.std()/data.mean()*100:.2f}%",
                'Rango Fechas': f"{data.index.min().strftime('%Y-%m-%d')} a {data.index.max().strftime('%Y-%m-%d')}"
            })
    
    # Crear DataFrame y ordenar
    summary_df = pd.DataFrame(summary_data)
    summary_df['Std Promedio Num'] = summary_df['Std Promedio (%)'].astype(float)
    summary_df = summary_df.sort_values('Std Promedio Num')
    summary_df = summary_df.drop('Std Promedio Num', axis=1)
    
    # Guardar reporte
    output_dir = '/home/nicole/SR/SOILING/graficos_analisis_integrado_py/analisis_varianza_intraday'
    report_path = os.path.join(output_dir, 'resumen_analisis_std_diaria_PVSTAND_SEPARATED_OCT_MAR.csv')
    summary_df.to_csv(report_path, index=False)
    
    logger.info(f"✅ Reporte con PVStand separado guardado en: {report_path}")
    
    return summary_df

def main():
    """Función principal del análisis con PVStand separado"""
    
    logger.info("=== ANÁLISIS CON PVSTAND SEPARADO (ISC Y PMAX) ===")
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
            logger.info(f"  DustIQ: Std promedio: {daily_std_dustiq.mean():.3f}%")
    
    # RefCells
    if not df_refcells.empty:
        daily_std_refcells = calculate_daily_std(df_refcells, 'SR_RefCells', 'RefCells')
        if not daily_std_refcells.empty:
            daily_std_data['RefCells'] = daily_std_refcells
            logger.info(f"  RefCells: {len(daily_std_refcells)} días con std confiable")
            logger.info(f"  RefCells: Std promedio: {daily_std_refcells.mean():.3f}%")
    
    # PVStand Isc
    if not df_pvstand_isc.empty:
        daily_std_pvstand_isc = calculate_daily_std(df_pvstand_isc, 'SR_PVStand_Isc', 'PVStand_Isc')
        if not daily_std_pvstand_isc.empty:
            daily_std_data['PVStand_Isc'] = daily_std_pvstand_isc
            logger.info(f"  PVStand Isc: {len(daily_std_pvstand_isc)} días con std confiable")
            logger.info(f"  PVStand Isc: Std promedio: {daily_std_pvstand_isc.mean():.3f}%")
    
    # PVStand Pmax
    if not df_pvstand_pmax.empty:
        daily_std_pvstand_pmax = calculate_daily_std(df_pvstand_pmax, 'SR_PVStand_Pmax', 'PVStand_Pmax')
        if not daily_std_pvstand_pmax.empty:
            daily_std_data['PVStand_Pmax'] = daily_std_pvstand_pmax
            logger.info(f"  PVStand Pmax: {len(daily_std_pvstand_pmax)} días con std confiable")
            logger.info(f"  PVStand Pmax: Std promedio: {daily_std_pvstand_pmax.mean():.3f}%")
    
    # Soiling Kit
    if not df_soiling.empty:
        daily_std_soiling = calculate_daily_std(df_soiling, 'SR_SoilingKit', 'Soiling Kit')
        if not daily_std_soiling.empty:
            daily_std_data['Soiling Kit'] = daily_std_soiling
            logger.info(f"  Soiling Kit: {len(daily_std_soiling)} días con std confiable")
            logger.info(f"  Soiling Kit: Std promedio: {daily_std_soiling.mean():.3f}%")
    
    if not daily_std_data:
        logger.error("No se pudieron calcular desviaciones estándar diarias")
        return False
    
    # Generar gráfico con PVStand separado
    logger.info("\nGenerando gráfico con PVStand separado...")
    plot_success = create_separated_comparison_plot(daily_std_data)
    
    # Generar reporte con PVStand separado
    logger.info("\nGenerando reporte con PVStand separado...")
    summary_df = generate_separated_summary_report(daily_std_data)
    
    # Mostrar resumen en consola
    logger.info("\n=== RESUMEN CON PVSTAND SEPARADO ===")
    print("\n" + "="*110)
    print("ANÁLISIS DE DESVIACIÓN ESTÁNDAR DIARIA - PVSTAND SEPARADO (ISC vs PMAX)")
    print("PERÍODO: OCTUBRE 2024 - MARZO 2025 (HORARIO UNIFICADO 13:00-18:00)")
    print("="*110)
    print(summary_df.to_string(index=False))
    print("="*110)
    
    # Ranking de estabilidad
    print("\n🏆 RANKING DE ESTABILIDAD (Menor std = Más estable):")
    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        print(f"{i}º {row['Metodología']}: {row['Std Promedio (%)']}% (n={row['Días Analizados']})")
    
    # Análisis específico de PVStand
    pvstand_isc_std = daily_std_data.get('PVStand_Isc', pd.Series())
    pvstand_pmax_std = daily_std_data.get('PVStand_Pmax', pd.Series())
    
    if not pvstand_isc_std.empty and not pvstand_pmax_std.empty:
        print(f"\n🔍 ANÁLISIS ESPECÍFICO PVSTAND:")
        print(f"   Isc: {pvstand_isc_std.mean():.3f}% (n={len(pvstand_isc_std)})")
        print(f"   Pmax: {pvstand_pmax_std.mean():.3f}% (n={len(pvstand_pmax_std)})")
        print(f"   Diferencia: {abs(pvstand_isc_std.mean() - pvstand_pmax_std.mean()):.3f}%")
        if pvstand_isc_std.mean() < pvstand_pmax_std.mean():
            print(f"   ➤ Isc es más estable que Pmax")
        else:
            print(f"   ➤ Pmax es más estable que Isc")
    
    print("\n✅ COMPARACIÓN JUSTA: Todas las metodologías usan el mismo rango horario (13:00-18:00)")
    print("✅ PVSTAND SEPARADO: Análisis individual de Isc y Pmax")
    
    if plot_success:
        logger.info("\n🎉 ¡ANÁLISIS CON PVSTAND SEPARADO COMPLETADO EXITOSAMENTE!")
        logger.info("Archivos generados:")
        logger.info("- Gráfico: daily_std_comparison_PVSTAND_SEPARATED_OCT_MAR.png")
        logger.info("- Reporte: resumen_analisis_std_diaria_PVSTAND_SEPARATED_OCT_MAR.csv")
        return True
    else:
        logger.error("Error generando el gráfico con PVStand separado")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
