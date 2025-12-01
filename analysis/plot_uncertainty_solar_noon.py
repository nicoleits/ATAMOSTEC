"""
Módulo para generar gráficos de análisis de incertidumbre entorno al mediodía solar.

Este módulo genera visualizaciones específicas para el período alrededor del mediodía solar,
que es crítico para análisis de energía solar debido a la máxima irradiancia.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import pytz
from pytz import timezone
import config.paths as paths
import config.settings as settings
import logging
from utils.solar_time import UtilsMedioDiaSolar

logger = logging.getLogger(__name__)

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Timezone para análisis
TZ_ANALYSIS = "America/Santiago"


def filter_by_solar_noon(df, hours_window=2.5):
    """
    Filtra un DataFrame por rangos de medio día solar dinámicos.
    
    Args:
        df: DataFrame con columna 'timestamp' en timezone local
        hours_window: Ventana en horas alrededor del medio día solar (total = 2 * hours_window)
    
    Returns:
        DataFrame filtrado por medio día solar
    """
    if df.empty:
        return df
    
    logger.info(f"Aplicando filtro de medio día solar (±{hours_window} horas alrededor del medio día solar real)")
    logger.info(f"DataFrame original: {len(df)} filas, rango: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    # Asegurar que timestamp esté en timezone local
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(TZ_ANALYSIS)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert(TZ_ANALYSIS)
    
    # Obtener rango de fechas del DataFrame
    start_date = df['timestamp'].min().date()
    end_date = df['timestamp'].max().date()
    
    logger.info(f"Rango de fechas para cálculo solar: {start_date} a {end_date}")
    
    # Inicializar utilidades de medio día solar
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
    
    logger.info(f"Intervalos solares calculados: {len(solar_intervals_df)}")
    
    # Convertir timestamp a UTC para comparación
    df_utc = df.copy()
    df_utc['timestamp_utc'] = df_utc['timestamp'].dt.tz_convert('UTC')
    
    # Crear máscara para filtrar por medio día solar
    mask = pd.Series(False, index=df_utc.index)
    
    # Aplicar cada intervalo de medio día solar
    for i, (_, row) in enumerate(solar_intervals_df.iterrows()):
        # Los intervalos están en UTC naive, convertirlos a UTC aware
        time_i_utc = pd.Timestamp(row[0]).tz_localize('UTC')
        time_f_utc = pd.Timestamp(row[1]).tz_localize('UTC')
        
        # Crear máscara para este intervalo
        interval_mask = (df_utc['timestamp_utc'] >= time_i_utc) & (df_utc['timestamp_utc'] <= time_f_utc)
        mask = mask | interval_mask
    
    # Aplicar máscara
    df_filtered = df_utc[mask].copy()
    df_filtered = df_filtered.drop(columns=['timestamp_utc'])
    
    logger.info(f"DataFrame filtrado por mediodía solar: {len(df_filtered)} filas ({len(df_filtered)/len(df)*100:.1f}% del total)")
    
    return df_filtered


def load_uncertainty_data_solar_noon(hours_window=2.5):
    """
    Carga los datos de incertidumbre y los filtra por mediodía solar.
    
    Args:
        hours_window: Ventana en horas alrededor del mediodía solar
    
    Returns:
        pd.DataFrame: DataFrame con datos filtrados o None si hay error
    """
    minute_file = paths.SR_MINUTE_WITH_UNCERTAINTY_FILE
    
    if not Path(minute_file).exists():
        logger.error(f"Archivo no encontrado: {minute_file}")
        return None
    
    logger.info(f"Cargando datos de incertidumbre desde: {minute_file}")
    
    try:
        # Leer datos
        df = pd.read_csv(minute_file, parse_dates=['timestamp'])
        
        # Asegurar timezone
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        # Convertir a timezone local para análisis
        df['timestamp'] = df['timestamp'].dt.tz_convert(TZ_ANALYSIS)
        
        # Filtrar datos válidos
        df = df[
            df['SR'].notna() &
            (df['SR'] >= 0) &
            (df['SR'] <= 200) &
            df['U_SR_k2_rel'].notna() &
            np.isfinite(df['U_SR_k2_rel']) &
            (df['U_SR_k2_rel'] >= 0) &
            (df['U_SR_k2_rel'] <= 0.5)
        ].copy()
        
        # Convertir incertidumbre a porcentaje
        df['U_SR_k2_rel_pct'] = df['U_SR_k2_rel'] * 100
        
        # Filtrar por mediodía solar
        df_solar_noon = filter_by_solar_noon(df, hours_window=hours_window)
        
        if df_solar_noon.empty:
            logger.warning("No hay datos después del filtro de mediodía solar")
            return None
        
        # Extraer componentes temporales
        df_solar_noon['fecha'] = df_solar_noon['timestamp'].dt.date
        df_solar_noon['hora'] = df_solar_noon['timestamp'].dt.hour
        df_solar_noon['minuto'] = df_solar_noon['timestamp'].dt.minute
        df_solar_noon['mes'] = df_solar_noon['timestamp'].dt.month
        df_solar_noon['año'] = df_solar_noon['timestamp'].dt.year
        df_solar_noon['año_mes'] = df_solar_noon['timestamp'].dt.to_period('M')
        
        # Calcular minutos desde mediodía solar (aproximado como 12:00 local)
        # Para análisis más preciso, se podría calcular el mediodía solar real por día
        df_solar_noon['minutos_desde_mediodia'] = (df_solar_noon['hora'] - 12) * 60 + df_solar_noon['minuto']
        
        logger.info(f"Datos de mediodía solar cargados: {len(df_solar_noon):,} registros válidos")
        logger.info(f"Rango temporal: {df_solar_noon['timestamp'].min()} a {df_solar_noon['timestamp'].max()}")
        logger.info(f"Horas incluidas: {df_solar_noon['hora'].min():02d}:00 - {df_solar_noon['hora'].max():02d}:00")
        
        return df_solar_noon
        
    except Exception as e:
        logger.error(f"Error cargando datos: {e}", exc_info=True)
        return None


def plot_uncertainty_by_minutes_from_noon(df):
    """
    Gráfico de incertidumbre por minutos desde el mediodía solar.
    
    Args:
        df: DataFrame con datos de incertidumbre filtrados por mediodía solar
    """
    logger.info("Generando gráfico de incertidumbre por minutos desde mediodía solar...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Boxplot por intervalo de minutos
    ax1 = axes[0]
    
    # Crear bins de 15 minutos
    df['minutos_bin'] = pd.cut(df['minutos_desde_mediodia'], 
                               bins=range(int(df['minutos_desde_mediodia'].min()), 
                                         int(df['minutos_desde_mediodia'].max()) + 15, 15),
                               include_lowest=True)
    
    # Ordenar por valor medio del bin
    bin_order = sorted(df['minutos_bin'].dropna().unique(), 
                       key=lambda x: x.mid if hasattr(x, 'mid') else x.left)
    
    data_by_bin = [df[df['minutos_bin'] == b]['U_SR_k2_rel_pct'].dropna() for b in bin_order]
    bin_labels = [f"{int(b.mid)}" if hasattr(b, 'mid') else f"{int(b.left)}" for b in bin_order]
    
    bp = ax1.boxplot(data_by_bin, tick_labels=bin_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Minutos desde mediodía solar', fontsize=12)
    ax1.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax1.set_title('Distribución de Incertidumbre por Minutos desde Mediodía Solar', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=len(bin_order)/2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Mediodía solar')
    ax1.legend()
    
    # Gráfico 2: Promedio y desviación estándar
    ax2 = axes[1]
    stats_by_bin = df.groupby('minutos_bin', observed=True)['U_SR_k2_rel_pct'].agg(['mean', 'std', 'count'])
    stats_by_bin = stats_by_bin.reindex(bin_order)
    
    x_pos = range(len(stats_by_bin))
    ax2.plot(x_pos, stats_by_bin['mean'], 'o-', label='Promedio', linewidth=2, markersize=6, color='blue')
    ax2.fill_between(x_pos,
                     stats_by_bin['mean'] - stats_by_bin['std'],
                     stats_by_bin['mean'] + stats_by_bin['std'],
                     alpha=0.3, label='±1 Desviación Estándar', color='blue')
    
    ax2.set_xlabel('Minutos desde mediodía solar', fontsize=12)
    ax2.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax2.set_title('Estadísticas de Incertidumbre por Minutos desde Mediodía Solar', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=len(stats_by_bin)/2, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR) / 'uncertainty_by_minutes_from_noon.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def plot_uncertainty_timeseries_solar_noon(df):
    """
    Series temporales de incertidumbre durante mediodía solar.
    
    Args:
        df: DataFrame con datos de incertidumbre filtrados por mediodía solar
    """
    logger.info("Generando series temporales de incertidumbre durante mediodía solar...")
    
    # Agregar por día
    df_daily = df.groupby('fecha')['U_SR_k2_rel_pct'].agg(['mean', 'std', 'count']).reset_index()
    df_daily['fecha'] = pd.to_datetime(df_daily['fecha'])
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Gráfico 1: Serie temporal diaria
    ax1 = axes[0]
    ax1.plot(df_daily['fecha'], df_daily['mean'], 'b-', linewidth=1.5, alpha=0.7, label='Promedio diario')
    ax1.fill_between(df_daily['fecha'], 
                     df_daily['mean'] - df_daily['std'],
                     df_daily['mean'] + df_daily['std'],
                     alpha=0.3, label='±1 Desviación Estándar')
    
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax1.set_title('Serie Temporal de Incertidumbre durante Mediodía Solar (Diaria)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Gráfico 2: Agregación semanal (convertir a UTC para evitar problemas de DST)
    df_weekly_utc = df.copy()
    df_weekly_utc['timestamp_utc'] = df_weekly_utc['timestamp'].dt.tz_convert('UTC')
    df_weekly_utc = df_weekly_utc.set_index('timestamp_utc')
    df_weekly = df_weekly_utc['U_SR_k2_rel_pct'].resample('W-SUN', label='right', closed='right').agg(['mean', 'std', 'count'])
    df_weekly = df_weekly[df_weekly['count'] > 0]
    
    ax2 = axes[1]
    ax2.plot(df_weekly.index, df_weekly['mean'], 'g-', linewidth=2, alpha=0.7, label='Promedio semanal')
    ax2.fill_between(df_weekly.index,
                     df_weekly['mean'] - df_weekly['std'],
                     df_weekly['mean'] + df_weekly['std'],
                     alpha=0.3, label='±1 Desviación Estándar')
    
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax2.set_title('Serie Temporal de Incertidumbre durante Mediodía Solar (Semanal)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Gráfico 3: Agregación mensual
    df_monthly = df.groupby('año_mes')['U_SR_k2_rel_pct'].agg(['mean', 'std', 'count'])
    
    ax3 = axes[2]
    x_pos = range(len(df_monthly))
    bars = ax3.bar(x_pos, df_monthly['mean'], yerr=df_monthly['std'],
                   capsize=3, alpha=0.7, edgecolor='black', color='coral')
    
    ax3.set_xlabel('Mes', fontsize=12)
    ax3.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax3.set_title('Incertidumbre Promedio durante Mediodía Solar por Mes', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([str(m) for m in df_monthly.index], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores
    for i, (bar, mean) in enumerate(zip(bars, df_monthly['mean'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR) / 'uncertainty_timeseries_solar_noon.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def plot_uncertainty_vs_sr_solar_noon(df):
    """
    Gráfico de incertidumbre vs SR durante mediodía solar.
    
    Args:
        df: DataFrame con datos de incertidumbre filtrados por mediodía solar
    """
    logger.info("Generando gráfico de incertidumbre vs SR durante mediodía solar...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Scatter plot
    ax1 = axes[0]
    scatter = ax1.scatter(df['SR'], df['U_SR_k2_rel_pct'], 
                         alpha=0.3, s=1, c=df['minutos_desde_mediodia'], cmap='viridis')
    ax1.set_xlabel('Soiling Ratio SR (%)', fontsize=12)
    ax1.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax1.set_title('Incertidumbre vs Soiling Ratio durante Mediodía Solar', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Minutos desde mediodía')
    
    # Ajustar línea de tendencia
    valid_data = df[df['SR'].notna() & df['U_SR_k2_rel_pct'].notna()]
    if len(valid_data) > 10:
        z = np.polyfit(valid_data['SR'], valid_data['U_SR_k2_rel_pct'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['SR'].min(), valid_data['SR'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, 
                label=f'Tendencia: y={z[0]:.4f}x+{z[1]:.2f}')
        ax1.legend()
    
    # Gráfico 2: Incertidumbre promedio por rangos de SR
    ax2 = axes[1]
    sr_bins = [0, 80, 90, 95, 100, 105, 200]
    df['sr_range'] = pd.cut(df['SR'], bins=sr_bins, include_lowest=True)
    
    range_stats = df.groupby('sr_range', observed=True)['U_SR_k2_rel_pct'].agg(['mean', 'std', 'count'])
    
    x_pos = range(len(range_stats))
    bars = ax2.bar(x_pos, range_stats['mean'], yerr=range_stats['std'],
                   capsize=5, alpha=0.7, edgecolor='black', color='coral')
    
    ax2.set_xlabel('Rango de SR (%)', fontsize=12)
    ax2.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax2.set_title('Incertidumbre Promedio por Rangos de SR durante Mediodía Solar', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(r) for r in range_stats.index], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores
    for i, (bar, mean, count) in enumerate(zip(bars, range_stats['mean'], range_stats['count'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}%\n(n={int(count):,})', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR) / 'uncertainty_vs_sr_solar_noon.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def plot_uncertainty_distribution_solar_noon(df):
    """
    Distribución de incertidumbre durante mediodía solar.
    
    Args:
        df: DataFrame con datos de incertidumbre filtrados por mediodía solar
    """
    logger.info("Generando gráficos de distribución de incertidumbre durante mediodía solar...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Histograma
    ax1 = axes[0, 0]
    ax1.hist(df['U_SR_k2_rel_pct'].dropna(), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.set_xlabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('Distribución de Incertidumbre durante Mediodía Solar', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(df['U_SR_k2_rel_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["U_SR_k2_rel_pct"].mean():.2f}%')
    ax1.axvline(df['U_SR_k2_rel_pct'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["U_SR_k2_rel_pct"].median():.2f}%')
    ax1.legend()
    
    # Gráfico 2: Boxplot
    ax2 = axes[0, 1]
    bp = ax2.boxplot([df['U_SR_k2_rel_pct'].dropna()], tick_labels=['Incertidumbre'],
                     patch_artist=True, vert=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax2.set_title('Boxplot de Incertidumbre durante Mediodía Solar', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 3: Q-Q plot
    ax3 = axes[1, 0]
    from scipy import stats
    stats.probplot(df['U_SR_k2_rel_pct'].dropna(), dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot de Incertidumbre durante Mediodía Solar', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Violin plot por mes
    ax4 = axes[1, 1]
    monthly_data = [df[df['mes'] == m]['U_SR_k2_rel_pct'].dropna() for m in sorted(df['mes'].unique())]
    if len(monthly_data) > 0 and all(len(d) > 0 for d in monthly_data):
        parts = ax4.violinplot(monthly_data, positions=range(len(monthly_data)), 
                               showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        ax4.set_xticks(range(len(monthly_data)))
        ax4.set_xticklabels([f'Mes {m}' for m in sorted(df['mes'].unique())])
        ax4.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
        ax4.set_title('Distribución de Incertidumbre por Mes durante Mediodía Solar', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR) / 'uncertainty_distribution_solar_noon.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def generate_all_solar_noon_plots(hours_window=2.5):
    """
    Función principal que genera todos los gráficos de análisis de incertidumbre durante mediodía solar.
    
    Args:
        hours_window: Ventana en horas alrededor del mediodía solar
    
    Returns:
        list: Lista de archivos generados
    """
    logger.info("="*80)
    logger.info("INICIANDO GENERACIÓN DE GRÁFICOS DE ANÁLISIS DE INCERTIDUMBRE - MEDIODÍA SOLAR")
    logger.info("="*80)
    
    # Cargar datos filtrados por mediodía solar
    df = load_uncertainty_data_solar_noon(hours_window=hours_window)
    
    if df is None or df.empty:
        logger.error("No se pudieron cargar los datos de incertidumbre para mediodía solar")
        return []
    
    # Asegurar que el directorio existe
    os.makedirs(paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR, exist_ok=True)
    
    generated_files = []
    
    try:
        # Generar todos los gráficos
        logger.info("\n📊 Generando gráficos de análisis de incertidumbre durante mediodía solar...\n")
        
        file1 = plot_uncertainty_by_minutes_from_noon(df)
        if file1:
            generated_files.append(file1)
        
        file2 = plot_uncertainty_timeseries_solar_noon(df)
        if file2:
            generated_files.append(file2)
        
        file3 = plot_uncertainty_vs_sr_solar_noon(df)
        if file3:
            generated_files.append(file3)
        
        file4 = plot_uncertainty_distribution_solar_noon(df)
        if file4:
            generated_files.append(file4)
        
        logger.info("\n" + "="*80)
        logger.info("✅ GENERACIÓN DE GRÁFICOS COMPLETADA")
        logger.info(f"Total de gráficos generados: {len(generated_files)}")
        logger.info(f"Ubicación: {paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR}")
        logger.info("="*80)
        
        if generated_files:
            print(f"\n✅ Se generaron {len(generated_files)} gráficos:")
            for f in generated_files:
                print(f"  - {f}")
        else:
            print("\n❌ No se generaron gráficos")
        
        return generated_files
        
    except Exception as e:
        logger.error(f"Error generando gráficos: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generar todos los gráficos
    generate_all_solar_noon_plots(hours_window=2.5)

