"""
Módulo para generar gráficos de análisis de dispersión de errores de incertidumbre.

Este módulo genera visualizaciones que permiten identificar:
- Patrones horarios de incertidumbre
- Variación por día de la semana
- Variación estacional/mensual
- Series temporales de incertidumbre
- Distribuciones y heatmaps
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
import config.paths as paths
import config.settings as settings
import logging

logger = logging.getLogger(__name__)

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Timezone para análisis
TZ_ANALYSIS = "America/Santiago"


def load_uncertainty_data():
    """
    Carga los datos de incertidumbre minuto a minuto.
    
    Returns:
        pd.DataFrame: DataFrame con datos de incertidumbre o None si hay error
    """
    minute_file = paths.SR_MINUTE_WITH_UNCERTAINTY_FILE
    
    if not Path(minute_file).exists():
        logger.error(f"Archivo no encontrado: {minute_file}")
        return None
    
    logger.info(f"Cargando datos de incertidumbre desde: {minute_file}")
    
    try:
        # Leer datos (puede ser grande, usar chunks si es necesario)
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
        
        # Extraer componentes temporales
        df['fecha'] = df['timestamp'].dt.date
        df['hora'] = df['timestamp'].dt.hour
        df['dia_semana'] = df['timestamp'].dt.day_name()
        df['mes'] = df['timestamp'].dt.month
        df['año'] = df['timestamp'].dt.year
        df['año_mes'] = df['timestamp'].dt.to_period('M')
        df['dia_mes'] = df['timestamp'].dt.day
        
        # Filtrar solo horas entre 05:00 y 21:00
        df = df[(df['hora'] >= 5) & (df['hora'] <= 21)].copy()
        
        # Ordenar por día de semana
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)
        
        logger.info(f"Datos cargados: {len(df):,} registros válidos (filtrados: 05:00-21:00)")
        logger.info(f"Rango temporal: {df['timestamp'].min()} a {df['timestamp'].max()}")
        logger.info(f"Horas incluidas: {df['hora'].min():02d}:00 - {df['hora'].max():02d}:00")
        
        return df
        
    except Exception as e:
        logger.error(f"Error cargando datos: {e}", exc_info=True)
        return None


def plot_uncertainty_by_hour(df):
    """
    Gráfico de incertidumbre por hora del día.
    
    Args:
        df: DataFrame con datos de incertidumbre
    """
    logger.info("Generando gráfico de incertidumbre por hora del día...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Boxplot por hora (solo 05:00-21:00)
    ax1 = axes[0]
    horas_validas = sorted(df['hora'].unique())
    data_by_hour = [df[df['hora'] == h]['U_SR_k2_rel_pct'].dropna() for h in horas_validas]
    
    bp = ax1.boxplot(data_by_hour, labels=horas_validas, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Hora del día', fontsize=12)
    ax1.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax1.set_title('Distribución de Incertidumbre por Hora del Día (05:00-21:00)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # Mostrar solo horas válidas, cada 2 horas
    tick_positions = [h for h in horas_validas if h % 2 == 0 or h == horas_validas[0] or h == horas_validas[-1]]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f'{h:02d}:00' for h in tick_positions])
    
    # Gráfico 2: Promedio y percentiles por hora
    ax2 = axes[1]
    hourly_stats = df.groupby('hora')['U_SR_k2_rel_pct'].agg(['mean', 'median', 'std', 'count'])
    
    ax2.plot(hourly_stats.index, hourly_stats['mean'], 'o-', label='Promedio', linewidth=2, markersize=6)
    ax2.fill_between(
        hourly_stats.index,
        hourly_stats['mean'] - hourly_stats['std'],
        hourly_stats['mean'] + hourly_stats['std'],
        alpha=0.3, label='±1 Desviación Estándar'
    )
    ax2.plot(hourly_stats.index, hourly_stats['median'], 's--', label='Mediana', linewidth=2, markersize=4)
    
    ax2.set_xlabel('Hora del día', fontsize=12)
    ax2.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax2.set_title('Estadísticas de Incertidumbre por Hora del Día (05:00-21:00)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    tick_positions = [h for h in horas_validas if h % 2 == 0 or h == horas_validas[0] or h == horas_validas[-1]]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f'{h:02d}:00' for h in tick_positions])
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'uncertainty_by_hour.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def plot_uncertainty_by_month_season(df):
    """
    Gráfico de incertidumbre por mes y temporada.
    
    Args:
        df: DataFrame con datos de incertidumbre
    """
    logger.info("Generando gráfico de incertidumbre por mes y temporada...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Definir temporadas (hemisferio sur)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Verano'
        elif month in [3, 4, 5]:
            return 'Otoño'
        elif month in [6, 7, 8]:
            return 'Invierno'
        else:
            return 'Primavera'
    
    df['temporada'] = df['mes'].apply(get_season)
    
    # Gráfico 1: Por mes
    ax1 = axes[0]
    monthly_stats = df.groupby('año_mes')['U_SR_k2_rel_pct'].agg(['mean', 'std', 'count'])
    
    x_pos = range(len(monthly_stats))
    bars = ax1.bar(x_pos, monthly_stats['mean'], yerr=monthly_stats['std'],
                   capsize=3, alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Mes', fontsize=12)
    ax1.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax1.set_title('Incertidumbre Promedio por Mes', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(p) for p in monthly_stats.index], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores
    for i, (bar, mean) in enumerate(zip(bars, monthly_stats['mean'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # Gráfico 2: Por temporada
    ax2 = axes[1]
    season_stats = df.groupby('temporada')['U_SR_k2_rel_pct'].agg(['mean', 'std', 'count'])
    season_order = ['Verano', 'Otoño', 'Invierno', 'Primavera']
    season_stats = season_stats.reindex([s for s in season_order if s in season_stats.index])
    
    x_pos = range(len(season_stats))
    colors_season = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax2.bar(x_pos, season_stats['mean'], yerr=season_stats['std'],
                   capsize=5, color=colors_season[:len(season_stats)], 
                   alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Temporada', fontsize=12)
    ax2.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax2.set_title('Incertidumbre Promedio por Temporada', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(season_stats.index, rotation=0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores
    for i, (bar, mean, std) in enumerate(zip(bars, season_stats['mean'], season_stats['std'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'uncertainty_by_month_season.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def plot_uncertainty_heatmap_hour_month(df):
    """
    Heatmap de incertidumbre: hora del día vs mes.
    
    Args:
        df: DataFrame con datos de incertidumbre
    """
    logger.info("Generando heatmap de incertidumbre (hora vs mes)...")
    
    # Crear matriz de incertidumbre promedio: hora vs mes
    pivot_data = df.pivot_table(
        values='U_SR_k2_rel_pct',
        index='año_mes',
        columns='hora',
        aggfunc='mean',
        observed=True
    )
    
    # Ordenar por fecha
    pivot_data = pivot_data.sort_index()
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    sns.heatmap(pivot_data, annot=False, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Incertidumbre U_SR_k2 (%)'},
                linewidths=0.1, linecolor='gray', ax=ax, vmin=0, vmax=15)
    
    ax.set_xlabel('Hora del día', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mes (Año-Mes)', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap de Incertidumbre: Hora del Día vs Mes (05:00-21:00)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Mejorar etiquetas - solo horas válidas (las columnas ya son las horas)
    horas_validas = sorted(pivot_data.columns)
    tick_positions = [h for h in horas_validas if h % 2 == 0 or h == horas_validas[0] or h == horas_validas[-1]]
    # Encontrar las posiciones de estas horas en las columnas
    tick_indices = [list(pivot_data.columns).index(h) for h in tick_positions]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{h:02d}:00' for h in tick_positions])
    ax.set_yticklabels([str(m) for m in pivot_data.index], rotation=0)
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'uncertainty_heatmap_hour_month.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def plot_uncertainty_timeseries(df):
    """
    Series temporales de incertidumbre.
    
    Args:
        df: DataFrame con datos de incertidumbre
    """
    logger.info("Generando series temporales de incertidumbre...")
    
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
    ax1.set_title('Serie Temporal de Incertidumbre (Diaria)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Gráfico 2: Agregación semanal
    df['semana'] = df['timestamp'].dt.to_period('W-SUN')
    weekly_stats = df.groupby('semana')['U_SR_k2_rel_pct'].agg(['mean', 'std']).reset_index()
    weekly_stats['semana'] = weekly_stats['semana'].astype(str)
    weekly_stats['fecha'] = pd.to_datetime([str(w).split('/')[0] for w in weekly_stats['semana']])
    
    ax2 = axes[1]
    ax2.plot(weekly_stats['fecha'], weekly_stats['mean'], 'g-', linewidth=2, marker='o', 
             markersize=4, label='Promedio semanal')
    ax2.fill_between(weekly_stats['fecha'],
                     weekly_stats['mean'] - weekly_stats['std'],
                     weekly_stats['mean'] + weekly_stats['std'],
                     alpha=0.3, label='±1 Desviación Estándar')
    
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax2.set_title('Serie Temporal de Incertidumbre (Semanal)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Gráfico 3: Agregación mensual
    monthly_stats = df.groupby('año_mes')['U_SR_k2_rel_pct'].agg(['mean', 'std']).reset_index()
    monthly_stats['fecha'] = pd.to_datetime([str(m) for m in monthly_stats['año_mes']])
    
    ax3 = axes[2]
    ax3.plot(monthly_stats['fecha'], monthly_stats['mean'], 'r-', linewidth=2.5, 
             marker='s', markersize=6, label='Promedio mensual')
    ax3.fill_between(monthly_stats['fecha'],
                     monthly_stats['mean'] - monthly_stats['std'],
                     monthly_stats['mean'] + monthly_stats['std'],
                     alpha=0.3, label='±1 Desviación Estándar')
    
    ax3.set_xlabel('Fecha', fontsize=12)
    ax3.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax3.set_title('Serie Temporal de Incertidumbre (Mensual)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'uncertainty_timeseries.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def plot_uncertainty_distribution(df):
    """
    Distribución de incertidumbre.
    
    Args:
        df: DataFrame con datos de incertidumbre
    """
    logger.info("Generando gráficos de distribución de incertidumbre...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Histograma
    ax1 = axes[0, 0]
    ax1.hist(df['U_SR_k2_rel_pct'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(df['U_SR_k2_rel_pct'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {df["U_SR_k2_rel_pct"].mean():.2f}%')
    ax1.axvline(df['U_SR_k2_rel_pct'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Mediana: {df["U_SR_k2_rel_pct"].median():.2f}%')
    ax1.set_xlabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('Distribución de Incertidumbre', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 2: Boxplot general
    ax2 = axes[0, 1]
    bp = ax2.boxplot([df['U_SR_k2_rel_pct']], labels=['Incertidumbre'], 
                     patch_artist=True, vert=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax2.set_title('Boxplot de Incertidumbre', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar estadísticas
    stats_text = f"Media: {df['U_SR_k2_rel_pct'].mean():.2f}%\n"
    stats_text += f"Mediana: {df['U_SR_k2_rel_pct'].median():.2f}%\n"
    stats_text += f"Std: {df['U_SR_k2_rel_pct'].std():.2f}%\n"
    stats_text += f"Min: {df['U_SR_k2_rel_pct'].min():.2f}%\n"
    stats_text += f"Max: {df['U_SR_k2_rel_pct'].max():.2f}%"
    ax2.text(0.7, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Gráfico 3: Q-Q plot (normalidad)
    ax3 = axes[1, 0]
    from scipy import stats
    data_norm = (df['U_SR_k2_rel_pct'] - df['U_SR_k2_rel_pct'].mean()) / df['U_SR_k2_rel_pct'].std()
    stats.probplot(data_norm, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normalidad)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Violin plot por hora (muestra)
    ax4 = axes[1, 1]
    # Seleccionar horas representativas para no saturar
    horas_seleccionadas = [6, 9, 12, 15, 18, 21]
    df_sample = df[df['hora'].isin(horas_seleccionadas)]
    
    # Filtrar horas que tienen datos
    data_violin = []
    horas_validas = []
    for h in horas_seleccionadas:
        data_h = df_sample[df_sample['hora'] == h]['U_SR_k2_rel_pct'].dropna()
        if len(data_h) > 0:
            data_violin.append(data_h)
            horas_validas.append(h)
    
    if len(data_violin) > 0:
        parts = ax4.violinplot(data_violin, positions=range(len(horas_validas)),
                               showmeans=True, showmedians=True)
    
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        ax4.set_xticks(range(len(horas_validas)))
        ax4.set_xticklabels([f'{h:02d}:00' for h in horas_validas])
    else:
        ax4.text(0.5, 0.5, 'No hay datos suficientes\npara este gráfico', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_xlabel('Hora del día', fontsize=12)
    ax4.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax4.set_title('Distribución de Incertidumbre por Hora (Selección)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar
    output_file = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'uncertainty_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def plot_uncertainty_vs_sr(df):
    """
    Gráfico de incertidumbre vs SR.
    
    Args:
        df: DataFrame con datos de incertidumbre
    """
    logger.info("Generando gráfico de incertidumbre vs SR...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Scatter plot
    ax1 = axes[0]
    scatter = ax1.scatter(df['SR'], df['U_SR_k2_rel_pct'], 
                         alpha=0.3, s=1, c=df['hora'], cmap='viridis')
    ax1.set_xlabel('Soiling Ratio SR (%)', fontsize=12)
    ax1.set_ylabel('Incertidumbre U_SR_k2 (%)', fontsize=12)
    ax1.set_title('Incertidumbre vs Soiling Ratio', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Hora del día', fontsize=10)
    
    # Agregar línea de tendencia
    z = np.polyfit(df['SR'].dropna(), df.loc[df['SR'].notna(), 'U_SR_k2_rel_pct'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['SR'].min(), df['SR'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label=f'Tendencia: y={z[0]:.4f}x+{z[1]:.2f}')
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
    ax2.set_title('Incertidumbre Promedio por Rangos de SR', fontsize=14, fontweight='bold')
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
    output_file = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'uncertainty_vs_sr.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado en: {output_file}")
    return output_file


def generate_all_uncertainty_plots():
    """
    Función principal que genera todos los gráficos de análisis de incertidumbre.
    
    Returns:
        list: Lista de archivos generados
    """
    logger.info("="*80)
    logger.info("INICIANDO GENERACIÓN DE GRÁFICOS DE ANÁLISIS DE INCERTIDUMBRE")
    logger.info("="*80)
    
    # Cargar datos
    df = load_uncertainty_data()
    
    if df is None or df.empty:
        logger.error("No se pudieron cargar los datos de incertidumbre")
        return []
    
    # Asegurar que el directorio existe
    os.makedirs(paths.PROPAGACION_ERRORES_REF_CELL_DIR, exist_ok=True)
    
    generated_files = []
    
    try:
        # Generar todos los gráficos
        logger.info("\n📊 Generando gráficos de análisis de incertidumbre...\n")
        
        file1 = plot_uncertainty_by_hour(df)
        if file1:
            generated_files.append(file1)
        
        file2 = plot_uncertainty_by_month_season(df)
        if file2:
            generated_files.append(file2)
        
        file3 = plot_uncertainty_heatmap_hour_month(df)
        if file3:
            generated_files.append(file3)
        
        file4 = plot_uncertainty_timeseries(df)
        if file4:
            generated_files.append(file4)
        
        file5 = plot_uncertainty_distribution(df)
        if file5:
            generated_files.append(file5)
        
        file6 = plot_uncertainty_vs_sr(df)
        if file6:
            generated_files.append(file6)
        
        logger.info("\n" + "="*80)
        logger.info(f"✅ GENERACIÓN DE GRÁFICOS COMPLETADA")
        logger.info(f"Total de gráficos generados: {len(generated_files)}")
        logger.info(f"Ubicación: {paths.PROPAGACION_ERRORES_REF_CELL_DIR}")
        logger.info("="*80)
        
        return generated_files
        
    except Exception as e:
        logger.error(f"Error generando gráficos: {e}", exc_info=True)
        return generated_files


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    files = generate_all_uncertainty_plots()
    
    if files:
        print(f"\n✅ Se generaron {len(files)} gráficos:")
        for f in files:
            print(f"  - {f}")
    else:
        print("\n❌ No se generaron gráficos")

