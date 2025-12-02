"""
Análisis estadístico y de errores entre Solys2 (PSDA) y Lalcktur

DESCRIPCIÓN:
------------
Este script realiza un análisis comparativo entre los datos de radiación solar
de dos fuentes:
- Solys2: desde PSDA.meteo6857
- Lalcktur: desde lalcktur.cr1000x

El análisis incluye:
- Estadísticas descriptivas comparativas
- Métricas de error (MAE, RMSE, MBE, R², MAPE)
- Análisis de correlación
- Visualizaciones comparativas
- Reporte detallado en HTML y texto

USO:
----
Ejecutar el script desde la línea de comandos:
    python compare_solys2_lalcktur.py

El script generará:
- Gráficos en: datos/comparison_solys2_lalcktur/plots/
- Reporte HTML: datos/comparison_solys2_lalcktur/report.html
- Reporte texto: datos/comparison_solys2_lalcktur/report.txt
"""

# ============================================================================
# SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directorios - ruta relativa al directorio del proyecto
# Obtener el directorio del script y construir la ruta relativa
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Subir un nivel desde analysis/
BASE_DIR = os.path.join(PROJECT_ROOT, "datos")
SOLYS2_FILE = os.path.join(BASE_DIR, "solys2", "raw_solys2_data.csv")
LALCKTUR_FILE = os.path.join(BASE_DIR, "lalcktur", "raw_lalcktur_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_solys2_lalcktur")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Variables a analizar
VARIABLES = ['GHI', 'DHI', 'DNI']

# Configuración de filtro de nubosidad
FILTER_CLOUDY_DAYS = True  # Activar filtro de días nublados
MIN_DAILY_GHI_MEAN = 200.0  # Umbral mínimo de GHI promedio diario (W/m²) para considerar día despejado
MAX_DHI_GHI_RATIO = 0.8  # Máxima relación DHI/GHI para considerar día despejado (días muy nublados tienen > 0.8)


# ============================================================================
# SECCIÓN 2: FUNCIONES DE CÁLCULO DE MÉTRICAS
# ============================================================================

def calculate_metrics(y_true, y_pred, variable_name):
    """
    Calcula métricas de error entre valores reales y predichos.
    
    Args:
        y_true: Valores reales (Lalcktur)
        y_pred: Valores predichos (Solys2)
        variable_name: Nombre de la variable
        
    Returns:
        dict: Diccionario con todas las métricas calculadas
    """
    # Filtrar valores válidos (no NaN, no infinitos)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            'n_points': 0,
            'mae': np.nan,
            'rmse': np.nan,
            'mbe': np.nan,
            'r2': np.nan,
            'mape': np.nan,
            'correlation': np.nan,
            'bias_percent': np.nan
        }
    
    # Métricas básicas
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mbe = np.mean(y_pred_clean - y_true_clean)  # Mean Bias Error
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Correlación de Pearson
    correlation, _ = stats.pearsonr(y_true_clean, y_pred_clean)
    
    # MAPE (Mean Absolute Percentage Error) - solo para valores positivos
    positive_mask = y_true_clean > 0
    if positive_mask.sum() > 0:
        mape = np.mean(np.abs((y_true_clean[positive_mask] - y_pred_clean[positive_mask]) / y_true_clean[positive_mask])) * 100
    else:
        mape = np.nan
    
    # Bias porcentual
    mean_true = np.mean(y_true_clean)
    if mean_true != 0:
        bias_percent = (mbe / mean_true) * 100
    else:
        bias_percent = np.nan
    
    return {
        'n_points': len(y_true_clean),
        'mae': mae,
        'rmse': rmse,
        'mbe': mbe,
        'r2': r2,
        'mape': mape,
        'correlation': correlation,
        'bias_percent': bias_percent,
        'mean_true': np.mean(y_true_clean),
        'mean_pred': np.mean(y_pred_clean),
        'std_true': np.std(y_true_clean),
        'std_pred': np.std(y_pred_clean)
    }


def calculate_statistics(df, prefix=""):
    """
    Calcula estadísticas descriptivas para un DataFrame.
    
    Args:
        df: DataFrame con las variables
        prefix: Prefijo para los nombres de las columnas
        
    Returns:
        DataFrame: Estadísticas descriptivas
    """
    stats_dict = {}
    for var in VARIABLES:
        if var in df.columns:
            data = df[var].dropna()
            if len(data) > 0:
                stats_dict[f'{prefix}{var}'] = {
                    'count': len(data),
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'median': data.median(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75)
                }
    
    return pd.DataFrame(stats_dict).T


# ============================================================================
# SECCIÓN 3: FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def plot_scatter_comparison(df, variable, output_path):
    """
    Genera gráfico de dispersión comparando Solys2 vs Lalcktur.
    
    Args:
        df: DataFrame con datos combinados
        variable: Nombre de la variable a graficar
        output_path: Ruta donde guardar el gráfico
    """
    solys2_col = f'{variable}_solys2'
    lalcktur_col = f'{variable}_lalcktur'
    
    if solys2_col not in df.columns or lalcktur_col not in df.columns:
        print(f"⚠️  No se encontraron columnas para {variable}")
        return
    
    # Filtrar valores válidos
    mask = df[solys2_col].notna() & df[lalcktur_col].notna()
    x = df.loc[mask, solys2_col]
    y = df.loc[mask, lalcktur_col]
    
    if len(x) == 0:
        print(f"⚠️  No hay datos válidos para {variable}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Gráfico de dispersión
    ax.scatter(x, y, alpha=0.3, s=10, edgecolors='none')
    
    # Línea 1:1
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Línea 1:1')
    
    # Calcular métricas para el título
    metrics = calculate_metrics(y.values, x.values, variable)
    
    # Título y etiquetas
    ax.set_xlabel(f'{variable} - Solys2 (PSDA) [W/m²]', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{variable} - Lalcktur [W/m²]', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparación {variable}: Solys2 vs Lalcktur\n'
                 f'R² = {metrics["r2"]:.4f} | RMSE = {metrics["rmse"]:.2f} W/m² | '
                 f'MAE = {metrics["mae"]:.2f} W/m² | Corr = {metrics["correlation"]:.4f}',
                 fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico guardado: {output_path}")


def plot_time_series_comparison(df, variable, output_path, sample_size=10000):
    """
    Genera gráfico de series temporales comparando ambas fuentes.
    
    Args:
        df: DataFrame con datos combinados
        variable: Nombre de la variable a graficar
        output_path: Ruta donde guardar el gráfico
        sample_size: Tamaño de muestra para el gráfico (para no sobrecargar)
    """
    solys2_col = f'{variable}_solys2'
    lalcktur_col = f'{variable}_lalcktur'
    
    if solys2_col not in df.columns or lalcktur_col not in df.columns:
        print(f"⚠️  No se encontraron columnas para {variable}")
        return
    
    # Muestrear si hay muchos datos
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size).sort_index()
    else:
        df_plot = df.sort_index()
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Graficar ambas series
    ax.plot(df_plot.index, df_plot[solys2_col], label='Solys2 (PSDA)', alpha=0.7, linewidth=1)
    ax.plot(df_plot.index, df_plot[lalcktur_col], label='Lalcktur', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{variable} [W/m²]', fontsize=12, fontweight='bold')
    ax.set_title(f'Series Temporales Comparativas: {variable}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico guardado: {output_path}")


def plot_error_distribution(df, variable, output_path):
    """
    Genera histograma de distribución de errores.
    
    Args:
        df: DataFrame con datos combinados
        variable: Nombre de la variable a graficar
        output_path: Ruta donde guardar el gráfico
    """
    solys2_col = f'{variable}_solys2'
    lalcktur_col = f'{variable}_lalcktur'
    
    if solys2_col not in df.columns or lalcktur_col not in df.columns:
        print(f"⚠️  No se encontraron columnas para {variable}")
        return
    
    # Calcular errores
    mask = df[solys2_col].notna() & df[lalcktur_col].notna()
    errors = df.loc[mask, solys2_col] - df.loc[mask, lalcktur_col]
    
    if len(errors) == 0:
        print(f"⚠️  No hay datos válidos para {variable}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograma
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Media: {errors.mean():.2f} W/m²')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(f'Error ({variable}) = Solys2 - Lalcktur [W/m²]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribución de Errores: {variable}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico guardado: {output_path}")


def plot_comprehensive_comparison(df, output_path, sample_size=50000):
    """
    Genera un gráfico comprensivo comparando todas las variables entre Solys2 y Lalcktur.
    
    Args:
        df: DataFrame con datos combinados
        output_path: Ruta donde guardar el gráfico
        sample_size: Tamaño de muestra para el gráfico (para no sobrecargar)
    """
    # Muestrear si hay muchos datos
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size).sort_index()
    else:
        df_plot = df.sort_index()
    
    # Crear figura con subplots para cada variable
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Comparación Completa: Solys2 (PSDA) vs Lalcktur', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, var in enumerate(VARIABLES):
        solys2_col = f'{var}_solys2'
        lalcktur_col = f'{var}_lalcktur'
        
        if solys2_col not in df_plot.columns or lalcktur_col not in df_plot.columns:
            continue
        
        # Filtrar valores válidos
        mask = df_plot[solys2_col].notna() & df_plot[lalcktur_col].notna()
        df_var = df_plot.loc[mask]
        
        if len(df_var) == 0:
            continue
        
        # Subplot izquierdo: Series temporales superpuestas
        ax1 = axes[idx, 0]
        ax1.plot(df_var.index, df_var[solys2_col], 
                label='Solys2 (PSDA)', alpha=0.7, linewidth=1, color='#2ecc71')
        ax1.plot(df_var.index, df_var[lalcktur_col], 
                label='Lalcktur', alpha=0.7, linewidth=1, color='#e74c3c')
        ax1.set_xlabel('Fecha', fontsize=11, fontweight='bold')
        ax1.set_ylabel(f'{var} [W/m²]', fontsize=11, fontweight='bold')
        ax1.set_title(f'{var} - Series Temporales', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Subplot derecho: Gráfico de dispersión con línea 1:1
        ax2 = axes[idx, 1]
        
        # Muestrear para el scatter si hay muchos puntos
        if len(df_var) > 10000:
            df_scatter = df_var.sample(n=10000)
        else:
            df_scatter = df_var
        
        ax2.scatter(df_scatter[solys2_col], df_scatter[lalcktur_col], 
                   alpha=0.3, s=10, edgecolors='none', color='#3498db')
        
        # Línea 1:1
        min_val = min(df_scatter[solys2_col].min(), df_scatter[lalcktur_col].min())
        max_val = max(df_scatter[solys2_col].max(), df_scatter[lalcktur_col].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Línea 1:1')
        
        # Calcular métricas para el título
        metrics = calculate_metrics(df_scatter[lalcktur_col].values, 
                                   df_scatter[solys2_col].values, var)
        
        ax2.set_xlabel(f'{var} - Solys2 (PSDA) [W/m²]', fontsize=11, fontweight='bold')
        ax2.set_ylabel(f'{var} - Lalcktur [W/m²]', fontsize=11, fontweight='bold')
        ax2.set_title(f'{var} - Dispersión\n'
                     f'R² = {metrics["r2"]:.4f} | RMSE = {metrics["rmse"]:.2f} W/m² | '
                     f'MAE = {metrics["mae"]:.2f} W/m²',
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico comprensivo guardado: {output_path}")


def plot_difference_analysis(df, output_path, sample_size=50000):
    """
    Genera un gráfico mostrando las diferencias entre Solys2 y Lalcktur.
    
    Args:
        df: DataFrame con datos combinados
        output_path: Ruta donde guardar el gráfico
        sample_size: Tamaño de muestra para el gráfico
    """
    # Muestrear si hay muchos datos
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size).sort_index()
    else:
        df_plot = df.sort_index()
    
    # Crear figura con subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Análisis de Diferencias: Solys2 (PSDA) - Lalcktur', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, var in enumerate(VARIABLES):
        solys2_col = f'{var}_solys2'
        lalcktur_col = f'{var}_lalcktur'
        
        if solys2_col not in df_plot.columns or lalcktur_col not in df_plot.columns:
            continue
        
        # Filtrar valores válidos
        mask = df_plot[solys2_col].notna() & df_plot[lalcktur_col].notna()
        df_var = df_plot.loc[mask]
        
        if len(df_var) == 0:
            continue
        
        # Calcular diferencias
        differences = df_var[solys2_col] - df_var[lalcktur_col]
        
        # Subplot izquierdo: Serie temporal de diferencias
        ax1 = axes[idx, 0]
        ax1.plot(df_var.index, differences, alpha=0.7, linewidth=1, color='#9b59b6')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axhline(y=differences.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: {differences.mean():.2f} W/m²')
        ax1.set_xlabel('Fecha', fontsize=11, fontweight='bold')
        ax1.set_ylabel(f'Diferencia {var} [W/m²]', fontsize=11, fontweight='bold')
        ax1.set_title(f'{var} - Diferencia Temporal (Solys2 - Lalcktur)', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Subplot derecho: Histograma de diferencias
        ax2 = axes[idx, 1]
        ax2.hist(differences, bins=50, alpha=0.7, edgecolor='black', color='#3498db')
        ax2.axvline(differences.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Media: {differences.mean():.2f} W/m²')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_xlabel(f'Diferencia {var} (Solys2 - Lalcktur) [W/m²]', 
                      fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
        ax2.set_title(f'{var} - Distribución de Diferencias\n'
                     f'Std: {differences.std():.2f} W/m² | '
                     f'Min: {differences.min():.2f} W/m² | '
                     f'Max: {differences.max():.2f} W/m²',
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de diferencias guardado: {output_path}")


# ============================================================================
# SECCIÓN 3.5: FUNCIÓN PARA FILTRAR DÍAS NUBLADOS
# ============================================================================

def filter_cloudy_days(df, ghi_col='GHI', dhi_col='DHI', min_daily_ghi=200.0, max_dhi_ghi_ratio=0.8):
    """
    Filtra días nublados basándose en el promedio diario de GHI y la relación DHI/GHI.
    
    Args:
        df: DataFrame con índice de fecha/hora
        ghi_col: Nombre de la columna GHI
        dhi_col: Nombre de la columna DHI
        min_daily_ghi: GHI promedio diario mínimo (W/m²) para considerar día despejado
        max_dhi_ghi_ratio: Máxima relación DHI/GHI para considerar día despejado
        
    Returns:
        DataFrame: DataFrame filtrado sin días nublados
    """
    if ghi_col not in df.columns or dhi_col not in df.columns:
        print("⚠️  No se encontraron columnas GHI o DHI para filtrar días nublados")
        return df
    
    # Crear columna de fecha (sin hora)
    df_copy = df.copy()
    df_copy['date'] = df_copy.index.date
    
    # Calcular estadísticas diarias
    daily_stats = df_copy.groupby('date').agg({
        ghi_col: ['mean', 'count'],
        dhi_col: 'mean'
    }).reset_index()
    
    # Aplanar nombres de columnas
    daily_stats.columns = ['date', 'ghi_mean', 'ghi_count', 'dhi_mean']
    
    # Calcular relación DHI/GHI (solo para días con GHI > 0)
    daily_stats['dhi_ghi_ratio'] = np.where(
        daily_stats['ghi_mean'] > 0,
        daily_stats['dhi_mean'] / daily_stats['ghi_mean'],
        np.nan
    )
    
    # Identificar días despejados:
    # - GHI promedio diario >= umbral mínimo
    # - Relación DHI/GHI <= umbral máximo
    clear_days = daily_stats[
        (daily_stats['ghi_mean'] >= min_daily_ghi) & 
        (daily_stats['dhi_ghi_ratio'] <= max_dhi_ghi_ratio)
    ]['date'].tolist()
    
    # Filtrar DataFrame original para mantener solo días despejados
    df_filtered = df_copy[df_copy['date'].isin(clear_days)].copy()
    df_filtered = df_filtered.drop(columns=['date'])
    
    n_original = len(df)
    n_filtered = len(df_filtered)
    n_removed = n_original - n_filtered
    pct_removed = (n_removed / n_original * 100) if n_original > 0 else 0
    
    print(f"   📊 Días analizados: {len(daily_stats)}")
    print(f"   ☀️  Días despejados: {len(clear_days)}")
    print(f"   ☁️  Días nublados eliminados: {len(daily_stats) - len(clear_days)}")
    print(f"   📉 Registros eliminados: {n_removed:,} ({pct_removed:.1f}%)")
    print(f"   ✅ Registros restantes: {n_filtered:,}")
    
    return df_filtered


# ============================================================================
# SECCIÓN 4: FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================================================

def main():
    """
    Función principal que ejecuta todo el análisis.
    """
    print("="*80)
    print("ANÁLISIS COMPARATIVO: SOLYS2 (PSDA) vs LALCKTUR")
    print("="*80)
    print()
    
    # 1. Cargar datos
    print("📂 Cargando datos...")
    try:
        df_solys2 = pd.read_csv(SOLYS2_FILE, index_col='fecha hora', parse_dates=True)
        df_lalcktur = pd.read_csv(LALCKTUR_FILE, index_col='fecha hora', parse_dates=True)
        print(f"✅ Solys2: {len(df_solys2)} registros")
        print(f"✅ Lalcktur: {len(df_lalcktur)} registros")
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return
    
    # Filtrar por rango de fechas: desde 08/11/2025 hasta la actualidad
    print("\n📅 Filtrando datos por rango de fechas...")
    start_date = pd.to_datetime('2025-11-08', utc=True)
    end_date = pd.Timestamp.now(tz='UTC')
    
    print(f"   Fecha inicio: {start_date}")
    print(f"   Fecha fin: {end_date}")
    
    # Filtrar ambos DataFrames
    df_solys2 = df_solys2[(df_solys2.index >= start_date) & (df_solys2.index <= end_date)]
    df_lalcktur = df_lalcktur[(df_lalcktur.index >= start_date) & (df_lalcktur.index <= end_date)]
    
    print(f"✅ Solys2 después del filtro: {len(df_solys2)} registros")
    print(f"✅ Lalcktur después del filtro: {len(df_lalcktur)} registros")
    
    # 2. Preparar datos para comparación
    print("\n🔄 Preparando datos para comparación...")
    
    # Renombrar columnas para evitar conflictos
    df_solys2_renamed = df_solys2[VARIABLES].copy()
    df_solys2_renamed.columns = [f'{col}_solys2' for col in df_solys2_renamed.columns]
    
    df_lalcktur_renamed = df_lalcktur[VARIABLES].copy()
    df_lalcktur_renamed.columns = [f'{col}_lalcktur' for col in df_lalcktur_renamed.columns]
    
    # Hacer merge por índice (fecha hora)
    df_combined = pd.merge(
        df_solys2_renamed,
        df_lalcktur_renamed,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    print(f"✅ Registros coincidentes: {len(df_combined)}")
    print(f"📅 Rango de fechas: {df_combined.index.min()} a {df_combined.index.max()}")
    
    # Filtrar días nublados si está activado
    if FILTER_CLOUDY_DAYS:
        print("\n☁️  Filtrando días nublados...")
        print(f"   Criterios:")
        print(f"   - GHI promedio diario >= {MIN_DAILY_GHI_MEAN} W/m²")
        print(f"   - Relación DHI/GHI <= {MAX_DHI_GHI_RATIO}")
        
        # Usar GHI de Lalcktur como referencia (o promedio de ambas)
        # Crear columna GHI combinada para el filtro
        ghi_solys2_col = 'GHI_solys2'
        ghi_lalcktur_col = 'GHI_lalcktur'
        dhi_solys2_col = 'DHI_solys2'
        dhi_lalcktur_col = 'DHI_lalcktur'
        
        # Usar promedio de ambas fuentes para el filtro
        if ghi_solys2_col in df_combined.columns and ghi_lalcktur_col in df_combined.columns:
            df_combined['GHI_mean'] = (df_combined[ghi_solys2_col] + df_combined[ghi_lalcktur_col]) / 2
        elif ghi_lalcktur_col in df_combined.columns:
            df_combined['GHI_mean'] = df_combined[ghi_lalcktur_col]
        else:
            df_combined['GHI_mean'] = df_combined[ghi_solys2_col]
        
        if dhi_solys2_col in df_combined.columns and dhi_lalcktur_col in df_combined.columns:
            df_combined['DHI_mean'] = (df_combined[dhi_solys2_col] + df_combined[dhi_lalcktur_col]) / 2
        elif dhi_lalcktur_col in df_combined.columns:
            df_combined['DHI_mean'] = df_combined[dhi_lalcktur_col]
        else:
            df_combined['DHI_mean'] = df_combined[dhi_solys2_col]
        
        # Aplicar filtro
        df_combined = filter_cloudy_days(
            df_combined,
            ghi_col='GHI_mean',
            dhi_col='DHI_mean',
            min_daily_ghi=MIN_DAILY_GHI_MEAN,
            max_dhi_ghi_ratio=MAX_DHI_GHI_RATIO
        )
        
        # Eliminar columnas temporales
        df_combined = df_combined.drop(columns=['GHI_mean', 'DHI_mean'], errors='ignore')
        
        print(f"✅ Después del filtro de nubosidad: {len(df_combined)} registros")
        print(f"📅 Rango de fechas filtrado: {df_combined.index.min()} a {df_combined.index.max()}")
    
    # 3. Calcular estadísticas descriptivas
    print("\n📊 Calculando estadísticas descriptivas...")
    stats_solys2 = calculate_statistics(df_solys2, prefix="")
    stats_lalcktur = calculate_statistics(df_lalcktur, prefix="")
    
    # 4. Calcular métricas de error para cada variable
    print("\n📈 Calculando métricas de error...")
    all_metrics = {}
    
    for var in VARIABLES:
        solys2_col = f'{var}_solys2'
        lalcktur_col = f'{var}_lalcktur'
        
        if solys2_col in df_combined.columns and lalcktur_col in df_combined.columns:
            metrics = calculate_metrics(
                df_combined[lalcktur_col].values,
                df_combined[solys2_col].values,
                var
            )
            all_metrics[var] = metrics
            print(f"  ✅ {var}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f} W/m²")
    
    # 5. Generar gráficos
    print("\n📊 Generando gráficos...")
    for var in VARIABLES:
        plot_scatter_comparison(df_combined, var, 
                               os.path.join(PLOTS_DIR, f'scatter_{var}.png'))
        plot_time_series_comparison(df_combined, var,
                                   os.path.join(PLOTS_DIR, f'timeseries_{var}.png'))
        plot_error_distribution(df_combined, var,
                               os.path.join(PLOTS_DIR, f'error_dist_{var}.png'))
    
    # Gráficos comprensivos adicionales
    print("\n📊 Generando gráficos comprensivos...")
    plot_comprehensive_comparison(df_combined,
                                 os.path.join(PLOTS_DIR, 'comprehensive_comparison.png'))
    plot_difference_analysis(df_combined,
                            os.path.join(PLOTS_DIR, 'difference_analysis.png'))
    
    # 6. Generar reporte
    print("\n📝 Generando reporte...")
    generate_report(df_combined, stats_solys2, stats_lalcktur, all_metrics)
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"📁 Resultados guardados en: {OUTPUT_DIR}")
    print(f"📊 Gráficos: {PLOTS_DIR}")
    print(f"📄 Reporte HTML: {os.path.join(OUTPUT_DIR, 'report.html')}")
    print(f"📄 Reporte texto: {os.path.join(OUTPUT_DIR, 'report.txt')}")


def generate_report(df_combined, stats_solys2, stats_lalcktur, all_metrics):
    """
    Genera reporte detallado en HTML y texto.
    
    Args:
        df_combined: DataFrame con datos combinados
        stats_solys2: Estadísticas de Solys2
        stats_lalcktur: Estadísticas de Lalcktur
        all_metrics: Diccionario con métricas de error
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reporte HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Análisis Comparativo: Solys2 vs Lalcktur</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
            .good {{ color: #27ae60; font-weight: bold; }}
            .warning {{ color: #f39c12; font-weight: bold; }}
            .bad {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Análisis Comparativo: Solys2 (PSDA) vs Lalcktur</h1>
        <p><strong>Fecha de análisis:</strong> {timestamp}</p>
        <p><strong>Total de registros comparados:</strong> {len(df_combined):,}</p>
        <p><strong>Rango de fechas:</strong> {df_combined.index.min()} a {df_combined.index.max()}</p>
        
        <h2>1. Resumen de Métricas de Error</h2>
        <table>
            <tr>
                <th>Variable</th>
                <th>N puntos</th>
                <th>R²</th>
                <th>RMSE (W/m²)</th>
                <th>MAE (W/m²)</th>
                <th>MBE (W/m²)</th>
                <th>Bias (%)</th>
                <th>MAPE (%)</th>
                <th>Correlación</th>
            </tr>
    """
    
    for var in VARIABLES:
        if var in all_metrics:
            m = all_metrics[var]
            mape_str = f"{m['mape']:.2f}" if not np.isnan(m['mape']) else 'N/A'
            html_content += f"""
            <tr>
                <td><strong>{var}</strong></td>
                <td>{m['n_points']:,}</td>
                <td>{m['r2']:.4f}</td>
                <td>{m['rmse']:.2f}</td>
                <td>{m['mae']:.2f}</td>
                <td>{m['mbe']:.2f}</td>
                <td>{m['bias_percent']:.2f}</td>
                <td>{mape_str}</td>
                <td>{m['correlation']:.4f}</td>
            </tr>
            """
    
    html_content += """
        </table>
        
        <h2>2. Estadísticas Descriptivas - Solys2 (PSDA)</h2>
        <table>
            <tr>
                <th>Variable</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>Median</th>
                <th>Q25</th>
                <th>Q75</th>
            </tr>
    """
    
    for var in VARIABLES:
        if var in stats_solys2.index:
            s = stats_solys2.loc[var]
            html_content += f"""
            <tr>
                <td><strong>{var}</strong></td>
                <td>{int(s['count']):,}</td>
                <td>{s['mean']:.2f}</td>
                <td>{s['std']:.2f}</td>
                <td>{s['min']:.2f}</td>
                <td>{s['max']:.2f}</td>
                <td>{s['median']:.2f}</td>
                <td>{s['q25']:.2f}</td>
                <td>{s['q75']:.2f}</td>
            </tr>
            """
    
    html_content += """
        </table>
        
        <h2>3. Estadísticas Descriptivas - Lalcktur</h2>
        <table>
            <tr>
                <th>Variable</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>Median</th>
                <th>Q25</th>
                <th>Q75</th>
            </tr>
    """
    
    for var in VARIABLES:
        if var in stats_lalcktur.index:
            s = stats_lalcktur.loc[var]
            html_content += f"""
            <tr>
                <td><strong>{var}</strong></td>
                <td>{int(s['count']):,}</td>
                <td>{s['mean']:.2f}</td>
                <td>{s['std']:.2f}</td>
                <td>{s['min']:.2f}</td>
                <td>{s['max']:.2f}</td>
                <td>{s['median']:.2f}</td>
                <td>{s['q25']:.2f}</td>
                <td>{s['q75']:.2f}</td>
            </tr>
            """
    
    html_content += """
        </table>
        
        <h2>4. Gráficos Generados</h2>
        <ul>
    """
    
    for var in VARIABLES:
        html_content += f"""
            <li><strong>{var}:</strong>
                <ul>
                    <li><a href="plots/scatter_{var}.png">Gráfico de dispersión</a></li>
                    <li><a href="plots/timeseries_{var}.png">Series temporales</a></li>
                    <li><a href="plots/error_dist_{var}.png">Distribución de errores</a></li>
                </ul>
            </li>
        """
    
    html_content += """
        </ul>
        
        <h2>5. Interpretación de Métricas</h2>
        <div class="metric">
            <p><strong>R² (Coeficiente de determinación):</strong> Mide qué tan bien los datos se ajustan a la línea 1:1. 
            Valores cercanos a 1 indican mejor ajuste.</p>
        </div>
        <div class="metric">
            <p><strong>RMSE (Root Mean Square Error):</strong> Error cuadrático medio. Mide la magnitud promedio del error. 
            Valores más bajos indican mejor precisión.</p>
        </div>
        <div class="metric">
            <p><strong>MAE (Mean Absolute Error):</strong> Error absoluto medio. Similar al RMSE pero menos sensible a valores extremos.</p>
        </div>
        <div class="metric">
            <p><strong>MBE (Mean Bias Error):</strong> Sesgo promedio. Valores positivos indican que Solys2 sobreestima, 
            valores negativos indican subestimación.</p>
        </div>
        <div class="metric">
            <p><strong>Bias (%):</strong> Sesgo porcentual. Indica el porcentaje de sesgo relativo al valor promedio.</p>
        </div>
        <div class="metric">
            <p><strong>MAPE (Mean Absolute Percentage Error):</strong> Error porcentual absoluto medio. 
            Útil para comparar errores relativos.</p>
        </div>
        <div class="metric">
            <p><strong>Correlación de Pearson:</strong> Mide la relación lineal entre ambas variables. 
            Valores cercanos a ±1 indican fuerte correlación.</p>
        </div>
    </body>
    </html>
    """
    
    # Guardar reporte HTML
    with open(os.path.join(OUTPUT_DIR, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Reporte texto
    txt_content = f"""
{'='*80}
ANÁLISIS COMPARATIVO: SOLYS2 (PSDA) vs LALCKTUR
{'='*80}

Fecha de análisis: {timestamp}
Total de registros comparados: {len(df_combined):,}
Rango de fechas: {df_combined.index.min()} a {df_combined.index.max()}

{'='*80}
1. RESUMEN DE MÉTRICAS DE ERROR
{'='*80}

"""
    
    for var in VARIABLES:
        if var in all_metrics:
            m = all_metrics[var]
            mape_str = f"{m['mape']:.2f}" if not np.isnan(m['mape']) else 'N/A'
            txt_content += f"""
{var}:
  - Número de puntos: {m['n_points']:,}
  - R²: {m['r2']:.4f}
  - RMSE: {m['rmse']:.2f} W/m²
  - MAE: {m['mae']:.2f} W/m²
  - MBE: {m['mbe']:.2f} W/m²
  - Bias (%): {m['bias_percent']:.2f}%
  - MAPE: {mape_str}%
  - Correlación: {m['correlation']:.4f}
  - Media Solys2: {m['mean_pred']:.2f} W/m²
  - Media Lalcktur: {m['mean_true']:.2f} W/m²
  - Std Solys2: {m['std_pred']:.2f} W/m²
  - Std Lalcktur: {m['std_true']:.2f} W/m²

"""
    
    txt_content += f"""
{'='*80}
2. ESTADÍSTICAS DESCRIPTIVAS - SOLYS2 (PSDA)
{'='*80}

{stats_solys2.to_string()}

{'='*80}
3. ESTADÍSTICAS DESCRIPTIVAS - LALCKTUR
{'='*80}

{stats_lalcktur.to_string()}

{'='*80}
4. INTERPRETACIÓN DE MÉTRICAS
{'='*80}

R² (Coeficiente de determinación): Mide qué tan bien los datos se ajustan a la línea 1:1.
  Valores cercanos a 1 indican mejor ajuste.

RMSE (Root Mean Square Error): Error cuadrático medio. Mide la magnitud promedio del error.
  Valores más bajos indican mejor precisión.

MAE (Mean Absolute Error): Error absoluto medio. Similar al RMSE pero menos sensible a valores extremos.

MBE (Mean Bias Error): Sesgo promedio. Valores positivos indican que Solys2 sobreestima,
  valores negativos indican subestimación.

Bias (%): Sesgo porcentual. Indica el porcentaje de sesgo relativo al valor promedio.

MAPE (Mean Absolute Percentage Error): Error porcentual absoluto medio.
  Útil para comparar errores relativos.

Correlación de Pearson: Mide la relación lineal entre ambas variables.
  Valores cercanos a ±1 indican fuerte correlación.

{'='*80}
"""
    
    # Guardar reporte texto
    with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(txt_content)
    
    print("✅ Reportes generados exitosamente")


if __name__ == "__main__":
    main()

