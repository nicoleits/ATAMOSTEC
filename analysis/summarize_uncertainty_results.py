"""
Script para generar un resumen visual de los resultados de incertidumbre de SR.

Este script lee los archivos CSV generados por el análisis de propagación de incertidumbre
y presenta un resumen estructurado similar al análisis manual.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime
import config.paths as paths


def format_number(value, decimals=2):
    """Formatea un número con separadores de miles."""
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_percentage(value, decimals=2):
    """Formatea un porcentaje."""
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def print_section(title, char="=", width=80):
    """Imprime una sección con título."""
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def print_subsection(title, char="-", width=80):
    """Imprime una subsección."""
    print(f"\n{title}")
    print(char * width)


def analyze_daily_results():
    """Analiza los resultados diarios."""
    daily_file = paths.SR_DAILY_ABS_WITH_U_FILE
    
    if not Path(daily_file).exists():
        print(f"⚠️  Archivo no encontrado: {daily_file}")
        return None
    
    df_daily = pd.read_csv(daily_file, parse_dates=['timestamp'])
    
    print_subsection("📊 RESUMEN DE DATOS DIARIOS")
    print(f"Total de días: {len(df_daily):,}")
    print(f"Rango de fechas: {df_daily['timestamp'].min().strftime('%Y-%m-%d')} a {df_daily['timestamp'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nIncertidumbre diaria (k=2):")
    print(f"  Mínima: {format_percentage(df_daily['U_rel_k2'].min(), 3)}")
    print(f"  Máxima: {format_percentage(df_daily['U_rel_k2'].max(), 3)}")
    print(f"  Promedio: {format_percentage(df_daily['U_rel_k2'].mean(), 3)}")
    print(f"  Mediana: {format_percentage(df_daily['U_rel_k2'].median(), 3)}")
    print(f"  Desviación estándar: {format_percentage(df_daily['U_rel_k2'].std(), 3)}")
    
    print(f"\nSR diario (Q25):")
    print(f"  Mínimo: {format_percentage(df_daily['SR_agg'].min(), 2)}")
    print(f"  Máximo: {format_percentage(df_daily['SR_agg'].max(), 2)}")
    print(f"  Promedio: {format_percentage(df_daily['SR_agg'].mean(), 2)}")
    print(f"  Mediana: {format_percentage(df_daily['SR_agg'].median(), 2)}")
    
    # Último día
    last_day = df_daily.iloc[-1]
    print(f"\nÚltimo día ({last_day['timestamp'].strftime('%Y-%m-%d')}):")
    print(f"  SR: {format_percentage(last_day['SR_agg'], 2)}")
    print(f"  Incertidumbre: {format_percentage(last_day['U_rel_k2'], 3)}")
    print(f"  Intervalo 95%: [{format_percentage(last_day['CI95_lo'], 2)}, {format_percentage(last_day['CI95_hi'], 2)}]")
    print(f"  Minutos: {int(last_day['n_minutes']):,}")
    
    return df_daily


def analyze_monthly_uncertainty():
    """Analiza la incertidumbre mensual."""
    monthly_file = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'sr_uncertainty_by_month.csv'
    
    if not monthly_file.exists():
        print(f"⚠️  Archivo no encontrado: {monthly_file}")
        return None
    
    df_monthly = pd.read_csv(monthly_file, parse_dates=['periodo'])
    
    print_subsection("📅 INCERTIDUMBRE POR PERÍODO MENSUAL")
    print(f"Total de meses: {len(df_monthly)}")
    print(f"Rango: {df_monthly['periodo'].min().strftime('%Y-%m')} a {df_monthly['periodo'].max().strftime('%Y-%m')}")
    
    print(f"\nIncertidumbre mensual (k=2):")
    min_idx = df_monthly['U_k2_rel_mean'].idxmin()
    max_idx = df_monthly['U_k2_rel_mean'].idxmax()
    
    print(f"  Mínima: {format_percentage(df_monthly.loc[min_idx, 'U_k2_rel_mean'], 3)} "
          f"({df_monthly.loc[min_idx, 'periodo'].strftime('%Y-%m')})")
    print(f"  Máxima: {format_percentage(df_monthly.loc[max_idx, 'U_k2_rel_mean'], 3)} "
          f"({df_monthly.loc[max_idx, 'periodo'].strftime('%Y-%m')})")
    print(f"  Promedio: {format_percentage(df_monthly['U_k2_rel_mean'].mean(), 3)}")
    print(f"  Mediana: {format_percentage(df_monthly['U_k2_rel_mean'].median(), 3)}")
    
    print(f"\nSR promedio mensual:")
    print(f"  Mínimo: {format_percentage(df_monthly['sr_mean'].min(), 2)}")
    print(f"  Máximo: {format_percentage(df_monthly['sr_mean'].max(), 2)}")
    print(f"  Promedio: {format_percentage(df_monthly['sr_mean'].mean(), 2)}")
    
    last_month = df_monthly.iloc[-1]
    print(f"\nÚltimo mes ({last_month['periodo'].strftime('%Y-%m')}):")
    print(f"  SR promedio: {format_percentage(last_month['sr_mean'], 2)}")
    print(f"  Incertidumbre: {format_percentage(last_month['U_k2_rel_mean'], 3)} ± {format_percentage(last_month['U_k2_rel_std'], 3)}")
    print(f"  Minutos: {int(last_month['n_minutes']):,}")
    
    # Tabla de todos los meses
    print(f"\n📋 Tabla completa por mes:")
    print(f"{'Mes':<12} {'U_k2 (%)':<12} {'U_std (%)':<12} {'SR (%)':<10} {'Minutos':<12}")
    print("-" * 70)
    for _, row in df_monthly.iterrows():
        month_str = row['periodo'].strftime('%Y-%m')
        print(f"{month_str:<12} {format_percentage(row['U_k2_rel_mean'], 3):<12} "
              f"{format_percentage(row['U_k2_rel_std'], 3):<12} "
              f"{format_percentage(row['sr_mean'], 2):<10} "
              f"{int(row['n_minutes']):>11,}")
    
    return df_monthly


def analyze_sr_range_uncertainty():
    """Analiza la incertidumbre por rangos de SR."""
    sr_range_file = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'sr_uncertainty_by_sr_range.csv'
    
    if not sr_range_file.exists():
        print(f"⚠️  Archivo no encontrado: {sr_range_file}")
        return None
    
    df_sr_range = pd.read_csv(sr_range_file)
    
    print_subsection("📊 INCERTIDUMBRE POR RANGOS DE SR")
    print("Muestra cómo varía la incertidumbre con el nivel de soiling.\n")
    
    print(f"{'Rango SR':<20} {'U_k2 (%)':<12} {'U_std (%)':<12} {'Minutos':<12} {'% Total':<10}")
    print("-" * 75)
    
    total_minutes = df_sr_range['n_minutes'].sum()
    
    for _, row in df_sr_range.iterrows():
        sr_range = row['sr_range']
        pct_total = (row['n_minutes'] / total_minutes) * 100
        print(f"{sr_range:<20} {format_percentage(row['U_k2_rel_mean'], 3):<12} "
              f"{format_percentage(row['U_k2_rel_std'], 3):<12} "
              f"{int(row['n_minutes']):>11,} "
              f"{format_percentage(pct_total, 2):<10}")
    
    print(f"\nTotal de minutos: {total_minutes:,}")
    
    # Identificar rangos con menor y mayor incertidumbre
    min_idx = df_sr_range['U_k2_rel_mean'].idxmin()
    max_idx = df_sr_range['U_k2_rel_mean'].idxmax()
    
    print(f"\n🔍 Observaciones:")
    print(f"  Rango con MENOR incertidumbre: {df_sr_range.loc[min_idx, 'sr_range']} "
          f"({format_percentage(df_sr_range.loc[min_idx, 'U_k2_rel_mean'], 3)})")
    print(f"  Rango con MAYOR incertidumbre: {df_sr_range.loc[max_idx, 'sr_range']} "
          f"({format_percentage(df_sr_range.loc[max_idx, 'U_k2_rel_mean'], 3)})")
    
    return df_sr_range


def analyze_summary_file():
    """Lee y muestra información del archivo de resumen."""
    summary_file = paths.SR_UNCERTAINTY_SUMMARY_FILE
    
    if not Path(summary_file).exists():
        print(f"⚠️  Archivo no encontrado: {summary_file}")
        return
    
    print_subsection("📄 RESUMEN DEL ARCHIVO DE ANÁLISIS")
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extraer información clave
    key_info = {}
    current_section = None
    
    for line in lines:
        line = line.strip()
        if 'Fecha de análisis:' in line:
            key_info['fecha'] = line.split(':', 1)[1].strip()
        elif 'Total de minutos válidos:' in line:
            key_info['minutos_validos'] = line.split(':', 1)[1].strip()
        elif 'Incertidumbre de campaña GLOBAL (k=2):' in line:
            key_info['U_global'] = line.split('=')[1].strip()
        elif 'SR promedio:' in line:
            key_info['sr_promedio'] = line.split(':', 1)[1].strip()
    
    if key_info:
        print("Información clave del análisis:")
        for key, value in key_info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n📁 Archivo completo disponible en: {summary_file}")


def analyze_minute_data():
    """Analiza los datos minutales (muestra estadísticas básicas)."""
    minute_file = paths.SR_MINUTE_WITH_UNCERTAINTY_FILE
    
    if not Path(minute_file).exists():
        print(f"⚠️  Archivo no encontrado: {minute_file}")
        return None
    
    print_subsection("⏱️  DATOS MINUTALES (Estadísticas básicas)")
    
    # Leer solo las primeras y últimas líneas para obtener rango temporal
    with open(minute_file, 'r') as f:
        first_line = f.readline().strip()
        # Contar líneas aproximadas
        f.seek(0, 2)  # Ir al final
        file_size = f.tell()
        # Leer últimas líneas
        f.seek(max(0, file_size - 5000))  # Últimos ~5KB
        last_lines = f.readlines()
    
    # Leer DataFrame completo para estadísticas (puede ser lento para archivos grandes)
    # Usar muestreo aleatorio para archivos grandes
    try:
        # Leer una muestra aleatoria para estadísticas
        df_sample = pd.read_csv(minute_file, parse_dates=['timestamp'], skiprows=lambda x: x > 0 and np.random.rand() > 0.01)
        df_minute = df_sample
        print(f"⚠️  Archivo muy grande. Analizando muestra aleatoria de {len(df_sample):,} registros (~1%).")
    except Exception as e:
        # Si falla, leer solo el header y primeras líneas
        try:
            df_minute = pd.read_csv(minute_file, parse_dates=['timestamp'], nrows=1000)
            print(f"⚠️  Analizando muestra de 1,000 registros.")
        except:
            df_minute = pd.read_csv(minute_file, nrows=1)
            print(f"⚠️  No se pudo leer el archivo. Mostrando solo información del archivo.")
    
    # Obtener información del archivo
    file_path = Path(minute_file)
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    # Contar líneas aproximadas
    with open(minute_file, 'r') as f:
        line_count = sum(1 for _ in f) - 1  # -1 para el header
    
    print(f"Tamaño del archivo: {file_size_mb:.2f} MB")
    print(f"Total de registros (aprox.): {line_count:,}")
    
    if len(df_minute) > 1:
        print(f"Rango temporal (muestra): {df_minute['timestamp'].min()} a {df_minute['timestamp'].max()}")
        
        if 'SR' in df_minute.columns:
            print(f"\nSR (muestra):")
            print(f"  Promedio: {format_percentage(df_minute['SR'].mean(), 2)}")
            print(f"  Mínimo: {format_percentage(df_minute['SR'].min(), 2)}")
            print(f"  Máximo: {format_percentage(df_minute['SR'].max(), 2)}")
        
        if 'U_SR_k2_rel' in df_minute.columns:
            # Filtrar valores extremos para estadísticas
            u_valid = df_minute['U_SR_k2_rel'].dropna()
            u_valid = u_valid[(u_valid >= 0) & (u_valid <= 0.5)]  # Filtrar valores razonables (0-50%)
            
            print(f"\nIncertidumbre (muestra, valores filtrados):")
            if len(u_valid) > 0:
                print(f"  Promedio: {format_percentage(u_valid.mean() * 100, 3)}")
                print(f"  Mínimo: {format_percentage(u_valid.min() * 100, 3)}")
                print(f"  Máximo: {format_percentage(u_valid.max() * 100, 3)}")
                print(f"  Mediana: {format_percentage(u_valid.median() * 100, 3)}")
                print(f"  Valores válidos: {len(u_valid):,} de {len(df_minute):,} ({len(u_valid)/len(df_minute)*100:.1f}%)")
            else:
                print(f"  No hay valores válidos en la muestra")
    
    print(f"\n📁 Archivo completo: {minute_file}")
    
    return df_minute


def generate_trend_analysis(df_daily, df_monthly):
    """Genera análisis de tendencias temporales."""
    if df_daily is None or df_monthly is None:
        return
    
    print_section("📈 ANÁLISIS DE TENDENCIAS TEMPORALES", "=", 80)
    
    # Tendencia de SR
    print_subsection("Tendencia del SR")
    df_daily['year_month'] = pd.to_datetime(df_daily['timestamp']).dt.to_period('M')
    monthly_sr = df_daily.groupby('year_month')['SR_agg'].agg(['mean', 'std', 'count'])
    
    print(f"SR promedio por mes (desde datos diarios):")
    print(f"{'Mes':<12} {'SR Promedio (%)':<15} {'Std (%)':<12} {'Días':<8}")
    print("-" * 50)
    for period, row in monthly_sr.iterrows():
        print(f"{str(period):<12} {format_percentage(row['mean'], 2):<15} "
              f"{format_percentage(row['std'], 2):<12} {int(row['count']):<8}")
    
    # Tendencia de incertidumbre
    print_subsection("\nTendencia de Incertidumbre")
    monthly_u = df_daily.groupby('year_month')['U_rel_k2'].agg(['mean', 'std', 'count'])
    
    print(f"Incertidumbre promedio por mes (desde datos diarios):")
    print(f"{'Mes':<12} {'U Promedio (%)':<15} {'Std (%)':<12} {'Días':<8}")
    print("-" * 50)
    for period, row in monthly_u.iterrows():
        print(f"{str(period):<12} {format_percentage(row['mean'], 3):<15} "
              f"{format_percentage(row['std'], 3):<12} {int(row['count']):<8}")
    
    # Correlación SR vs Incertidumbre
    print_subsection("\nCorrelación SR vs Incertidumbre")
    correlation = df_daily['SR_agg'].corr(df_daily['U_rel_k2'])
    print(f"Correlación entre SR e Incertidumbre: {correlation:.4f}")
    
    if abs(correlation) > 0.3:
        if correlation > 0:
            print("  → Correlación positiva: Mayor SR tiende a mayor incertidumbre")
        else:
            print("  → Correlación negativa: Mayor SR tiende a menor incertidumbre")
    else:
        print("  → Correlación débil: No hay relación clara entre SR e incertidumbre")


def main(output_file=None):
    """
    Función principal que genera el resumen completo.
    
    Args:
        output_file: Ruta opcional para guardar el resumen en un archivo de texto.
                    Si es None, solo imprime en consola.
    """
    import io
    from contextlib import redirect_stdout
    
    # Si se especifica archivo de salida, capturar la salida
    if output_file:
        f = io.StringIO()
        with redirect_stdout(f):
            _generate_summary()
        output = f.getvalue()
        
        # Guardar en archivo
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(output)
        print(f"✅ Resumen guardado en: {output_file}")
        print(f"📄 También mostrando en consola:\n")
        print(output)
    else:
        _generate_summary()


def _generate_summary():
    """Genera el resumen (función interna para poder capturar la salida)."""
    print_section("📊 RESUMEN DE RESULTADOS DE INCERTIDUMBRE DE SR", "=", 80)
    print(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analizar cada tipo de resultado
    df_daily = analyze_daily_results()
    df_monthly = analyze_monthly_uncertainty()
    df_sr_range = analyze_sr_range_uncertainty()
    
    # Análisis adicionales
    analyze_minute_data()
    analyze_summary_file()
    
    # Análisis de tendencias
    if df_daily is not None and df_monthly is not None:
        generate_trend_analysis(df_daily, df_monthly)
    
    # Resumen final
    print_section("✅ RESUMEN EJECUTIVO", "=", 80)
    
    if df_daily is not None:
        print(f"✓ Total de días analizados: {len(df_daily):,}")
        print(f"✓ Rango temporal: {df_daily['timestamp'].min().strftime('%Y-%m-%d')} a {df_daily['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"✓ Incertidumbre diaria promedio: {format_percentage(df_daily['U_rel_k2'].mean(), 3)}")
        print(f"✓ SR promedio: {format_percentage(df_daily['SR_agg'].mean(), 2)}")
    
    if df_monthly is not None:
        print(f"✓ Total de meses analizados: {len(df_monthly)}")
        print(f"✓ Incertidumbre mensual promedio: {format_percentage(df_monthly['U_k2_rel_mean'].mean(), 3)}")
        print(f"✓ Rango de incertidumbre mensual: {format_percentage(df_monthly['U_k2_rel_mean'].min(), 3)} - {format_percentage(df_monthly['U_k2_rel_mean'].max(), 3)}")
    
    if df_sr_range is not None:
        print(f"✓ Rangos de SR analizados: {len(df_sr_range)}")
        optimal_range = df_sr_range.loc[df_sr_range['U_k2_rel_mean'].idxmin(), 'sr_range']
        print(f"✓ Rango óptimo (menor incertidumbre): {optimal_range}")
    
    print("\n" + "=" * 80)
    print("📁 Archivos analizados:")
    print(f"  - Diarios: {paths.SR_DAILY_ABS_WITH_U_FILE}")
    print(f"  - Mensual: {Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'sr_uncertainty_by_month.csv'}")
    print(f"  - Rangos SR: {Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'sr_uncertainty_by_sr_range.csv'}")
    print(f"  - Minutal: {paths.SR_MINUTE_WITH_UNCERTAINTY_FILE}")
    print(f"  - Resumen: {paths.SR_UNCERTAINTY_SUMMARY_FILE}")
    print(f"\n📂 Ubicación: {paths.PROPAGACION_ERRORES_REF_CELL_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Genera un resumen visual de los resultados de incertidumbre de SR'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Archivo de salida para guardar el resumen (opcional)'
    )
    
    args = parser.parse_args()
    
    # Si no se especifica archivo, usar uno por defecto
    if args.output is None:
        # Preguntar si quiere guardar
        save_file = input("\n¿Deseas guardar el resumen en un archivo? (s/n, default=n): ").strip().lower()
        if save_file == 's':
            default_output = Path(paths.PROPAGACION_ERRORES_REF_CELL_DIR) / 'uncertainty_results_summary.txt'
            args.output = str(default_output)
    
    main(output_file=args.output)

