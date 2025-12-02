import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Configurar rutas relativas al proyecto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Subir un nivel desde analysis/

# Configurar estilo de los gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def cargar_datos():
    """Cargar y preparar los datos para análisis"""
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'resultados_diferencias_masas.csv'))
    
    # Filtrar para excluir datos de "2 Meses"
    df = df[df['Periodo'] != '2 Meses'].copy()
    
    # Convertir fechas
    df['Inicio_Exposicion'] = pd.to_datetime(df['Inicio_Exposicion'])
    df['Fin_Exposicion'] = pd.to_datetime(df['Fin_Exposicion'])
    
    # Crear columna de masa promedio (promedio de A, B, C)
    df['Diferencia_Promedio_mg'] = (df['Diferencia_Masa_A_mg'] + df['Diferencia_Masa_B_mg'] + df['Diferencia_Masa_C_mg']) / 3
    
    # Ordenar períodos para gráficos (sin "2 Meses")
    orden_periodos = ['semanal', '2 semanas', 'Mensual', 'Trimestral', 'Cuatrimestral', 'Semestral', '1 año']
    df['Periodo_Cat'] = pd.Categorical(df['Periodo'], categories=orden_periodos, ordered=True)
    
    return df

def crear_tabla_resumen_por_periodo(df):
    """Crear tabla resumen estadístico por período"""
    print("=" * 80)
    print("📊 TABLA RESUMEN POR PERÍODO DE EXPOSICIÓN")
    print("=" * 80)
    
    # Agrupar por período
    resumen = df.groupby('Periodo').agg({
        'Exposicion_dias': ['count', 'mean'],
        'Diferencia_Masa_A_mg': ['mean', 'std', 'min', 'max'],
        'Diferencia_Masa_B_mg': ['mean', 'std', 'min', 'max'],
        'Diferencia_Masa_C_mg': ['mean', 'std', 'min', 'max'],
        'Diferencia_Promedio_mg': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Simplificar nombres de columnas
    resumen.columns = ['N_muestras', 'Días_prom', 
                      'A_mean', 'A_std', 'A_min', 'A_max',
                      'B_mean', 'B_std', 'B_min', 'B_max', 
                      'C_mean', 'C_std', 'C_min', 'C_max',
                      'Promedio_mean', 'Promedio_std', 'Promedio_min', 'Promedio_max']
    
    print(resumen)
    
    # Guardar tabla
    resumen.to_csv(os.path.join(PROJECT_ROOT, 'tabla_resumen_periodos.csv'))
    print(f"\n✅ Tabla guardada en: tabla_resumen_periodos.csv")
    
    return resumen

def crear_tabla_estadisticas_generales(df):
    """Crear tabla de estadísticas generales"""
    print("\n" + "=" * 80)
    print("📈 ESTADÍSTICAS GENERALES")
    print("=" * 80)
    
    stats = {
        'Métrica': [],
        'Masa_A_mg': [],
        'Masa_B_mg': [],
        'Masa_C_mg': [],
        'Promedio_mg': []
    }
    
    metricas = ['Media', 'Mediana', 'Desv_Std', 'Mínimo', 'Máximo', 'Q25', 'Q75']
    columnas = ['Diferencia_Masa_A_mg', 'Diferencia_Masa_B_mg', 'Diferencia_Masa_C_mg', 'Diferencia_Promedio_mg']
    
    for metrica in metricas:
        stats['Métrica'].append(metrica)
        for col in columnas:
            if metrica == 'Media':
                valor = df[col].mean()
            elif metrica == 'Mediana':
                valor = df[col].median()
            elif metrica == 'Desv_Std':
                valor = df[col].std()
            elif metrica == 'Mínimo':
                valor = df[col].min()
            elif metrica == 'Máximo':
                valor = df[col].max()
            elif metrica == 'Q25':
                valor = df[col].quantile(0.25)
            elif metrica == 'Q75':
                valor = df[col].quantile(0.75)
            
            stats[col.replace('Diferencia_', '').replace('_mg', '_mg')].append(round(valor, 2))
    
    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False))
    
    # Guardar tabla
    df_stats.to_csv(os.path.join(PROJECT_ROOT, 'estadisticas_generales.csv'), index=False)
    print(f"\n✅ Estadísticas guardadas en: estadisticas_generales.csv")

def grafico_boxplot_por_periodo(df):
    """Crear boxplot de diferencias por período (gráficos originales)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribución de Diferencias de Masa por Período', fontsize=16, fontweight='bold')
    
    # Masa A
    sns.boxplot(data=df, x='Periodo_Cat', y='Diferencia_Masa_A_mg', ax=axes[0,0])
    axes[0,0].set_title('Masa A (mg)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Masa B
    sns.boxplot(data=df, x='Periodo_Cat', y='Diferencia_Masa_B_mg', ax=axes[0,1])
    axes[0,1].set_title('Masa B (mg)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Masa C
    sns.boxplot(data=df, x='Periodo_Cat', y='Diferencia_Masa_C_mg', ax=axes[1,0])
    axes[1,0].set_title('Masa C (mg)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Promedio
    sns.boxplot(data=df, x='Periodo_Cat', y='Diferencia_Promedio_mg', ax=axes[1,1])
    axes[1,1].set_title('Masa Promedio (mg)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'boxplot_masas_por_periodo.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Boxplot guardado en: graficos_analisis_integrado_py/boxplot_masas_por_periodo.png")

def grafico_scatter_dias_vs_masa(df):
    """Crear scatter plot de días de exposición vs diferencias"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Relación: Días de Exposición vs Diferencia de Masa', fontsize=16, fontweight='bold')
    
    # Masa A
    axes[0,0].scatter(df['Exposicion_dias'], df['Diferencia_Masa_A_mg'], alpha=0.7, color='blue')
    axes[0,0].set_xlabel('Días de Exposición')
    axes[0,0].set_ylabel('Diferencia Masa A (mg)')
    axes[0,0].set_title('Masa A')
    axes[0,0].grid(True, alpha=0.3)
    
    # Masa B
    axes[0,1].scatter(df['Exposicion_dias'], df['Diferencia_Masa_B_mg'], alpha=0.7, color='green')
    axes[0,1].set_xlabel('Días de Exposición')
    axes[0,1].set_ylabel('Diferencia Masa B (mg)')
    axes[0,1].set_title('Masa B')
    axes[0,1].grid(True, alpha=0.3)
    
    # Masa C
    axes[1,0].scatter(df['Exposicion_dias'], df['Diferencia_Masa_C_mg'], alpha=0.7, color='red')
    axes[1,0].set_xlabel('Días de Exposición')
    axes[1,0].set_ylabel('Diferencia Masa C (mg)')
    axes[1,0].set_title('Masa C')
    axes[1,0].grid(True, alpha=0.3)
    
    # Promedio
    axes[1,1].scatter(df['Exposicion_dias'], df['Diferencia_Promedio_mg'], alpha=0.7, color='purple')
    axes[1,1].set_xlabel('Días de Exposición')
    axes[1,1].set_ylabel('Diferencia Masa Promedio (mg)')
    axes[1,1].set_title('Masa Promedio')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'scatter_dias_vs_masa.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Scatter plot guardado en: graficos_analisis_integrado_py/scatter_dias_vs_masa.png")

def grafico_scatter_dias_vs_promedio(df):
    """Crear scatter plot de días de exposición vs masa promedio"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear scatter plot con colores por período
    colores_periodo = plt.cm.Set3(np.linspace(0, 1, len(df['Periodo'].unique())))
    
    for i, periodo in enumerate(df['Periodo'].unique()):
        datos_periodo = df[df['Periodo'] == periodo]
        ax.scatter(datos_periodo['Exposicion_dias'], datos_periodo['Diferencia_Promedio_mg'], 
                  alpha=0.7, s=80, label=periodo, color=colores_periodo[i])
    
    ax.set_xlabel('Días de Exposición')
    ax.set_ylabel('Diferencia Promedio de Masa (mg)')
    ax.set_title('Relación: Días de Exposición vs Masa Promedio', fontsize=14, fontweight='bold')
    ax.legend(title='Período', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'scatter_dias_vs_promedio.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Scatter plot de promedios guardado en: graficos_analisis_integrado_py/scatter_dias_vs_promedio.png")



def grafico_barras_promedio_por_periodo(df):
    """Crear gráfico de barras con promedios por período"""
    # Calcular promedios por período
    promedios = df.groupby('Periodo_Cat')[['Diferencia_Masa_A_mg', 'Diferencia_Masa_B_mg', 
                                          'Diferencia_Masa_C_mg']].mean()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(promedios.index))
    width = 0.25
    
    bars1 = ax.bar(x - width, promedios['Diferencia_Masa_A_mg'], width, label='Masa A', color='skyblue')
    bars2 = ax.bar(x, promedios['Diferencia_Masa_B_mg'], width, label='Masa B', color='lightgreen')
    bars3 = ax.bar(x + width, promedios['Diferencia_Masa_C_mg'], width, label='Masa C', color='salmon')
    
    ax.set_xlabel('Período de Exposición')
    ax.set_ylabel('Diferencia Promedio (mg)')
    ax.set_title('Diferencias Promedio de Masa por Período', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(promedios.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'barras_promedio_periodo.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico de barras guardado en: graficos_analisis_integrado_py/barras_promedio_periodo.png")

def grafico_boxplot_agrupado_por_periodo(df):
    """Crear boxplot agrupado por período con todas las masas juntas"""
    # Preparar datos en formato largo para agrupación
    df_melted = df.melt(
        id_vars=['Periodo_Cat', 'Exposicion_dias'], 
        value_vars=['Diferencia_Masa_A_mg', 'Diferencia_Masa_B_mg', 'Diferencia_Masa_C_mg'],
        var_name='Tipo_Masa', 
        value_name='Diferencia_mg'
    )
    df_melted['Tipo_Masa'] = df_melted['Tipo_Masa'].str.replace('Diferencia_Masa_', '').str.replace('_mg', '')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Análisis Agrupado por Período de Exposición', fontsize=16, fontweight='bold')
    
    # Boxplot agrupado por período y tipo de masa
    sns.boxplot(data=df_melted, x='Periodo_Cat', y='Diferencia_mg', hue='Tipo_Masa', ax=ax1)
    ax1.set_title('Distribución por Período (A, B, C agrupadas)')
    ax1.set_xlabel('Período de Exposición')
    ax1.set_ylabel('Diferencia de Masa (mg)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Tipo de Masa')
    ax1.grid(True, alpha=0.3)
    
    # Boxplot de masa promedio por período
    sns.boxplot(data=df, x='Periodo_Cat', y='Diferencia_Promedio_mg', ax=ax2)
    ax2.set_title('Distribución de Masa Promedio por Período')
    ax2.set_xlabel('Período de Exposición')
    ax2.set_ylabel('Diferencia Promedio (mg)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'boxplot_agrupado_por_periodo.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Boxplot agrupado guardado en: graficos_analisis_integrado_py/boxplot_agrupado_por_periodo.png")

def grafico_barras_individuales_por_periodo(df):
    """Crear gráfico de barras individuales por masa y período"""
    # Calcular promedios por período con orden correcto
    orden_periodos = ['semanal', '2 semanas', 'Mensual', 'Trimestral', 
                     'Cuatrimestral', 'Semestral', '1 año']
    
    promedios = df.groupby('Periodo')[['Diferencia_Masa_A_mg', 'Diferencia_Masa_B_mg', 
                                      'Diferencia_Masa_C_mg']].mean()
    
    # Reordenar por duración de exposición
    promedios = promedios.reindex(orden_periodos)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Barras individuales por masa
    x = np.arange(len(promedios.index))
    width = 0.25
    
    bars1 = ax.bar(x - width, promedios['Diferencia_Masa_A_mg'], width, label='Masa A', color='skyblue')
    bars2 = ax.bar(x, promedios['Diferencia_Masa_B_mg'], width, label='Masa B', color='lightgreen')
    bars3 = ax.bar(x + width, promedios['Diferencia_Masa_C_mg'], width, label='Masa C', color='salmon')
    
    ax.set_xlabel('Período de Exposición')
    ax.set_ylabel('Diferencia Promedio (mg)')
    ax.set_title('Promedios por Masa Individual por Período', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(promedios.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'barras_individuales_por_periodo.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico de barras individuales guardado en: graficos_analisis_integrado_py/barras_individuales_por_periodo.png")

def grafico_barras_promedio_general(df):
    """Crear gráfico de barras del promedio general por período"""
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
    
    # Calcular promedios por período con orden correcto
    orden_periodos = ['semanal', '2 semanas', 'Mensual', 'Trimestral', 
                     'Cuatrimestral', 'Semestral', '1 año']
    
    promedios = df.groupby('Periodo')['Diferencia_Promedio_mg'].mean()
    
    # Reordenar por duración de exposición
    promedios = promedios.reindex(orden_periodos)
    
    # Traducir los índices al inglés
    promedios.index = [traduccion_periodos.get(periodo, periodo) for periodo in promedios.index]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Barras del promedio general
    bars = ax.bar(promedios.index, promedios.values, 
                  color='mediumpurple', alpha=0.8, edgecolor='darkslateblue', linewidth=1.5)
    
    ax.set_xlabel('Exposure Period')
    ax.set_ylabel('General Average Difference (mg)')
    ax.set_title('General Average of Soiling by Period\n(A+B+C)/3', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'barras_promedio_general.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico de barras promedio general guardado en: graficos_analisis_integrado_py/barras_promedio_general.png")

def grafico_barras_promedio_general_sin_anual(df):
    """Crear gráfico de barras del promedio general por período SIN el período anual"""
    # Calcular promedios por período con orden correcto (sin el anual)
    orden_periodos = ['semanal', '2 semanas', 'Mensual', 'Trimestral', 
                     'Cuatrimestral', 'Semestral']
    
    # Mapeo para traducir las etiquetas del eje x al inglés
    traduccion_periodos = {
        'semanal': 'Weekly',
        '2 semanas': '2 weeks',
        'Mensual': 'Monthly',
        'Trimestral': 'Quarterly',
        'Cuatrimestral': '4-month',
        'Semestral': 'Semester'
    }
    
    promedios = df.groupby('Periodo')['Diferencia_Promedio_mg'].mean()
    
    # Reordenar por duración de exposición (sin el anual)
    promedios = promedios.reindex(orden_periodos)
    
    # Crear etiquetas traducidas para el eje x
    etiquetas_traducidas = [traduccion_periodos[periodo] for periodo in promedios.index]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    # Barras del promedio general
    bars = ax.bar(range(len(promedios)), promedios.values, 
                  color='mediumpurple', alpha=0.8, edgecolor='darkslateblue', linewidth=1.5)
    
    # Configurar etiquetas del eje x
    ax.set_xticks(range(len(promedios)))
    ax.set_xticklabels(etiquetas_traducidas)
    
    ax.set_xlabel('Period of exposure', fontsize=14)
    ax.set_ylabel('Mass of accumulated soiling (mg)', fontsize=14)
    # ax.set_title('Promedio General de Soiling por Período\n(A+B+C)/3 - Sin Período Anual', fontsize=14, fontweight='bold')
    ax.set_title('Mean mass of accumulated soiling per period', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)#, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'barras_promedio_general_sin_anual.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Gráfico de barras promedio general (sin anual) guardado en: graficos_analisis_integrado_py/barras_promedio_general_sin_anual.png")

def grafico_heatmap_periodo_masa(df):
    """Crear heatmap de correlación entre períodos y masas con orden correcto"""
    # Crear matriz de promedios por período
    pivot_data = df.groupby('Periodo')[['Diferencia_Masa_A_mg', 'Diferencia_Masa_B_mg', 
                                       'Diferencia_Masa_C_mg', 'Diferencia_Promedio_mg']].mean()
    
    # Ordenar por duración de exposición (orden correcto)
    orden_periodos_heatmap = ['semanal', '2 semanas', 'Mensual', 'Trimestral', 
                             'Cuatrimestral', 'Semestral', '1 año']
    
    # Reindexar para mantener el orden correcto
    pivot_data = pivot_data.reindex(orden_periodos_heatmap)
    
    # Renombrar columnas para el heatmap
    pivot_data.columns = ['Masa A', 'Masa B', 'Masa C', 'Promedio']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear heatmap
    sns.heatmap(pivot_data.T, annot=True, cmap='YlOrRd', fmt='.2f', ax=ax, 
                cbar_kws={'label': 'Diferencia Promedio (mg)'})
    ax.set_title('Heatmap: Diferencias Promedio por Período', fontsize=14, fontweight='bold')
    ax.set_xlabel('Período de Exposición')
    ax.set_ylabel('Tipo de Masa')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'graficos_analisis_integrado_py', 'heatmap_periodo_masa.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Heatmap guardado en: graficos_analisis_integrado_py/heatmap_periodo_masa.png")

def crear_tabla_detallada_por_periodo(df):
    """Crear tabla detallada con estadísticas por período"""
    print("\n" + "=" * 100)
    print("📋 TABLA DETALLADA POR PERÍODO (Enfoque en Agrupación por Período)")
    print("=" * 100)
    
    tabla_detallada = []
    
    for periodo in df['Periodo'].unique():
        datos_periodo = df[df['Periodo'] == periodo]
        
        fila = {
            'Período': periodo,
            'N_Muestras': len(datos_periodo),
            'Días_Promedio': datos_periodo['Exposicion_dias'].mean(),
            'Masa_A_Media': datos_periodo['Diferencia_Masa_A_mg'].mean(),
            'Masa_A_Max': datos_periodo['Diferencia_Masa_A_mg'].max(),
            'Masa_B_Media': datos_periodo['Diferencia_Masa_B_mg'].mean(),
            'Masa_B_Max': datos_periodo['Diferencia_Masa_B_mg'].max(),
            'Masa_C_Media': datos_periodo['Diferencia_Masa_C_mg'].mean(),
            'Masa_C_Max': datos_periodo['Diferencia_Masa_C_mg'].max(),
            'Promedio_Media': datos_periodo['Diferencia_Promedio_mg'].mean(),
            'Promedio_Max': datos_periodo['Diferencia_Promedio_mg'].max(),
            'Coef_Variacion_Promedio': (datos_periodo['Diferencia_Promedio_mg'].std() / 
                                       datos_periodo['Diferencia_Promedio_mg'].mean()) * 100
        }
        tabla_detallada.append(fila)
    
    df_tabla = pd.DataFrame(tabla_detallada).round(2)
    print(df_tabla.to_string(index=False))
    
    # Guardar tabla
    df_tabla.to_csv(os.path.join(PROJECT_ROOT, 'tabla_detallada_por_periodo.csv'), index=False)
    print(f"\n✅ Tabla detallada guardada en: tabla_detallada_por_periodo.csv")
    
    return df_tabla

def main():
    """Función principal para generar todos los análisis"""
    print("🚀 Iniciando análisis visual de diferencias de masa por soiling...")
    print("=" * 80)
    
    # Cargar datos
    df = cargar_datos()
    print(f"📊 Datos cargados: {len(df)} registros")
    
    # Crear tablas resumen
    resumen_periodos = crear_tabla_resumen_por_periodo(df)
    crear_tabla_estadisticas_generales(df)
    
    # Crear gráficos originales
    print("\n🎨 Generando visualizaciones originales...")
    grafico_boxplot_por_periodo(df)
    grafico_scatter_dias_vs_masa(df)
    grafico_scatter_dias_vs_promedio(df)
    grafico_barras_promedio_por_periodo(df)
    
    # Crear nuevos gráficos agrupados por período
    print("\n🎯 Generando análisis agrupados por período...")
    crear_tabla_detallada_por_periodo(df)
    grafico_boxplot_agrupado_por_periodo(df)
    grafico_barras_individuales_por_periodo(df)
    grafico_barras_promedio_general(df)
    grafico_barras_promedio_general_sin_anual(df)
    grafico_heatmap_periodo_masa(df)
    
    print("\n" + "=" * 80)
    print("✅ ¡Análisis completo!")
    print("📁 Archivos generados:")
    print("   📊 TABLAS:")
    print("     • tabla_resumen_periodos.csv")
    print("     • estadisticas_generales.csv")
    print("     • tabla_detallada_por_periodo.csv")
    print("   📈 GRÁFICOS ORIGINALES:")
    print("     • boxplot_masas_por_periodo.png")
    print("     • scatter_dias_vs_masa.png") 
    print("     • scatter_dias_vs_promedio.png")
    print("     • barras_promedio_periodo.png")
    print("   🎯 GRÁFICOS AGRUPADOS POR PERÍODO:")
    print("     • boxplot_agrupado_por_periodo.png")
    print("     • barras_individuales_por_periodo.png")
    print("     • barras_promedio_general.png")
    print("     • barras_promedio_general_sin_anual.png")
    print("     • heatmap_periodo_masa.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
