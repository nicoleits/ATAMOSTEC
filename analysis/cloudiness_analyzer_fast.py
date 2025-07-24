#!/usr/bin/env python3
"""
Analizador Rápido de Nubosidad basado en Irradiancia
Versión optimizada para análisis rápido con muestreo de datos
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.logging_config import logger
from config import paths
from utils.helpers import save_plot_matplotlib, save_df_csv
import polars as pl

class FastCloudinessAnalyzer:
    """
    Analizador rápido de nubosidad con muestreo de datos
    """
    
    def __init__(self, irradiance_column: str = "1RC411(w.m-2)", 
                 latitude: float = -33.4489, longitude: float = -70.6693, altitude: float = 520):
        self.irradiance_column = irradiance_column
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        
        # Crear objeto de localización para pvlib
        import pvlib
        self.location = pvlib.location.Location(
            latitude=latitude, 
            longitude=longitude, 
            altitude=altitude,
            tz='UTC'
        )
        
        # Parámetros de clasificación
        self.csi_thresholds = {
            'clear_sky': 0.8,      # CSI > 0.8: Cielo despejado
            'partly_cloudy': 0.5,  # 0.5 < CSI <= 0.8: Parcialmente nublado
            'cloudy': 0.2,         # 0.2 < CSI <= 0.5: Nublado
            'heavy_cloud': 0.2     # CSI <= 0.2: Muy nublado
        }
        
    def load_and_sample_data(self, filepath: str, sample_freq: str = '1H') -> pd.DataFrame:
        """
        Carga y muestrea los datos de irradiancia para análisis rápido
        
        Args:
            filepath: Ruta al archivo de datos
            sample_freq: Frecuencia de muestreo ('1H' = cada hora, '30T' = cada 30 min)
        """
        logger.info(f"Cargando y muestreando datos de irradiancia desde: {filepath}")
        
        try:
            # Cargar datos
            df = pd.read_csv(filepath)
            
            # Convertir columna de tiempo
            df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
            df.set_index('_time', inplace=True)
            
            # Verificar que existe la columna de irradiancia
            if self.irradiance_column not in df.columns:
                available_cols = df.columns.tolist()
                logger.error(f"Columna {self.irradiance_column} no encontrada. Columnas disponibles: {available_cols}")
                raise ValueError(f"Columna {self.irradiance_column} no encontrada")
            
            # Convertir irradiancia a numérico
            df[self.irradiance_column] = pd.to_numeric(df[self.irradiance_column], errors='coerce')
            
            # Eliminar valores nulos
            df.dropna(subset=[self.irradiance_column], inplace=True)
            
            # Filtrar valores negativos y extremadamente altos
            df = df[(df[self.irradiance_column] >= 0) & (df[self.irradiance_column] <= 1200)]
            
            # Muestrear datos para análisis rápido
            df_sampled = df.resample(sample_freq).mean()
            df_sampled = df_sampled.dropna()
            
            logger.info(f"Datos originales: {len(df)} filas")
            logger.info(f"Datos muestreados ({sample_freq}): {len(df_sampled)} filas")
            logger.info(f"Rango temporal: {df_sampled.index.min()} a {df_sampled.index.max()}")
            logger.info(f"Rango de irradiancia: {df_sampled[self.irradiance_column].min():.1f} - {df_sampled[self.irradiance_column].max():.1f} W/m²")
            
            return df_sampled
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise
    
    def calculate_theoretical_irradiance_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la irradiancia teórica de forma optimizada
        """
        logger.info("Calculando irradiancia teórica (versión rápida)...")
        
        # Convertir a UTC si no lo está
        df_utc = df.copy()
        if df_utc.index.tz is None:
            df_utc.index = df_utc.index.tz_localize('UTC')
        
        # Usar pvlib para calcular irradiancia teórica en lote
        import pvlib
        clear_sky = self.location.get_clearsky(df_utc.index, model='ineichen', linke_turbidity=2.0)
        
        # Agregar irradiancia teórica al DataFrame
        df_with_theoretical = df_utc.copy()
        df_with_theoretical['irradiance_theoretical'] = clear_sky['ghi']
        
        # Calcular Clear Sky Index
        df_with_theoretical['clear_sky_index'] = (
            df_with_theoretical[self.irradiance_column] / 
            df_with_theoretical['irradiance_theoretical']
        )
        
        # Limitar CSI a valores razonables
        df_with_theoretical['clear_sky_index'] = df_with_theoretical['clear_sky_index'].clip(0, 1.2)
        
        # Filtrar casos donde la irradiancia teórica es muy baja (sol bajo horizonte)
        df_with_theoretical = df_with_theoretical[df_with_theoretical['irradiance_theoretical'] > 10]
        
        logger.info(f"Irradiancia teórica calculada para {len(df_with_theoretical)} puntos válidos")
        logger.info(f"Rango CSI: {df_with_theoretical['clear_sky_index'].min():.3f} - {df_with_theoretical['clear_sky_index'].max():.3f}")
        
        return df_with_theoretical
    
    def classify_cloudiness_by_csi(self, csi: float) -> str:
        """
        Clasifica la nubosidad basada en el Clear Sky Index
        """
        if csi >= self.csi_thresholds['clear_sky']:
            return "Cielo Despejado"
        elif csi >= self.csi_thresholds['partly_cloudy']:
            return "Parcialmente Nublado"
        elif csi >= self.csi_thresholds['cloudy']:
            return "Nublado"
        else:
            return "Muy Nublado"
    
    def calculate_cloudiness_metrics_fast(self, df: pd.DataFrame) -> Dict:
        """
        Calcula métricas de nubosidad de forma rápida
        """
        logger.info("Calculando métricas de nubosidad (versión rápida)...")
        
        # Clasificación de nubosidad basada en CSI
        cloudiness_classes = df['clear_sky_index'].apply(self.classify_cloudiness_by_csi)
        
        # Estadísticas básicas
        stats = {
            'total_measurements': len(df),
            'mean_observed_irradiance': df[self.irradiance_column].mean(),
            'mean_theoretical_irradiance': df['irradiance_theoretical'].mean(),
            'std_observed_irradiance': df[self.irradiance_column].std(),
            'min_observed_irradiance': df[self.irradiance_column].min(),
            'max_observed_irradiance': df[self.irradiance_column].max(),
            'median_observed_irradiance': df[self.irradiance_column].median(),
        }
        
        # Estadísticas del Clear Sky Index
        csi_stats = df['clear_sky_index'].describe()
        stats['clear_sky_index_stats'] = {
            'mean': csi_stats['mean'],
            'std': csi_stats['std'],
            'min': csi_stats['min'],
            'max': csi_stats['max'],
            'median': csi_stats['50%'],
            'q25': csi_stats['25%'],
            'q75': csi_stats['75%']
        }
        
        # Distribución de nubosidad
        cloudiness_distribution = cloudiness_classes.value_counts()
        stats['cloudiness_distribution'] = cloudiness_distribution.to_dict()
        stats['cloudiness_percentages'] = (cloudiness_distribution / len(df) * 100).to_dict()
        
        # Análisis temporal simplificado
        df_with_class = df.copy()
        df_with_class['cloudiness_class'] = cloudiness_classes
        
        # Análisis por hora del día
        df_with_class['hour'] = df_with_class.index.hour
        hourly_cloudiness = df_with_class.groupby(['hour', 'cloudiness_class']).size().unstack(fill_value=0)
        stats['hourly_cloudiness'] = hourly_cloudiness.to_dict()
        
        # Análisis por mes
        df_with_class['month'] = df_with_class.index.month
        monthly_cloudiness = df_with_class.groupby(['month', 'cloudiness_class']).size().unstack(fill_value=0)
        stats['monthly_cloudiness'] = monthly_cloudiness.to_dict()
        
        # Análisis de correlación
        correlation = df[self.irradiance_column].corr(df['irradiance_theoretical'])
        stats['correlation_observed_theoretical'] = correlation
        
        # Análisis de sesgo
        bias = df[self.irradiance_column].mean() - df['irradiance_theoretical'].mean()
        stats['bias_observed_vs_theoretical'] = bias
        
        # RMSE
        rmse = np.sqrt(((df[self.irradiance_column] - df['irradiance_theoretical']) ** 2).mean())
        stats['rmse_observed_vs_theoretical'] = rmse
        
        logger.info("Métricas de nubosidad calculadas exitosamente")
        return stats
    
    def create_fast_visualizations(self, df: pd.DataFrame, stats: Dict, output_dir: str):
        """
        Crea visualizaciones rápidas del análisis de nubosidad
        """
        logger.info("Creando visualizaciones rápidas...")
        
        # Configurar estilo de gráficos
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Análisis Rápido de Nubosidad - Sensor {self.irradiance_column}', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot observado vs teórico
        axes[0, 0].scatter(df['irradiance_theoretical'], df[self.irradiance_column], alpha=0.6, s=10)
        axes[0, 0].plot([0, df['irradiance_theoretical'].max()], [0, df['irradiance_theoretical'].max()], 'r--', label='Línea 1:1')
        axes[0, 0].set_xlabel('Irradiancia Teórica (W/m²)')
        axes[0, 0].set_ylabel('Irradiancia Observada (W/m²)')
        axes[0, 0].set_title('Observado vs Teórico')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribución del Clear Sky Index
        axes[0, 1].hist(df['clear_sky_index'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        for threshold, label in self.csi_thresholds.items():
            axes[0, 1].axvline(label, color='red', linestyle='--', alpha=0.7, label=f'{label}')
        axes[0, 1].set_xlabel('Clear Sky Index')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución del Clear Sky Index')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribución de clases de nubosidad
        cloudiness_classes = df['clear_sky_index'].apply(self.classify_cloudiness_by_csi)
        cloudiness_counts = cloudiness_classes.value_counts()
        colors = ['lightgreen', 'orange', 'lightblue', 'darkblue']
        axes[1, 0].pie(cloudiness_counts.values, labels=cloudiness_counts.index, autopct='%1.1f%%', colors=colors)
        axes[1, 0].set_title('Distribución de Clases de Nubosidad (CSI)')
        
        # 4. Serie temporal de CSI
        sample_csi = df['clear_sky_index'].resample('D').mean()  # Promedio diario para visualización
        axes[1, 1].plot(sample_csi.index, sample_csi.values, alpha=0.7, linewidth=1, color='purple')
        for threshold, value in self.csi_thresholds.items():
            axes[1, 1].axhline(value, color='red', linestyle='--', alpha=0.7, label=f'{value}')
        axes[1, 1].set_xlabel('Fecha')
        axes[1, 1].set_ylabel('Clear Sky Index Promedio Diario')
        axes[1, 1].set_title('Evolución Temporal del CSI')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Guardar gráfico
        save_plot_matplotlib(fig, 'cloudiness_analysis_fast.png', output_dir, dpi=300)
        plt.close()
        
        logger.info("Visualizaciones rápidas creadas exitosamente")
    
    def generate_fast_report(self, stats: Dict, output_dir: str):
        """
        Genera un reporte rápido del análisis de nubosidad
        """
        logger.info("Generando reporte rápido...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("REPORTE RÁPIDO DE ANÁLISIS DE NUBOSIDAD")
        report_lines.append("=" * 80)
        report_lines.append(f"Sensor analizado: {self.irradiance_column}")
        report_lines.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Ubicación: Lat {self.latitude}°, Lon {self.longitude}°, Alt {self.altitude}m")
        report_lines.append("")
        
        # Estadísticas generales
        report_lines.append("ESTADÍSTICAS GENERALES")
        report_lines.append("-" * 40)
        report_lines.append(f"Total de mediciones: {stats['total_measurements']:,}")
        report_lines.append(f"Irradiancia observada promedio: {stats['mean_observed_irradiance']:.1f} W/m²")
        report_lines.append(f"Irradiancia teórica promedio: {stats['mean_theoretical_irradiance']:.1f} W/m²")
        report_lines.append(f"Correlación observado-teórico: {stats['correlation_observed_theoretical']:.3f}")
        report_lines.append(f"Sesgo (observado - teórico): {stats['bias_observed_vs_theoretical']:.1f} W/m²")
        report_lines.append(f"RMSE: {stats['rmse_observed_vs_theoretical']:.1f} W/m²")
        report_lines.append("")
        
        # Estadísticas del Clear Sky Index
        report_lines.append("ESTADÍSTICAS DEL CLEAR SKY INDEX")
        report_lines.append("-" * 40)
        csi_stats = stats['clear_sky_index_stats']
        report_lines.append(f"CSI promedio: {csi_stats['mean']:.3f}")
        report_lines.append(f"CSI máximo: {csi_stats['max']:.3f}")
        report_lines.append(f"CSI mínimo: {csi_stats['min']:.3f}")
        report_lines.append(f"CSI mediano: {csi_stats['median']:.3f}")
        report_lines.append(f"Desviación estándar CSI: {csi_stats['std']:.3f}")
        report_lines.append("")
        
        # Distribución de nubosidad
        report_lines.append("DISTRIBUCIÓN DE NUBOSIDAD (BASADA EN CSI)")
        report_lines.append("-" * 40)
        for class_name, count in stats['cloudiness_distribution'].items():
            percentage = stats['cloudiness_percentages'][class_name]
            report_lines.append(f"{class_name}: {count:,} mediciones ({percentage:.1f}%)")
        report_lines.append("")
        
        # Interpretación rápida
        report_lines.append("INTERPRETACIÓN RÁPIDA")
        report_lines.append("-" * 40)
        
        # Calidad del modelo
        correlation = stats['correlation_observed_theoretical']
        if correlation > 0.8:
            report_lines.append("• Buena correlación entre observado y teórico")
        elif correlation > 0.6:
            report_lines.append("• Correlación moderada entre observado y teórico")
        else:
            report_lines.append("• Baja correlación - revisar parámetros del modelo")
        
        # Análisis de nubosidad
        clear_sky_pct = stats['cloudiness_percentages'].get('Cielo Despejado', 0)
        very_cloudy_pct = stats['cloudiness_percentages'].get('Muy Nublado', 0)
        
        if clear_sky_pct > 40:
            report_lines.append("• Predominio de cielos despejados")
        elif very_cloudy_pct > 20:
            report_lines.append("• Frecuente nubosidad pesada")
        else:
            report_lines.append("• Nubosidad variable")
        
        # Análisis del CSI
        mean_csi = csi_stats['mean']
        if mean_csi > 0.7:
            report_lines.append("• Alto Clear Sky Index promedio")
        elif mean_csi < 0.5:
            report_lines.append("• Bajo Clear Sky Index promedio")
        else:
            report_lines.append("• Clear Sky Index moderado")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Guardar reporte
        report_path = os.path.join(output_dir, 'cloudiness_analysis_fast_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Reporte rápido guardado en: {report_path}")
        
        # También imprimir en consola
        print('\n'.join(report_lines))
    
    def run_fast_analysis(self, data_filepath: str, output_dir: str = None, 
                         sample_freq: str = '1H') -> Dict:
        """
        Ejecuta el análisis rápido de nubosidad
        """
        if output_dir is None:
            output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "cloudiness_analysis_fast")
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=== INICIANDO ANÁLISIS RÁPIDO DE NUBOSIDAD ===")
        
        try:
            # 1. Cargar y muestrear datos
            df = self.load_and_sample_data(data_filepath, sample_freq)
            
            # 2. Calcular irradiancia teórica
            df_with_theoretical = self.calculate_theoretical_irradiance_fast(df)
            
            # 3. Calcular métricas
            stats = self.calculate_cloudiness_metrics_fast(df_with_theoretical)
            
            # 4. Crear visualizaciones
            self.create_fast_visualizations(df_with_theoretical, stats, output_dir)
            
            # 5. Generar reporte
            self.generate_fast_report(stats, output_dir)
            
            # 6. Guardar datos procesados
            df_with_class = df_with_theoretical.copy()
            df_with_class['cloudiness_class'] = df_with_theoretical['clear_sky_index'].apply(self.classify_cloudiness_by_csi)
            
            # Guardar como CSV
            output_csv_path = os.path.join(output_dir, 'cloudiness_fast_processed_data.csv')
            df_with_class.to_csv(output_csv_path)
            logger.info(f"Datos procesados guardados en: {output_csv_path}")
            
            logger.info("=== ANÁLISIS RÁPIDO DE NUBOSIDAD COMPLETADO ===")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error en el análisis rápido de nubosidad: {e}", exc_info=True)
            raise


def run_fast_cloudiness_analysis(data_filepath: str = None, output_dir: str = None, 
                                sample_freq: str = '1H') -> bool:
    """
    Función principal para ejecutar el análisis rápido de nubosidad
    """
    if data_filepath is None:
        data_filepath = os.path.join(paths.BASE_INPUT_DIR, "refcells_data.csv")
    
    try:
        analyzer = FastCloudinessAnalyzer()
        results = analyzer.run_fast_analysis(data_filepath, output_dir, sample_freq)
        
        print(f"\n✅ Análisis rápido de nubosidad completado exitosamente")
        print(f"📊 Resultados guardados en: {output_dir or os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, 'cloudiness_analysis_fast')}")
        print(f"⏱️ Frecuencia de muestreo: {sample_freq}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error ejecutando análisis rápido de nubosidad: {e}")
        return False


if __name__ == "__main__":
    # Permitir ejecución directa
    success = run_fast_cloudiness_analysis()
    
    if success:
        print("✅ Análisis rápido de nubosidad completado")
        sys.exit(0)
    else:
        print("❌ Error en el análisis rápido de nubosidad")
        sys.exit(1) 