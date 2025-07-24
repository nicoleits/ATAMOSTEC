#!/usr/bin/env python3
"""
Analizador de Nubosidad basado en Irradiancia
Estudia la nubosidad utilizando la irradiancia medida por el sensor 1RC411
Incluye cálculo de irradiancia teórica para análisis robusto
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

class ClearSkyModel:
    """
    Modelo de cielo despejado para calcular irradiancia teórica
    Usa pvlib para cálculos precisos de posición solar y modelos de cielo despejado
    """
    
    def __init__(self, latitude: float = -33.4489, longitude: float = -70.6693, altitude: float = 520):
        """
        Inicializa el modelo de cielo despejado
        
        Args:
            latitude: Latitud del sitio (grados)
            longitude: Longitud del sitio (grados)
            altitude: Altitud del sitio (metros)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        
        # Constantes solares
        self.G0 = 1367  # Constante solar (W/m²)
        
        # Crear objeto de localización para pvlib
        import pvlib
        self.location = pvlib.location.Location(
            latitude=latitude, 
            longitude=longitude, 
            altitude=altitude,
            tz='UTC'
        )
        
    def calculate_solar_zenith_angle(self, timestamp: pd.Timestamp) -> float:
        """
        Calcula el ángulo cenital solar usando pvlib
        
        Args:
            timestamp: Timestamp para el cálculo
            
        Returns:
            Ángulo cenital en grados
        """
        # Convertir a UTC si no lo está
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        # Usar pvlib para calcular posición solar
        solar_position = self.location.get_solarposition(timestamp)
        zenith_angle = solar_position['zenith'].iloc[0]
        
        return zenith_angle
    
    def calculate_air_mass(self, zenith_angle: float) -> float:
        """
        Calcula la masa de aire usando la fórmula de Kasten-Young
        """
        # Fórmula de Kasten-Young (1989)
        air_mass = 1 / (np.cos(np.radians(zenith_angle)) + 0.50572 * 
                       (96.07995 - zenith_angle) ** (-1.6364))
        
        return air_mass
    
    def calculate_clear_sky_irradiance_kasten(self, timestamp: pd.Timestamp, 
                                            turbidity: float = 2.0) -> float:
        """
        Calcula irradiancia de cielo despejado usando modelo de Ineichen-Perez (pvlib)
        
        Args:
            timestamp: Timestamp para el cálculo
            turbidity: Coeficiente de turbidez de Linke (default: 2.0)
            
        Returns:
            Irradiancia de cielo despejado en W/m²
        """
        # Convertir a UTC si no lo está
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        # Usar pvlib para calcular irradiancia de cielo despejado
        import pvlib
        clear_sky = self.location.get_clearsky(timestamp, model='ineichen', 
                                              linke_turbidity=turbidity)
        
        # Retornar la irradiancia global horizontal
        return clear_sky['ghi'].iloc[0]
    
    def calculate_clear_sky_irradiance_bird(self, timestamp: pd.Timestamp,
                                          turbidity: float = 2.0,
                                          precipitable_water: float = 1.0,
                                          ozone: float = 0.35) -> float:
        """
        Calcula irradiancia de cielo despejado usando modelo de Bird (pvlib)
        
        Args:
            timestamp: Timestamp para el cálculo
            turbidity: Coeficiente de turbidez de Angstrom
            precipitable_water: Agua precipitable en cm
            ozone: Contenido de ozono en cm
            
        Returns:
            Irradiancia de cielo despejado en W/m²
        """
        # Convertir a UTC si no lo está
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        # Usar pvlib para calcular irradiancia de cielo despejado con modelo Bird
        import pvlib
        clear_sky = self.location.get_clearsky(timestamp, model='bird', 
                                              linke_turbidity=turbidity)
        
        # Retornar la irradiancia global horizontal
        return clear_sky['ghi'].iloc[0]
    
    def calculate_clear_sky_irradiance_simple(self, timestamp: pd.Timestamp) -> float:
        """
        Modelo simplificado de cielo despejado para comparación rápida
        """
        zenith_angle = self.calculate_solar_zenith_angle(timestamp)
        
        if zenith_angle > 90:
            return 0.0
        
        # Modelo simplificado basado en coseno del ángulo cenital
        # Con corrección por masa de aire
        air_mass = self.calculate_air_mass(zenith_angle)
        
        # Irradiancia extraterrestre
        day_of_year = timestamp.timetuple().tm_yday
        E0 = self.G0 * (1 + 0.034 * np.cos(2 * np.pi * (day_of_year - 1) / 365.25))
        
        # Transmitancia atmosférica simplificada
        atmospheric_transmittance = 0.75 * np.exp(-0.1 * air_mass)
        
        G_clear = E0 * atmospheric_transmittance * np.cos(np.radians(zenith_angle))
        
        return max(0, G_clear)


class CloudinessAnalyzer:
    """
    Analizador de nubosidad basado en irradiancia solar con modelos de cielo despejado
    """
    
    def __init__(self, irradiance_column: str = "1RC411(w.m-2)", 
                 latitude: float = -33.4489, longitude: float = -70.6693, altitude: float = 520):
        self.irradiance_column = irradiance_column
        self.clear_sky_model = ClearSkyModel(latitude, longitude, altitude)
        self.analysis_results = {}
        
        # Parámetros de clasificación mejorados
        self.csi_thresholds = {
            'clear_sky': 0.8,      # CSI > 0.8: Cielo despejado
            'partly_cloudy': 0.5,  # 0.5 < CSI <= 0.8: Parcialmente nublado
            'cloudy': 0.2,         # 0.2 < CSI <= 0.5: Nublado
            'heavy_cloud': 0.2     # CSI <= 0.2: Muy nublado
        }
        
    def load_and_preprocess_data(self, filepath: str) -> pd.DataFrame:
        """
        Carga y preprocesa los datos de irradiancia
        """
        logger.info(f"Cargando datos de irradiancia desde: {filepath}")
        
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
            
            # Filtrar valores negativos (no físicamente posibles)
            df = df[df[self.irradiance_column] >= 0]
            
            # Filtrar valores extremadamente altos (posibles errores de sensor)
            max_reasonable_irradiance = 1200  # W/m²
            df = df[df[self.irradiance_column] <= max_reasonable_irradiance]
            
            logger.info(f"Datos cargados exitosamente: {len(df)} filas")
            logger.info(f"Rango temporal: {df.index.min()} a {df.index.max()}")
            logger.info(f"Rango de irradiancia: {df[self.irradiance_column].min():.1f} - {df[self.irradiance_column].max():.1f} W/m²")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise
    
    def calculate_theoretical_irradiance(self, df: pd.DataFrame, model_type: str = 'kasten') -> pd.DataFrame:
        """
        Calcula la irradiancia teórica para todos los timestamps (optimizado)
        
        Args:
            df: DataFrame con datos observados
            model_type: Tipo de modelo ('kasten', 'bird', 'simple')
            
        Returns:
            DataFrame con irradiancia teórica agregada
        """
        logger.info(f"Calculando irradiancia teórica usando modelo: {model_type}")
        
        # Convertir a UTC si no lo está
        df_utc = df.copy()
        if df_utc.index.tz is None:
            df_utc.index = df_utc.index.tz_localize('UTC')
        
        # Usar pvlib para calcular irradiancia teórica en lote
        import pvlib
        
        # Calcular irradiancia teórica para todo el DataFrame de una vez
        if model_type == 'kasten':
            clear_sky = self.clear_sky_model.location.get_clearsky(df_utc.index, model='ineichen', linke_turbidity=2.0)
        elif model_type == 'bird':
            clear_sky = self.clear_sky_model.location.get_clearsky(df_utc.index, model='bird', linke_turbidity=2.0)
        else:
            clear_sky = self.clear_sky_model.location.get_clearsky(df_utc.index, model='ineichen', linke_turbidity=2.0)
        
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
    
    def calculate_cloudiness_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calcula métricas de nubosidad usando irradiancia teórica
        """
        logger.info("Calculando métricas de nubosidad con modelo teórico...")
        
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
        
        # Análisis temporal
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
        
        # Análisis de variabilidad diaria
        daily_csi_std = df['clear_sky_index'].resample('D').std()
        stats['daily_csi_variability'] = {
            'mean': daily_csi_std.mean(),
            'std': daily_csi_std.std(),
            'high_variability_days': len(daily_csi_std[daily_csi_std > daily_csi_std.quantile(0.75)])
        }
        
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
    
    def create_cloudiness_visualizations(self, df: pd.DataFrame, stats: Dict, output_dir: str):
        """
        Crea visualizaciones del análisis de nubosidad mejorado
        """
        logger.info("Creando visualizaciones de nubosidad mejoradas...")
        
        # Configurar estilo de gráficos
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Comparación observado vs teórico
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Análisis Robusto de Nubosidad - Sensor {self.irradiance_column}', fontsize=16, fontweight='bold')
        
        # Scatter plot observado vs teórico
        axes[0, 0].scatter(df['irradiance_theoretical'], df[self.irradiance_column], alpha=0.6, s=1)
        axes[0, 0].plot([0, df['irradiance_theoretical'].max()], [0, df['irradiance_theoretical'].max()], 'r--', label='Línea 1:1')
        axes[0, 0].set_xlabel('Irradiancia Teórica (W/m²)')
        axes[0, 0].set_ylabel('Irradiancia Observada (W/m²)')
        axes[0, 0].set_title('Observado vs Teórico')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribución del Clear Sky Index
        axes[0, 1].hist(df['clear_sky_index'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        for threshold, label in self.csi_thresholds.items():
            axes[0, 1].axvline(label, color='red', linestyle='--', alpha=0.7, label=f'{label}')
        axes[0, 1].set_xlabel('Clear Sky Index')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución del Clear Sky Index')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribución de clases de nubosidad
        cloudiness_classes = df['clear_sky_index'].apply(self.classify_cloudiness_by_csi)
        cloudiness_counts = cloudiness_classes.value_counts()
        colors = ['lightgreen', 'orange', 'lightblue', 'darkblue']
        axes[0, 2].pie(cloudiness_counts.values, labels=cloudiness_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0, 2].set_title('Distribución de Clases de Nubosidad (CSI)')
        
        # Serie temporal de CSI
        sample_csi = df['clear_sky_index'].resample('H').mean()
        axes[1, 0].plot(sample_csi.index, sample_csi.values, alpha=0.7, linewidth=0.8, color='purple')
        for threshold, value in self.csi_thresholds.items():
            axes[1, 0].axhline(value, color='red', linestyle='--', alpha=0.7, label=f'{value}')
        axes[1, 0].set_xlabel('Fecha')
        axes[1, 0].set_ylabel('Clear Sky Index Promedio')
        axes[1, 0].set_title('Evolución Temporal del CSI')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Perfil diario de CSI
        df_with_hour = df.copy()
        df_with_hour['hour'] = df_with_hour.index.hour
        hourly_csi = df_with_hour.groupby('hour')['clear_sky_index'].mean()
        axes[1, 1].plot(hourly_csi.index, hourly_csi.values, marker='o', linewidth=2, markersize=6, color='purple')
        for threshold, value in self.csi_thresholds.items():
            axes[1, 1].axhline(value, color='red', linestyle='--', alpha=0.7, label=f'{value}')
        axes[1, 1].set_xlabel('Hora del Día')
        axes[1, 1].set_ylabel('Clear Sky Index Promedio')
        axes[1, 1].set_title('Perfil Diario del CSI')
        axes[1, 1].set_xticks(range(0, 24, 2))
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Análisis estacional de CSI
        df_with_month = df.copy()
        df_with_month['month'] = df_with_month.index.month
        monthly_csi = df_with_month.groupby('month')['clear_sky_index'].mean()
        monthly_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        axes[1, 2].bar(monthly_csi.index, monthly_csi.values, alpha=0.7, color='teal')
        for threshold, value in self.csi_thresholds.items():
            axes[1, 2].axhline(value, color='red', linestyle='--', alpha=0.7, label=f'{value}')
        axes[1, 2].set_xlabel('Mes')
        axes[1, 2].set_ylabel('Clear Sky Index Promedio')
        axes[1, 2].set_title('Análisis Estacional del CSI')
        axes[1, 2].set_xticks(range(1, 13))
        axes[1, 2].set_xticklabels(monthly_names)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico principal
        save_plot_matplotlib(fig, 'cloudiness_analysis_robust', output_dir, dpi=300)
        plt.close()
        
        # 2. Análisis detallado de variabilidad
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis Detallado de Variabilidad y Calidad del Modelo', fontsize=16, fontweight='bold')
        
        # Residuales (observado - teórico)
        residuals = df[self.irradiance_column] - df['irradiance_theoretical']
        axes[0, 0].scatter(df['irradiance_theoretical'], residuals, alpha=0.6, s=1)
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Irradiancia Teórica (W/m²)')
        axes[0, 0].set_ylabel('Residuales (Observado - Teórico)')
        axes[0, 0].set_title('Análisis de Residuales')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histograma de residuales
        axes[0, 1].hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Residuales (W/m²)')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Residuales')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Variabilidad diaria del CSI
        daily_csi_std = df['clear_sky_index'].resample('D').std()
        axes[1, 0].plot(daily_csi_std.index, daily_csi_std.values, alpha=0.7, linewidth=0.8, color='brown')
        axes[1, 0].axhline(daily_csi_std.quantile(0.75), color='red', linestyle='--', alpha=0.7, label='Alta Variabilidad (75% percentil)')
        axes[1, 0].set_xlabel('Fecha')
        axes[1, 0].set_ylabel('Desviación Estándar Diaria del CSI')
        axes[1, 0].set_title('Variabilidad Diaria del Clear Sky Index')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Comparación de distribuciones
        axes[1, 1].hist(df[self.irradiance_column], bins=50, alpha=0.5, label='Observado', color='blue')
        axes[1, 1].hist(df['irradiance_theoretical'], bins=50, alpha=0.5, label='Teórico', color='red')
        axes[1, 1].set_xlabel('Irradiancia (W/m²)')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Comparación de Distribuciones')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico de variabilidad
        save_plot_matplotlib(fig, 'cloudiness_variability_robust', output_dir, dpi=300)
        plt.close()
        
        logger.info("Visualizaciones de nubosidad mejoradas creadas exitosamente")
    
    def generate_cloudiness_report(self, stats: Dict, output_dir: str):
        """
        Genera un reporte detallado del análisis de nubosidad mejorado
        """
        logger.info("Generando reporte de nubosidad mejorado...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("REPORTE DE ANÁLISIS ROBUSTO DE NUBOSIDAD")
        report_lines.append("=" * 80)
        report_lines.append(f"Sensor analizado: {self.irradiance_column}")
        report_lines.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Ubicación: Lat {self.clear_sky_model.latitude}°, Lon {self.clear_sky_model.longitude}°, Alt {self.clear_sky_model.altitude}m")
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
        
        # Análisis de variabilidad
        report_lines.append("ANÁLISIS DE VARIABILIDAD")
        report_lines.append("-" * 40)
        variability_stats = stats['daily_csi_variability']
        report_lines.append(f"Variabilidad diaria promedio del CSI: {variability_stats['mean']:.3f}")
        report_lines.append(f"Días con alta variabilidad: {variability_stats['high_variability_days']}")
        report_lines.append("")
        
        # Interpretación mejorada
        report_lines.append("INTERPRETACIÓN MEJORADA")
        report_lines.append("-" * 40)
        
        # Calidad del modelo
        correlation = stats['correlation_observed_theoretical']
        if correlation > 0.9:
            report_lines.append("• Excelente correlación entre observado y teórico - modelo confiable")
        elif correlation > 0.8:
            report_lines.append("• Buena correlación entre observado y teórico - modelo adecuado")
        elif correlation > 0.7:
            report_lines.append("• Correlación moderada - modelo aceptable")
        else:
            report_lines.append("• Baja correlación - revisar parámetros del modelo")
        
        # Análisis de nubosidad
        clear_sky_pct = stats['cloudiness_percentages'].get('Cielo Despejado', 0)
        very_cloudy_pct = stats['cloudiness_percentages'].get('Muy Nublado', 0)
        
        if clear_sky_pct > 40:
            report_lines.append("• Predominio de cielos despejados en el sitio")
        elif very_cloudy_pct > 20:
            report_lines.append("• Frecuente nubosidad pesada en el sitio")
        else:
            report_lines.append("• Nubosidad variable en el sitio")
        
        # Análisis de variabilidad
        if variability_stats['high_variability_days'] > stats['total_measurements'] / 365 * 0.25:
            report_lines.append("• Alta variabilidad diaria - condiciones meteorológicas cambiantes")
        else:
            report_lines.append("• Variabilidad diaria moderada - condiciones relativamente estables")
        
        # Análisis del CSI
        mean_csi = csi_stats['mean']
        if mean_csi > 0.7:
            report_lines.append("• Alto Clear Sky Index promedio - excelentes condiciones solares")
        elif mean_csi < 0.5:
            report_lines.append("• Bajo Clear Sky Index promedio - frecuente nubosidad")
        else:
            report_lines.append("• Clear Sky Index moderado - condiciones solares variables")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Guardar reporte
        report_path = os.path.join(output_dir, 'cloudiness_analysis_robust_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Reporte mejorado guardado en: {report_path}")
        
        # También imprimir en consola
        print('\n'.join(report_lines))
    
    def run_cloudiness_analysis(self, data_filepath: str, output_dir: str = None, 
                               model_type: str = 'kasten') -> Dict:
        """
        Ejecuta el análisis completo de nubosidad con modelo teórico
        """
        if output_dir is None:
            output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "cloudiness_analysis_robust")
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=== INICIANDO ANÁLISIS ROBUSTO DE NUBOSIDAD ===")
        
        try:
            # 1. Cargar y preprocesar datos
            df = self.load_and_preprocess_data(data_filepath)
            
            # 2. Calcular irradiancia teórica
            df_with_theoretical = self.calculate_theoretical_irradiance(df, model_type)
            
            # 3. Calcular métricas
            stats = self.calculate_cloudiness_metrics(df_with_theoretical)
            
            # 4. Crear visualizaciones
            self.create_cloudiness_visualizations(df_with_theoretical, stats, output_dir)
            
            # 5. Generar reporte
            self.generate_cloudiness_report(stats, output_dir)
            
            # 6. Guardar datos procesados
            df_with_class = df_with_theoretical.copy()
            df_with_class['cloudiness_class'] = df_with_theoretical['clear_sky_index'].apply(self.classify_cloudiness_by_csi)
            
            # Guardar como CSV
            output_csv_path = os.path.join(output_dir, 'cloudiness_robust_processed_data.csv')
            df_with_class.to_csv(output_csv_path)
            logger.info(f"Datos procesados guardados en: {output_csv_path}")
            
            # Guardar estadísticas como JSON
            import json
            stats_output_path = os.path.join(output_dir, 'cloudiness_robust_statistics.json')
            # Convertir numpy types a Python types para JSON
            stats_json = {}
            for key, value in stats.items():
                if isinstance(value, dict):
                    stats_json[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                     for k, v in value.items()}
                elif isinstance(value, (np.integer, np.floating)):
                    stats_json[key] = float(value)
                else:
                    stats_json[key] = value
            
            with open(stats_output_path, 'w') as f:
                json.dump(stats_json, f, indent=2, default=str)
            
            logger.info(f"Estadísticas guardadas en: {stats_output_path}")
            
            self.analysis_results = stats
            logger.info("=== ANÁLISIS ROBUSTO DE NUBOSIDAD COMPLETADO ===")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error en el análisis de nubosidad: {e}", exc_info=True)
            raise


def run_cloudiness_analysis(data_filepath: str = None, output_dir: str = None, 
                           model_type: str = 'kasten') -> bool:
    """
    Función principal para ejecutar el análisis robusto de nubosidad
    """
    if data_filepath is None:
        data_filepath = os.path.join(paths.BASE_INPUT_DIR, "refcells_data.csv")
    
    try:
        analyzer = CloudinessAnalyzer()
        results = analyzer.run_cloudiness_analysis(data_filepath, output_dir, model_type)
        
        print(f"\n✅ Análisis robusto de nubosidad completado exitosamente")
        print(f"📊 Resultados guardados en: {output_dir or os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, 'cloudiness_analysis_robust')}")
        print(f"🔬 Modelo utilizado: {model_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error ejecutando análisis de nubosidad: {e}")
        return False


if __name__ == "__main__":
    # Permitir ejecución directa
    success = run_cloudiness_analysis()
    
    if success:
        print("✅ Análisis robusto de nubosidad completado")
        sys.exit(0)
    else:
        print("❌ Error en el análisis de nubosidad")
        sys.exit(1) 