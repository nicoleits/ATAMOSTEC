"""
Analizador de Desviaciones Estadísticas para Sistema SOILING
Detecta anomalías y desviaciones en datos de sensores usando múltiples métodos estadísticos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

from config.logging_config import logger
from config import paths, settings
from utils.helpers import save_plot_matplotlib

# Configurar warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class StatisticalDeviationAnalyzer:
    """
    Analizador de desviaciones estadísticas para detectar anomalías en datos de sensores
    """
    
    def __init__(self):
        self.results = {}
        self.anomaly_summary = {}
        self.output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "statistical_deviations")
        self.csv_output_dir = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "statistical_deviations")
        
        # Crear directorios de salida
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.csv_output_dir, exist_ok=True)
        
        # Configurar estilo de gráficos
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def detect_z_score_anomalies(self, data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detecta anomalías usando Z-Score
        
        Args:
            data: Serie de datos a analizar
            threshold: Umbral de Z-Score (default: 3.0)
            
        Returns:
            Serie booleana indicando anomalías
        """
        if data.empty or data.isna().all():
            return pd.Series(dtype=bool, index=data.index)
            
        z_scores = np.abs(zscore(data.dropna()))
        anomalies = pd.Series(False, index=data.index)
        anomalies.loc[data.dropna().index] = z_scores > threshold
        
        return anomalies
    
    def detect_iqr_anomalies(self, data: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        Detecta anomalías usando método IQR (Interquartile Range)
        
        Args:
            data: Serie de datos a analizar
            factor: Factor multiplicador para IQR (default: 1.5)
            
        Returns:
            Serie booleana indicando anomalías
        """
        if data.empty or data.isna().all():
            return pd.Series(dtype=bool, index=data.index)
            
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        anomalies = (data < lower_bound) | (data > upper_bound)
        return anomalies.fillna(False)
    
    def detect_isolation_forest_anomalies(self, data: pd.Series, contamination: float = 0.1) -> pd.Series:
        """
        Detecta anomalías usando Isolation Forest
        
        Args:
            data: Serie de datos a analizar
            contamination: Proporción esperada de anomalías (default: 0.1)
            
        Returns:
            Serie booleana indicando anomalías
        """
        if data.empty or data.isna().all():
            return pd.Series(dtype=bool, index=data.index)
            
        # Preparar datos
        clean_data = data.dropna()
        if len(clean_data) < 10:  # Mínimo de datos necesarios
            return pd.Series(False, index=data.index)
            
        # Reshape para sklearn
        X = clean_data.values.reshape(-1, 1)
        
        # Aplicar Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        
        # Convertir a serie booleana
        anomalies = pd.Series(False, index=data.index)
        anomalies.loc[clean_data.index] = anomaly_labels == -1
        
        return anomalies
    
    def detect_dbscan_anomalies(self, data: pd.Series, eps: float = 0.5, min_samples: int = 5) -> pd.Series:
        """
        Detecta anomalías usando DBSCAN clustering
        
        Args:
            data: Serie de datos a analizar
            eps: Distancia máxima entre puntos en un cluster
            min_samples: Número mínimo de puntos para formar un cluster
            
        Returns:
            Serie booleana indicando anomalías
        """
        if data.empty or data.isna().all():
            return pd.Series(dtype=bool, index=data.index)
            
        # Preparar datos
        clean_data = data.dropna()
        if len(clean_data) < min_samples * 2:
            return pd.Series(False, index=data.index)
            
        # Normalizar datos
        scaler = StandardScaler()
        X = scaler.fit_transform(clean_data.values.reshape(-1, 1))
        
        # Aplicar DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        
        # Puntos con label -1 son anomalías
        anomalies = pd.Series(False, index=data.index)
        anomalies.loc[clean_data.index] = cluster_labels == -1
        
        return anomalies
    
    def detect_rolling_deviation_anomalies(self, data: pd.Series, window: int = 24, 
                                         threshold: float = 3.0) -> pd.Series:
        """
        Detecta anomalías usando desviación de media móvil
        
        Args:
            data: Serie de datos a analizar
            window: Ventana para media móvil (default: 24 horas)
            threshold: Umbral de desviación en desviaciones estándar
            
        Returns:
            Serie booleana indicando anomalías
        """
        if data.empty or data.isna().all():
            return pd.Series(dtype=bool, index=data.index)
            
        # Calcular media y desviación estándar móvil
        rolling_mean = data.rolling(window=window, center=True).mean()
        rolling_std = data.rolling(window=window, center=True).std()
        
        # Calcular desviación normalizada
        deviation = np.abs(data - rolling_mean) / rolling_std
        
        # Detectar anomalías
        anomalies = deviation > threshold
        return anomalies.fillna(False)
    
    def detect_seasonal_anomalies(self, data: pd.Series, period: int = 24) -> pd.Series:
        """
        Detecta anomalías considerando patrones estacionales
        
        Args:
            data: Serie de datos a analizar
            period: Período estacional (default: 24 para patrón diario)
            
        Returns:
            Serie booleana indicando anomalías
        """
        if data.empty or data.isna().all():
            return pd.Series(dtype=bool, index=data.index)
            
        # Agrupar por período (ej: hora del día)
        if isinstance(data.index, pd.DatetimeIndex):
            seasonal_groups = data.groupby(data.index.hour)
        else:
            seasonal_groups = data.groupby(data.index % period)
        
        anomalies = pd.Series(False, index=data.index)
        
        for group_key, group_data in seasonal_groups:
            if len(group_data) > 5:  # Mínimo de datos para análisis
                # Detectar anomalías dentro de cada grupo estacional
                group_anomalies = self.detect_z_score_anomalies(group_data, threshold=2.5)
                anomalies.loc[group_data.index] = group_anomalies
        
        return anomalies
    
    def analyze_sensor_data(self, data: pd.Series, sensor_name: str, 
                          methods: List[str] = None) -> Dict:
        """
        Analiza un sensor específico con múltiples métodos
        
        Args:
            data: Serie de datos del sensor
            sensor_name: Nombre del sensor
            methods: Lista de métodos a usar (default: todos)
            
        Returns:
            Diccionario con resultados del análisis
        """
        if methods is None:
            methods = ['z_score', 'iqr', 'isolation_forest', 'rolling_deviation', 'seasonal']
        
        logger.info(f"Analizando sensor {sensor_name} con {len(methods)} métodos")
        
        results = {
            'sensor_name': sensor_name,
            'data_points': len(data),
            'valid_points': data.count(),
            'anomalies': {},
            'statistics': {}
        }
        
        # Estadísticas básicas
        results['statistics'] = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'median': data.median(),
            'skewness': stats.skew(data.dropna()),
            'kurtosis': stats.kurtosis(data.dropna())
        }
        
        # Aplicar métodos de detección
        for method in methods:
            try:
                if method == 'z_score':
                    anomalies = self.detect_z_score_anomalies(data)
                elif method == 'iqr':
                    anomalies = self.detect_iqr_anomalies(data)
                elif method == 'isolation_forest':
                    anomalies = self.detect_isolation_forest_anomalies(data)
                elif method == 'dbscan':
                    anomalies = self.detect_dbscan_anomalies(data)
                elif method == 'rolling_deviation':
                    anomalies = self.detect_rolling_deviation_anomalies(data)
                elif method == 'seasonal':
                    anomalies = self.detect_seasonal_anomalies(data)
                else:
                    logger.warning(f"Método {method} no reconocido")
                    continue
                
                anomaly_count = anomalies.sum()
                anomaly_percentage = (anomaly_count / len(data)) * 100
                
                results['anomalies'][method] = {
                    'count': int(anomaly_count),
                    'percentage': round(anomaly_percentage, 2),
                    'indices': anomalies[anomalies].index.tolist(),
                    'values': data[anomalies].tolist()
                }
                
                logger.info(f"Método {method}: {anomaly_count} anomalías ({anomaly_percentage:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error en método {method} para sensor {sensor_name}: {e}")
                results['anomalies'][method] = {'error': str(e)}
        
        return results
    
    def create_anomaly_visualization(self, data: pd.Series, anomalies_dict: Dict, 
                                   sensor_name: str) -> None:
        """
        Crea visualización de anomalías detectadas
        
        Args:
            data: Serie de datos original
            anomalies_dict: Diccionario con anomalías por método
            sensor_name: Nombre del sensor
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Análisis de Anomalías - {sensor_name}', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Serie temporal con anomalías Z-Score
        ax1 = axes[0, 0]
        ax1.plot(data.index, data.values, 'b-', alpha=0.7, label='Datos originales')
        
        if 'z_score' in anomalies_dict:
            anomaly_indices = anomalies_dict['z_score'].get('indices', [])
            if anomaly_indices:
                anomaly_values = [data.loc[idx] for idx in anomaly_indices if idx in data.index]
                ax1.scatter(anomaly_indices, anomaly_values, color='red', s=50, 
                          label=f'Anomalías Z-Score ({len(anomaly_indices)})', zorder=5)
        
        ax1.set_title('Detección por Z-Score')
        ax1.set_ylabel('Valor del Sensor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Serie temporal con anomalías IQR
        ax2 = axes[0, 1]
        ax2.plot(data.index, data.values, 'b-', alpha=0.7, label='Datos originales')
        
        if 'iqr' in anomalies_dict:
            anomaly_indices = anomalies_dict['iqr'].get('indices', [])
            if anomaly_indices:
                anomaly_values = [data.loc[idx] for idx in anomaly_indices if idx in data.index]
                ax2.scatter(anomaly_indices, anomaly_values, color='orange', s=50, 
                          label=f'Anomalías IQR ({len(anomaly_indices)})', zorder=5)
        
        ax2.set_title('Detección por IQR')
        ax2.set_ylabel('Valor del Sensor')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Histograma con distribución
        ax3 = axes[1, 0]
        clean_data = data.dropna()
        ax3.hist(clean_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(clean_data.mean(), color='red', linestyle='--', 
                   label=f'Media: {clean_data.mean():.2f}')
        ax3.axvline(clean_data.mean() + 3*clean_data.std(), color='orange', linestyle='--', 
                   label=f'+3σ: {clean_data.mean() + 3*clean_data.std():.2f}')
        ax3.axvline(clean_data.mean() - 3*clean_data.std(), color='orange', linestyle='--', 
                   label=f'-3σ: {clean_data.mean() - 3*clean_data.std():.2f}')
        ax3.set_title('Distribución de Datos')
        ax3.set_xlabel('Valor del Sensor')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Comparación de métodos
        ax4 = axes[1, 1]
        methods = []
        counts = []
        percentages = []
        
        for method, result in anomalies_dict.items():
            if isinstance(result, dict) and 'count' in result:
                methods.append(method.replace('_', ' ').title())
                counts.append(result['count'])
                percentages.append(result['percentage'])
        
        if methods:
            bars = ax4.bar(methods, percentages, color=['red', 'orange', 'green', 'purple', 'brown'][:len(methods)])
            ax4.set_title('Comparación de Métodos de Detección')
            ax4.set_ylabel('Porcentaje de Anomalías (%)')
            ax4.tick_params(axis='x', rotation=45)
            
            # Añadir valores en las barras
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(count)}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
        
        # Formatear fechas si es necesario
        if isinstance(data.index, pd.DatetimeIndex):
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Guardar gráfico
        filename = f"anomalies_{sensor_name.lower().replace(' ', '_')}.png"
        save_plot_matplotlib(fig, filename, self.output_dir)
        
        plt.close(fig)
        logger.info(f"Gráfico de anomalías guardado: {filename}")
    
    def create_summary_report(self, all_results: Dict) -> None:
        """
        Crea un reporte resumen de todos los análisis
        
        Args:
            all_results: Diccionario con resultados de todos los sensores
        """
        # Crear DataFrame resumen
        summary_data = []
        
        for sensor_name, results in all_results.items():
            row = {
                'Sensor': sensor_name,
                'Puntos_Datos': results['data_points'],
                'Puntos_Válidos': results['valid_points'],
                'Media': round(results['statistics']['mean'], 3),
                'Desv_Std': round(results['statistics']['std'], 3),
                'Asimetría': round(results['statistics']['skewness'], 3),
                'Curtosis': round(results['statistics']['kurtosis'], 3)
            }
            
            # Añadir anomalías por método
            for method, anomaly_data in results['anomalies'].items():
                if isinstance(anomaly_data, dict) and 'count' in anomaly_data:
                    row[f'Anomalías_{method}'] = anomaly_data['count']
                    row[f'Porcentaje_{method}'] = anomaly_data['percentage']
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        
        # Guardar resumen en CSV
        summary_file = os.path.join(self.csv_output_dir, "anomaly_detection_summary.csv")
        df_summary.to_csv(summary_file, index=False)
        logger.info(f"Resumen guardado en: {summary_file}")
        
        # Crear gráfico resumen
        self.create_summary_visualization(df_summary)
    
    def create_summary_visualization(self, df_summary: pd.DataFrame) -> None:
        """
        Crea visualización resumen de todos los sensores
        
        Args:
            df_summary: DataFrame con resumen de anomalías
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Resumen de Análisis de Anomalías - Todos los Sensores', 
                     fontsize=16, fontweight='bold')
        
        # Gráfico 1: Número total de anomalías por sensor
        ax1 = axes[0, 0]
        anomaly_cols = [col for col in df_summary.columns if col.startswith('Anomalías_')]
        if anomaly_cols:
            df_anomalies = df_summary[['Sensor'] + anomaly_cols].set_index('Sensor')
            df_anomalies.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title('Número de Anomalías por Sensor y Método')
            ax1.set_ylabel('Número de Anomalías')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Porcentaje de anomalías por sensor
        ax2 = axes[0, 1]
        percentage_cols = [col for col in df_summary.columns if col.startswith('Porcentaje_')]
        if percentage_cols:
            df_percentages = df_summary[['Sensor'] + percentage_cols].set_index('Sensor')
            df_percentages.plot(kind='bar', ax=ax2, width=0.8)
            ax2.set_title('Porcentaje de Anomalías por Sensor y Método')
            ax2.set_ylabel('Porcentaje de Anomalías (%)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.tick_params(axis='x', rotation=45)
        
        # Gráfico 3: Estadísticas descriptivas
        ax3 = axes[1, 0]
        stats_cols = ['Media', 'Desv_Std', 'Asimetría', 'Curtosis']
        available_stats = [col for col in stats_cols if col in df_summary.columns]
        
        if available_stats:
            df_stats = df_summary[['Sensor'] + available_stats].set_index('Sensor')
            # Normalizar para mejor visualización
            df_stats_norm = (df_stats - df_stats.mean()) / df_stats.std()
            df_stats_norm.plot(kind='bar', ax=ax3, width=0.8)
            ax3.set_title('Estadísticas Descriptivas Normalizadas')
            ax3.set_ylabel('Valor Normalizado')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
        
        # Gráfico 4: Calidad de datos
        ax4 = axes[1, 1]
        if 'Puntos_Datos' in df_summary.columns and 'Puntos_Válidos' in df_summary.columns:
            df_summary['Porcentaje_Válidos'] = (df_summary['Puntos_Válidos'] / 
                                               df_summary['Puntos_Datos']) * 100
            
            bars = ax4.bar(df_summary['Sensor'], df_summary['Porcentaje_Válidos'], 
                          color='lightgreen', alpha=0.7)
            ax4.set_title('Calidad de Datos por Sensor')
            ax4.set_ylabel('Porcentaje de Datos Válidos (%)')
            ax4.set_ylim(0, 105)
            ax4.tick_params(axis='x', rotation=45)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, df_summary['Porcentaje_Válidos']):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        filename = "anomaly_detection_summary.png"
        save_plot_matplotlib(fig, filename, self.output_dir)
        
        plt.close(fig)
        logger.info(f"Gráfico resumen guardado: {filename}")

def analyze_all_sensors_deviations() -> Dict:
    """
    Función principal para analizar desviaciones en todos los sensores disponibles
    
    Returns:
        Diccionario con resultados de todos los análisis
    """
    logger.info("=== INICIANDO ANÁLISIS DE DESVIACIONES ESTADÍSTICAS ===")
    
    analyzer = StatisticalDeviationAnalyzer()
    all_results = {}
    
    # Definir archivos de datos y sensores a analizar
    data_sources = {
        'DustIQ_C11': {
            'file': os.path.join(paths.BASE_INPUT_DIR, 'raw_dustiq_data.csv'),
            'time_col': '_time',
            'value_col': 'SR_C11_Avg'
        },
        'DustIQ_C12': {
            'file': os.path.join(paths.BASE_INPUT_DIR, 'raw_dustiq_data.csv'),
            'time_col': '_time',
            'value_col': 'SR_C12_Avg'
        },
        'RefCell_410': {
            'file': os.path.join(paths.BASE_INPUT_DIR, 'refcells_data.csv'),
            'time_col': '_time',
            'value_col': '1RC410(w.m-2)'
        },
        'RefCell_411': {
            'file': os.path.join(paths.BASE_INPUT_DIR, 'refcells_data.csv'),
            'time_col': '_time',
            'value_col': '1RC411(w.m-2)'
        },
        'RefCell_412': {
            'file': os.path.join(paths.BASE_INPUT_DIR, 'refcells_data.csv'),
            'time_col': '_time',
            'value_col': '1RC412(w.m-2)'
        },
        'SoilingKit_Isc_Exposed': {
            'file': os.path.join(paths.BASE_INPUT_DIR, 'soiling_kit_raw_data.csv'),
            'time_col': '_time',
            'value_col': 'Isc(e)'
        },
        'SoilingKit_Isc_Protected': {
            'file': os.path.join(paths.BASE_INPUT_DIR, 'soiling_kit_raw_data.csv'),
            'time_col': '_time',
            'value_col': 'Isc(p)'
        }
    }
    
    # Analizar cada sensor
    for sensor_name, config in data_sources.items():
        try:
            logger.info(f"Procesando sensor: {sensor_name}")
            
            # Cargar datos
            if not os.path.exists(config['file']):
                logger.warning(f"Archivo no encontrado: {config['file']}")
                continue
            
            df = pd.read_csv(config['file'])
            
            # Procesar columna de tiempo
            if config['time_col'] in df.columns:
                df[config['time_col']] = pd.to_datetime(df[config['time_col']], errors='coerce')
                df.set_index(config['time_col'], inplace=True)
            
            # Extraer serie del sensor
            if config['value_col'] not in df.columns:
                logger.warning(f"Columna {config['value_col']} no encontrada en {config['file']}")
                continue
            
            sensor_data = df[config['value_col']].copy()
            sensor_data = pd.to_numeric(sensor_data, errors='coerce')
            
            # Filtrar datos válidos
            sensor_data = sensor_data.dropna()
            
            if len(sensor_data) < 100:  # Mínimo de datos para análisis
                logger.warning(f"Datos insuficientes para sensor {sensor_name}: {len(sensor_data)}")
                continue
            
            # Realizar análisis
            results = analyzer.analyze_sensor_data(sensor_data, sensor_name)
            all_results[sensor_name] = results
            
            # Crear visualización
            analyzer.create_anomaly_visualization(sensor_data, results['anomalies'], sensor_name)
            
        except Exception as e:
            logger.error(f"Error procesando sensor {sensor_name}: {e}")
            continue
    
    # Crear reporte resumen
    if all_results:
        analyzer.create_summary_report(all_results)
        logger.info(f"Análisis completado para {len(all_results)} sensores")
    else:
        logger.warning("No se pudieron analizar sensores")
    
    return all_results

def run_analysis():
    """
    Función de entrada para el analizador de desviaciones estadísticas
    """
    try:
        results = analyze_all_sensors_deviations()
        logger.info("=== ANÁLISIS DE DESVIACIONES ESTADÍSTICAS COMPLETADO ===")
        return results
    except Exception as e:
        logger.error(f"Error en análisis de desviaciones estadísticas: {e}")
        return {}

if __name__ == "__main__":
    run_analysis() 