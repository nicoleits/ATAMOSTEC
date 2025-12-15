import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging
from datetime import datetime
import numpy as np
import sys

# Agregar el directorio padre al path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings, paths

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ref_cells_photocells():
    """Cargar datos SR Photocells de RefCells (1RC411)"""
    try:
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "ref_cells", "ref_cells_sr_semanal_q25.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Convertir la columna de tiempo a datetime (puede ser 'timestamp' o '_time')
            time_col = 'timestamp' if 'timestamp' in df.columns else '_time'
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
            
            # Buscar la columna que corresponde a 1RC411 (Photocells)
            photocells_col = None
            for col in df.columns:
                if '1RC411' in col or 'photocells' in col.lower():
                    photocells_col = col
                    break
            
            if photocells_col:
                result_df = pd.DataFrame({photocells_col: df[photocells_col]})
                logger.info(f"Datos RefCells Photocells cargados: {result_df.shape}")
                return result_df
            else:
                logger.warning("No se encontró columna de Photocells en RefCells")
                return pd.DataFrame()
        else:
            logger.warning(f"Archivo no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos RefCells Photocells: {e}")
        return pd.DataFrame()

def load_dustiq_mediodia_solar():
    """Cargar datos DustIQ Q25 Semanal (mediodía solar)"""
    try:
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "dustiq", "dustiq_sr_mediodia_semanal_q25.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"Archivo DustIQ encontrado: {csv_path}")
            logger.info(f"Columnas disponibles: {list(df.columns)}")
            logger.info(f"Primeras filas: {df.head()}")
            
            # Buscar la columna de tiempo correcta
            time_col = None
            for col in ['timestamp', '_time', 'time', 'date']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col:
                # Convertir la columna de tiempo a datetime
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                logger.info(f"Datos DustIQ mediodía solar cargados: {df.shape}")
                logger.info(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
                return df
            else:
                logger.error(f"No se encontró columna de tiempo en DustIQ. Columnas disponibles: {list(df.columns)}")
                return pd.DataFrame()
        else:
            logger.warning(f"Archivo no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos DustIQ mediodía solar: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def load_soiling_kit_raw_q25():
    """Cargar datos Soiling Kit SR Raw Q25 Semanal"""
    try:
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "soiling_kit", "soiling_kit_sr_raw_weekly_q25.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Convertir la columna de tiempo a datetime
            df['Original_Timestamp_Col'] = pd.to_datetime(df['Original_Timestamp_Col'])
            df.set_index('Original_Timestamp_Col', inplace=True)
            logger.info(f"Datos Soiling Kit Raw Q25 cargados: {df.shape}")
            return df
        else:
            logger.warning(f"Archivo no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos Soiling Kit Raw: {e}")
        return pd.DataFrame()

def load_pvstand_solar_noon_corrected():
    """Cargar y procesar datos semanales Q25 de PVStand (SR_Pmax_Corrected_Raw_NoOffset y SR_Isc_Corrected_Raw_NoOffset) igual que el gráfico individual"""
    try:
        # Intentar primero la ruta original, luego la ruta estándar
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "pv_stand", "solar_noon", "pvstand_sr_raw_no_offset_solar_noon.csv")
        if not os.path.exists(csv_path):
            # Si no existe, usar la ruta estándar
            csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "pv_stand", "pvstand_sr_raw_no_offset.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Buscar columna de tiempo
            time_col = None
            for col in ['_time', 'timestamp', 'time', 'date']:
                if col in df.columns:
                    time_col = col
                    break
            if not time_col:
                logger.error("No se encontró columna de tiempo en PVStand raw")
                return pd.DataFrame()
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
            # Procesar ambas columnas
            result = {}
            for col_name, label in zip([
                'SR_Pmax_Corrected_Raw_NoOffset',
                'SR_Isc_Corrected_Raw_NoOffset'],
                ['SR_PVStand_Semanal_Q25_Pmax', 'SR_PVStand_Semanal_Q25_Isc']):
                if col_name in df.columns:
                    series = df[col_name].dropna()
                    # Filtrar SR >= 80
                    series = series[series >= 80.0]
                    if not series.empty:
                        # Calcular Q25 semanal
                        weekly_q25 = series.resample('1W').quantile(0.25).dropna()
                        # Eliminar solo la primera semana
                        weekly_q25_trimmed = weekly_q25.iloc[1:].copy()
                        if not weekly_q25_trimmed.empty:
                            # Normalizar al primer valor tras el recorte
                            first_value = weekly_q25_trimmed.iloc[0]
                            if first_value > 0:
                                weekly_normalized = (weekly_q25_trimmed / first_value) * 100.0
                                result[label] = weekly_normalized
            if result:
                result_df = pd.DataFrame(result)
                logger.info(f"Datos PVStand semanal Q25 (Pmax e Isc) cargados: {result_df.shape}")
                return result_df
            else:
                logger.warning("No se encontraron datos válidos de PVStand para Pmax o Isc")
                return pd.DataFrame()
        else:
            logger.warning(f"Archivo no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos PVStand semanal Q25 (raw): {e}")
        return pd.DataFrame()

def load_iv600_both_curves():
    """Cargar ambas curvas de IV600 (Pmax e Isc)"""
    try:
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "iv600", "iv600_sr_semanal_q25.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Verificar qué columna de tiempo existe
            time_col = None
            for col in ['timestamp', '_time', 'time', 'date']:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col:
                # Convertir la columna de tiempo a datetime
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                logger.info(f"Datos IV600 cargados: {df.shape}")
                return df
            else:
                logger.error("No se encontró columna de tiempo en IV600")
                return pd.DataFrame()
        else:
            logger.warning(f"Archivo no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos IV600: {e}")
        return pd.DataFrame()

def normalize_series_to_100(series, name=""):
    """Normalizar serie al primer valor (100%)"""
    if series.empty:
        return series
    
    # Encontrar el primer valor no-NaN
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return series
    
    first_value = series[first_valid_idx]
    if first_value == 0:
        logger.warning(f"Primer valor de {name} es 0, no se puede normalizar")
        return series
    
    normalized = series * (100 / first_value)
    logger.info(f"Serie {name} normalizada: primer valor {first_value:.2f} → 100.00")
    return normalized

def create_consolidated_weekly_q25_plot():
    """Crear gráfico consolidado con las curvas específicas solicitadas"""
    
    logger.info("Iniciando generación de gráfico consolidado con curvas específicas...")
    
    # Cargar todos los datos específicos
    logger.info("=== CARGANDO DATOS ===")
    
    ref_cells_data = load_ref_cells_photocells()
    logger.info(f"RefCells cargados: {not ref_cells_data.empty} - Shape: {ref_cells_data.shape if not ref_cells_data.empty else 'N/A'}")
    
    dustiq_data = load_dustiq_mediodia_solar()
    logger.info(f"DustIQ cargados: {not dustiq_data.empty} - Shape: {dustiq_data.shape if not dustiq_data.empty else 'N/A'}")
    if not dustiq_data.empty:
        logger.info(f"DustIQ columnas: {list(dustiq_data.columns)}")
        logger.info(f"DustIQ rango fechas: {dustiq_data.index.min()} a {dustiq_data.index.max()}")
    
    soiling_kit_data = load_soiling_kit_raw_q25()
    logger.info(f"Soiling Kit cargados: {not soiling_kit_data.empty} - Shape: {soiling_kit_data.shape if not soiling_kit_data.empty else 'N/A'}")
    
    pvstand_data = load_pvstand_solar_noon_corrected()
    logger.info(f"PVStand cargados: {not pvstand_data.empty} - Shape: {pvstand_data.shape if not pvstand_data.empty else 'N/A'}")
    
    iv600_data = load_iv600_both_curves()
    logger.info(f"IV600 cargados: {not iv600_data.empty} - Shape: {iv600_data.shape if not iv600_data.empty else 'N/A'}")
    
    logger.info("=== FIN CARGA DE DATOS ===")
    
    # Verificar si hay datos para graficar
    if (ref_cells_data.empty and dustiq_data.empty and 
        soiling_kit_data.empty and pvstand_data.empty and iv600_data.empty):
        logger.warning("No se encontraron datos para graficar")
        return False
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Colores intermedios/pastel para cada tipo de análisis
    colors = {
        'ref_cells': '#4FA3DF',      # Azul intermedio
        'dustiq': '#FFB84D',         # Naranja pastel
        'soiling_kit': '#6FCF97',    # Verde pastel
        'pvstand_isc': '#FF7F7F',    # Rojo pastel
        'pvstand_pmax': '#B39DDB',   # Púrpura pastel
        'iv600_pmax': '#C2B280',     # Marrón pastel
        'iv600_isc': '#F48FB1'       # Rosa pastel
    }
    
    plotted_series = []
    
    # Graficar RefCells Photocells
    if not ref_cells_data.empty:
        for col in ref_cells_data.columns:
            series = ref_cells_data[col].dropna()
            if not series.empty:
                series_norm = normalize_series_to_100(series, f"RefCells {col}")
                ax.plot(series_norm.index, series_norm.values, 'o-', color=colors['ref_cells'], 
                       alpha=0.9, linewidth=2, markersize=6, 
                       label=f'RefCells - Photocells')
                plotted_series.append(f'RefCells - Photocells')
    
    # Graficar DustIQ Mediodía Solar
    logger.info("=== GRAFICANDO DUSTIQ ===")
    if not dustiq_data.empty:
        logger.info(f"DustIQ tiene {len(dustiq_data.columns)} columnas: {list(dustiq_data.columns)}")
        for col in dustiq_data.columns:
            series = dustiq_data[col].dropna()
            logger.info(f"Columna {col}: {len(series)} puntos no-nulos")
            if not series.empty:
                logger.info(f"Serie {col} - Primer valor: {series.iloc[0]:.2f}, Último valor: {series.iloc[-1]:.2f}")
                series_norm = normalize_series_to_100(series, f"DustIQ {col}")
                ax.plot(series_norm.index, series_norm.values, 'o-', color=colors['dustiq'], 
                       alpha=0.9, linewidth=2, markersize=6, 
                       label='DustIQ')
                plotted_series.append('DustIQ - Q25 Semanal')
                logger.info(f"DustIQ graficado exitosamente")
            else:
                logger.warning(f"Serie {col} está vacía después de dropna()")
    else:
        logger.warning("DustIQ está vacío, no se puede graficar")
    logger.info("=== FIN GRAFICADO DUSTIQ ===")
    
    # Graficar Soiling Kit Raw Q25
    if not soiling_kit_data.empty:
        for col in soiling_kit_data.columns:
            series = soiling_kit_data[col].dropna()
            if not series.empty:
                series_norm = normalize_series_to_100(series, f"Soiling Kit {col}")
                ax.plot(series_norm.index, series_norm.values, 'o-', color=colors['soiling_kit'], 
                       alpha=0.9, linewidth=2, markersize=6, 
                       label='Soiling Kit')
                plotted_series.append('Soiling Kit - SR Raw Q25')
    
    # Graficar PVStand Semanal Q25 (ambas curvas)
    if not pvstand_data.empty:
        color_map = {'SR_PVStand_Semanal_Q25_Pmax': colors['pvstand_pmax'],
                     'SR_PVStand_Semanal_Q25_Isc': colors['pvstand_isc']}
        label_map = {'SR_PVStand_Semanal_Q25_Pmax': 'PVStand Pmax',
                     'SR_PVStand_Semanal_Q25_Isc': 'PVStand Isc'}
        for col in pvstand_data.columns:
            series = pvstand_data[col].dropna()
            if not series.empty:
                series_norm = normalize_series_to_100(series, label_map.get(col, col))
                ax.plot(series_norm.index, series_norm.values, 'o-', color=color_map.get(col, None),
                       alpha=0.9, linewidth=2, markersize=6,
                       label=label_map.get(col, col))
                plotted_series.append(label_map.get(col, col))
    
    # Graficar IV600 ambas curvas
    if not iv600_data.empty:
        for i, col in enumerate(iv600_data.columns):
            series = iv600_data[col].dropna()
            if not series.empty:
                series_norm = normalize_series_to_100(series, f"{col}")
                color = colors['iv600_pmax'] if 'Pmax' in col or 'Pmp' in col else colors['iv600_isc']
                ax.plot(series_norm.index, series_norm.values, 'o-', color=color, 
                       alpha=0.9, linewidth=2, markersize=6, 
                       label=f'{col}')
                plotted_series.append(f'{col}')
    
    # Configurar el gráfico
    if plotted_series:
        ax.set_title('Intercomparison Soiling Ratio Q25', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=15, frameon=True, fancybox=True, framealpha=0.8)
        ax.set_ylim([90, 110])
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Soiling Ratio [%]', fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        
        # Formatear eje X - Usar el patrón exitoso de dustiq_analyzer.py
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=15)
        
        # Establecer límites del eje x dinámicamente (patrón exacto de dustiq_analyzer.py)
        start_date_for_xlim = pd.Timestamp('2024-07-23')
        # Encontrar la fecha máxima real de los datos disponibles
        max_available_date = pd.Timestamp('2025-07-31')  # Valor por defecto
        
        for series_data in [ref_cells_data, dustiq_data, soiling_kit_data, pvstand_data, iv600_data]:
            if not series_data.empty:
                series_max = series_data.index.max()
                if hasattr(series_max, 'tz') and series_max.tz is not None:
                    series_max = series_max.tz_localize(None)
                if series_max > max_available_date:
                    max_available_date = series_max
        
        # Usar la fecha máxima de los datos reales, pero nunca exceder el final del año actual
        end_date_for_xlim = min(max_available_date, pd.Timestamp('2025-12-31'))
        ax.set_xlim([start_date_for_xlim, end_date_for_xlim])
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar gráfico
        output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "consolidados")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = "consolidated_weekly_q25_no_trend.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico consolidado guardado en: {plot_path}")
        
        if settings.SHOW_FIGURES:
            plt.show()
        
        plt.close()
        return True
    else:
        logger.warning("No se pudieron graficar series de datos")
        return False

def create_synchronized_weekly_q25_plot():
    """Crear gráfico consolidado con curvas sincronizadas que parten en la misma fecha e inician en 100%"""
    
    logger.info("Iniciando generación de gráfico consolidado sincronizado...")
    
    # Cargar todos los datos específicos
    ref_cells_data = load_ref_cells_photocells()
    dustiq_data = load_dustiq_mediodia_solar()
    soiling_kit_data = load_soiling_kit_raw_q25()
    pvstand_data = load_pvstand_solar_noon_corrected()
    iv600_data = load_iv600_both_curves()
    
    # Verificar si hay datos para graficar
    if (ref_cells_data.empty and dustiq_data.empty and 
        soiling_kit_data.empty and pvstand_data.empty and iv600_data.empty):
        logger.warning("No se encontraron datos para graficar")
        return False
    
    # Recolectar todas las series de datos
    all_series = {}
    
    # RefCells Photocells
    if not ref_cells_data.empty:
        for col in ref_cells_data.columns:
            series = ref_cells_data[col].dropna()
            if not series.empty:
                all_series['RefCells - Photocells'] = series
    
    # DustIQ Mediodía Solar
    if not dustiq_data.empty:
        for col in dustiq_data.columns:
            series = dustiq_data[col].dropna()
            if not series.empty:
                all_series['DustIQ'] = series
    
    # Soiling Kit Raw Q25
    if not soiling_kit_data.empty:
        for col in soiling_kit_data.columns:
            series = soiling_kit_data[col].dropna()
            if not series.empty:
                all_series['Soiling Kit'] = series
    
    # PVStand Semanal Q25 (ambas curvas)
    if not pvstand_data.empty:
        label_map = {'SR_PVStand_Semanal_Q25_Pmax': 'PVStand SRPmax',
                     'SR_PVStand_Semanal_Q25_Isc': 'PVStand SRIsc'}
        for col in pvstand_data.columns:
            series = pvstand_data[col].dropna()
            if not series.empty:
                all_series[label_map.get(col, col)] = series
    
    # IV600 ambas curvas
    if not iv600_data.empty:
        for col in iv600_data.columns:
            series = iv600_data[col].dropna()
            if not series.empty:
                # Mapear nombres de columnas IV600 a los nuevos nombres
                if 'Pmax' in col:
                    new_name = 'IV Curve Tracer 2 SRPmax'
                elif 'Isc' in col:
                    new_name = 'IV Curve Tracer 2 SRIsc'
                else:
                    new_name = f'IV Curve Tracer 2 {col}'
                all_series[new_name] = series
    
    if not all_series:
        logger.warning("No se encontraron series válidas para sincronizar")
        return False
    
    # Encontrar la fecha de inicio más tardía (donde todas las series tienen datos)
    start_dates = []
    for series in all_series.values():
        if not series.empty:
            # Asegurar que la fecha sea tz-naive
            start_date = series.index[0]
            if start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            start_dates.append(start_date)
    
    if not start_dates:
        logger.warning("No se pudieron determinar fechas de inicio")
        return False
    
    # Encontrar la fecha más tardía
    latest_start_date = max(start_dates)
    logger.info(f"Fecha de inicio sincronizada: {latest_start_date}")
    
    # Definir fecha de fin (primera semana de marzo)
    end_date = pd.Timestamp('2025-03-05')  # Primera semana de marzo
    logger.info(f"Fecha de fin establecida: {end_date}")
    
    # Recortar todas las series desde la fecha más tardía hasta la primera semana de enero
    synchronized_series = {}
    for name, series in all_series.items():
        # Asegurar que el índice sea tz-naive (sin zona horaria)
        if series.index.tz is not None:
            series = series.copy()
            series.index = series.index.tz_localize(None)
        
        # Filtrar desde la fecha más tardía hasta la fecha de fin
        filtered_series = series[(series.index >= latest_start_date) & (series.index <= end_date)]
        if not filtered_series.empty:
            # Normalizar al primer valor (100%)
            first_value = filtered_series.iloc[0]
            if first_value > 0:
                normalized_series = (filtered_series / first_value) * 100.0
                synchronized_series[name] = normalized_series
                logger.info(f"Serie {name} sincronizada: {len(normalized_series)} puntos, primer valor: {first_value:.2f} → 100.00, último valor: {normalized_series.iloc[-1]:.2f}")
    
    if not synchronized_series:
        logger.warning("No se pudieron sincronizar las series")
        return False
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Colores pasteles más oscuros para cada tipo de análisis
    colors = {
        'ref_cells': '#3B82F6',      # Azul pastel más oscuro
        'dustiq': '#F97316',         # Naranja pastel más oscuro
        'soiling_kit': '#22C55E',    # Verde pastel más oscuro
        'pvstand_isc': '#EF4444',    # Rojo pastel más oscuro
        'pvstand_pmax': '#8B5CF6',   # Púrpura pastel más oscuro
        'iv600_pmax': '#B45309',     # Marrón pastel más oscuro
        'iv600_isc': '#EC4899'       # Rosa pastel más oscuro
    }
    
    # Graficar todas las series sincronizadas
    for name, series in synchronized_series.items():
        # Debug: mostrar las primeras fechas de cada serie
        logger.info(f"Serie {name} - Primera fecha: {series.index[0]}, Última fecha: {series.index[-1]}")
        # Determinar color basado en el nombre
        if 'Photocells' in name:
            color = colors['ref_cells']
        elif 'DustIQ' in name:
            color = colors['dustiq']
        elif 'Soiling Kit' in name:
            color = colors['soiling_kit']
        elif 'Tracer 1' in name and 'SRIsc' in name:
            color = colors['pvstand_isc']
        elif 'Tracer 1' in name and 'SRPmax' in name:
            color = colors['pvstand_pmax']
        elif 'Tracer 2' in name and 'SRPmax' in name:
            color = colors['iv600_pmax']
        elif 'Tracer 2' in name and 'SRIsc' in name:
            color = colors['iv600_isc']
        else:
            color = '#000000'  # Negro por defecto
        
        # Usar ax.plot para asegurar fechas correctas en el eje X
        ax.plot(series.index, series.values, 'o--', color=color, 
                alpha=0.9, linewidth=2, markersize=8, 
                label=name, marker='o')
    
    # Configurar el gráfico
    ax.set_title('Soiling Ratio - Intercomparison', fontsize=28)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=20, frameon=True, fancybox=True, framealpha=0.8)
    ax.set_ylim([90, 110])
    ax.set_xlabel('Date', fontsize=24)
    ax.set_ylabel('Soiling Ratio [%]', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    
    # Formatear eje X - Usar el patrón exitoso de dustiq_analyzer.py
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20)
    
    # Establecer límites del eje x dinámicamente (patrón de dustiq_analyzer.py)
    # Obtener el rango de fechas de los datos
    all_dates = []
    for series in synchronized_series.values():
        all_dates.extend(series.index.tolist())
    
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        logger.info(f"Rango de fechas en el gráfico: {min_date} a {max_date}")
        logger.info(f"Tipo de min_date: {type(min_date)}")
        logger.info(f"Tipo de max_date: {type(max_date)}")
        ax.set_xlim([min_date, max_date])
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar gráfico
    output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "consolidados")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = "consolidated_weekly_q25_synchronized.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    logger.info(f"Gráfico consolidado sincronizado guardado en: {plot_path}")
    
    if settings.SHOW_FIGURES:
        plt.show()
    
    plt.close()
    return True

def load_pv_glasses_promedios():
    """Cargar promedios de PV glasses por período desde el análisis tradicional"""
    logger.info("Cargando promedios de PV glasses tradicional...")
    
    try:
        # Ruta al archivo de soiling ratios tradicional de PV glasses
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "seleccion_irradiancia_con_sr.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"Archivo de PV glasses tradicional no encontrado: {csv_path}")
            return pd.DataFrame()
        
        # Cargar datos
        df = pd.read_csv(csv_path, parse_dates=['_time'])
        
        if df.empty:
            logger.warning("Datos de PV glasses tradicional vacíos")
            return pd.DataFrame()
        
        logger.info(f"Datos de PV glasses tradicional cargados: {df.shape}")
        
        # Calcular promedios por período
        promedios_por_periodo = calcular_promedios_por_periodo_pv_glasses(df)
        
        return promedios_por_periodo
        
    except Exception as e:
        logger.error(f"Error cargando datos PV glasses tradicional: {e}")
        return pd.DataFrame()

def calcular_promedios_por_periodo_pv_glasses(df):
    """Calcula promedios individuales de cada FC por período y los asigna a fechas específicas del gráfico"""
    
    # Definir períodos y sus fechas objetivo en el gráfico
    periodos_fechas = {
        'semanal': pd.Timestamp('2024-07-29'),      # Primera semana
        '2 semanas': pd.Timestamp('2024-08-05'),    # Segunda semana  
        'Mensual': pd.Timestamp('2024-08-23'),      # Primer mes
        'Trimestral': pd.Timestamp('2024-10-23'),   # Primer trimestre
        'Cuatrimestral': pd.Timestamp('2024-11-23'), # Primer cuatrimestre
        'Semestral': pd.Timestamp('2025-01-23'),    # Primer semestre
        '1 año': pd.Timestamp('2025-07-23')         # Primer año
    }
    
    # Columnas SR de PV glasses y correspondencia con masas
    sr_columns = ['SR_R_FC3', 'SR_R_FC4', 'SR_R_FC5']
    correspondencia_masa = {
        'SR_R_FC3': 'Masa_C_Referencia',
        'SR_R_FC4': 'Masa_B_Referencia', 
        'SR_R_FC5': 'Masa_A_Referencia'
    }
    
    # Mapeo de nombres de FC para la leyenda
    fc_names = {
        'SR_R_FC3': 'FC3',
        'SR_R_FC4': 'FC4', 
        'SR_R_FC5': 'FC5'
    }
    
    resultados = []
    
    for periodo, fecha_objetivo in periodos_fechas.items():
        # Filtrar datos del período
        df_periodo = df[df['Periodo_Referencia'] == periodo].copy()
        
        if df_periodo.empty:
            logger.info(f"No hay datos para período {periodo}")
            continue
        
        # Calcular promedio individual para cada FC por separado
        for sr_col in sr_columns:
            if sr_col in df_periodo.columns:
                masa_col = correspondencia_masa[sr_col]
                
                if masa_col in df_periodo.columns:
                    # Filtrar solo filas donde la masa correspondiente > 0
                    df_filtrado = df_periodo[df_periodo[masa_col] > 0].copy()
                    
                    if not df_filtrado.empty and not df_filtrado[sr_col].isna().all():
                        # Calcular promedio individual para este FC
                        valores_validos = df_filtrado[sr_col].dropna().values
                        if len(valores_validos) > 0:
                            promedio_fc = valores_validos.mean() * 100
                            # Aplicar corrección de +7.5%
                            promedio_corregido = promedio_fc + 7.5
                            
                            fc_name = fc_names[sr_col]
                            
                            resultados.append({
                                'fecha': fecha_objetivo,
                                'valor': promedio_corregido,
                                'serie': f'PV Glasses {fc_name}',
                                'periodo': periodo,
                                'fc': fc_name
                            })
                            
                            logger.info(f"Período {periodo}, {fc_name}: promedio = {promedio_fc:.2f}%, corregido (+7.5%) = {promedio_corregido:.2f}% (basado en {len(valores_validos)} valores)")
    
    if resultados:
        df_resultado = pd.DataFrame(resultados)
        df_resultado.set_index('fecha', inplace=True)
        return df_resultado
    else:
        return pd.DataFrame()

def create_consolidated_weekly_q25_plot_with_pv_glasses():
    """Crear gráfico consolidado con las curvas específicas solicitadas + PV Glasses"""
    
    logger.info("Iniciando generación de gráfico consolidado con PV Glasses...")
    
    # Cargar todos los datos específicos
    logger.info("=== CARGANDO DATOS ===")
    
    ref_cells_data = load_ref_cells_photocells()
    logger.info(f"RefCells cargados: {not ref_cells_data.empty} - Shape: {ref_cells_data.shape if not ref_cells_data.empty else 'N/A'}")
    
    dustiq_data = load_dustiq_mediodia_solar()
    logger.info(f"DustIQ cargados: {not dustiq_data.empty} - Shape: {dustiq_data.shape if not dustiq_data.empty else 'N/A'}")
    if not dustiq_data.empty:
        logger.info(f"DustIQ columnas: {list(dustiq_data.columns)}")
        logger.info(f"DustIQ rango fechas: {dustiq_data.index.min()} a {dustiq_data.index.max()}")
    
    soiling_kit_data = load_soiling_kit_raw_q25()
    logger.info(f"Soiling Kit cargados: {not soiling_kit_data.empty} - Shape: {soiling_kit_data.shape if not soiling_kit_data.empty else 'N/A'}")
    
    pvstand_data = load_pvstand_solar_noon_corrected()
    logger.info(f"PVStand cargados: {not pvstand_data.empty} - Shape: {pvstand_data.shape if not pvstand_data.empty else 'N/A'}")
    
    iv600_data = load_iv600_both_curves()
    logger.info(f"IV600 cargados: {not iv600_data.empty} - Shape: {iv600_data.shape if not iv600_data.empty else 'N/A'}")
    
    pv_glasses_data = load_pv_glasses_promedios()
    logger.info(f"PV Glasses cargados: {not pv_glasses_data.empty} - Shape: {pv_glasses_data.shape if not pv_glasses_data.empty else 'N/A'}")
    
    logger.info("=== FIN CARGA DE DATOS ===")
    
    # Verificar si hay datos para graficar
    if (ref_cells_data.empty and dustiq_data.empty and 
        soiling_kit_data.empty and pvstand_data.empty and iv600_data.empty and pv_glasses_data.empty):
        logger.warning("No se encontraron datos para graficar")
        return False
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Colores intermedios/pastel para cada tipo de análisis
    colors = {
        'ref_cells': '#8B4513',      # Marrón saddle
        'dustiq': '#FF8C00',         # Naranja oscuro
        'soiling_kit': '#2E8B57',    # Verde mar
        'pvstand_pmax': '#4169E1',   # Azul real
        'pvstand_isc': '#8A2BE2',    # Azul violeta
        'iv600_pmax': '#DC143C',     # Carmesí
        'iv600_isc': '#FF1493',      # Rosa profundo
        # Colores para PV glasses (tonos verdes/amarillos para diferenciarse)
        'pv_glasses_fc3': '#228B22',  # Verde bosque
        'pv_glasses_fc4': '#32CD32',  # Verde lima  
        'pv_glasses_fc5': '#9ACD32'   # Verde amarillento
    }
    
    plotted_series = []
    
    # Graficar RefCells Photocells
    if not ref_cells_data.empty:
        for col in ref_cells_data.columns:
            series = ref_cells_data[col].dropna()
            if not series.empty:
                # Normalizar a 100% en el primer punto válido
                normalized_series = normalize_series_to_100(series, f"RefCells {col}")
                if not normalized_series.empty:
                    ax.plot(normalized_series.index, normalized_series.values, 
                           color=colors['ref_cells'], linewidth=2,
                           label='RefCells - Photocells')
                    plotted_series.append('RefCells - Photocells')
    
    # Graficar DustIQ Mediodía Solar
    if not dustiq_data.empty:
        for col in dustiq_data.columns:
            series = dustiq_data[col].dropna()
            if not series.empty:
                # Normalizar a 100% en el primer punto válido
                normalized_series = normalize_series_to_100(series, f"DustIQ {col}")
                if not normalized_series.empty:
                    ax.plot(normalized_series.index, normalized_series.values, 
                           color=colors['dustiq'], linewidth=2,
                           label='DustIQ - Q25')
                    plotted_series.append('DustIQ - Q25')
    
    # Graficar Soiling Kit Raw Q25 Semanal
    if not soiling_kit_data.empty:
        for col in soiling_kit_data.columns:
            series = soiling_kit_data[col].dropna()
            if not series.empty:
                # Normalizar a 100% en el primer punto válido
                normalized_series = normalize_series_to_100(series, f"Soiling Kit Raw Q25 {col}")
                if not normalized_series.empty:
                    ax.plot(normalized_series.index, normalized_series.values, 
                           color=colors['soiling_kit'], linewidth=2,
                           label='Soiling Kit')
                    plotted_series.append('Soiling Kit')
    
    # Graficar PVStand Semanal Q25 (ambas curvas)
    if not pvstand_data.empty:
        label_map = {'SR_PVStand_Semanal_Q25_Pmax': 'PVStand Pmax',
                     'SR_PVStand_Semanal_Q25_Isc': 'PVStand Isc'}
        for col in pvstand_data.columns:
            series = pvstand_data[col].dropna()
            if not series.empty:
                # Normalizar a 100% en el primer punto válido
                normalized_series = normalize_series_to_100(series, f"PVStand {col}")
                if not normalized_series.empty:
                    color_key = 'pvstand_pmax' if 'Pmax' in col else 'pvstand_isc'
                    ax.plot(normalized_series.index, normalized_series.values, 
                           color=colors[color_key], linewidth=2,
                           label=label_map.get(col, col))
                    plotted_series.append(label_map.get(col, col))
    
    # Graficar IV600 ambas curvas
    if not iv600_data.empty:
        label_map = {'SR_Pmax_IV600': 'SR Pmax IV600',
                     'SR_Isc_IV600': 'SR Isc IV600'}
        for col in iv600_data.columns:
            series = iv600_data[col].dropna()
            if not series.empty:
                # IV600 mantiene sus valores originales (sin normalizar)
                color_key = 'iv600_pmax' if 'Pmax' in col else 'iv600_isc'
                ax.plot(series.index, series.values, 
                       color=colors[color_key], linewidth=2,
                       label=label_map.get(col, col))
                plotted_series.append(label_map.get(col, col))
    
    # Graficar PV Glasses como marcadores discretos
    if not pv_glasses_data.empty:
        # Agrupar por FC para graficar cada una por separado
        for fc in ['FC3', 'FC4', 'FC5']:
            fc_data = pv_glasses_data[pv_glasses_data['fc'] == fc]
            
            if not fc_data.empty:
                color_key = f'pv_glasses_{fc.lower()}'
                if color_key in colors:
                    ax.scatter(fc_data.index, fc_data['valor'], 
                              color=colors[color_key], s=100, marker='o',
                              label=f'PV Glasses {fc}', zorder=5)
                    plotted_series.append(f'PV Glasses {fc}')
    
    # Configurar el gráfico
    if plotted_series:
        ax.set_title('Intercomparison Soiling Ratio Q25 + PV Glasses', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=15, frameon=True, fancybox=True, framealpha=0.8)
        
        # Ajustar límites Y para incluir todos los datos de PV glasses
        y_min, y_max = 70, 110  # Rango ampliado para incluir valores de PV glasses
        if not pv_glasses_data.empty:
            pv_min, pv_max = pv_glasses_data['valor'].min(), pv_glasses_data['valor'].max()
            y_min = min(y_min, pv_min - 5)  # Margen inferior
            y_max = max(y_max, pv_max + 5)  # Margen superior
            logger.info(f"Límites Y ajustados: [{y_min:.1f}, {y_max:.1f}] para incluir PV glasses [{pv_min:.1f}, {pv_max:.1f}]")
        
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Soiling Ratio [%]', fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        
        # Formatear eje X - Usar el patrón exitoso de dustiq_analyzer.py
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=15)
        
        # Establecer límites del eje x dinámicamente (patrón exacto de dustiq_analyzer.py)
        start_date_for_xlim = pd.Timestamp('2024-07-23')
        # Encontrar la fecha máxima real de los datos disponibles
        max_available_date = pd.Timestamp('2025-07-31')  # Valor por defecto
        
        for series_data in [ref_cells_data, dustiq_data, soiling_kit_data, pvstand_data, iv600_data]:
            if not series_data.empty:
                series_max = series_data.index.max()
                if hasattr(series_max, 'tz') and series_max.tz is not None:
                    series_max = series_max.tz_localize(None)
                if series_max > max_available_date:
                    max_available_date = series_max
        
        # También considerar fechas de PV glasses
        if not pv_glasses_data.empty:
            pv_max = pv_glasses_data.index.max()
            if hasattr(pv_max, 'tz') and pv_max.tz is not None:
                pv_max = pv_max.tz_localize(None)
            if pv_max > max_available_date:
                max_available_date = pv_max
        
        # Usar la fecha máxima de los datos reales, pero nunca exceder el final del año actual
        end_date_for_xlim = min(max_available_date, pd.Timestamp('2025-12-31'))
        ax.set_xlim([start_date_for_xlim, end_date_for_xlim])
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar gráfico
        output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "consolidados")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = "consolidated_weekly_q25_with_pv_glasses.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico consolidado con PV Glasses guardado en: {plot_path}")
        
        if settings.SHOW_FIGURES:
            plt.show()
        
        plt.close()
        return True
    else:
        logger.warning("No se encontraron series para graficar")
        return False

def create_consolidated_weekly_q25_plot_oct_mar():
    """Crear gráfico consolidado desde Octubre 2024 hasta Marzo 2025, normalizando al 100% el primer dato de octubre"""
    
    logger.info("Iniciando generación de gráfico consolidado Octubre 2024 - Marzo 2025...")
    
    # Cargar todos los datos específicos
    logger.info("=== CARGANDO DATOS ===")
    
    ref_cells_data = load_ref_cells_photocells()
    logger.info(f"RefCells cargados: {not ref_cells_data.empty} - Shape: {ref_cells_data.shape if not ref_cells_data.empty else 'N/A'}")
    
    dustiq_data = load_dustiq_mediodia_solar()
    logger.info(f"DustIQ cargados: {not dustiq_data.empty} - Shape: {dustiq_data.shape if not dustiq_data.empty else 'N/A'}")
    if not dustiq_data.empty:
        logger.info(f"DustIQ columnas: {list(dustiq_data.columns)}")
        logger.info(f"DustIQ rango fechas: {dustiq_data.index.min()} a {dustiq_data.index.max()}")
    
    soiling_kit_data = load_soiling_kit_raw_q25()
    logger.info(f"Soiling Kit cargados: {not soiling_kit_data.empty} - Shape: {soiling_kit_data.shape if not soiling_kit_data.empty else 'N/A'}")
    
    pvstand_data = load_pvstand_solar_noon_corrected()
    logger.info(f"PVStand cargados: {not pvstand_data.empty} - Shape: {pvstand_data.shape if not pvstand_data.empty else 'N/A'}")
    
    iv600_data = load_iv600_both_curves()
    logger.info(f"IV600 cargados: {not iv600_data.empty} - Shape: {iv600_data.shape if not iv600_data.empty else 'N/A'}")
    
    pv_glasses_data = load_pv_glasses_promedios()
    logger.info(f"PV Glasses cargados: {not pv_glasses_data.empty} - Shape: {pv_glasses_data.shape if not pv_glasses_data.empty else 'N/A'}")
    
    logger.info("=== FIN CARGA DE DATOS ===")
    
    # Verificar si hay datos para graficar
    if (ref_cells_data.empty and dustiq_data.empty and 
        soiling_kit_data.empty and pvstand_data.empty and iv600_data.empty and pv_glasses_data.empty):
        logger.warning("No se encontraron datos para graficar")
        return False
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlabel('Date', fontsize=18)
    # Colores intermedios/pastel para cada tipo de análisis
    colors = {
        'ref_cells': '#8B4513',      # Marrón saddle
        'dustiq': '#FF8C00',         # Naranja oscuro
        'soiling_kit': '#2E8B57',    # Verde mar
        'pvstand_pmax': '#4169E1',   # Azul real
        'pvstand_isc': '#8A2BE2',    # Azul violeta
        'iv600_pmax': '#DC143C',     # Carmesí
        'iv600_isc': '#FF1493',      # Rosa profundo
        # Colores para PV glasses (tonos verdes/amarillos para diferenciarse)
        'pv_glasses_fc3': '#228B22',  # Verde bosque
        'pv_glasses_fc4': '#32CD32',  # Verde lima  
        'pv_glasses_fc5': '#9ACD32'   # Verde amarillento
    }
    
    plotted_series = []
    
    # Diccionario de etiquetas para la leyenda
    legend_labels = {
        'ref_cells': 'Photocells',
        'dustiq': 'DustIQ',
        'soiling_kit': 'Pilot',
        'pvstand_pmax': 'PVStand Pmax',
        'pvstand_isc': 'PVStand Isc',
        'iv600_pmax': 'SR_Pmax_IV600',
        'iv600_isc': 'SR_Isc_IV600'
    }
    
    # Función para filtrar y normalizar datos desde octubre
    def filter_and_normalize_from_october(data, data_name):
        if data.empty:
            return None
        
        # Filtrar solo datos desde octubre 2024
        oct_start = pd.Timestamp('2024-10-01')
        
        # Asegurar que el índice sea tz-naive para comparaciones
        data_copy = data.copy()
        if data_copy.index.tz is not None:
            data_copy.index = data_copy.index.tz_localize(None)
        
        filtered_data = data_copy[data_copy.index >= oct_start]
        
        if filtered_data.empty:
            logger.warning(f"No hay datos de {data_name} desde octubre 2024")
            return None
        
        # Normalizar al primer valor de octubre (100%)
        first_value = filtered_data.iloc[0]
        if first_value > 0:
            normalized_data = (filtered_data / first_value) * 100.0
            logger.info(f"{data_name} normalizado: primer valor {first_value:.2f} → 100.00, último valor: {normalized_data.iloc[-1]:.2f}")
            return normalized_data
        else:
            logger.warning(f"Primer valor de {data_name} es 0, no se puede normalizar")
            return None
    
    # Graficar RefCells Photocells
    if not ref_cells_data.empty:
        for col in ref_cells_data.columns:
            series = ref_cells_data[col].dropna()
            if not series.empty:
                normalized_series = filter_and_normalize_from_october(series, f"RefCells {col}")
                if normalized_series is not None:
                    ax.plot(normalized_series.index, normalized_series.values, 
                           color=colors['ref_cells'], linewidth=2,
                           label=legend_labels['ref_cells'])
                    plotted_series.append(legend_labels['ref_cells'])
    
    # Graficar DustIQ Mediodía Solar
    if not dustiq_data.empty:
        for col in dustiq_data.columns:
            series = dustiq_data[col].dropna()
            if not series.empty:
                normalized_series = filter_and_normalize_from_october(series, f"DustIQ {col}")
                if normalized_series is not None:
                    ax.plot(normalized_series.index, normalized_series.values, 
                           color=colors['dustiq'], linewidth=2,
                           label=legend_labels['dustiq'])
                    plotted_series.append(legend_labels['dustiq'])
    
    # Graficar Soiling Kit Raw Q25 Semanal
    if not soiling_kit_data.empty:
        for col in soiling_kit_data.columns:
            series = soiling_kit_data[col].dropna()
            if not series.empty:
                normalized_series = filter_and_normalize_from_october(series, f"Soiling Kit {col}")
                if normalized_series is not None:
                    ax.plot(normalized_series.index, normalized_series.values, 
                           color=colors['soiling_kit'], linewidth=2,
                           label=legend_labels['soiling_kit'])
                    plotted_series.append(legend_labels['soiling_kit'])
    
    # Graficar PVStand Semanal Q25 (ambas curvas)
    if not pvstand_data.empty:
        label_map = {'SR_PVStand_Semanal_Q25_Pmax': 'PVStand Pmax',
                     'SR_PVStand_Semanal_Q25_Isc': 'PVStand Isc'}
        for col in pvstand_data.columns:
            series = pvstand_data[col].dropna()
            if not series.empty:
                normalized_series = filter_and_normalize_from_october(series, f"PVStand {col}")
                if normalized_series is not None:
                    color_key = 'pvstand_pmax' if 'Pmax' in col else 'pvstand_isc'
                    ax.plot(normalized_series.index, normalized_series.values, 
                           color=colors[color_key], linewidth=2,
                           label=legend_labels[color_key])
                    plotted_series.append(legend_labels[color_key])
    
    # Graficar IV600 ambas curvas
    if not iv600_data.empty:
        label_map = {'SR_Pmax_IV600': 'SR_Pmax_IV600',
                     'SR_Isc_IV600': 'SR_Isc_IV600'}
        for col in iv600_data.columns:
            series = iv600_data[col].dropna()
            if not series.empty:
                # IV600 mantiene sus valores originales (sin normalizar)
                color_key = 'iv600_pmax' if 'Pmax' in col else 'iv600_isc'
                ax.plot(series.index, series.values, 
                       color=colors[color_key], linewidth=2,
                       label=label_map.get(col, col))
                plotted_series.append(label_map.get(col, col))
    
    # Graficar PV Glasses como marcadores discretos - Desplazados para empezar en octubre
    if not pv_glasses_data.empty:
        # Asegurar que el índice sea tz-naive para comparaciones
        pv_glasses_copy = pv_glasses_data.copy()
        if pv_glasses_copy.index.tz is not None:
            pv_glasses_copy.index = pv_glasses_copy.index.tz_localize(None)
        
        # Calcular el desplazamiento para que empiecen en octubre + 1 semana de desfase
        oct_start = pd.Timestamp('2024-10-01')
        one_week = pd.Timedelta(days=7)
        target_start = oct_start + one_week  # Octubre + 1 semana
        first_pv_date = pv_glasses_copy.index.min()
        time_offset = target_start - first_pv_date
        
        # Desplazar todos los datos de PV Glasses
        pv_glasses_displaced = pv_glasses_copy.copy()
        pv_glasses_displaced.index = pv_glasses_displaced.index + time_offset
        
        # Agrupar por FC para graficar cada una por separado (misma lógica que el gráfico exitoso)
        fc_labels = {'FC3': 'PV Glasses I', 'FC4': 'PV Glasses II', 'FC5': 'PV Glasses III'}
        for fc in ['FC3', 'FC4', 'FC5']:
            fc_data = pv_glasses_displaced[pv_glasses_displaced['fc'] == fc]
            
            if not fc_data.empty:
                color_key = f'pv_glasses_{fc.lower()}'
                if color_key in colors:
                    ax.scatter(fc_data.index, fc_data['valor'], 
                              color=colors[color_key], s=100, marker='o',
                              label=fc_labels[fc], zorder=5)
                    plotted_series.append(fc_labels[fc])
    
    # Configurar el gráfico
    if plotted_series:
#        ax.set_title('Intercomparison Soiling Ratio Q25 + PV Glasses - Octubre 2024 a Marzo 2025', fontsize=20, fontweight='bold')
        ax.set_title('Intercomparison Soiling Ratio', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=11, frameon=True, fancybox=True, framealpha=0.8)
        
        # Ajustar límites Y para incluir todos los datos de PV glasses
        y_min, y_max = 70, 110  # Rango ampliado para incluir valores de PV glasses
        if not pv_glasses_data.empty:
            pv_min, pv_max = pv_glasses_data['valor'].min(), pv_glasses_data['valor'].max()
            y_min = min(y_min, pv_min - 5)  # Margen inferior
            y_max = max(y_max, pv_max + 5)  # Margen superior
            logger.info(f"Límites Y ajustados: [{y_min:.1f}, {y_max:.1f}] para incluir PV glasses [{pv_min:.1f}, {pv_max:.1f}]")
        
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Soiling Ratio [%]', fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        
        # Formatear eje X - Usar el patrón exitoso de dustiq_analyzer.py
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=15)
        
        # Establecer límites del eje x al período específico: Octubre 2024 - Marzo 2025
        start_date_for_xlim = pd.Timestamp('2024-10-01')
        end_date_for_xlim = pd.Timestamp('2025-03-31')
        ax.set_xlim([start_date_for_xlim, end_date_for_xlim])
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar gráfico
        output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "consolidados")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = "consolidated_weekly_q25_oct_mar.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico consolidado Octubre-Marzo con PV Glasses guardado en: {plot_path}")
        
        if settings.SHOW_FIGURES:
            plt.show()
        
        plt.close()
        return True
    else:
        logger.warning("No se encontraron series para graficar")
        return False


def create_consolidated_weekly_q25_plot_monthly_labels():
    """Crear gráfico consolidado que finaliza el 01/06/2025 con etiquetas Month 1, Month 2, etc."""
    
    logger.info("Iniciando generación de gráfico consolidado con etiquetas mensuales...")
    
    # Cargar todos los datos específicos
    logger.info("=== CARGANDO DATOS ===")
    
    ref_cells_data = load_ref_cells_photocells()
    logger.info(f"RefCells cargados: {not ref_cells_data.empty} - Shape: {ref_cells_data.shape if not ref_cells_data.empty else 'N/A'}")
    
    dustiq_data = load_dustiq_mediodia_solar()
    logger.info(f"DustIQ cargados: {not dustiq_data.empty} - Shape: {dustiq_data.shape if not dustiq_data.empty else 'N/A'}")
    
    soiling_kit_data = load_soiling_kit_raw_q25()
    logger.info(f"Soiling Kit cargados: {not soiling_kit_data.empty} - Shape: {soiling_kit_data.shape if not soiling_kit_data.empty else 'N/A'}")
    
    pvstand_data = load_pvstand_solar_noon_corrected()
    logger.info(f"PVStand cargados: {not pvstand_data.empty} - Shape: {pvstand_data.shape if not pvstand_data.empty else 'N/A'}")
    
    iv600_data = load_iv600_both_curves()
    logger.info(f"IV600 cargados: {not iv600_data.empty} - Shape: {iv600_data.shape if not iv600_data.empty else 'N/A'}")
    
    pv_glasses_data = load_pv_glasses_promedios()
    logger.info(f"PV Glasses cargados: {not pv_glasses_data.empty} - Shape: {pv_glasses_data.shape if not pv_glasses_data.empty else 'N/A'}")
    
    logger.info("=== FIN CARGA DE DATOS ===")
    
    # Verificar si hay datos para graficar
    if (ref_cells_data.empty and dustiq_data.empty and 
        soiling_kit_data.empty and pvstand_data.empty and iv600_data.empty and pv_glasses_data.empty):
        logger.warning("No se encontraron datos para graficar")
        return False
    
    # Fecha de fin: 01/06/2025
    end_date = pd.Timestamp('2025-06-01')
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Colores intermedios/pastel para cada tipo de análisis
    colors = {
        'ref_cells': '#8B4513',      # Marrón saddle
        'dustiq': '#FF8C00',         # Naranja oscuro
        'soiling_kit': '#2E8B57',    # Verde mar
        'pvstand_pmax': '#4169E1',   # Azul real
        'pvstand_isc': '#8A2BE2',    # Azul violeta
        'iv600_pmax': '#DC143C',     # Carmesí
        'iv600_isc': '#FF1493',      # Rosa profundo
        # Colores para PV glasses (tonos verdes/amarillos para diferenciarse)
        'pv_glasses_fc3': '#228B22',  # Verde bosque
        'pv_glasses_fc4': '#32CD32',  # Verde lima  
        'pv_glasses_fc5': '#9ACD32'   # Verde amarillento
    }
    
    plotted_series = []
    
    # Función para filtrar datos hasta la fecha de fin
    def filter_to_end_date(data, data_name):
        if data.empty:
            return None
        
        # Asegurar que el índice sea tz-naive para comparaciones
        data_copy = data.copy()
        if data_copy.index.tz is not None:
            data_copy.index = data_copy.index.tz_localize(None)
        
        filtered_data = data_copy[data_copy.index <= end_date]
        
        if filtered_data.empty:
            logger.warning(f"No hay datos de {data_name} hasta {end_date}")
            return None
        
        return filtered_data
    
    # Graficar RefCells Photocells
    if not ref_cells_data.empty:
        for col in ref_cells_data.columns:
            series = ref_cells_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"RefCells {col}")
                if filtered_series is not None and not filtered_series.empty:
                    normalized_series = normalize_series_to_100(filtered_series, f"RefCells {col}")
                    if not normalized_series.empty:
                        ax.plot(normalized_series.index, normalized_series.values, 
                               color=colors['ref_cells'], linewidth=2,
                               label='RefCells - Photocells')
                        plotted_series.append('RefCells - Photocells')
    
    # Graficar DustIQ Mediodía Solar
    if not dustiq_data.empty:
        for col in dustiq_data.columns:
            series = dustiq_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"DustIQ {col}")
                if filtered_series is not None and not filtered_series.empty:
                    normalized_series = normalize_series_to_100(filtered_series, f"DustIQ {col}")
                    if not normalized_series.empty:
                        ax.plot(normalized_series.index, normalized_series.values, 
                               color=colors['dustiq'], linewidth=2,
                               label='DustIQ - Q25')
                        plotted_series.append('DustIQ - Q25')
    
    # Graficar Soiling Kit Raw Q25 Semanal
    if not soiling_kit_data.empty:
        for col in soiling_kit_data.columns:
            series = soiling_kit_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"Soiling Kit {col}")
                if filtered_series is not None and not filtered_series.empty:
                    normalized_series = normalize_series_to_100(filtered_series, f"Soiling Kit {col}")
                    if not normalized_series.empty:
                        ax.plot(normalized_series.index, normalized_series.values, 
                               color=colors['soiling_kit'], linewidth=2,
                               label='Soiling Kit')
                        plotted_series.append('Soiling Kit')
    
    # Graficar PVStand Semanal Q25 (ambas curvas)
    if not pvstand_data.empty:
        label_map = {'SR_PVStand_Semanal_Q25_Pmax': 'PVStand Pmax',
                     'SR_PVStand_Semanal_Q25_Isc': 'PVStand Isc'}
        for col in pvstand_data.columns:
            series = pvstand_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"PVStand {col}")
                if filtered_series is not None and not filtered_series.empty:
                    normalized_series = normalize_series_to_100(filtered_series, f"PVStand {col}")
                    if not normalized_series.empty:
                        color_key = 'pvstand_pmax' if 'Pmax' in col else 'pvstand_isc'
                        ax.plot(normalized_series.index, normalized_series.values, 
                               color=colors[color_key], linewidth=2,
                               label=label_map.get(col, col))
                        plotted_series.append(label_map.get(col, col))
    
    # Graficar IV600 ambas curvas
    if not iv600_data.empty:
        label_map = {'SR_Pmax_IV600': 'SR Pmax IV600',
                     'SR_Isc_IV600': 'SR Isc IV600'}
        for col in iv600_data.columns:
            series = iv600_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"IV600 {col}")
                if filtered_series is not None and not filtered_series.empty:
                    color_key = 'iv600_pmax' if 'Pmax' in col else 'iv600_isc'
                    ax.plot(filtered_series.index, filtered_series.values, 
                           color=colors[color_key], linewidth=2,
                           label=label_map.get(col, col))
                    plotted_series.append(label_map.get(col, col))
    
    # Graficar PV Glasses como marcadores discretos
    if not pv_glasses_data.empty:
        # Asegurar que el índice sea tz-naive para comparaciones
        pv_glasses_copy = pv_glasses_data.copy()
        if pv_glasses_copy.index.tz is not None:
            pv_glasses_copy.index = pv_glasses_copy.index.tz_localize(None)
        
        # Filtrar hasta la fecha de fin
        pv_glasses_filtered = pv_glasses_copy[pv_glasses_copy.index <= end_date]
        
        # Agrupar por FC para graficar cada una por separado
        for fc in ['FC3', 'FC4', 'FC5']:
            fc_data = pv_glasses_filtered[pv_glasses_filtered['fc'] == fc]
            
            if not fc_data.empty:
                color_key = f'pv_glasses_{fc.lower()}'
                if color_key in colors:
                    ax.scatter(fc_data.index, fc_data['valor'], 
                              color=colors[color_key], s=100, marker='o',
                              label=f'PV Glasses {fc}', zorder=5)
                    plotted_series.append(f'PV Glasses {fc}')
    
    # Configurar el gráfico
    if plotted_series:
        ax.set_title('Intercomparison Soiling Ratio Q25 + PV Glasses', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=15, frameon=True, fancybox=True, framealpha=0.8)
        
        # Ajustar límites Y para incluir todos los datos de PV glasses
        y_min, y_max = 70, 110
        if not pv_glasses_data.empty:
            pv_min, pv_max = pv_glasses_data['valor'].min(), pv_glasses_data['valor'].max()
            y_min = min(y_min, pv_min - 5)
            y_max = max(y_max, pv_max + 5)
            logger.info(f"Límites Y ajustados: [{y_min:.1f}, {y_max:.1f}] para incluir PV glasses [{pv_min:.1f}, {pv_max:.1f}]")
        
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Soiling Ratio [%]', fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        
        # Establecer límites del eje x
        start_date_for_xlim = pd.Timestamp('2024-07-23')
        end_date_for_xlim = end_date
        ax.set_xlim([start_date_for_xlim, end_date_for_xlim])
        
        # Crear formateador personalizado para mostrar Month 1, Month 2, etc.
        # Usar agosto 2024 como fecha de referencia (Month 1 = agosto 2024)
        reference_date = pd.Timestamp('2024-08-01')
        
        # Calcular número de mes desde agosto 2024
        def month_formatter(x, pos):
            # Convertir el valor numérico de matplotlib a fecha
            try:
                date = mdates.num2date(x)
                # Asegurar que sea tz-naive
                if hasattr(date, 'tz') and date.tz is not None:
                    date = date.replace(tzinfo=None)
                
                # Calcular diferencia en meses desde agosto 2024
                # Usar el primer día del mes para comparación consistente
                ref_month_start = pd.Timestamp(reference_date.year, reference_date.month, 1)
                date_month_start = pd.Timestamp(date.year, date.month, 1)
                
                months_diff = (date_month_start.year - ref_month_start.year) * 12 + (date_month_start.month - ref_month_start.month) + 1
                return f'Month {months_diff}'
            except:
                return ''
        
        # Configurar el formateador y localizador
        ax.xaxis.set_major_formatter(plt.FuncFormatter(month_formatter))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=15)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar gráfico
        output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "consolidados")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = "consolidated_weekly_q25_monthly_labels.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico consolidado con etiquetas mensuales guardado en: {plot_path}")
        
        if settings.SHOW_FIGURES:
            plt.show()
        
        plt.close()
        return True
    else:
        logger.warning("No se encontraron series para graficar")
        return False

def load_uncertainty_data_weekly():
    """Cargar datos de incertidumbre semanal para todas las metodologías"""
    uncertainty_data = {}
    
    # RefCells
    try:
        if os.path.exists(paths.SR_WEEKLY_ABS_WITH_U_FILE):
            df = pd.read_csv(paths.SR_WEEKLY_ABS_WITH_U_FILE)
            time_col = 'timestamp' if 'timestamp' in df.columns else '_time'
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                uncertainty_data['ref_cells'] = df
                logger.info(f"Datos de incertidumbre RefCells cargados: {len(df)} puntos")
    except Exception as e:
        logger.warning(f"Error cargando incertidumbre RefCells: {e}")
    
    # DustIQ
    try:
        if os.path.exists(paths.DUSTIQ_SR_WEEKLY_ABS_WITH_U_FILE):
            df = pd.read_csv(paths.DUSTIQ_SR_WEEKLY_ABS_WITH_U_FILE, index_col='timestamp', parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            uncertainty_data['dustiq'] = df
            logger.info(f"Datos de incertidumbre DustIQ cargados: {len(df)} puntos")
    except Exception as e:
        logger.warning(f"Error cargando incertidumbre DustIQ: {e}")
    
    # Soiling Kit
    try:
        if os.path.exists(paths.SOILING_KIT_SR_WEEKLY_ABS_WITH_U_FILE):
            df = pd.read_csv(paths.SOILING_KIT_SR_WEEKLY_ABS_WITH_U_FILE)
            # Soiling Kit usa 'Original_Timestamp_Col' como columna de tiempo
            time_col = 'Original_Timestamp_Col' if 'Original_Timestamp_Col' in df.columns else ('timestamp' if 'timestamp' in df.columns else '_time')
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                uncertainty_data['soiling_kit'] = df
                logger.info(f"Datos de incertidumbre Soiling Kit cargados: {len(df)} puntos")
    except Exception as e:
        logger.warning(f"Error cargando incertidumbre Soiling Kit: {e}", exc_info=True)
    
    # PVStand Isc
    try:
        if os.path.exists(paths.PROPAGACION_ERRORES_PVSTAND_DIR):
            isc_file = os.path.join(paths.PROPAGACION_ERRORES_PVSTAND_DIR, "sr_isc_weekly_abs_with_U.csv")
            if os.path.exists(isc_file):
                df = pd.read_csv(isc_file)
                if '_time' in df.columns:
                    df['_time'] = pd.to_datetime(df['_time'])
                    df.set_index('_time', inplace=True)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    uncertainty_data['pvstand_isc'] = df
                    logger.info(f"Datos de incertidumbre PVStand Isc cargados: {len(df)} puntos")
    except Exception as e:
        logger.warning(f"Error cargando incertidumbre PVStand Isc: {e}")
    
    # PVStand Pmax
    try:
        if os.path.exists(paths.PROPAGACION_ERRORES_PVSTAND_DIR):
            pmax_file = os.path.join(paths.PROPAGACION_ERRORES_PVSTAND_DIR, "sr_pmax_weekly_abs_with_U.csv")
            if os.path.exists(pmax_file):
                df = pd.read_csv(pmax_file)
                if '_time' in df.columns:
                    df['_time'] = pd.to_datetime(df['_time'])
                    df.set_index('_time', inplace=True)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    uncertainty_data['pvstand_pmax'] = df
                    logger.info(f"Datos de incertidumbre PVStand Pmax cargados: {len(df)} puntos")
    except Exception as e:
        logger.warning(f"Error cargando incertidumbre PVStand Pmax: {e}")
    
    # IV600 - Tiene múltiples columnas de incertidumbre, necesitamos mapear a las columnas usadas en el gráfico
    try:
        if os.path.exists(paths.IV600_SR_WEEKLY_ABS_WITH_U_FILE):
            df = pd.read_csv(paths.IV600_SR_WEEKLY_ABS_WITH_U_FILE)
            time_col = 'timestamp' if 'timestamp' in df.columns else '_time'
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # IV600 tiene múltiples comparaciones, usar la que corresponde a 434vs439
                # Mapear a las columnas usadas en el gráfico consolidado
                df_iv600_mapped = pd.DataFrame(index=df.index)
                
                # Para SR_Pmax_IV600 -> usar U_SR_Pmax_1MD434vs1MD439_k2_rel
                if 'U_SR_Pmax_1MD434vs1MD439_k2_rel' in df.columns:
                    df_iv600_mapped['U_rel_k2_Pmax'] = df['U_SR_Pmax_1MD434vs1MD439_k2_rel']
                
                # Para SR_Isc_IV600 -> usar U_SR_Isc_1MD434vs1MD439_k2_rel
                if 'U_SR_Isc_1MD434vs1MD439_k2_rel' in df.columns:
                    df_iv600_mapped['U_rel_k2_Isc'] = df['U_SR_Isc_1MD434vs1MD439_k2_rel']
                
                # Si no hay columnas específicas, intentar usar un promedio o la primera disponible
                if df_iv600_mapped.empty:
                    # Buscar cualquier columna U_SR_*_k2_rel
                    u_cols = [col for col in df.columns if 'U_SR' in col and 'k2_rel' in col]
                    if u_cols:
                        # Usar promedio de todas las incertidumbres disponibles
                        df_iv600_mapped['U_rel_k2'] = df[u_cols].mean(axis=1)
                        logger.info(f"IV600: Usando promedio de {len(u_cols)} columnas de incertidumbre")
                
                if not df_iv600_mapped.empty:
                    uncertainty_data['iv600'] = df_iv600_mapped
                    logger.info(f"Datos de incertidumbre IV600 cargados: {len(df_iv600_mapped)} puntos, columnas: {df_iv600_mapped.columns.tolist()}")
    except Exception as e:
        logger.warning(f"Error cargando incertidumbre IV600: {e}", exc_info=True)
    
    # PV Glasses - Tiene 3 celdas (FC3, FC4, FC5), cada una con su propia incertidumbre
    try:
        if os.path.exists(paths.PV_GLASSES_SR_WEEKLY_ABS_WITH_U_FILE):
            df = pd.read_csv(paths.PV_GLASSES_SR_WEEKLY_ABS_WITH_U_FILE)
            time_col = '_time' if '_time' in df.columns else ('timestamp' if 'timestamp' in df.columns else 'index')
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # PV Glasses tiene columnas U_SR_R_FC3_k2_rel, U_SR_R_FC4_k2_rel, U_SR_R_FC5_k2_rel
                # Mapear a un formato que get_error_bars_for_series pueda usar
                df_pv_glasses_mapped = pd.DataFrame(index=df.index)
                
                # Mapear cada celda a su columna de incertidumbre
                if 'U_SR_R_FC3_k2_rel' in df.columns:
                    df_pv_glasses_mapped['U_rel_k2_FC3'] = df['U_SR_R_FC3_k2_rel']
                if 'U_SR_R_FC4_k2_rel' in df.columns:
                    df_pv_glasses_mapped['U_rel_k2_FC4'] = df['U_SR_R_FC4_k2_rel']
                if 'U_SR_R_FC5_k2_rel' in df.columns:
                    df_pv_glasses_mapped['U_rel_k2_FC5'] = df['U_SR_R_FC5_k2_rel']
                
                # También agregar una columna genérica U_rel_k2 como promedio (para casos donde no se especifica la celda)
                u_cols = [col for col in df.columns if 'U_SR_R_FC' in col and 'k2_rel' in col]
                if u_cols:
                    df_pv_glasses_mapped['U_rel_k2'] = df[u_cols].mean(axis=1)
                
                if not df_pv_glasses_mapped.empty:
                    uncertainty_data['pv_glasses'] = df_pv_glasses_mapped
                    logger.info(f"Datos de incertidumbre PV Glasses cargados: {len(df_pv_glasses_mapped)} puntos, columnas: {df_pv_glasses_mapped.columns.tolist()}")
    except Exception as e:
        logger.warning(f"Error cargando incertidumbre PV Glasses: {e}", exc_info=True)
    
    return uncertainty_data

def get_error_bars_for_series(series, uncertainty_data, data_key, col_name=None):
    """Obtener barras de error para una serie basándose en datos de incertidumbre"""
    if uncertainty_data is None or data_key not in uncertainty_data:
        return None
    
    df_unc = uncertainty_data[data_key]
    
    # Determinar qué columna de incertidumbre usar
    u_col = None
    if 'U_rel_k2' in df_unc.columns:
        u_col = 'U_rel_k2'
    elif col_name and 'Pmax' in col_name and 'U_rel_k2_Pmax' in df_unc.columns:
        u_col = 'U_rel_k2_Pmax'
    elif col_name and 'Isc' in col_name and 'U_rel_k2_Isc' in df_unc.columns:
        u_col = 'U_rel_k2_Isc'
    elif col_name and 'FC3' in col_name and 'U_rel_k2_FC3' in df_unc.columns:
        u_col = 'U_rel_k2_FC3'
    elif col_name and 'FC4' in col_name and 'U_rel_k2_FC4' in df_unc.columns:
        u_col = 'U_rel_k2_FC4'
    elif col_name and 'FC5' in col_name and 'U_rel_k2_FC5' in df_unc.columns:
        u_col = 'U_rel_k2_FC5'
    elif 'U_rel_k2_Pmax' in df_unc.columns:
        u_col = 'U_rel_k2_Pmax'
    elif 'U_rel_k2_Isc' in df_unc.columns:
        u_col = 'U_rel_k2_Isc'
    else:
        # Buscar cualquier columna que contenga U_rel_k2
        u_cols = [col for col in df_unc.columns if 'U_rel_k2' in col]
        if u_cols:
            u_col = u_cols[0]
            logger.info(f"Usando columna de incertidumbre: {u_col} para {data_key}")
    
    if u_col is None or u_col not in df_unc.columns:
        return None
    
    yerr = []
    uncertainty_index = df_unc.index
    
    # Asegurar timezone consistente
    if series.index.tz is None and uncertainty_index.tz is not None:
        series.index = series.index.tz_localize('UTC')
    elif series.index.tz is not None and uncertainty_index.tz is None:
        uncertainty_index = uncertainty_index.tz_localize('UTC')
    elif series.index.tz is not None and uncertainty_index.tz is not None:
        uncertainty_index = uncertainty_index.tz_convert(series.index.tz)
    
    for date in series.index:
        sr_val = series.loc[date]
        if pd.notna(sr_val):
            if date in uncertainty_index:
                u_rel = df_unc.loc[date, u_col]
            else:
                # Buscar fecha más cercana
                # Para PV Glasses, usar un umbral más amplio (7 días) ya que son puntos discretos
                # que pueden no coincidir exactamente con las fechas semanales
                max_days = 7 if data_key == 'pv_glasses' else 3
                time_diffs = abs(uncertainty_index - date)
                closest_idx = time_diffs.argmin()
                if time_diffs[closest_idx] <= pd.Timedelta(days=max_days):
                    u_rel = df_unc.iloc[closest_idx][u_col]
                else:
                    u_rel = np.nan
            
            if pd.notna(u_rel):
                # Determinar si u_rel está en fracción o porcentaje basándose en el data_key
                # IV600 y PV Glasses guardan en fracción (0.0295 = 2.95%), otros en porcentaje (1.815 = 1.815%)
                if data_key in ['iv600', 'pv_glasses']:
                    # IV600 y PV Glasses: u_rel está en fracción, multiplicar directamente
                    yerr.append(u_rel * sr_val)
                else:
                    # Otros: u_rel está en porcentaje, usar fórmula estándar
                    # Incertidumbre absoluta = incertidumbre relativa (en %) * valor / 100
                    yerr.append(u_rel * sr_val / 100.0)
            else:
                yerr.append(0)
        else:
            yerr.append(0)
    
    return yerr if any(err > 0 for err in yerr) else None

def create_consolidated_weekly_q25_plot_with_uncertainty():
    """Crear gráfico consolidado con barras de error de propagación de incertidumbre"""
    
    logger.info("Iniciando generación de gráfico consolidado con propagación de errores...")
    
    # Cargar todos los datos específicos
    logger.info("=== CARGANDO DATOS ===")
    
    ref_cells_data = load_ref_cells_photocells()
    dustiq_data = load_dustiq_mediodia_solar()
    soiling_kit_data = load_soiling_kit_raw_q25()
    pvstand_data = load_pvstand_solar_noon_corrected()
    iv600_data = load_iv600_both_curves()
    pv_glasses_data = load_pv_glasses_promedios()
    
    # Cargar datos de incertidumbre
    logger.info("=== CARGANDO DATOS DE INCERTIDUMBRE ===")
    uncertainty_data = load_uncertainty_data_weekly()
    
    logger.info("=== FIN CARGA DE DATOS ===")
    
    # Verificar si hay datos para graficar
    if (ref_cells_data.empty and dustiq_data.empty and 
        soiling_kit_data.empty and pvstand_data.empty and iv600_data.empty and pv_glasses_data.empty):
        logger.warning("No se encontraron datos para graficar")
        return False
    
    # Fecha de fin: 01/06/2025
    end_date = pd.Timestamp('2025-06-01')
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Colores intermedios/pastel para cada tipo de análisis
    colors = {
        'ref_cells': '#8B4513',      # Marrón saddle
        'dustiq': '#FF8C00',         # Naranja oscuro
        'soiling_kit': '#2E8B57',    # Verde mar
        'pvstand_pmax': '#4169E1',   # Azul real
        'pvstand_isc': '#8A2BE2',    # Azul violeta
        'iv600_pmax': '#DC143C',     # Carmesí
        'iv600_isc': '#FF1493',      # Rosa profundo
        'pv_glasses_fc3': '#228B22',  # Verde bosque
        'pv_glasses_fc4': '#32CD32',  # Verde lima  
        'pv_glasses_fc5': '#9ACD32'   # Verde amarillento
    }
    
    plotted_series = []
    
    # Función para filtrar datos hasta la fecha de fin
    def filter_to_end_date(data, data_name):
        if data.empty:
            return None
        data_copy = data.copy()
        if data_copy.index.tz is not None:
            data_copy.index = data_copy.index.tz_localize(None)
        filtered_data = data_copy[data_copy.index <= end_date]
        if filtered_data.empty:
            logger.warning(f"No hay datos de {data_name} hasta {end_date}")
            return None
        return filtered_data
    
    # Graficar RefCells Photocells
    if not ref_cells_data.empty:
        for col in ref_cells_data.columns:
            series = ref_cells_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"RefCells {col}")
                if filtered_series is not None and not filtered_series.empty:
                    # Calcular barras de error sobre valores originales
                    yerr_original = get_error_bars_for_series(filtered_series, uncertainty_data, 'ref_cells')
                    normalized_series = normalize_series_to_100(filtered_series, f"RefCells {col}")
                    if not normalized_series.empty:
                        # Escalar las barras de error proporcionalmente
                        if yerr_original is not None and len(yerr_original) == len(normalized_series):
                            first_value = filtered_series.iloc[0]
                            normalization_factor = 100.0 / first_value if first_value != 0 else 1.0
                            yerr = [err * normalization_factor for err in yerr_original]
                        else:
                            yerr = None
                        
                        if yerr is not None:
                            ax.errorbar(normalized_series.index, normalized_series.values, yerr=yerr,
                                       color=colors['ref_cells'], linewidth=2, capsize=3, capthick=1.5,
                                       elinewidth=1.5, ecolor=colors['ref_cells'], alpha=0.8,
                                       label='RefCells - Photocells')
                        else:
                            ax.plot(normalized_series.index, normalized_series.values, 
                                   color=colors['ref_cells'], linewidth=2,
                                   label='RefCells - Photocells')
                        plotted_series.append('RefCells - Photocells')
    
    # Graficar DustIQ Mediodía Solar
    if not dustiq_data.empty:
        for col in dustiq_data.columns:
            series = dustiq_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"DustIQ {col}")
                if filtered_series is not None and not filtered_series.empty:
                    # Calcular barras de error sobre valores originales
                    yerr_original = get_error_bars_for_series(filtered_series, uncertainty_data, 'dustiq')
                    normalized_series = normalize_series_to_100(filtered_series, f"DustIQ {col}")
                    if not normalized_series.empty:
                        # Escalar las barras de error proporcionalmente
                        if yerr_original is not None and len(yerr_original) == len(normalized_series):
                            first_value = filtered_series.iloc[0]
                            normalization_factor = 100.0 / first_value if first_value != 0 else 1.0
                            yerr = [err * normalization_factor for err in yerr_original]
                        else:
                            yerr = None
                        
                        if yerr is not None:
                            ax.errorbar(normalized_series.index, normalized_series.values, yerr=yerr,
                                       color=colors['dustiq'], linewidth=2, capsize=3, capthick=1.5,
                                       elinewidth=1.5, ecolor=colors['dustiq'], alpha=0.8,
                                       label='DustIQ - Q25')
                        else:
                            ax.plot(normalized_series.index, normalized_series.values, 
                                   color=colors['dustiq'], linewidth=2,
                                   label='DustIQ - Q25')
                        plotted_series.append('DustIQ - Q25')
    
    # Graficar Soiling Kit Raw Q25 Semanal
    if not soiling_kit_data.empty:
        for col in soiling_kit_data.columns:
            series = soiling_kit_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"Soiling Kit {col}")
                if filtered_series is not None and not filtered_series.empty:
                    # Calcular barras de error sobre valores originales (antes de normalizar)
                    yerr_original = get_error_bars_for_series(filtered_series, uncertainty_data, 'soiling_kit')
                    normalized_series = normalize_series_to_100(filtered_series, f"Soiling Kit {col}")
                    if not normalized_series.empty:
                        # Escalar las barras de error proporcionalmente a la normalización
                        if yerr_original is not None and len(yerr_original) == len(normalized_series):
                            first_value = filtered_series.iloc[0]
                            normalization_factor = 100.0 / first_value if first_value != 0 else 1.0
                            yerr = [err * normalization_factor for err in yerr_original]
                        else:
                            yerr = None
                        
                        if yerr is not None:
                            ax.errorbar(normalized_series.index, normalized_series.values, yerr=yerr,
                                       color=colors['soiling_kit'], linewidth=2, capsize=3, capthick=1.5,
                                       elinewidth=1.5, ecolor=colors['soiling_kit'], alpha=0.8,
                                       label='Soiling Kit')
                        else:
                            ax.plot(normalized_series.index, normalized_series.values, 
                                   color=colors['soiling_kit'], linewidth=2,
                                   label='Soiling Kit')
                        plotted_series.append('Soiling Kit')
    
    # Graficar PVStand Semanal Q25 (ambas curvas)
    if not pvstand_data.empty:
        label_map = {'SR_PVStand_Semanal_Q25_Pmax': 'PVStand Pmax',
                     'SR_PVStand_Semanal_Q25_Isc': 'PVStand Isc'}
        for col in pvstand_data.columns:
            series = pvstand_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"PVStand {col}")
                if filtered_series is not None and not filtered_series.empty:
                    color_key = 'pvstand_pmax' if 'Pmax' in col else 'pvstand_isc'
                    uncertainty_key = 'pvstand_pmax' if 'Pmax' in col else 'pvstand_isc'
                    # Calcular barras de error sobre valores originales
                    yerr_original = get_error_bars_for_series(filtered_series, uncertainty_data, uncertainty_key)
                    normalized_series = normalize_series_to_100(filtered_series, f"PVStand {col}")
                    if not normalized_series.empty:
                        # Escalar las barras de error proporcionalmente
                        if yerr_original is not None and len(yerr_original) == len(normalized_series):
                            first_value = filtered_series.iloc[0]
                            normalization_factor = 100.0 / first_value if first_value != 0 else 1.0
                            yerr = [err * normalization_factor for err in yerr_original]
                        else:
                            yerr = None
                        
                        if yerr is not None:
                            ax.errorbar(normalized_series.index, normalized_series.values, yerr=yerr,
                                       color=colors[color_key], linewidth=2, capsize=3, capthick=1.5,
                                       elinewidth=1.5, ecolor=colors[color_key], alpha=0.8,
                                       label=label_map.get(col, col))
                        else:
                            ax.plot(normalized_series.index, normalized_series.values, 
                                   color=colors[color_key], linewidth=2,
                                   label=label_map.get(col, col))
                        plotted_series.append(label_map.get(col, col))
    
    # Graficar IV600 ambas curvas
    if not iv600_data.empty:
        label_map = {'SR_Pmax_IV600': 'SR Pmax IV600',
                     'SR_Isc_IV600': 'SR Isc IV600'}
        for col in iv600_data.columns:
            series = iv600_data[col].dropna()
            if not series.empty:
                filtered_series = filter_to_end_date(series, f"IV600 {col}")
                if filtered_series is not None and not filtered_series.empty:
                    color_key = 'iv600_pmax' if 'Pmax' in col else 'iv600_isc'
                    # IV600 no se normaliza en el gráfico consolidado, calcular barras de error directamente
                    yerr = get_error_bars_for_series(filtered_series, uncertainty_data, 'iv600', col_name=col)
                    if yerr is not None:
                        ax.errorbar(filtered_series.index, filtered_series.values, yerr=yerr,
                                   color=colors[color_key], linewidth=2, capsize=3, capthick=1.5,
                                   elinewidth=1.5, ecolor=colors[color_key], alpha=0.8,
                                   label=label_map.get(col, col))
                    else:
                        ax.plot(filtered_series.index, filtered_series.values, 
                               color=colors[color_key], linewidth=2,
                               label=label_map.get(col, col))
                    plotted_series.append(label_map.get(col, col))
    
    # Graficar PV Glasses como marcadores discretos con barras de error
    if not pv_glasses_data.empty:
        pv_glasses_copy = pv_glasses_data.copy()
        if pv_glasses_copy.index.tz is not None:
            pv_glasses_copy.index = pv_glasses_copy.index.tz_localize(None)
        pv_glasses_filtered = pv_glasses_copy[pv_glasses_copy.index <= end_date]
        for fc in ['FC3', 'FC4', 'FC5']:
            fc_data = pv_glasses_filtered[pv_glasses_filtered['fc'] == fc]
            if not fc_data.empty:
                color_key = f'pv_glasses_{fc.lower()}'
                if color_key in colors:
                    # Crear Series para obtener barras de error
                    fc_series = pd.Series(fc_data['valor'].values, index=fc_data.index)
                    
                    # Obtener barras de error usando col_name para especificar la celda
                    yerr_fc = get_error_bars_for_series(
                        fc_series, 
                        uncertainty_data, 
                        'pv_glasses', 
                        col_name=fc  # 'FC3', 'FC4', 'FC5'
                    )
                    
                    # Graficar con barras de error si están disponibles
                    if yerr_fc is not None:
                        ax.errorbar(fc_data.index, fc_data['valor'], 
                                   yerr=yerr_fc,
                                   color=colors[color_key], 
                                   fmt='o', markersize=8, capsize=3, capthick=1.5,
                                   label=f'PV Glasses {fc}', zorder=5, alpha=0.8)
                    else:
                        # Si no hay barras de error, graficar sin ellas
                        ax.scatter(fc_data.index, fc_data['valor'], 
                                  color=colors[color_key], s=100, marker='o',
                                  label=f'PV Glasses {fc}', zorder=5)
                    plotted_series.append(f'PV Glasses {fc}')
    
    # Configurar el gráfico
    if plotted_series:
        ax.set_title('Intercomparison Soiling Ratio Q25 + PV Glasses (with Uncertainty)', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=15, frameon=True, fancybox=True, framealpha=0.8)
        
        # Establecer límites fijos del eje Y
        ax.set_ylim([50, 110])
        ax.set_xlabel('Date', fontsize=18)
        ax.set_ylabel('Soiling Ratio [%]', fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        
        start_date_for_xlim = pd.Timestamp('2024-07-23')
        end_date_for_xlim = end_date
        ax.set_xlim([start_date_for_xlim, end_date_for_xlim])
        
        # Formateador de meses (Month 1 = agosto 2024)
        reference_date = pd.Timestamp('2024-08-01')
        def month_formatter(x, pos):
            try:
                date = mdates.num2date(x)
                if hasattr(date, 'tz') and date.tz is not None:
                    date = date.replace(tzinfo=None)
                ref_month_start = pd.Timestamp(reference_date.year, reference_date.month, 1)
                date_month_start = pd.Timestamp(date.year, date.month, 1)
                months_diff = (date_month_start.year - ref_month_start.year) * 12 + (date_month_start.month - ref_month_start.month) + 1
                return f'Month {months_diff}'
            except:
                return ''
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(month_formatter))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=15)
        
        plt.tight_layout()
        
        output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "consolidados")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = "consolidated_weekly_q25_with_uncertainty.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico consolidado con incertidumbre guardado en: {plot_path}")
        
        if settings.SHOW_FIGURES:
            plt.show()
        
        plt.close()
        return True
    else:
        logger.warning("No se encontraron series para graficar")
        return False

def main():
    """Función principal"""
    try:
        # Ejecutar función original sincronizada para debug
        success2 = create_synchronized_weekly_q25_plot()
        if success2:
            print("✅ Gráfico consolidado sincronizado generado exitosamente")
        else:
            print("❌ No se pudo generar el gráfico consolidado sincronizado")
        
        # Ejecutar nueva función con PV Glasses
        success3 = create_consolidated_weekly_q25_plot_with_pv_glasses()
        if success3:
            print("✅ Gráfico consolidado con PV Glasses generado exitosamente")
        else:
            print("❌ No se pudo generar el gráfico consolidado con PV Glasses")
        
        # Ejecutar nueva función con etiquetas mensuales
        success4 = create_consolidated_weekly_q25_plot_monthly_labels()
        if success4:
            print("✅ Gráfico consolidado con etiquetas mensuales generado exitosamente")
        else:
            print("❌ No se pudo generar el gráfico consolidado con etiquetas mensuales")
        
        # Ejecutar nueva función con propagación de errores
        success5 = create_consolidated_weekly_q25_plot_with_uncertainty()
        if success5:
            print("✅ Gráfico consolidado con propagación de errores generado exitosamente")
        else:
            print("❌ No se pudo generar el gráfico consolidado con propagación de errores")
            
    except Exception as e:
        logger.error(f"Error en la generación de los gráficos consolidados: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
    