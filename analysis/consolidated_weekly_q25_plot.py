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
            # Convertir la columna de tiempo a datetime
            df['_time'] = pd.to_datetime(df['_time'])
            df.set_index('_time', inplace=True)
            
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
            # Convertir la columna de tiempo a datetime
            df['_time'] = pd.to_datetime(df['_time'])
            df.set_index('_time', inplace=True)
            logger.info(f"Datos DustIQ mediodía solar cargados: {df.shape}")
            return df
        else:
            logger.warning(f"Archivo no encontrado: {csv_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error cargando datos DustIQ mediodía solar: {e}")
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
        csv_path = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "pv_stand", "solar_noon", "pvstand_sr_raw_no_offset_solar_noon.csv")
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
                        # Eliminar las dos primeras semanas
                        weekly_q25_trimmed = weekly_q25.iloc[2:].copy()
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
                series_norm.plot(ax=ax, style='o-', color=colors['ref_cells'], 
                               alpha=0.8, linewidth=2, markersize=6, 
                               label=f'Photocells Methodology', marker='o')
                plotted_series.append(f'RefCells - Photocells')
    
    # Graficar DustIQ Mediodía Solar
    if not dustiq_data.empty:
        for col in dustiq_data.columns:
            series = dustiq_data[col].dropna()
            if not series.empty:
                series_norm = normalize_series_to_100(series, f"DustIQ {col}")
                series_norm.plot(ax=ax, style='o-', color=colors['dustiq'], 
                               alpha=0.8, linewidth=2, markersize=6, 
                               label='DustIQ', marker='o')
                plotted_series.append('DustIQ - Q25 Semanal')
    
    # Graficar Soiling Kit Raw Q25
    if not soiling_kit_data.empty:
        for col in soiling_kit_data.columns:
            series = soiling_kit_data[col].dropna()
            if not series.empty:
                series_norm = normalize_series_to_100(series, f"Soiling Kit {col}")
                series_norm.plot(ax=ax, style='o-', color=colors['soiling_kit'], 
                               alpha=0.8, linewidth=2, markersize=6, 
                               label='SoilRatio', marker='o')
                plotted_series.append('Soiling Kit - SR Raw Q25')
    
    # Graficar PVStand Semanal Q25 (ambas curvas)
    if not pvstand_data.empty:
        color_map = {'SR_PVStand_Semanal_Q25_Pmax': colors['pvstand_pmax'],
                     'SR_PVStand_Semanal_Q25_Isc': colors['pvstand_isc']}
        label_map = {'SR_PVStand_Semanal_Q25_Pmax': 'IV Curve Tracer 1 Pmax',
                     'SR_PVStand_Semanal_Q25_Isc': 'IV Curve Tracer 1 Isc'}
        for col in pvstand_data.columns:
            series = pvstand_data[col].dropna()
            if not series.empty:
                series_norm = normalize_series_to_100(series, label_map.get(col, col))
                series_norm.plot(ax=ax, style='o-', color=color_map.get(col, None),
                                alpha=0.8, linewidth=2, markersize=6,
                                label=label_map.get(col, col), marker='o')
                plotted_series.append(label_map.get(col, col))
    
    # Graficar IV600 ambas curvas
    if not iv600_data.empty:
        for i, col in enumerate(iv600_data.columns):
            series = iv600_data[col].dropna()
            if not series.empty:
                series_norm = normalize_series_to_100(series, f"IV600 {col}")
                color = colors['iv600_pmax'] if 'Pmax' in col or 'Pmp' in col else colors['iv600_isc']
                series_norm.plot(ax=ax, style='o-', color=color, 
                               alpha=0.8, linewidth=2, markersize=6, 
                               label=f'IV Curva Tracer 2 {col}', marker='o')
                plotted_series.append(f'IV Curva Tracer 2 {col}')
    
    # Configurar el gráfico
    if plotted_series:
        ax.set_title('Intercomparison Soiling Ratio Q25', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=15, frameon=True, fancybox=True, framealpha=0.8)
        ax.set_ylim([90, 110])
        ax.set_xlabel('Fecha', fontsize=18)
        ax.set_ylabel('Soiling Ratio [%]', fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        
        # Formatear eje X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45, ha='right')
        
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
                all_series['Photocells Methodology'] = series
    
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
                all_series['SoilRatio'] = series
    
    # PVStand Semanal Q25 (ambas curvas)
    if not pvstand_data.empty:
        label_map = {'SR_PVStand_Semanal_Q25_Pmax': 'IV Curve Tracer 1 SRPmax',
                     'SR_PVStand_Semanal_Q25_Isc': 'IV Curve Tracer 1 SRIsc'}
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
        elif 'SoilRatio' in name:
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
                alpha=0.8, linewidth=2, markersize=8, 
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

def main():
    """Función principal"""
    try:
        # Solo ejecutar la función sincronizada para debug
        success2 = create_synchronized_weekly_q25_plot()
        if success2:
            print("✅ Gráfico consolidado sincronizado generado exitosamente")
        else:
            print("❌ No se pudo generar el gráfico consolidado sincronizado")
            
    except Exception as e:
        logger.error(f"Error en la generación de los gráficos consolidados: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 