import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import numpy as np
from scipy import stats
import matplotlib.dates as mdates
import itertools
from .classes_codes import medio_dia_solar
from config import paths
from config.settings import (
    ANALYSIS_START_DATE_GENERAL_STR,
    ANALYSIS_END_DATE_GENERAL_STR,
    DUSTIQ_SR_FILTER_THRESHOLD,
    DUSTIQ_LOCAL_TIMEZONE_STR
)

# Configurar logging
logger = logging.getLogger(__name__)

def save_plot_matplotlib(fig, filename_base, output_dir, subfolder=None, dpi=300):
    """Guarda una figura de matplotlib en el directorio especificado."""
    if subfolder:
        output_path = os.path.join(output_dir, subfolder)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = output_dir

    full_path = os.path.join(output_path, filename_base)
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Gráfico guardado en: {full_path}")

def calcular_tendencia(x, y):
    """
    Calcula la línea de tendencia, pendiente y R² para una serie de datos.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    return slope, intercept, r_squared

def analyze_dustiq_data(
    data_filepath: str,
    output_graph_dir: str,
    analysis_start_date_str: str | None = None, # Formato 'YYYY-MM-DD'
    analysis_end_date_str: str | None = None,   # Formato 'YYYY-MM-DD'
    save_figures: bool = True,
    show_figures: bool = False,
    generar_grafico1_1_mediodia_solar: bool = True, # Semanal, 3 slots
    duracion_slot_minutos_grafico1_1: int = 60,    # Para Gráfico 1.1 y 1.3
    generar_grafico1_2_promedio_diario_solar: bool = True, # Diario, 1 franja
    duracion_franja_diaria_solar_minutos: int = 120,    # Para Gráfico 1.2
    generar_grafico1_3_promedios_diarios_3slots_solar: bool = True # Nuevo para Gráfico 1.3
):
    """
    Carga datos procesados de DustIQ desde un CSV, los analiza y genera gráficos.
    """
    logger.info(f"--- Iniciando Análisis de DustIQ desde: {data_filepath} ---")
    
    if not os.path.exists(data_filepath):
        logger.error(f"Archivo de datos DustIQ no encontrado en {data_filepath}")
        return

    logger.info(f"Cargando datos de DustIQ desde {data_filepath}...")
    try:
        df = pd.read_csv(data_filepath, index_col='timestamp', parse_dates=True)
    except Exception as e:
        logger.error(f"Error cargando el archivo CSV de DustIQ: {e}")
        return

    logger.info("Datos de DustIQ cargados exitosamente. Preprocesando...")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            logger.info("Índice convertido a DatetimeIndex.")
        except Exception as e:
            logger.error(f"No se pudo convertir el índice a DatetimeIndex: {e}")
            return

    if df.index.tz is not None:
        logger.info(f"Convirtiendo índice principal de DustIQ (df.index) a naive. Zona horaria original: {df.index.tz}")
        df.index = df.index.tz_localize(None)

    # Filtrado estricto de fechas justo después de cargar y preparar el índice
    # Usar la fecha de inicio del análisis si está disponible, sino usar la fecha por defecto
    if analysis_start_date_str:
        fecha_inicio = pd.to_datetime(analysis_start_date_str)
    else:
        fecha_inicio = pd.to_datetime('2024-06-24')  # Sincronizar con start_date_dustiq_str
    df = df[df.index >= fecha_inicio]
    logger.info(f"Filtrado estricto: solo datos desde {fecha_inicio.date()}. Filas restantes: {len(df)}")

    try:
        if analysis_start_date_str:
            df = df.loc[df.index >= pd.to_datetime(analysis_start_date_str)]
        if analysis_end_date_str:
            df = df.loc[df.index <= pd.to_datetime(analysis_end_date_str)]
        logger.info(f"Filtro de fechas aplicado. Filas restantes: {len(df)}")
    except Exception as e:
        logger.error(f"Error al parsear o aplicar el filtro de fechas (índice ya es naive): {e}")
        if df.empty: return

    if df.empty:
        logger.warning("No hay datos de DustIQ disponibles para el rango de fechas especificado (después del filtro).")
        return

    logger.info(f"Datos de DustIQ listos para análisis: {len(df)} filas")

    sr_c11_col_name = "SR_C11_Avg" 
    if sr_c11_col_name in df.columns:
        sr_c11_filtered = df[sr_c11_col_name][df[sr_c11_col_name] > DUSTIQ_SR_FILTER_THRESHOLD].copy() 
        logger.info(f"Filtrado {sr_c11_col_name} > {DUSTIQ_SR_FILTER_THRESHOLD}. {len(sr_c11_filtered)} puntos restantes.")
    else:
        logger.warning(f"Advertencia: La columna '{sr_c11_col_name}' no se encontró.")
        sr_c11_filtered = pd.Series(dtype=float) 

    # Análisis de Propagación de Incertidumbre de SR
    logger.info("Iniciando análisis de propagación de incertidumbre de SR (DustIQ)...")
    try:
        from analysis.sr_uncertainty_dustiq import run_uncertainty_propagation_analysis
        # Ejecutar análisis con SR_C11_Avg (columna principal)
        uncertainty_success = run_uncertainty_propagation_analysis(
            df,
            sr_col=sr_c11_col_name,
            use_average=False
        )
        if uncertainty_success:
            logger.info("✅ Análisis de propagación de incertidumbre completado exitosamente (DustIQ).")
        else:
            logger.warning("⚠️  El análisis de propagación de incertidumbre no se completó exitosamente.")
    except ImportError as e:
        logger.error(f"No se pudo importar el módulo 'sr_uncertainty_dustiq': {e}")
    except Exception as e:
        logger.error(f"Error al ejecutar el análisis de propagación de incertidumbre: {e}", exc_info=True)
    # Continuar con el resto del análisis aunque falle la incertidumbre

    dustiq_graph_subdir = "dustiq"
    os.makedirs(os.path.join(output_graph_dir, dustiq_graph_subdir), exist_ok=True)

    # Cargar datos de incertidumbre semanal y diaria para barras de error
    uncertainty_data_weekly = None
    uncertainty_data_daily = None
    try:
        uncertainty_file_weekly = paths.DUSTIQ_SR_WEEKLY_ABS_WITH_U_FILE
        if os.path.exists(uncertainty_file_weekly):
            df_uncertainty = pd.read_csv(uncertainty_file_weekly, index_col='timestamp', parse_dates=True)
            if df_uncertainty.index.tz is None:
                df_uncertainty.index = df_uncertainty.index.tz_localize('UTC')
            uncertainty_data_weekly = df_uncertainty
            logger.info(f"Datos de incertidumbre semanal cargados: {len(uncertainty_data_weekly)} puntos")
        else:
            logger.warning(f"Archivo de incertidumbre semanal no encontrado: {uncertainty_file_weekly}")
        
        uncertainty_file_daily = paths.DUSTIQ_SR_DAILY_ABS_WITH_U_FILE
        if os.path.exists(uncertainty_file_daily):
            df_uncertainty_daily = pd.read_csv(uncertainty_file_daily, index_col='timestamp', parse_dates=True)
            if df_uncertainty_daily.index.tz is None:
                df_uncertainty_daily.index = df_uncertainty_daily.index.tz_localize('UTC')
            uncertainty_data_daily = df_uncertainty_daily
            logger.info(f"Datos de incertidumbre diaria cargados: {len(uncertainty_data_daily)} puntos")
        else:
            logger.warning(f"Archivo de incertidumbre diaria no encontrado: {uncertainty_file_daily}")
    except Exception as e:
        logger.warning(f"No se pudieron cargar datos de incertidumbre: {e}")

    # Gráfico 1: Medias semanales de SR_C11_Avg para franjas horarias FIJAS
    if not sr_c11_filtered.empty:
        fig1 = None
        try:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            logger.info(f"Generando Gráfico 1 de DustIQ usando franjas horarias fijas.")
            
            # Datos para cada franja horaria
            franjas = {
                '12:00-13:00': ('12:00', '13:00'),
                '14:00-15:00': ('14:00', '15:00'),
                '16:00-17:00': ('16:00', '17:00')
            }
            
            colores = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_iter = itertools.cycle(colores)
            handles = []
            labels = []
            for label, (start, end) in franjas.items():
                data = sr_c11_filtered.between_time(start, end).resample('1W', origin='start').quantile(0.25)
                # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
                data = data[data.index >= pd.Timestamp('2024-07-23')]
                if not data.empty:
                    color = next(color_iter)
                    x = data.index
                    x_num = x.map(lambda d: d.toordinal())
                    y = data.values
                    z = np.polyfit(x_num, y, 1)
                    p = np.poly1d(z)
                    residuos = y - p(x_num)
                    ss_res = np.sum(residuos**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2 = 1 - (ss_res / ss_tot)
                    
                    # Agregar barras de error si hay datos de incertidumbre
                    if uncertainty_data_weekly is not None and 'U_rel_k2' in uncertainty_data_weekly.columns:
                        yerr = []
                        uncertainty_index = uncertainty_data_weekly.index
                        # Asegurar que las fechas tengan el mismo timezone
                        if x.tz is None and uncertainty_index.tz is not None:
                            x = x.tz_localize('UTC')
                        elif x.tz is not None and uncertainty_index.tz is None:
                            uncertainty_index = uncertainty_index.tz_localize('UTC')
                        elif x.tz is not None and uncertainty_index.tz is not None:
                            uncertainty_index = uncertainty_index.tz_convert(x.tz)
                        
                        for i, date in enumerate(x):
                            sr_val = y[i]  # Usar valor directamente de y
                            if pd.notna(sr_val):
                                # Buscar fecha más cercana en datos de incertidumbre
                                if date in uncertainty_index:
                                    u_rel = uncertainty_data_weekly.loc[date, 'U_rel_k2']
                                else:
                                    time_diffs = abs(uncertainty_index - date)
                                    closest_idx = time_diffs.argmin()
                                    if time_diffs[closest_idx] <= pd.Timedelta(days=3):
                                        u_rel = uncertainty_data_weekly.iloc[closest_idx]['U_rel_k2']
                                    else:
                                        u_rel = np.nan
                                
                                if pd.notna(u_rel):
                                    yerr.append(u_rel * sr_val / 100.0)
                                else:
                                    yerr.append(0)
                            else:
                                yerr.append(0)
                        
                        errorbar_result = ax1.errorbar(x, y, yerr=yerr, fmt='o-', alpha=0.75, label=None, 
                                             color=color, markersize=3, capsize=3, capthick=1.5,
                                             elinewidth=1.5, ecolor=color)
                        puntos = errorbar_result[0]  # errorbar devuelve (line, caplines, barlinecols)
                    else:
                        puntos, = ax1.plot(x, y, 'o-', alpha=0.75, label=None, color=color, markersize=3)
                    
                    tendencia, = ax1.plot(x, p(x_num), '--', alpha=0.7, label=None, color=color)
                    handles.append(puntos)
                    labels.append(f'{label}')
                    handles.append(tendencia)
                    labels.append(f'trend: {z[0]*7:.3f}[%/semana] R²={r2:.3f}')
            
            # Eje x: solo meses y años
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax1.legend(handles, labels, loc='upper right', fontsize=10, frameon=True)
            ax1.set_ylim([90, 110])
            # Establecer límites del eje x dinámicamente
            start_date_for_xlim = pd.Timestamp('2024-07-23')
            # Usar la fecha máxima de los datos reales, pero nunca exceder el final del año actual
            max_available_date = sr_c11_filtered.index.max() if not sr_c11_filtered.empty else pd.Timestamp('2025-07-31')
            end_date_for_xlim = min(max_available_date, pd.Timestamp('2025-12-31'))
            ax1.set_xlim([start_date_for_xlim, end_date_for_xlim])
            ax1.set_ylabel('Soiling Ratio [%]')
            ax1.set_xlabel('Week')
            ax1.set_title(f'Weekly Q25 Soiling Ratio - DustIQ (fixed time slots)')
            ax1.grid(True)
            plt.tight_layout()
            if save_figures:
                save_plot_matplotlib(fig1, 'SR_DustIQ_FranjasHorarias_Fijas.png', output_graph_dir, subfolder=dustiq_graph_subdir)
                fig1 = None
            if show_figures: plt.show()
        except Exception as e:
            logger.error(f"Error generando Gráfico 1 de DustIQ (fijas): {e}", exc_info=True)
        finally:
            if fig1 is not None and plt.fignum_exists(fig1.number):
                if not show_figures: plt.close(fig1)
    else:
        logger.info(f"No hay datos de {sr_c11_col_name} filtrados para Gráfico 1.")

    # Gráfico 1.4: Franjas horarias específicas sin tendencia (10-11 AM, 12-1 PM, 2-3 PM, 4-5 PM)
    if not sr_c11_filtered.empty:
        fig1_4 = None
        try:
            fig1_4, ax1_4 = plt.subplots(figsize=(12, 6))
            logger.info(f"Generando Gráfico 1.4 de DustIQ usando franjas horarias específicas sin tendencia.")
            
            # Datos para cada franja horaria específica
            franjas_especificas = {
                'SR 10 AM - 11 AM (Fijo)': ('10:00', '11:00'),
                'SR 12 PM - 1 PM (Fijo)': ('12:00', '13:00'),
                'SR 2 PM - 3 PM (Fijo)': ('14:00', '15:00'),
                'SR 4 PM - 5 PM (Fijo)': ('16:00', '17:00')
            }
            
            colores = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_iter = itertools.cycle(colores)
            
            for label, (start, end) in franjas_especificas.items():
                data = sr_c11_filtered.between_time(start, end).resample('1W', origin='start').quantile(0.25)
                # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
                data = data[data.index >= pd.Timestamp('2024-07-23')]
                if not data.empty:
                    color = next(color_iter)
                    x = data.index
                    y = data.values
                    # Solo graficar los puntos sin línea de tendencia
                    ax1_4.plot(x, y, 'o-', alpha=0.75, label=label, color=color, markersize=3)
            
            # Eje x: solo meses y años
            ax1_4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax1_4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax1_4.legend(loc='upper right', fontsize=10, frameon=True)
            ax1_4.set_ylim([90, 110])
            # Establecer límites del eje x dinámicamente
            ax1_4.set_xlim([start_date_for_xlim, end_date_for_xlim])
            ax1_4.set_ylabel('Soiling Ratio [%]')
            ax1_4.set_xlabel('Week')
            ax1_4.set_title(f'Weekly Q25 Soiling Ratio - DustIQ (fixed time slot)')
            ax1_4.grid(True)
            plt.tight_layout()
            if save_figures:
                save_plot_matplotlib(fig1_4, 'SR_DustIQ_FranjasHorarias_Especificas_SinTendencia.png', output_graph_dir, subfolder=dustiq_graph_subdir)
                fig1_4 = None
            if show_figures: plt.show()
        except Exception as e:
            logger.error(f"Error generando Gráfico 1.4 de DustIQ (franjas específicas sin tendencia): {e}", exc_info=True)
        finally:
            if fig1_4 is not None and plt.fignum_exists(fig1_4.number):
                if not show_figures: plt.close(fig1_4)
    else:
        logger.info(f"No hay datos de {sr_c11_col_name} filtrados para Gráfico 1.4.")

    # Gráfico 1.5: Franjas horarias específicas CON TENDENCIAS
    if not sr_c11_filtered.empty:
        fig1_5 = None
        try:
            fig1_5, ax1_5 = plt.subplots(figsize=(12, 6))
            
            franjas_especificas = {
                'SR 10 AM - 11 AM (Fijo)': ('10:00', '11:00'),
                'SR 12 PM - 1 PM (Fijo)': ('12:00', '13:00'),
                'SR 2 PM - 3 PM (Fijo)': ('14:00', '15:00'),
                'SR 4 PM - 5 PM (Fijo)': ('16:00', '17:00')
            }
            
            colores = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_iter = itertools.cycle(colores)
            
            for label, (start, end) in franjas_especificas.items():
                data = sr_c11_filtered.between_time(start, end).resample('1W', origin='start').quantile(0.25)
                # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
                data = data[data.index >= pd.Timestamp('2024-07-23')]
                if not data.empty:
                    color = next(color_iter)
                    x = data.index
                    y = data.values
                    
                    # Validar datos antes de la regresión
                    valid_mask = ~np.isnan(y)
                    if valid_mask.sum() >= 2:
                        # Convertir fechas a números para la regresión
                        x_num = x.map(lambda d: d.toordinal())
                        x_clean = x_num[valid_mask]
                        y_clean = y[valid_mask]
                        
                        # Ajustar la regresión lineal
                        z = np.polyfit(x_clean, y_clean, 1)
                        p = np.poly1d(z)
                        
                        # Calcular R²
                        residuos = y_clean - p(x_clean)
                        ss_res = np.sum(residuos**2)
                        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
                        r2 = 1 - (ss_res / ss_tot)
                        
                        # Graficar datos y línea de tendencia
                        ax1_5.plot(x, y, 'o-', alpha=0.75, label=f'{label}', color=color, markersize=3)
                        x_valid = x[valid_mask]
                        ax1_5.plot(x_valid, p(x_clean), '--', alpha=0.7, 
                                  label=f'Trend: {z[0]*7:.3f}[%/week], R²={r2:.3f}', color=color, linewidth=2)
                    else:
                        # Si no hay suficientes datos válidos, graficar solo los datos
                        ax1_5.plot(x, y, 'o-', alpha=0.75, label=f'{label} (No trend)', color=color, markersize=3)
            
            # Eje x: solo meses y años
            ax1_5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax1_5.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax1_5.legend(loc='upper right', fontsize=9, frameon=True)
            ax1_5.set_ylim([90, 110])
            # Establecer límites del eje x dinámicamente
            ax1_5.set_xlim([start_date_for_xlim, end_date_for_xlim])
            ax1_5.set_ylabel('Soiling Ratio [%]')
            ax1_5.set_xlabel('Week')
            ax1_5.set_title(f'Q25 Soiling Ratio - DustIQ (Time Slots)')
            ax1_5.grid(True)
            plt.tight_layout()
            if save_figures:
                save_plot_matplotlib(fig1_5, 'SR_DustIQ_FranjasHorarias_Especificas_ConTendencia.png', output_graph_dir, subfolder=dustiq_graph_subdir)
                fig1_5 = None
            if show_figures: plt.show()
        except Exception as e:
            logger.error(f"Error generando Gráfico 1.5 de DustIQ (franjas específicas con tendencia): {e}", exc_info=True)
        finally:
            if fig1_5 is not None and plt.fignum_exists(fig1_5.number):
                if not show_figures: plt.close(fig1_5)
    else:
        logger.info(f"No hay datos de {sr_c11_col_name} filtrados para Gráfico 1.5.")

    # Gráfico 2: Media semanal de SR_C11_Avg (procesado con un offset) para el horario de 14:00-18:00
    if not sr_c11_filtered.empty:
        fig2 = None
        try:
            sr_dustiq_semanal = sr_c11_filtered.between_time('14:00','18:00').resample('1W', origin='start').quantile(0.25)
            # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
            sr_dustiq_semanal = sr_dustiq_semanal[sr_dustiq_semanal.index >= pd.Timestamp('2024-07-23')]
            if not sr_dustiq_semanal.empty:
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                
                # Eje x: fechas resampleadas
                x2 = sr_dustiq_semanal.index
                # Convertir fechas a números para la regresión
                x2_num = x2.map(lambda d: d.toordinal())
                y2 = sr_dustiq_semanal.values
                # Ajustar la regresión lineal
                z2 = np.polyfit(x2_num, y2, 1)
                p2 = np.poly1d(z2)
                # Calcular R²
                residuos2 = y2 - p2(x2_num)
                ss_res2 = np.sum(residuos2**2)
                ss_tot2 = np.sum((y2 - np.mean(y2))**2)
                r2_2 = 1 - (ss_res2 / ss_tot2)
                # Usar el primer color del ciclo de matplotlib
                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
                # Graficar datos SOLO con matplotlib
                ax2.plot(x2, y2, 'o-', alpha=0.75, label='DustIQ Q25 Weekly (14-18h)', color=color, markersize=3)
                # Graficar línea de tendencia sobre los mismos puntos
                ax2.plot(x2, p2(x2_num), '--', alpha=0.7, label=f'Trend= {z2[0]*7:.3f}[%/week], R²={r2_2:.3f})', color=color)
                # Eje x: solo meses y años
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                
                ax2.legend(loc='best', frameon=True)
                ax2.set_ylim([95, 102])
                # Establecer límites del eje x dinámicamente
                ax2.set_xlim([start_date_for_xlim, end_date_for_xlim])
                ax2.set_ylabel('Soiling Ratio [%]')
                ax2.set_xlabel('Week')
                ax2.set_title('Weekly Q25 Soiling Ratio DustIQ')
                ax2.grid(True)
                plt.tight_layout()
                if save_figures:
                    save_plot_matplotlib(fig2, 'SR_DustIQ_Procesado_Semanal.png', output_graph_dir, subfolder=dustiq_graph_subdir)
                    fig2 = None
                if show_figures: plt.show()
            else:
                logger.info("No hay datos para sr_dustiq_semanal (Gráfico 2) después del procesamiento.")
        except Exception as e:
            logger.error(f"Error generando Gráfico 2 de DustIQ: {e}", exc_info=True)
        finally:
            if fig2 is not None and plt.fignum_exists(fig2.number):
                if not show_figures: plt.close(fig2)
    else:
        logger.info(f"No hay datos de {sr_c11_col_name} filtrados para Gráfico 2.")

    # Gráfico 1.1: Medias SEMANALES de SR_C11_Avg para 3 franjas relativas al MEDIODÍA SOLAR
    df_mediodia_solar_comun = None
    if not sr_c11_filtered.empty:
        try:
            min_date_str_comun = sr_c11_filtered.index.min().strftime('%Y-%m-%d')
            max_date_str_comun = sr_c11_filtered.index.max().strftime('%Y-%m-%d')
            if 'medio_dia_solar' in globals():
                calculador_md_comun = medio_dia_solar(datei=min_date_str_comun, datef=max_date_str_comun, freq="1d", inter=0)
                df_mediodia_pd_temp_comun = calculador_md_comun.msd()
                if df_mediodia_pd_temp_comun is not None and not df_mediodia_pd_temp_comun.empty:
                    df_mediodia_solar_comun = pd.DataFrame()
                    df_mediodia_solar_comun['SolarNoon_Center'] = pd.to_datetime(df_mediodia_pd_temp_comun.iloc[:, 0])
                    df_mediodia_solar_comun['fecha_para_join'] = df_mediodia_solar_comun['SolarNoon_Center'].dt.normalize()
                    df_mediodia_solar_comun.set_index('fecha_para_join', inplace=True)
                    df_mediodia_solar_comun = df_mediodia_solar_comun[['SolarNoon_Center']]
        except Exception as e:
            logger.error(f"Error calculando mediodía solar común: {e}")
            df_mediodia_solar_comun = None

    if not sr_c11_filtered.empty and df_mediodia_solar_comun is not None:
        # Gráfico 1.1: Semanal, 3 slots relativos al mediodía solar
        try:
            fig1_1, ax1_1 = plt.subplots(figsize=(12, 6))
            df_to_plot_g1_1 = sr_c11_filtered.to_frame()
            df_to_plot_g1_1['fecha_para_join'] = df_to_plot_g1_1.index.normalize()
            df_merged_g1_1 = pd.merge(df_to_plot_g1_1, df_mediodia_solar_comun, left_on='fecha_para_join', right_index=True, how='left')
            slot_duration_td = timedelta(minutes=60)
            half_slot_td = slot_duration_td / 2
            df_merged_g1_1['slot1_start_dt'] = df_merged_g1_1['SolarNoon_Center'] - slot_duration_td - half_slot_td
            df_merged_g1_1['slot1_end_dt']   = df_merged_g1_1['SolarNoon_Center'] - half_slot_td
            df_merged_g1_1['slot2_start_dt'] = df_merged_g1_1['SolarNoon_Center'] - half_slot_td
            df_merged_g1_1['slot2_end_dt']   = df_merged_g1_1['SolarNoon_Center'] + half_slot_td
            df_merged_g1_1['slot3_start_dt'] = df_merged_g1_1['SolarNoon_Center'] + half_slot_td
            df_merged_g1_1['slot3_end_dt']   = df_merged_g1_1['SolarNoon_Center'] + slot_duration_td + half_slot_td
            df_merged_g1_1['in_slot1'] = (df_merged_g1_1.index >= df_merged_g1_1['slot1_start_dt']) & (df_merged_g1_1.index < df_merged_g1_1['slot1_end_dt'])
            df_merged_g1_1['in_slot2'] = (df_merged_g1_1.index >= df_merged_g1_1['slot2_start_dt']) & (df_merged_g1_1.index < df_merged_g1_1['slot2_end_dt'])
            df_merged_g1_1['in_slot3'] = (df_merged_g1_1.index >= df_merged_g1_1['slot3_start_dt']) & (df_merged_g1_1.index < df_merged_g1_1['slot3_end_dt'])
            label_slot1 = f'SN -90 a -30 min'
            label_slot2 = f'SN +/- 30 min (Centro)'
            label_slot3 = f'SN +30 a +90 min'
            colores = plt.rcParams['axes.prop_cycle'].by_key()['color']
            # Preparar datos con filtro de fecha para cada slot
            slot1_data = df_merged_g1_1[df_merged_g1_1['in_slot1']].resample('1W', origin='start')[sr_c11_col_name].quantile(0.25)
            slot1_data = slot1_data[slot1_data.index >= pd.Timestamp('2024-07-23')]
            slot2_data = df_merged_g1_1[df_merged_g1_1['in_slot2']].resample('1W', origin='start')[sr_c11_col_name].quantile(0.25)
            slot2_data = slot2_data[slot2_data.index >= pd.Timestamp('2024-07-23')]
            slot3_data = df_merged_g1_1[df_merged_g1_1['in_slot3']].resample('1W', origin='start')[sr_c11_col_name].quantile(0.25)
            slot3_data = slot3_data[slot3_data.index >= pd.Timestamp('2024-07-23')]
            
            ax1_1.plot(slot1_data.index, slot1_data.values, 'o-', alpha=0.75, label=label_slot1, color=colores[0], markersize=3)
            ax1_1.plot(slot2_data.index, slot2_data.values, 'o-', alpha=0.75, label=label_slot2, color=colores[1], markersize=3)
            ax1_1.plot(slot3_data.index, slot3_data.values, 'o-', alpha=0.75, label=label_slot3, color=colores[2], markersize=3)
            ax1_1.legend()
            ax1_1.set_ylim([90, 110])
            # Establecer límites del eje x dinámicamente
            ax1_1.set_xlim([start_date_for_xlim, end_date_for_xlim])
            ax1_1.set_ylabel('Soiling Ratio [%]')
            ax1_1.set_xlabel('Month')
            ax1_1.set_title(f'Weekly DustIQ Soiling Ratio - Solar Noon')
            ax1_1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax1_1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax1_1.grid(True)
            plt.tight_layout()
            save_plot_matplotlib(fig1_1, 'SR_DustIQ_FranjasHorarias_MediodiaSolar_Semanal.png', output_graph_dir, subfolder=dustiq_graph_subdir)
            plt.close(fig1_1)
        except Exception as e:
            logger.error(f"Error generando Gráfico 1.1 (mediodía solar semanal): {e}")

        # Gráfico 1.2: Promedio diario en franja solar única CON LÍNEA DE TENDENCIA
        try:
            fig1_2, ax1_2 = plt.subplots(figsize=(12, 6))
            df_to_plot_g1_2 = sr_c11_filtered.to_frame()
            df_to_plot_g1_2['fecha_para_join'] = df_to_plot_g1_2.index.normalize()
            df_merged_g1_2 = pd.merge(df_to_plot_g1_2, df_mediodia_solar_comun, left_on='fecha_para_join', right_index=True, how='left')
            half_franja_td = timedelta(minutes=60)
            df_merged_g1_2['franja_start_dt'] = df_merged_g1_2['SolarNoon_Center'] - half_franja_td
            df_merged_g1_2['franja_end_dt']   = df_merged_g1_2['SolarNoon_Center'] + half_franja_td
            df_merged_g1_2['in_franja_diaria'] = (df_merged_g1_2.index >= df_merged_g1_2['franja_start_dt']) & (df_merged_g1_2.index < df_merged_g1_2['franja_end_dt'])
            sr_diario_en_franja = df_merged_g1_2[df_merged_g1_2['in_franja_diaria']][sr_c11_col_name].resample('D').quantile(0.25)
            
            if not sr_diario_en_franja.empty:
                # Eje x: fechas diarias
                x1_2 = sr_diario_en_franja.index
                # Convertir fechas a números para la regresión
                x1_2_num = x1_2.map(lambda d: d.toordinal())
                y1_2 = sr_diario_en_franja.values
                
                # Validar datos antes de la regresión
                logger.info(f"G1.2 - Datos para regresión: {len(y1_2)} puntos")
                logger.info(f"G1.2 - Rango de fechas: {x1_2.min()} a {x1_2.max()}")
                logger.info(f"G1.2 - Valores NaN en y1_2: {np.isnan(y1_2).sum()}")
                logger.info(f"G1_2 - Rango de valores y1_2: {np.nanmin(y1_2):.3f} a {np.nanmax(y1_2):.3f}")
                
                # Eliminar valores NaN antes de la regresión
                valid_mask = ~np.isnan(y1_2)
                if valid_mask.sum() < 2:
                    logger.warning("G1.2 - No hay suficientes datos válidos para la regresión")
                    # Graficar solo los datos sin tendencia
                    ax1_2.plot(x1_2, y1_2, 'o-', alpha=0.75, label='Soiling Ratio Q25 (+/- 60 min)', color='blue', markersize=3)
                    ax1_2.set_title(f'Daily Q25 Soiling Ratio Solar Noon (No Trend - Insufficient Data)')
                else:
                    x1_2_clean = x1_2_num[valid_mask]
                    y1_2_clean = y1_2[valid_mask]
                    
                    logger.info(f"G1.2 - Datos válidos para regresión: {len(y1_2_clean)} puntos")
                    
                    # Ajustar la regresión lineal
                    z1_2 = np.polyfit(x1_2_clean, y1_2_clean, 1)
                    p1_2 = np.poly1d(z1_2)
                    
                    # Calcular R²
                    residuos1_2 = y1_2_clean - p1_2(x1_2_clean)
                    ss_res1_2 = np.sum(residuos1_2**2)
                    ss_tot1_2 = np.sum((y1_2_clean - np.mean(y1_2_clean))**2)
                    r2_1_2 = 1 - (ss_res1_2 / ss_tot1_2)
                    
                    logger.info(f"G1.2 - Pendiente: {z1_2[0]:.6f}, R²: {r2_1_2:.6f}")
                    
                    # Usar el primer color del ciclo de matplotlib
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
                    
                    # Agregar barras de error si hay datos de incertidumbre diaria
                    if uncertainty_data_daily is not None and 'U_rel_k2' in uncertainty_data_daily.columns:
                        yerr_1_2 = []
                        uncertainty_index = uncertainty_data_daily.index
                        # Asegurar que las fechas tengan el mismo timezone
                        if x1_2.tz is None and uncertainty_index.tz is not None:
                            x1_2 = x1_2.tz_localize('UTC')
                        elif x1_2.tz is not None and uncertainty_index.tz is None:
                            uncertainty_index = uncertainty_index.tz_localize('UTC')
                        elif x1_2.tz is not None and uncertainty_index.tz is not None:
                            uncertainty_index = uncertainty_index.tz_convert(x1_2.tz)
                        
                        for i, date in enumerate(x1_2):
                            sr_val = y1_2[i]
                            if pd.notna(sr_val):
                                if date in uncertainty_index:
                                    u_rel = uncertainty_data_daily.loc[date, 'U_rel_k2']
                                else:
                                    time_diffs = abs(uncertainty_index - date)
                                    closest_idx = time_diffs.argmin()
                                    if time_diffs[closest_idx] <= pd.Timedelta(days=1):
                                        u_rel = uncertainty_data_daily.iloc[closest_idx]['U_rel_k2']
                                    else:
                                        u_rel = np.nan
                                
                                if pd.notna(u_rel):
                                    yerr_1_2.append(u_rel * sr_val / 100.0)
                                else:
                                    yerr_1_2.append(0)
                            else:
                                yerr_1_2.append(0)
                        
                        errorbar_result = ax1_2.errorbar(x1_2, y1_2, yerr=yerr_1_2, fmt='o-', alpha=0.75, 
                                                       label='Soiling Ratio Q25 (+/- 60 min)', color=color, markersize=3,
                                                       capsize=3, capthick=1.5, elinewidth=1.5, ecolor=color)
                    else:
                        # Graficar datos sin barras de error
                        ax1_2.plot(x1_2, y1_2, 'o-', alpha=0.75, label='Soiling Ratio Q25 (+/- 60 min)', color=color, markersize=3)
                    
                    # Graficar línea de tendencia solo para datos válidos
                    x1_2_valid = x1_2[valid_mask]
                    ax1_2.plot(x1_2_valid, p1_2(x1_2_clean), '--', alpha=0.7, 
                              label=f'Trend: {z1_2[0]:.6f}[%/day], R²={r2_1_2:.3f}', color=color, linewidth=2)
                
                # Eje x: solo meses y años
                ax1_2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax1_2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                
                ax1_2.legend(loc='upper right', frameon=True)
                ax1_2.set_ylim([90, 110])
                # Establecer límites del eje x dinámicamente
                ax1_2.set_xlim([start_date_for_xlim, end_date_for_xlim])
                ax1_2.set_ylabel('Soiling Ratio [%]')
                ax1_2.set_xlabel('Date')
                ax1_2.set_title(f'Daily Q25 Soiling Ratio Solar Noon')
                ax1_2.grid(True)
                plt.tight_layout()
                save_plot_matplotlib(fig1_2, 'SR_DustIQ_PromedioDiario_MediodiaSolar_ConTendencia.png', output_graph_dir, subfolder=dustiq_graph_subdir)
                plt.close(fig1_2)
            else:
                logger.info("No hay datos para sr_diario_en_franja (Gráfico 1.2) después del procesamiento.")
        except Exception as e:
            logger.error(f"Error generando Gráfico 1.2 (promedio diario solar con tendencia): {e}")

        # Gráfico 1.3: Promedios diarios en 3 franjas solares
        try:
            fig1_3, ax1_3 = plt.subplots(figsize=(12, 6))
            df_to_plot_g1_3 = sr_c11_filtered.to_frame()
            df_to_plot_g1_3['fecha_para_join'] = df_to_plot_g1_3.index.normalize()
            df_merged_g1_3 = pd.merge(df_to_plot_g1_3, df_mediodia_solar_comun, left_on='fecha_para_join', right_index=True, how='left')
            slot_duration_td = timedelta(minutes=60)
            half_slot_td = slot_duration_td / 2
            df_merged_g1_3['slot1_start_dt'] = df_merged_g1_3['SolarNoon_Center'] - slot_duration_td - half_slot_td
            df_merged_g1_3['slot1_end_dt']   = df_merged_g1_3['SolarNoon_Center'] - half_slot_td
            df_merged_g1_3['slot2_start_dt'] = df_merged_g1_3['SolarNoon_Center'] - half_slot_td
            df_merged_g1_3['slot2_end_dt']   = df_merged_g1_3['SolarNoon_Center'] + half_slot_td
            df_merged_g1_3['slot3_start_dt'] = df_merged_g1_3['SolarNoon_Center'] + half_slot_td
            df_merged_g1_3['slot3_end_dt']   = df_merged_g1_3['SolarNoon_Center'] + slot_duration_td + half_slot_td
            df_merged_g1_3['in_slot1'] = (df_merged_g1_3.index >= df_merged_g1_3['slot1_start_dt']) & (df_merged_g1_3.index < df_merged_g1_3['slot1_end_dt'])
            df_merged_g1_3['in_slot2'] = (df_merged_g1_3.index >= df_merged_g1_3['slot2_start_dt']) & (df_merged_g1_3.index < df_merged_g1_3['slot2_end_dt'])
            df_merged_g1_3['in_slot3'] = (df_merged_g1_3.index >= df_merged_g1_3['slot3_start_dt']) & (df_merged_g1_3.index < df_merged_g1_3['slot3_end_dt'])
            sr_diario_slot1 = df_merged_g1_3[df_merged_g1_3['in_slot1']][sr_c11_col_name].resample('D').quantile(0.25)
            sr_diario_slot2 = df_merged_g1_3[df_merged_g1_3['in_slot2']][sr_c11_col_name].resample('D').quantile(0.25)
            sr_diario_slot3 = df_merged_g1_3[df_merged_g1_3['in_slot3']][sr_c11_col_name].resample('D').quantile(0.25)
            ax1_3.plot(sr_diario_slot1.index, sr_diario_slot1.values, '-o', alpha=0.7, label='Daily Q25 Pre-Noon', color=colores[0], markersize=3)
            ax1_3.plot(sr_diario_slot2.index, sr_diario_slot2.values, '-o', alpha=0.7, label='Daily Q25 Noon', color=colores[1], markersize=3)
            ax1_3.plot(sr_diario_slot3.index, sr_diario_slot3.values, '-o', alpha=0.7, label='Daily Q25 Post-Noon', color=colores[2], markersize=3)
            ax1_3.legend()
            ax1_3.set_ylim([90, 110])
            # Establecer límites del eje x dinámicamente
            ax1_3.set_xlim([start_date_for_xlim, end_date_for_xlim])
            ax1_3.set_ylabel('Soiling Ratio [%]')
            ax1_3.set_xlabel('Date')
            ax1_3.set_title(f'Daily Q25 Soiling Ratio Solar Noon Time Slots')
            ax1_3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax1_3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax1_3.grid(True)
            plt.tight_layout()
            save_plot_matplotlib(fig1_3, 'SR_DustIQ_PromediosDiarios_3Slots_MediodiaSolar.png', output_graph_dir, subfolder=dustiq_graph_subdir)
            plt.close(fig1_3)
        except Exception as e:
            logger.error(f"Error generando Gráfico 1.3 (promedios diarios 3 franjas solares): {e}")

    # Gráfico 2.1: Media semanal de SR_C11_Avg para mediodía solar (estilo similar al Gráfico 2)
    if not sr_c11_filtered.empty and df_mediodia_solar_comun is not None:
        fig2_1 = None
        try:
            # Usar los datos del mediodía solar (franja única +/- 60 min)
            df_to_plot_mediodia = sr_c11_filtered.to_frame()
            df_to_plot_mediodia['fecha_para_join'] = df_to_plot_mediodia.index.normalize()
            df_merged_mediodia = pd.merge(df_to_plot_mediodia, df_mediodia_solar_comun, left_on='fecha_para_join', right_index=True, how='left')
            
            half_franja_td = timedelta(minutes=60)
            df_merged_mediodia['franja_start_dt'] = df_merged_mediodia['SolarNoon_Center'] - half_franja_td
            df_merged_mediodia['franja_end_dt']   = df_merged_mediodia['SolarNoon_Center'] + half_franja_td
            df_merged_mediodia['in_franja_mediodia'] = (df_merged_mediodia.index >= df_merged_mediodia['franja_start_dt']) & (df_merged_mediodia.index < df_merged_mediodia['franja_end_dt'])
            
            sr_mediodia_semanal = df_merged_mediodia[df_merged_mediodia['in_franja_mediodia']][sr_c11_col_name].resample('1W', origin='start').quantile(0.25)
            # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
            sr_mediodia_semanal = sr_mediodia_semanal[sr_mediodia_semanal.index >= pd.Timestamp('2024-07-23')]
            
            if not sr_mediodia_semanal.empty:
                fig2_1, ax2_1 = plt.subplots(figsize=(12, 6))
                
                # Eje x: fechas resampleadas
                x2_1 = sr_mediodia_semanal.index
                # Convertir fechas a números para la regresión
                x2_1_num = x2_1.map(lambda d: d.toordinal())
                y2_1 = sr_mediodia_semanal.values
                # Ajustar la regresión lineal
                z2_1 = np.polyfit(x2_1_num, y2_1, 1)
                p2_1 = np.poly1d(z2_1)
                # Calcular R²
                residuos2_1 = y2_1 - p2_1(x2_1_num)
                ss_res2_1 = np.sum(residuos2_1**2)
                ss_tot2_1 = np.sum((y2_1 - np.mean(y2_1))**2)
                r2_2_1 = 1 - (ss_res2_1 / ss_tot2_1)
                # Usar el primer color del ciclo de matplotlib
                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
                
                # Agregar barras de error si hay datos de incertidumbre semanal
                if uncertainty_data_weekly is not None and 'U_rel_k2' in uncertainty_data_weekly.columns:
                    yerr_2_1 = []
                    uncertainty_index = uncertainty_data_weekly.index
                    # Asegurar que las fechas tengan el mismo timezone
                    if x2_1.tz is None and uncertainty_index.tz is not None:
                        x2_1 = x2_1.tz_localize('UTC')
                    elif x2_1.tz is not None and uncertainty_index.tz is None:
                        uncertainty_index = uncertainty_index.tz_localize('UTC')
                    elif x2_1.tz is not None and uncertainty_index.tz is not None:
                        uncertainty_index = uncertainty_index.tz_convert(x2_1.tz)
                    
                    for i, date in enumerate(x2_1):
                        sr_val = y2_1[i]
                        if pd.notna(sr_val):
                            if date in uncertainty_index:
                                u_rel = uncertainty_data_weekly.loc[date, 'U_rel_k2']
                            else:
                                time_diffs = abs(uncertainty_index - date)
                                closest_idx = time_diffs.argmin()
                                if time_diffs[closest_idx] <= pd.Timedelta(days=3):
                                    u_rel = uncertainty_data_weekly.iloc[closest_idx]['U_rel_k2']
                                else:
                                    u_rel = np.nan
                            
                            if pd.notna(u_rel):
                                yerr_2_1.append(u_rel * sr_val / 100.0)
                            else:
                                yerr_2_1.append(0)
                        else:
                            yerr_2_1.append(0)
                    
                    errorbar_result = ax2_1.errorbar(x2_1, y2_1, yerr=yerr_2_1, fmt='o-', alpha=0.75, 
                                                   label='DustIQ Q25 Weekly', color=color, markersize=3,
                                                   capsize=3, capthick=1.5, elinewidth=1.5, ecolor=color)
                else:
                    # Graficar datos sin barras de error
                    ax2_1.plot(x2_1, y2_1, 'o-', alpha=0.75, label='DustIQ Q25 Weekly', color=color, markersize=3)
                
                # Graficar línea de tendencia sobre los mismos puntos
                ax2_1.plot(x2_1, p2_1(x2_1_num), '--', alpha=0.7, label=f'Trend= {z2_1[0]*7:.3f}[%/week], R²={r2_2_1:.3f})', color=color)
                # Eje x: solo meses y años
                ax2_1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax2_1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                
                # Leyenda dentro del cuadro del gráfico
                ax2_1.legend(loc='upper right', frameon=True)
                ax2_1.set_ylim([90, 110])
                # Establecer límites del eje x dinámicamente
                ax2_1.set_xlim([start_date_for_xlim, end_date_for_xlim])
                ax2_1.set_ylabel('Soiling Ratio [%]')
                ax2_1.set_xlabel('Week')
                ax2_1.set_title('Weekly DustIQ Soiling Ratio - Solar Noon')
                ax2_1.grid(True)
                plt.tight_layout()
                if save_figures:
                    save_plot_matplotlib(fig2_1, 'SR_DustIQ_MediodiaSolar_Semanal.png', output_graph_dir, subfolder=dustiq_graph_subdir)
                    fig2_1 = None
                if show_figures: plt.show()
            else:
                logger.info("No hay datos para sr_mediodia_semanal (Gráfico 2.1) después del procesamiento.")
        except Exception as e:
            logger.error(f"Error generando Gráfico 2.1 de DustIQ (mediodía solar semanal): {e}", exc_info=True)
        finally:
            if fig2_1 is not None and plt.fignum_exists(fig2_1.number):
                if not show_figures: plt.close(fig2_1)
    else:
        logger.info("No hay datos disponibles para Gráfico 2.1 (mediodía solar semanal).")

    # --- Generar Excel consolidado con todos los datos procesados ---
    try:
        import openpyxl
        
        # Crear directorio específico para Excel
        dustiq_csv_subdir = "/home/nicole/SR/SOILING/datos_procesados_analisis_integrado_py/dustiq"
        os.makedirs(dustiq_csv_subdir, exist_ok=True)
        
        excel_filename = "dustiq_datos_completos_agregados_Q25.xlsx"
        excel_filepath = os.path.join(dustiq_csv_subdir, excel_filename)
        
        logger.info(f"Generando Excel consolidado de DustIQ: {excel_filepath}")
        
        with pd.ExcelWriter(excel_filepath, engine='openpyxl', 
                           date_format='YYYY-MM-DD', 
                           datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
            
            # Hoja 1: Datos de franjas horarias fijas (semanales)
            if not sr_c11_filtered.empty:
                franjas_data = pd.DataFrame()
                franjas = {
                    '12:00-13:00': ('12:00', '13:00'),
                    '14:00-15:00': ('14:00', '15:00'),
                    '16:00-17:00': ('16:00', '17:00')
                }
                
                for label, (start, end) in franjas.items():
                    data = sr_c11_filtered.between_time(start, end).resample('1W', origin='start').quantile(0.25)
                    # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
                    data = data[data.index >= pd.Timestamp('2024-07-23')]
                    if not data.empty:
                        if data.index.tz is not None:
                            data.index = data.index.tz_localize(None)
                        franjas_data[f'SR_Q25_{label}'] = data
                
                if not franjas_data.empty:
                    franjas_data.to_excel(writer, sheet_name="Franjas_Horarias_Fijas_Semanal")
            
            # Hoja 1.1: Datos de franjas horarias específicas sin tendencia (semanales)
            if not sr_c11_filtered.empty:
                franjas_especificas_data = pd.DataFrame()
                franjas_especificas = {
                    '10:00-11:00': ('10:00', '11:00'),
                    '12:00-13:00': ('12:00', '13:00'),
                    '14:00-15:00': ('14:00', '15:00'),
                    '16:00-17:00': ('16:00', '17:00')
                }
                
                for label, (start, end) in franjas_especificas.items():
                    data = sr_c11_filtered.between_time(start, end).resample('1W', origin='start').quantile(0.25)
                    # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
                    data = data[data.index >= pd.Timestamp('2024-07-23')]
                    if not data.empty:
                        if data.index.tz is not None:
                            data.index = data.index.tz_localize(None)
                        franjas_especificas_data[f'SR_Q25_{label}'] = data
                
                if not franjas_especificas_data.empty:
                    franjas_especificas_data.to_excel(writer, sheet_name="Franjas_Especificas_Semanal")
            
            # Hoja 1.2: Datos mediodía solar semanal (estilo gráfico 2)
            if not sr_c11_filtered.empty and df_mediodia_solar_comun is not None:
                try:
                    df_to_plot_mediodia = sr_c11_filtered.to_frame()
                    df_to_plot_mediodia['fecha_para_join'] = df_to_plot_mediodia.index.normalize()
                    df_merged_mediodia = pd.merge(df_to_plot_mediodia, df_mediodia_solar_comun, left_on='fecha_para_join', right_index=True, how='left')
                    
                    half_franja_td = timedelta(minutes=60)
                    df_merged_mediodia['franja_start_dt'] = df_merged_mediodia['SolarNoon_Center'] - half_franja_td
                    df_merged_mediodia['franja_end_dt']   = df_merged_mediodia['SolarNoon_Center'] + half_franja_td
                    df_merged_mediodia['in_franja_mediodia'] = (df_merged_mediodia.index >= df_merged_mediodia['franja_start_dt']) & (df_merged_mediodia.index < df_merged_mediodia['franja_end_dt'])
                    
                    sr_mediodia_semanal = df_merged_mediodia[df_merged_mediodia['in_franja_mediodia']][sr_c11_col_name].resample('1W', origin='start').quantile(0.25)
                    # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
                    sr_mediodia_semanal = sr_mediodia_semanal[sr_mediodia_semanal.index >= pd.Timestamp('2024-07-23')]
                    
                    if not sr_mediodia_semanal.empty:
                        df_mediodia_semanal = pd.DataFrame()
                        if sr_mediodia_semanal.index.tz is not None:
                            sr_mediodia_semanal.index = sr_mediodia_semanal.index.tz_localize(None)
                        df_mediodia_semanal['SR_Q25_MediodiaSolar_Semanal'] = sr_mediodia_semanal
                        df_mediodia_semanal.to_excel(writer, sheet_name="MediodiaSolar_Semanal")
                
                except Exception as e:
                    logger.error(f"Error procesando datos de mediodía solar semanal para Excel: {e}")
            
            # Hoja 2: Datos 14:00-18:00 (semanal)
            if not sr_c11_filtered.empty:
                sr_dustiq_semanal = sr_c11_filtered.between_time('14:00','18:00').resample('1W', origin='start').quantile(0.25)
                # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
                sr_dustiq_semanal = sr_dustiq_semanal[sr_dustiq_semanal.index >= pd.Timestamp('2024-07-23')]
                if not sr_dustiq_semanal.empty:
                    df_semanal = pd.DataFrame()
                    if sr_dustiq_semanal.index.tz is not None:
                        sr_dustiq_semanal.index = sr_dustiq_semanal.index.tz_localize(None)
                    df_semanal['SR_Q25_14-18h_Semanal'] = sr_dustiq_semanal
                    df_semanal.to_excel(writer, sheet_name="Franja_14-18h_Semanal")
            
            # Hoja 3: Datos mediodía solar (semanales) - 3 slots
            if not sr_c11_filtered.empty and df_mediodia_solar_comun is not None:
                try:
                    df_to_plot_g1_1 = sr_c11_filtered.to_frame()
                    df_to_plot_g1_1['fecha_para_join'] = df_to_plot_g1_1.index.normalize()
                    df_merged_g1_1 = pd.merge(df_to_plot_g1_1, df_mediodia_solar_comun, left_on='fecha_para_join', right_index=True, how='left')
                    
                    slot_duration_td = timedelta(minutes=60)
                    half_slot_td = slot_duration_td / 2
                    df_merged_g1_1['slot1_start_dt'] = df_merged_g1_1['SolarNoon_Center'] - slot_duration_td - half_slot_td
                    df_merged_g1_1['slot1_end_dt']   = df_merged_g1_1['SolarNoon_Center'] - half_slot_td
                    df_merged_g1_1['slot2_start_dt'] = df_merged_g1_1['SolarNoon_Center'] - half_slot_td
                    df_merged_g1_1['slot2_end_dt']   = df_merged_g1_1['SolarNoon_Center'] + half_slot_td
                    df_merged_g1_1['slot3_start_dt'] = df_merged_g1_1['SolarNoon_Center'] + half_slot_td
                    df_merged_g1_1['slot3_end_dt']   = df_merged_g1_1['SolarNoon_Center'] + slot_duration_td + half_slot_td
                    df_merged_g1_1['in_slot1'] = (df_merged_g1_1.index >= df_merged_g1_1['slot1_start_dt']) & (df_merged_g1_1.index < df_merged_g1_1['slot1_end_dt'])
                    df_merged_g1_1['in_slot2'] = (df_merged_g1_1.index >= df_merged_g1_1['slot2_start_dt']) & (df_merged_g1_1.index < df_merged_g1_1['slot2_end_dt'])
                    df_merged_g1_1['in_slot3'] = (df_merged_g1_1.index >= df_merged_g1_1['slot3_start_dt']) & (df_merged_g1_1.index < df_merged_g1_1['slot3_end_dt'])
                    
                    mediodia_solar_semanal = pd.DataFrame()
                    
                    slot1_data = df_merged_g1_1[df_merged_g1_1['in_slot1']].resample('1W', origin='start')[sr_c11_col_name].quantile(0.25)
                    slot1_data = slot1_data[slot1_data.index >= pd.Timestamp('2024-07-23')]
                    slot2_data = df_merged_g1_1[df_merged_g1_1['in_slot2']].resample('1W', origin='start')[sr_c11_col_name].quantile(0.25)
                    slot2_data = slot2_data[slot2_data.index >= pd.Timestamp('2024-07-23')]
                    slot3_data = df_merged_g1_1[df_merged_g1_1['in_slot3']].resample('1W', origin='start')[sr_c11_col_name].quantile(0.25)
                    slot3_data = slot3_data[slot3_data.index >= pd.Timestamp('2024-07-23')]
                    
                    if not slot1_data.empty:
                        if slot1_data.index.tz is not None:
                            slot1_data.index = slot1_data.index.tz_localize(None)
                        mediodia_solar_semanal['SR_Q25_Pre_Mediodia'] = slot1_data
                    if not slot2_data.empty:
                        if slot2_data.index.tz is not None:
                            slot2_data.index = slot2_data.index.tz_localize(None)
                        mediodia_solar_semanal['SR_Q25_Mediodia_Centro'] = slot2_data
                    if not slot3_data.empty:
                        if slot3_data.index.tz is not None:
                            slot3_data.index = slot3_data.index.tz_localize(None)
                        mediodia_solar_semanal['SR_Q25_Post_Mediodia'] = slot3_data
                    
                    if not mediodia_solar_semanal.empty:
                        mediodia_solar_semanal.to_excel(writer, sheet_name="Mediodia_Solar_3Slots_Semanal")
                
                except Exception as e:
                    logger.error(f"Error procesando datos de mediodía solar semanales para Excel: {e}")
            
            # Hoja 4: Datos mediodía solar diario (franja única)
            if not sr_c11_filtered.empty and df_mediodia_solar_comun is not None:
                try:
                    df_to_plot_g1_2 = sr_c11_filtered.to_frame()
                    df_to_plot_g1_2['fecha_para_join'] = df_to_plot_g1_2.index.normalize()
                    df_merged_g1_2 = pd.merge(df_to_plot_g1_2, df_mediodia_solar_comun, left_on='fecha_para_join', right_index=True, how='left')
                    
                    half_franja_td = timedelta(minutes=60)
                    df_merged_g1_2['franja_start_dt'] = df_merged_g1_2['SolarNoon_Center'] - half_franja_td
                    df_merged_g1_2['franja_end_dt']   = df_merged_g1_2['SolarNoon_Center'] + half_franja_td
                    df_merged_g1_2['in_franja_diaria'] = (df_merged_g1_2.index >= df_merged_g1_2['franja_start_dt']) & (df_merged_g1_2.index < df_merged_g1_2['franja_end_dt'])
                    
                    sr_diario_en_franja = df_merged_g1_2[df_merged_g1_2['in_franja_diaria']][sr_c11_col_name].resample('D').quantile(0.25)
                    
                    if not sr_diario_en_franja.empty:
                        df_diario_solar = pd.DataFrame()
                        if sr_diario_en_franja.index.tz is not None:
                            sr_diario_en_franja.index = sr_diario_en_franja.index.tz_localize(None)
                        df_diario_solar['SR_Q25_Mediodia_Solar_Diario'] = sr_diario_en_franja
                        df_diario_solar.to_excel(writer, sheet_name="Mediodia_Solar_Franja_Diario")
                        
                        # Hoja 4.1: Datos con tendencia (incluyendo parámetros de regresión)
                        try:
                            # Calcular parámetros de tendencia
                            x1_2 = sr_diario_en_franja.index
                            x1_2_num = x1_2.map(lambda d: d.toordinal())
                            y1_2 = sr_diario_en_franja.values
                            
                            # Ajustar la regresión lineal
                            z1_2 = np.polyfit(x1_2_num, y1_2, 1)
                            p1_2 = np.poly1d(z1_2)
                            
                            # Calcular R²
                            residuos1_2 = y1_2 - p1_2(x1_2_num)
                            ss_res1_2 = np.sum(residuos1_2**2)
                            ss_tot1_2 = np.sum((y1_2 - np.mean(y1_2))**2)
                            r2_1_2 = 1 - (ss_res1_2 / ss_tot1_2)
                            
                            # Crear DataFrame con datos y tendencia
                            df_tendencia = pd.DataFrame({
                                'Fecha': x1_2,
                                'SR_Q25_Mediodia_Solar_Diario': y1_2,
                                'Tendencia_Lineal': p1_2(x1_2_num),
                                'Residuos': residuos1_2
                            })
                            df_tendencia.set_index('Fecha', inplace=True)
                            
                            # Agregar parámetros de regresión como comentario en la primera fila
                            df_tendencia.to_excel(writer, sheet_name="Mediodia_Solar_Diario_ConTendencia")
                            
                            # Crear hoja adicional con parámetros de regresión
                            parametros_regresion = pd.DataFrame({
                                'Parametro': [
                                    'Pendiente (%/dia)',
                                    'Pendiente (%/año)', 
                                    'Intercepto',
                                    'R_cuadrado',
                                    'Numero_Puntos',
                                    'Fecha_Inicio',
                                    'Fecha_Fin'
                                ],
                                'Valor': [
                                    f"{z1_2[0]:.6f}",
                                    f"{z1_2[0]*365:.3f}",
                                    f"{z1_2[1]:.3f}",
                                    f"{r2_1_2:.6f}",
                                    len(y1_2_clean),
                                    x1_2_valid.min().strftime('%Y-%m-%d'),
                                    x1_2_valid.max().strftime('%Y-%m-%d')
                                ]
                            })
                            parametros_regresion.to_excel(writer, sheet_name="Parametros_Tendencia_Diaria", index=False)
                            
                        except Exception as e:
                            logger.error(f"Error calculando parámetros de tendencia para Excel: {e}")
                
                except Exception as e:
                    logger.error(f"Error procesando datos de mediodía solar diarios para Excel: {e}")
            
            # Hoja 5: Datos mediodía solar diario (3 franjas)
            if not sr_c11_filtered.empty and df_mediodia_solar_comun is not None:
                try:
                    df_to_plot_g1_3 = sr_c11_filtered.to_frame()
                    df_to_plot_g1_3['fecha_para_join'] = df_to_plot_g1_3.index.normalize()
                    df_merged_g1_3 = pd.merge(df_to_plot_g1_3, df_mediodia_solar_comun, left_on='fecha_para_join', right_index=True, how='left')
                    
                    slot_duration_td = timedelta(minutes=60)
                    half_slot_td = slot_duration_td / 2
                    df_merged_g1_3['slot1_start_dt'] = df_merged_g1_3['SolarNoon_Center'] - slot_duration_td - half_slot_td
                    df_merged_g1_3['slot1_end_dt']   = df_merged_g1_3['SolarNoon_Center'] - half_slot_td
                    df_merged_g1_3['slot2_start_dt'] = df_merged_g1_3['SolarNoon_Center'] - half_slot_td
                    df_merged_g1_3['slot2_end_dt']   = df_merged_g1_3['SolarNoon_Center'] + half_slot_td
                    df_merged_g1_3['slot3_start_dt'] = df_merged_g1_3['SolarNoon_Center'] + half_slot_td
                    df_merged_g1_3['slot3_end_dt']   = df_merged_g1_3['SolarNoon_Center'] + slot_duration_td + half_slot_td
                    df_merged_g1_3['in_slot1'] = (df_merged_g1_3.index >= df_merged_g1_3['slot1_start_dt']) & (df_merged_g1_3.index < df_merged_g1_3['slot1_end_dt'])
                    df_merged_g1_3['in_slot2'] = (df_merged_g1_3.index >= df_merged_g1_3['slot2_start_dt']) & (df_merged_g1_3.index < df_merged_g1_3['slot2_end_dt'])
                    df_merged_g1_3['in_slot3'] = (df_merged_g1_3.index >= df_merged_g1_3['slot3_start_dt']) & (df_merged_g1_3.index < df_merged_g1_3['slot3_end_dt'])
                    
                    mediodia_solar_diario = pd.DataFrame()
                    
                    sr_diario_slot1 = df_merged_g1_3[df_merged_g1_3['in_slot1']][sr_c11_col_name].resample('D').quantile(0.25)
                    sr_diario_slot2 = df_merged_g1_3[df_merged_g1_3['in_slot2']][sr_c11_col_name].resample('D').quantile(0.25)
                    sr_diario_slot3 = df_merged_g1_3[df_merged_g1_3['in_slot3']][sr_c11_col_name].resample('D').quantile(0.25)
                    
                    if not sr_diario_slot1.empty:
                        if sr_diario_slot1.index.tz is not None:
                            sr_diario_slot1.index = sr_diario_slot1.index.tz_localize(None)
                        mediodia_solar_diario['SR_Q25_Pre_Mediodia_Diario'] = sr_diario_slot1
                    if not sr_diario_slot2.empty:
                        if sr_diario_slot2.index.tz is not None:
                            sr_diario_slot2.index = sr_diario_slot2.index.tz_localize(None)
                        mediodia_solar_diario['SR_Q25_Mediodia_Centro_Diario'] = sr_diario_slot2
                    if not sr_diario_slot3.empty:
                        if sr_diario_slot3.index.tz is not None:
                            sr_diario_slot3.index = sr_diario_slot3.index.tz_localize(None)
                        mediodia_solar_diario['SR_Q25_Post_Mediodia_Diario'] = sr_diario_slot3
                    
                    if not mediodia_solar_diario.empty:
                        mediodia_solar_diario.to_excel(writer, sheet_name="Mediodia_Solar_3Slots_Diario")
                
                except Exception as e:
                    logger.error(f"Error procesando datos de mediodía solar diarios (3 slots) para Excel: {e}")
            
            # Hoja 5.1: Datos de franjas horarias específicas con tendencias (Gráfico 1.5)
            if not sr_c11_filtered.empty:
                try:
                    franjas_especificas_excel = {
                        'SR_Q25_10AM_11AM': ('10:00', '11:00'),
                        'SR_Q25_12PM_1PM': ('12:00', '13:00'),
                        'SR_Q25_2PM_3PM': ('14:00', '15:00'),
                        'SR_Q25_4PM_5PM': ('16:00', '17:00')
                    }
                    
                    franjas_con_tendencias = pd.DataFrame()
                    
                    for label, (start, end) in franjas_especificas_excel.items():
                        data = sr_c11_filtered.between_time(start, end).resample('1W', origin='start').quantile(0.25)
                        # Filtrar datos para eliminar fechas anteriores al 23 de julio de 2024
                        data = data[data.index >= pd.Timestamp('2024-07-23')]
                        if not data.empty:
                            if data.index.tz is not None:
                                data.index = data.index.tz_localize(None)
                            franjas_con_tendencias[label] = data
                    
                    if not franjas_con_tendencias.empty:
                        franjas_con_tendencias.to_excel(writer, sheet_name="Franjas_Especificas_ConTendencias")
                        
                        # Hoja adicional con parámetros de tendencia para cada franja
                        try:
                            parametros_tendencias_franjas = []
                            
                            for label, (start, end) in franjas_especificas_excel.items():
                                if label in franjas_con_tendencias.columns:
                                    data = franjas_con_tendencias[label].dropna()
                                    if len(data) >= 2:
                                        x_num = data.index.map(lambda d: d.toordinal())
                                        y = data.values
                                        
                                        # Ajustar la regresión lineal
                                        z = np.polyfit(x_num, y, 1)
                                        
                                        # Calcular R²
                                        p = np.poly1d(z)
                                        residuos = y - p(x_num)
                                        ss_res = np.sum(residuos**2)
                                        ss_tot = np.sum((y - np.mean(y))**2)
                                        r2 = 1 - (ss_res / ss_tot)
                                        
                                        parametros_tendencias_franjas.append({
                                            'Franja': f'{start}-{end}',
                                            'Pendiente (%/semana)': f"{z[0]*7:.6f}",
                                            'Pendiente (%/año)': f"{z[0]*365:.3f}",
                                            'Intercepto': f"{z[1]:.3f}",
                                            'R_cuadrado': f"{r2:.6f}",
                                            'Numero_Puntos': len(y),
                                            'Fecha_Inicio': data.index.min().strftime('%Y-%m-%d'),
                                            'Fecha_Fin': data.index.max().strftime('%Y-%m-%d')
                                        })
                            
                            if parametros_tendencias_franjas:
                                df_parametros_franjas = pd.DataFrame(parametros_tendencias_franjas)
                                df_parametros_franjas.to_excel(writer, sheet_name="Parametros_Tendencias_Franjas", index=False)
                        
                        except Exception as e:
                            logger.error(f"Error calculando parámetros de tendencia para franjas específicas: {e}")
                
                except Exception as e:
                    logger.error(f"Error procesando datos de franjas específicas con tendencias para Excel: {e}")
            
            # Hoja 6: Estadísticas generales
            if not sr_c11_filtered.empty:
                stats_generales = pd.DataFrame({
                    'Estadistica': [
                        'Total_Puntos_Originales',
                        'Puntos_Filtrados_Umbral',
                        'Porcentaje_Datos_Validos',
                        'Valor_Minimo',
                        'Valor_Maximo',
                        'Promedio_General',
                        'Mediana_General',
                        'Desviacion_Estandar',
                        'Cuartil_25',
                        'Cuartil_75',
                        'Rango_Fechas'
                    ],
                    'Valor': [
                        len(df[sr_c11_col_name]) if sr_c11_col_name in df.columns else 0,
                        len(sr_c11_filtered),
                        f"{len(sr_c11_filtered) / len(df[sr_c11_col_name]) * 100:.2f}%" if sr_c11_col_name in df.columns and len(df[sr_c11_col_name]) > 0 else "N/A",
                        f"{sr_c11_filtered.min():.3f}%",
                        f"{sr_c11_filtered.max():.3f}%",
                        f"{sr_c11_filtered.mean():.3f}%",
                        f"{sr_c11_filtered.median():.3f}%",
                        f"{sr_c11_filtered.std():.3f}%",
                        f"{sr_c11_filtered.quantile(0.25):.3f}%",
                        f"{sr_c11_filtered.quantile(0.75):.3f}%",
                        f"{sr_c11_filtered.index.min().strftime('%Y-%m-%d')} a {sr_c11_filtered.index.max().strftime('%Y-%m-%d')}"
                    ]
                })
                stats_generales.to_excel(writer, sheet_name="Estadisticas_Generales", index=False)
            
            # Hoja 7: Datos originales filtrados (muestra)
            if not sr_c11_filtered.empty:
                # Tomar una muestra de los datos originales para revisión
                sample_size = min(1000, len(sr_c11_filtered))
                df_sample = sr_c11_filtered.tail(sample_size).to_frame()
                df_sample.columns = ['SR_C11_Filtrado']
                df_sample.to_excel(writer, sheet_name="Datos_Originales_Muestra")
        
        logger.info(f"Excel consolidado de DustIQ generado exitosamente: {excel_filepath}")
        
    except ImportError:
        logger.warning("openpyxl no está disponible. No se pudo generar el archivo Excel consolidado para DustIQ.")
    except Exception as e:
        logger.error(f"Error generando Excel consolidado para DustIQ: {e}")

    # --- Generar archivos CSV semanales Q25 para consolidación ---
    logger.info("Generando archivos CSV semanales Q25 para consolidación...")
    
    try:
        # Crear directorio CSV si no existe
        csv_output_dir = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "dustiq")
        os.makedirs(csv_output_dir, exist_ok=True)
        
        # CSV 1: Datos semanales Q25 generales
        if not sr_c11_filtered.empty:
            weekly_q25_general = sr_c11_filtered.resample('1W', origin='start').quantile(0.25).dropna()
            weekly_q25_general = weekly_q25_general[weekly_q25_general.index >= pd.Timestamp('2024-07-23')]
            if not weekly_q25_general.empty:
                csv_filename = os.path.join(csv_output_dir, 'dustiq_sr_semanal_q25.csv')
                weekly_q25_general.to_csv(csv_filename)
                logger.info(f"CSV semanal Q25 general guardado: {csv_filename}")
        
        # CSV 2: Datos semanales Q25 por franjas horarias específicas
        if not sr_c11_filtered.empty:
            franjas_especificas = {
                'SR_Q25_10AM_11AM': ('10:00', '11:00'),
                'SR_Q25_12PM_1PM': ('12:00', '13:00'),
                'SR_Q25_2PM_3PM': ('14:00', '15:00'),
                'SR_Q25_4PM_5PM': ('16:00', '17:00')
            }
            
            franjas_df = pd.DataFrame()
            for label, (start, end) in franjas_especificas.items():
                data = sr_c11_filtered.between_time(start, end).resample('1W', origin='start').quantile(0.25)
                data = data[data.index >= pd.Timestamp('2024-07-23')]
                if not data.empty:
                    franjas_df[label] = data
            
            if not franjas_df.empty:
                csv_filename = os.path.join(csv_output_dir, 'dustiq_sr_franjas_semanal_q25.csv')
                franjas_df.to_csv(csv_filename)
                logger.info(f"CSV semanal Q25 por franjas guardado: {csv_filename}")
        
        # CSV 3: Datos semanales Q25 mediodía solar
        if not sr_c11_filtered.empty and 'df_merged_mediodia' in locals():
            sr_mediodia_semanal = df_merged_mediodia[df_merged_mediodia['in_franja_mediodia']][sr_c11_col_name].resample('1W', origin='start').quantile(0.25)
            sr_mediodia_semanal = sr_mediodia_semanal[sr_mediodia_semanal.index >= pd.Timestamp('2024-07-23')]
            if not sr_mediodia_semanal.empty:
                csv_filename = os.path.join(csv_output_dir, 'dustiq_sr_mediodia_semanal_q25.csv')
                sr_mediodia_semanal.to_csv(csv_filename)
                logger.info(f"CSV semanal Q25 mediodía solar guardado: {csv_filename}")
        
    except Exception as e:
        logger.error(f"Error generando archivos CSV semanales Q25: {e}")

    logger.info("--- Análisis y gráficos de DustIQ finalizados. ---")

def run_analysis():
    """
    Función principal que ejecuta el análisis de DustIQ.
    Esta función es importada por main.py
    """
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Directorios base
    BASE_INPUT_DIR = paths.BASE_INPUT_DIR
    BASE_OUTPUT_GRAPH_DIR = paths.BASE_OUTPUT_GRAPH_DIR

    # Archivo de datos DustIQ
    DUSTIQ_CSV_FILEPATH = paths.DUSTIQ_RAW_DATA_FILE

    # Fechas de análisis
    start_date_dustiq_str = '2024-06-24'  # Última semana de junio 2024
    end_date_dustiq_str = ANALYSIS_END_DATE_GENERAL_STR

    logger.info("--- Ejecutando Análisis de DustIQ ---")
    logger.info(f"Rango de fechas para análisis DustIQ: {start_date_dustiq_str} a {end_date_dustiq_str}")
        
    analyze_dustiq_data(
        data_filepath=DUSTIQ_CSV_FILEPATH,
        output_graph_dir=BASE_OUTPUT_GRAPH_DIR,
        analysis_start_date_str=start_date_dustiq_str,
        analysis_end_date_str=end_date_dustiq_str,
        save_figures=True,
        show_figures=False,
        generar_grafico1_1_mediodia_solar=True,
        duracion_slot_minutos_grafico1_1=60
    )

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando análisis de DustIQ...")
    raw_data_filepath = paths.DUSTIQ_RAW_DATA_FILE
    analyze_dustiq_data(raw_data_filepath, paths.BASE_OUTPUT_GRAPH_DIR) 