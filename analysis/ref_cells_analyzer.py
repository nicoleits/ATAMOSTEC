# analysis/ref_cells_analyzer.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import re
from datetime import timezone
from typing import Optional
import scipy.stats as stats

from config.logging_config import logger
from config import paths, settings
from utils.helpers import save_plot_matplotlib
from utils.solar_time import UtilsMedioDiaSolar
from datetime import timezone

def _adjust_series_start_to_100(series: pd.Series, series_name_for_log: str) -> pd.Series:
    """
    Ajusta una serie para que su primer valor válido sea 100.
    """
    if series.empty or series.dropna().empty:
        logger.warning(f"Serie '{series_name_for_log}' está vacía o no tiene datos válidos, no se puede ajustar/normalizar.")
        return series 
    
    first_valid_value = series.dropna().iloc[0]
    offset = 100 - first_valid_value
    logger.info(f"Ajustando serie '{series_name_for_log}' con offset: {offset:.2f} (primer valor válido: {first_valid_value:.2f})")
    return series + offset

def _calculate_ylim_from_data(series: pd.Series, margin_percent: float = 0.1) -> tuple:
    """
    Calcula los límites del eje Y basándose en los datos reales, agregando un margen.
    
    Args:
        series: Serie de datos para calcular los límites
        margin_percent: Porcentaje de margen a agregar arriba y abajo (default: 10%)
    
    Returns:
        tuple: (ymin, ymax) para usar en set_ylim
    """
    if series.empty or series.dropna().empty:
        return (0, 200)  # Valores por defecto si no hay datos
    
    valid_data = series.dropna()
    y_min = valid_data.min()
    y_max = valid_data.max()
    
    # Calcular el rango y agregar margen
    y_range = y_max - y_min
    if y_range == 0:
        # Si todos los valores son iguales, usar un rango mínimo
        y_range = max(abs(y_min) * 0.1, 10)
    
    margin = y_range * margin_percent
    y_min_adj = y_min - margin
    y_max_adj = y_max + margin
    
    # Asegurar que no sea negativo si todos los valores son positivos
    if y_min >= 0 and y_min_adj < 0:
        y_min_adj = 0
    
    return (y_min_adj, y_max_adj)

def analyze_ref_cells_data(raw_data_filepath: str) -> bool:
    """
    Analiza los datos de las celdas de referencia basándose en la lógica del notebook.
    Calcula Soiling Ratios (SR), los filtra, los ajusta opcionalmente,
    y genera CSVs y gráficos de resultados.
    """
    print("[INFO] Ejecutando analyze_ref_cells_data...")
    logger.info("--- Iniciando Análisis de Datos de Celdas de Referencia (Lógica Notebook) ---")
    
    os.makedirs(paths.REFCELLS_OUTPUT_SUBDIR_CSV, exist_ok=True)
    os.makedirs(paths.REFCELLS_OUTPUT_SUBDIR_GRAPH, exist_ok=True)

    try:
        # --- 1. Carga y Preprocesamiento de Datos ---
        logger.info(f"Cargando datos de celdas de referencia desde: {raw_data_filepath}")
        df_ref_cells = pd.read_csv(raw_data_filepath, index_col=settings.REFCELLS_TIME_COLUMN)
        logger.info(f"Datos de celdas de referencia cargados inicialmente: {len(df_ref_cells)} filas. Tipo de índice: {type(df_ref_cells.index)}")

        # Convertir índice a pd.to_datetime, manejar NaT, asegurar UTC
        df_ref_cells.index = pd.to_datetime(df_ref_cells.index, format=settings.REFCELLS_TIME_FORMAT, errors='coerce')
        
        rows_before_nat_drop = len(df_ref_cells)
        if df_ref_cells.index.hasnans:
            logger.info(f"Índice contiene NaTs después de pd.to_datetime(..., errors='coerce').")
            df_ref_cells = df_ref_cells[pd.notnull(df_ref_cells.index)]
            logger.info(f"Filas con NaT en el índice eliminadas. Antes: {rows_before_nat_drop}, Después: {len(df_ref_cells)}.")

        if df_ref_cells.empty:
            logger.warning("DataFrame de celdas de referencia vacío después de eliminar NaTs del índice.")
            print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
            return False # No se puede continuar sin índice válido

        if isinstance(df_ref_cells.index, pd.DatetimeIndex):
            if df_ref_cells.index.tz is None:
                logger.info("Índice es DatetimeIndex pero naive. Localizando a UTC (asumiendo que los tiempos crudos son UTC)...")
                df_ref_cells.index = df_ref_cells.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            elif df_ref_cells.index.tz != timezone.utc:
                logger.info(f"Índice es DatetimeIndex con zona horaria {df_ref_cells.index.tz}. Convirtiendo a UTC...")
                df_ref_cells.index = df_ref_cells.index.tz_convert('UTC')
            else:
                logger.info("Índice ya es DatetimeIndex y está en UTC.")
        else:
            logger.error(f"Índice no es DatetimeIndex después de la conversión. Tipo actual: {type(df_ref_cells.index)}. No se puede continuar.")
            return False
        
        df_ref_cells.sort_index(inplace=True)
        logger.info(f"Índice procesado a UTC y ordenado. Rango: {df_ref_cells.index.min()} a {df_ref_cells.index.max() if not df_ref_cells.empty else 'N/A'}")

        # --- 2. Filtrado por Fechas Globales ---
        start_date_analysis = pd.to_datetime(settings.ANALYSIS_START_DATE_GENERAL_STR, utc=True, errors='coerce')
        end_date_analysis = pd.to_datetime(settings.ANALYSIS_END_DATE_GENERAL_STR, utc=True, errors='coerce')
        
        # Para que la fecha final sea inclusiva hasta el final del día
        if pd.notna(end_date_analysis) and end_date_analysis.hour == 0 and end_date_analysis.minute == 0 and end_date_analysis.second == 0:
            end_date_analysis = end_date_analysis + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        original_rows_before_date_filter = len(df_ref_cells)
        if pd.notna(start_date_analysis):
            df_ref_cells = df_ref_cells[df_ref_cells.index >= start_date_analysis]
        if pd.notna(end_date_analysis):
            df_ref_cells = df_ref_cells[df_ref_cells.index <= end_date_analysis]
        
        if len(df_ref_cells) < original_rows_before_date_filter:
            logger.info(f"Datos filtrados por fecha ({settings.ANALYSIS_START_DATE_GENERAL_STR} - {settings.ANALYSIS_END_DATE_GENERAL_STR}). Antes: {original_rows_before_date_filter}, Después: {len(df_ref_cells)}")
        
        if df_ref_cells.empty:
            logger.warning("No hay datos de celdas de referencia después del filtrado por fecha.")
            print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
            return True # No es un error, simplemente no hay datos en el rango.

        # --- 3. Validación de Columnas ---
        ref_col = settings.REFCELLS_REFERENCE_COLUMN
        soiled_cols_config = settings.REFCELLS_SOILED_COLUMNS_TO_ANALYZE
        
        if ref_col not in df_ref_cells.columns:
            logger.error(f"Columna de referencia '{ref_col}' no encontrada. Columnas disponibles: {df_ref_cells.columns.tolist()}")
            print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
            return False
            
        valid_soiled_cols = [col for col in soiled_cols_config if col in df_ref_cells.columns]
        if not valid_soiled_cols:
            logger.error(f"Ninguna de las columnas 'sucias' especificadas {soiled_cols_config} se encontró. Columnas disponibles: {df_ref_cells.columns.tolist()}")
            print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
            return False
        logger.info(f"Columna de referencia: {ref_col}. Columnas 'sucias' válidas para análisis: {valid_soiled_cols}")

        # Convertir columnas de datos a numérico, por si acaso
        cols_to_convert = [ref_col] + valid_soiled_cols + settings.REFCELLS_IRRADIANCE_COLUMNS_TO_PLOT
        for col in list(set(cols_to_convert)): # set to avoid duplicates
             if col in df_ref_cells.columns:
                df_ref_cells[col] = pd.to_numeric(df_ref_cells[col], errors='coerce')
        df_ref_cells.dropna(subset=[ref_col] + valid_soiled_cols, how='any', inplace=True) # Requiere que todas las celdas involucradas en SR tengan valor

        if df_ref_cells.empty:
            logger.warning(f"No hay datos después de convertir a numérico y eliminar NaNs en columnas de trabajo ({ref_col}, {valid_soiled_cols}).")
            print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
            return True

        # --- 4. Cálculo de Soiling Ratios (SR) ---
        # Umbral mínimo de irradiancia para calcular SR (filtrar datos de baja irradiancia/noche)
        min_irradiance_threshold = settings.MIN_IRRADIANCE  # 200 W/m² por defecto
        
        # Filtrar por irradiancia mínima: la celda de referencia debe tener irradiancia >= umbral
        # Esto elimina datos nocturnos y de baja irradiancia donde el SR no es confiable
        mask_irradiance_valid = df_ref_cells[ref_col] >= min_irradiance_threshold
        n_filtered_by_irradiance = (~mask_irradiance_valid).sum()
        if n_filtered_by_irradiance > 0:
            logger.info(f"Filtrados {n_filtered_by_irradiance} datos con irradiancia < {min_irradiance_threshold} W/m² antes de calcular SR")
        
        sr_df = pd.DataFrame(index=df_ref_cells.index)
        for soiled_col in valid_soiled_cols:
            # Asegurar que el denominador no sea cero Y que la irradiancia sea >= umbral
            mask_denom_valid = mask_irradiance_valid & (df_ref_cells[ref_col] > 0)
            
            sr_df[soiled_col] = np.nan # Inicializar
            sr_df.loc[mask_denom_valid, soiled_col] = (df_ref_cells.loc[mask_denom_valid, soiled_col] / df_ref_cells.loc[mask_denom_valid, ref_col]) * 100
        
        logger.info("Soiling Ratios calculados.")
        sr_df.dropna(how='all', inplace=True)
        if sr_df.empty:
            logger.warning("DataFrame de SR vacío después del cálculo y dropna.")
            print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
            return True

        # --- 4.5. Análisis de Propagación de Incertidumbre de SR ---
        logger.info("Iniciando análisis de propagación de incertidumbre de SR...")
        try:
            from analysis.sr_uncertainty_propagation import run_uncertainty_propagation_analysis
            # Pasar el DataFrame original con las columnas S y C
            # La función de incertidumbre se encargará de sus propios filtros de calidad.
            uncertainty_success = run_uncertainty_propagation_analysis(
                df_ref_cells,
                soiled_col=settings.SOILED_COL,
                clean_col=settings.CLEAN_COL
            )
            if uncertainty_success:
                logger.info("✅ Análisis de propagación de incertidumbre completado exitosamente.")
            else:
                logger.warning("⚠️  El análisis de propagación de incertidumbre no se completó exitosamente.")
        except ImportError as e:
            logger.error(f"No se pudo importar el módulo 'sr_uncertainty_propagation': {e}")
        except Exception as e:
            logger.error(f"Error al ejecutar el análisis de propagación de incertidumbre: {e}", exc_info=True)
        # Continuar con el resto del análisis aunque falle la incertidumbre

        # --- 5. Filtrado de SR ---
        sr_min_val = settings.REFCELLS_SR_MIN_FILTER * 100
        sr_max_val = settings.REFCELLS_SR_MAX_FILTER * 100
        
        sr_filtered_df = sr_df.copy()
        for col in valid_soiled_cols: # Usar valid_soiled_cols que son las keys en sr_df
            if col in sr_filtered_df.columns:
                condition = (sr_filtered_df[col] >= sr_min_val) & (sr_filtered_df[col] <= sr_max_val)
                sr_filtered_df[col] = sr_filtered_df[col].where(condition)
        
        sr_filtered_df.dropna(how='all', inplace=True)
        logger.info(f"Soiling Ratios filtrados ({sr_min_val}% - {sr_max_val}%). Filas restantes: {len(sr_filtered_df)}")
        if sr_filtered_df.empty:
            logger.warning("DataFrame de SR filtrado está vacío.")
            # Aún así, generar gráfico de irradiancia si es posible
            # Intentar generar gráfico de irradiancia antes de retornar
            pass # Continuar para el gráfico de irradiancia

        # --- 6. Preparar DataFrames para CSV (Minutal, Diario Q25, Semanal Q25) ---
        # Guardar SRs filtrados (minutales)
        df_minutal_sr_to_save = sr_filtered_df.copy()
        
        # Diario Q25 (sobre datos filtrados)
        df_daily_sr_q25 = sr_filtered_df[valid_soiled_cols].resample('D').quantile(0.25).dropna(how='all')
        
        # Semanal Q25 (sobre datos filtrados, 'W-SUN' para consistencia con '1W' del notebook)
        df_weekly_sr_q25 = sr_filtered_df[valid_soiled_cols].resample('W-SUN').quantile(0.25).dropna(how='all')

        # --- 7. Guardar CSVs ---
        logger.info("Guardando resultados de SR de Celdas de Referencia en CSVs...")
        df_minutal_sr_to_save.to_csv(os.path.join(paths.REFCELLS_OUTPUT_SUBDIR_CSV, 'ref_cells_sr_minutal_filtrado.csv'))
        if not df_daily_sr_q25.empty:
            df_daily_sr_q25.to_csv(os.path.join(paths.REFCELLS_OUTPUT_SUBDIR_CSV, 'ref_cells_sr_diario_q25.csv'))
        if not df_weekly_sr_q25.empty:
            df_weekly_sr_q25.to_csv(os.path.join(paths.REFCELLS_OUTPUT_SUBDIR_CSV, 'ref_cells_sr_semanal_q25.csv'))

        # --- 8. Análisis de Incertidumbre de SR ---
        # Módulo sr_uncertainty eliminado - comentado temporalmente
        # logger.info("Iniciando análisis de incertidumbre de SR...")
        # try:
        #     from analysis.sr_uncertainty import calculate_sr_uncertainty, save_uncertainty_results
        #     
        #     # Calcular incertidumbre usando los datos originales filtrados
        #     uncertainty_results = calculate_sr_uncertainty(df_ref_cells)
        #     
        #     if uncertainty_results:
        #         # Guardar resultados de incertidumbre
        #         save_success = save_uncertainty_results(uncertainty_results)
        #         if save_success:
        #             logger.info("Análisis de incertidumbre completado exitosamente")
        #         else:
        #             logger.warning("Error guardando resultados de incertidumbre")
        #     else:
        #         logger.warning("No se obtuvieron resultados de incertidumbre")
        #         
        # except Exception as e:
        #     logger.error(f"Error en análisis de incertidumbre: {e}")
        #     # Continuar con el resto del análisis aunque falle la incertidumbre

        # --- 8. Generación de Gráficos ---
        logger.info("Generando gráficos para Celdas de Referencia...")
        plt.style.use('seaborn-v0_8-whitegrid') # Estilo similar al notebook

        # --- 8.1 Gráfico de Irradiancias ---
        irr_cols_to_plot = settings.REFCELLS_IRRADIANCE_COLUMNS_TO_PLOT
        actual_irr_cols_to_plot = [col for col in irr_cols_to_plot if col in df_ref_cells.columns]

        if actual_irr_cols_to_plot:
            fig_irr, ax_irr = plt.subplots(figsize=(12, 6))
            plotted_irradiance = False
            for col_irr in actual_irr_cols_to_plot:
                # Usar df_ref_cells que tiene los datos de irradiancia originales (después de filtro de fecha y NaNs)
                serie_irr_daily = df_ref_cells[col_irr].resample('D').mean().dropna()
                if not serie_irr_daily.empty:
                    serie_irr_daily.plot(ax=ax_irr, style='-', alpha=0.8, label=f'{col_irr} (Media Diaria)')
                    plotted_irradiance = True
            
            if plotted_irradiance:
                ax_irr.legend(loc='best')
                ax_irr.set_ylabel('Irradiance [W/m²]')
                ax_irr.set_xlabel('Day')
                ax_irr.set_title('Daily Irradiance - Selected Reference Cells')
                ax_irr.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                if settings.SAVE_FIGURES:
                    save_plot_matplotlib(fig_irr, 'refcells_irradiancia_seleccionada_diaria.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
                if settings.SHOW_FIGURES: plt.show()
                elif settings.SAVE_FIGURES: plt.close(fig_irr)
                if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES: plt.close(fig_irr)
            else:
                logger.warning(f"No se graficaron datos de irradiancia para las celdas seleccionadas {irr_cols_to_plot} (series diarias vacías).")
                plt.close(fig_irr) # Cerrar figura vacía
        else:
            logger.warning(f"Ninguna de las columnas de irradiancia especificadas {irr_cols_to_plot} se encontró en el DataFrame. No se generará el gráfico de irradiancias.")

        # Para los siguientes gráficos de SR, solo proceder si sr_filtered_df no está vacío.
        if sr_filtered_df.empty:
            logger.warning("DataFrame de SR filtrado está vacío, no se generarán gráficos de SR.")
            logger.info("--- Fin Análisis de Datos de Celdas de Referencia (Lógica Notebook) ---")
            print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
            return True # Fue exitoso hasta este punto (irradiance plot pudo haberse generado)


        # --- 8.2 Gráfico SR Semanal Combinado (Ajustado si aplica) ---
        fig_w, ax_w = plt.subplots(figsize=(12, 6))
        plotted_weekly_sr = False
        for col in valid_soiled_cols:
            if col in df_weekly_sr_q25.columns and not df_weekly_sr_q25[col].dropna().empty:
                serie_w = df_weekly_sr_q25[col].copy() # Usar la serie ya calculada y no vacía
                
                if settings.REFCELLS_ADJUST_TO_100_FLAG:
                    serie_w_adj = _adjust_series_start_to_100(serie_w, f"{col} Semanal Q25")
                else:
                    serie_w_adj = serie_w
                
                if not serie_w_adj.dropna().empty:
                    serie_w_adj.plot(ax=ax_w, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)
                    plotted_weekly_sr = True
        
        if plotted_weekly_sr:
            ax_w.legend(loc='best')
            ax_w.set_ylabel(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
            ax_w.set_xlabel('Week')
            ax_w.set_title(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} - Reference Cells (Weekly Q25)')
            # Límites fijos del eje Y: 50% a 110%
            ax_w.set_ylim([50, 110])
            ax_w.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if settings.SAVE_FIGURES:
                save_plot_matplotlib(fig_w, 'refcells_sr_combinado_semanal.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
            if settings.SHOW_FIGURES: plt.show()
            elif settings.SAVE_FIGURES: plt.close(fig_w)
            if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES: plt.close(fig_w)
        else:
            logger.warning("No se graficaron datos semanales combinados de SR para Celdas de Referencia.")
            if 'fig_w' in locals() and plt.fignum_exists(fig_w.number): plt.close(fig_w)

        # --- 8.3 Gráfico SR Diario Combinado (Ajustado si aplica) ---
        # Cargar datos de incertidumbre diaria para barras de error
        uncertainty_data_daily = None
        try:
            uncertainty_file = paths.SR_DAILY_ABS_WITH_U_FILE
            if os.path.exists(uncertainty_file):
                df_uncertainty = pd.read_csv(uncertainty_file, index_col='timestamp', parse_dates=True)
                if df_uncertainty.index.tz is None:
                    df_uncertainty.index = df_uncertainty.index.tz_localize('UTC')
                uncertainty_data_daily = df_uncertainty
        except Exception as e:
            logger.warning(f"No se pudieron cargar datos de incertidumbre diaria para gráfico combinado: {e}")
        
        fig_d, ax_d = plt.subplots(figsize=(12, 6))
        plotted_daily_sr = False
        for col in valid_soiled_cols:
            if col in df_daily_sr_q25.columns and not df_daily_sr_q25[col].dropna().empty:
                serie_d = df_daily_sr_q25[col].copy() # Usar la serie ya calculada y no vacía

                if settings.REFCELLS_ADJUST_TO_100_FLAG:
                    serie_d_adj = _adjust_series_start_to_100(serie_d, f"{col} Diario Q25")
                else:
                    serie_d_adj = serie_d
                
                if not serie_d_adj.dropna().empty:
                    # Intentar agregar barras de error si hay datos de incertidumbre
                    if uncertainty_data_daily is not None and 'U_rel_k2' in uncertainty_data_daily.columns:
                        uncertainty_index = uncertainty_data_daily.index
                        yerr = []
                        for date in serie_d_adj.index:
                            sr_val = serie_d_adj.loc[date]
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
                                    yerr.append(u_rel * sr_val / 100.0)
                                else:
                                    yerr.append(0)
                            else:
                                yerr.append(0)
                        
                        ax_d.errorbar(serie_d_adj.index, serie_d_adj.values, yerr=yerr, fmt='o-', 
                                     alpha=0.85, label=f'SR Photocells', linewidth=1, capsize=3, 
                                     capthick=1.5, elinewidth=1.5, ecolor='lightblue')
                    else:
                        serie_d_adj.plot(ax=ax_d, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)
                    plotted_daily_sr = True

        if plotted_daily_sr:
            ax_d.legend(loc='best')
            ax_d.set_ylabel(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
            ax_d.set_xlabel('Day')
            ax_d.set_title(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} - Reference Cells (Daily Q25)')
            # Límites fijos del eje Y: 50% a 110%
            ax_d.set_ylim([50, 110])
            ax_d.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            if settings.SAVE_FIGURES:
                save_plot_matplotlib(fig_d, 'refcells_sr_combinado_diario.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
            if settings.SHOW_FIGURES: plt.show()
            elif settings.SAVE_FIGURES: plt.close(fig_d)
            if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES: plt.close(fig_d)
        else:
            logger.warning("No se graficaron datos diarios combinados de SR para Celdas de Referencia.")
            if 'fig_d' in locals() and plt.fignum_exists(fig_d.number): plt.close(fig_d)
            
        # --- 8.4 Gráficos SR Individuales (Ajustados si aplica, Diario y Semanal) ---
        for col in valid_soiled_cols:
            safe_col_name = re.sub(r'[^a-zA-Z0-9_]', '', col)
            
            # Semanal Individual
            if col in df_weekly_sr_q25.columns and not df_weekly_sr_q25[col].dropna().empty:
                serie_w_ind = df_weekly_sr_q25[col].copy()
                
                if settings.REFCELLS_ADJUST_TO_100_FLAG:
                    serie_w_ind_adj = _adjust_series_start_to_100(serie_w_ind, f"{col} Semanal Ind. Q25")
                else:
                    serie_w_ind_adj = serie_w_ind

                if not serie_w_ind_adj.dropna().empty:
                    fig_w_ind, ax_w_ind = plt.subplots(figsize=(10, 5))
                    serie_w_ind_adj.plot(ax=ax_w_ind, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)
                    ax_w_ind.set_ylabel(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
                    ax_w_ind.set_xlabel('Week')
                    ax_w_ind.set_title('Soiling Ratio with Reference Cells')
                    ax_w_ind.grid(True, which='both', linestyle='--'); ax_w_ind.legend(loc='best')
                    ax_w_ind.set_ylim([50, 110])
                    ax_w_ind.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
                    if settings.SAVE_FIGURES:
                        save_plot_matplotlib(fig_w_ind, f'refcell_{safe_col_name}_sr_semanal.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
                    if settings.SHOW_FIGURES: plt.show()
                    elif settings.SAVE_FIGURES: plt.close(fig_w_ind)
                    if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES: plt.close(fig_w_ind)
                else: logger.info(f"No hay datos para graficar semanalmente (Q25 ajustado) la celda {col}.")
            else:
                logger.info(f"No hay datos en df_weekly_sr_q25 para {col} o está vacío, no se graficará semanal individual.")

            # Diario Individual
            if col in df_daily_sr_q25.columns and not df_daily_sr_q25[col].dropna().empty:
                serie_d_ind = df_daily_sr_q25[col].copy()

                if settings.REFCELLS_ADJUST_TO_100_FLAG:
                    serie_d_ind_adj = _adjust_series_start_to_100(serie_d_ind, f"{col} Diario Ind. Q25")
                else:
                    serie_d_ind_adj = serie_d_ind
                
                if not serie_d_ind_adj.dropna().empty:
                    fig_d_ind, ax_d_ind = plt.subplots(figsize=(10, 5))
                    serie_d_ind_adj.plot(ax=ax_d_ind, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)
                    ax_d_ind.set_ylabel(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
                    ax_d_ind.set_xlabel('Day')
                    ax_d_ind.set_title(f'SR{" Adj." if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} - {col} (Daily Q25)')
                    ax_d_ind.grid(True, which='both', linestyle='--'); ax_d_ind.legend(loc='best')
                    ax_d_ind.set_ylim([50, 110])
                    ax_d_ind.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
                    if settings.SAVE_FIGURES:
                        save_plot_matplotlib(fig_d_ind, f'refcell_{safe_col_name}_sr_diario.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
                    if settings.SHOW_FIGURES: plt.show()
                    elif settings.SAVE_FIGURES: plt.close(fig_d_ind)
                    if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES: plt.close(fig_d_ind)
                else: logger.info(f"No hay datos para graficar diariamente (Q25 ajustado) la celda {col}.")
            else:
                logger.info(f"No hay datos en df_daily_sr_q25 para {col} o está vacío, no se graficará diario individual.")

        # Generar gráfico específico para 1RC411
        _generate_specific_cell_plot(
            df_weekly_sr_q25,
            '1RC411(w.m-2)',
            '2024-07-23',
            '2025-05-22',  # Segunda semana de mayo 2025
            sr_min_val,
            sr_max_val
        )
        # Generar gráfico de los primeros 3 meses para 1RC411
        _generate_first_3_months_plot(
            df_weekly_sr_q25,
            '1RC411(w.m-2)',
            sr_min_val,
            sr_max_val
        )
        
        # Generar gráfico de los primeros 3 meses usando datos semanales pero con eje X como Month 1, Month 2, Month 3.
        _generate_first_3_months_weekly_plot(
            df_weekly_sr_q25,
            '1RC411(w.m-2)',
            sr_min_val,
            sr_max_val
        )
        
        # Generar gráfico de SR diario Q25 con tendencia para 1RC411
        _generate_daily_q25_trend_plot(
            df_daily_sr_q25,
            '1RC411(w.m-2)',
            sr_min_val,
            sr_max_val
        )
        
        # --- Análisis de Días Nublados durante Mediodía Solar ---
        logger.info("Iniciando análisis de días nublados durante mediodía solar...")
        _analyze_cloudy_days_solar_noon(df_ref_cells, ref_col, valid_soiled_cols)

        logger.info("--- Fin Análisis de Datos de Celdas de Referencia (Lógica Notebook) ---")
        print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
        return True

    except FileNotFoundError:
        logger.error(f"Archivo de datos de celdas de referencia no encontrado: {raw_data_filepath}")
    except pd.errors.EmptyDataError:
        logger.error(f"El archivo de datos {raw_data_filepath} está vacío o no se pudo parsear.")
    except KeyError as e_key:
        logger.error(f"Error de clave (columna faltante o mal configurada) en celdas de referencia: {e_key}")
    except Exception as e:
        logger.error(f"Error inesperado procesando celdas de referencia: {e}", exc_info=True)
    
    return False 


def filter_by_solar_noon(df: pd.DataFrame, hours_window: float = 2.5) -> pd.DataFrame:
    """
    Filtra un DataFrame por rangos de medio día solar dinámicos.
    
    Args:
        df: DataFrame con índice DatetimeIndex en UTC
        hours_window: Ventana en horas alrededor del medio día solar (total = 2 * hours_window)
    
    Returns:
        DataFrame filtrado por medio día solar
    """
    if df.empty:
        return df
        
    logger.info(f"Aplicando filtro de medio día solar (±{hours_window} horas alrededor del medio día solar real)")
    logger.info(f"DataFrame original: {len(df)} filas, rango: {df.index.min()} a {df.index.max()}")
    
    # Obtener rango de fechas del DataFrame
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    
    logger.info(f"Rango de fechas para cálculo solar: {start_date} a {end_date}")
    
    # Inicializar utilidades de medio día solar con parámetros correctos
    solar_utils = UtilsMedioDiaSolar(
        datei=start_date,
        datef=end_date,
        freq='D',
        inter=int(hours_window * 2 * 60),  # Convertir horas a minutos
        tz_local_str=settings.DUSTIQ_LOCAL_TIMEZONE_STR,
        lat=settings.SITE_LATITUDE,
        lon=settings.SITE_LONGITUDE,
        alt=settings.SITE_ALTITUDE
    )
    
    # Obtener intervalos de medio día solar
    solar_intervals_df = solar_utils.msd()
    
    if solar_intervals_df.empty:
        logger.warning("No se pudieron calcular intervalos de medio día solar. Retornando DataFrame vacío.")
        return pd.DataFrame()
    
    logger.info(f"Intervalos solares calculados: {len(solar_intervals_df)}")
    
    # Crear máscara para filtrar por medio día solar
    mask = pd.Series(False, index=df.index)
    
    # Asegurar que el índice del DataFrame esté en UTC
    df_utc = df.copy()
    if df_utc.index.tz is None:
        df_utc.index = df_utc.index.tz_localize('UTC')
    elif df_utc.index.tz != timezone.utc:
        df_utc.index = df_utc.index.tz_convert('UTC')
    
    logger.info(f"DataFrame convertido a UTC: {len(df_utc)} filas, rango: {df_utc.index.min()} a {df_utc.index.max()}")
    
    # Aplicar cada intervalo de medio día solar
    for i, (_, row) in enumerate(solar_intervals_df.iterrows()):
        # Los intervalos ya están en UTC naive, convertirlos a UTC aware
        start_time = pd.Timestamp(row[0]).tz_localize('UTC')
        end_time = pd.Timestamp(row[1]).tz_localize('UTC')
        
        # Aplicar máscara para este intervalo
        interval_mask = (df_utc.index >= start_time) & (df_utc.index <= end_time)
        mask = mask | interval_mask
    
    filtered_df = df_utc[mask]
    logger.info(f"Filtro de medio día solar aplicado: {len(df)} -> {len(filtered_df)} puntos ({len(filtered_df)/len(df)*100:.1f}%)")
    
    if filtered_df.empty:
        logger.warning("⚠️ DataFrame vacío después del filtro solar.")
    
    return filtered_df


def analyze_ref_cells_data_solar_noon(raw_data_filepath: str, hours_window: float = 2.5) -> bool:
    """
    Analiza los datos de las celdas de referencia filtrando solo por horario de mediodía solar.
    Similar a analyze_ref_cells_data pero aplica filtro de mediodía solar antes del procesamiento.
    Los resultados se guardan en carpetas separadas (mediodia_solar).
    
    Args:
        raw_data_filepath: Ruta al archivo CSV con datos crudos de refcells
        hours_window: Ventana en horas alrededor del medio día solar (default: 2.5 horas)
    
    Returns:
        bool: True si el análisis fue exitoso, False en caso contrario
    """
    print("[INFO] Ejecutando analyze_ref_cells_data_solar_noon...")
    logger.info("--- Iniciando Análisis de Datos de Celdas de Referencia (Mediodía Solar) ---")
    
    # Crear directorios de salida para mediodía solar
    os.makedirs(paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_CSV, exist_ok=True)
    os.makedirs(paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_GRAPH, exist_ok=True)
    
    try:
        # --- 1. Carga y Preprocesamiento de Datos ---
        logger.info(f"Cargando datos de celdas de referencia desde: {raw_data_filepath}")
        df_ref_cells = pd.read_csv(raw_data_filepath, index_col=settings.REFCELLS_TIME_COLUMN)
        logger.info(f"Datos de celdas de referencia cargados inicialmente: {len(df_ref_cells)} filas.")
        
        # Convertir índice a pd.to_datetime, manejar NaT, asegurar UTC
        df_ref_cells.index = pd.to_datetime(df_ref_cells.index, format=settings.REFCELLS_TIME_FORMAT, errors='coerce')
        
        rows_before_nat_drop = len(df_ref_cells)
        if df_ref_cells.index.hasnans:
            df_ref_cells = df_ref_cells[pd.notnull(df_ref_cells.index)]
            logger.info(f"Filas con NaT eliminadas. Antes: {rows_before_nat_drop}, Después: {len(df_ref_cells)}.")
        
        if df_ref_cells.empty:
            logger.warning("DataFrame de celdas de referencia vacío después de eliminar NaTs.")
            return False
        
        if isinstance(df_ref_cells.index, pd.DatetimeIndex):
            if df_ref_cells.index.tz is None:
                df_ref_cells.index = df_ref_cells.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            elif df_ref_cells.index.tz != timezone.utc:
                df_ref_cells.index = df_ref_cells.index.tz_convert('UTC')
        
        df_ref_cells.sort_index(inplace=True)
        logger.info(f"Índice procesado a UTC. Rango: {df_ref_cells.index.min()} a {df_ref_cells.index.max()}")
        
        # --- 1.5. FILTRAR POR MEDIODÍA SOLAR ---
        logger.info(f"Aplicando filtro de mediodía solar (ventana: ±{hours_window} horas)...")
        df_ref_cells = filter_by_solar_noon(df_ref_cells, hours_window)
        
        if df_ref_cells.empty:
            logger.warning("No hay datos después del filtro de mediodía solar.")
            return False
        
        logger.info(f"Datos después del filtro de mediodía solar: {len(df_ref_cells)} filas")
        
        # --- 2. Filtrado por Fechas Globales ---
        start_date_analysis = pd.to_datetime(settings.ANALYSIS_START_DATE_GENERAL_STR, utc=True, errors='coerce')
        end_date_analysis = pd.to_datetime(settings.ANALYSIS_END_DATE_GENERAL_STR, utc=True, errors='coerce')
        
        if pd.notna(end_date_analysis) and end_date_analysis.hour == 0 and end_date_analysis.minute == 0:
            end_date_analysis = end_date_analysis + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        
        original_rows_before_date_filter = len(df_ref_cells)
        if pd.notna(start_date_analysis):
            df_ref_cells = df_ref_cells[df_ref_cells.index >= start_date_analysis]
        if pd.notna(end_date_analysis):
            df_ref_cells = df_ref_cells[df_ref_cells.index <= end_date_analysis]
        
        if len(df_ref_cells) < original_rows_before_date_filter:
            logger.info(f"Datos filtrados por fecha. Antes: {original_rows_before_date_filter}, Después: {len(df_ref_cells)}")
        
        if df_ref_cells.empty:
            logger.warning("No hay datos después del filtrado por fecha.")
            return True
        
        # --- 3. Validación de Columnas ---
        ref_col = settings.REFCELLS_REFERENCE_COLUMN
        soiled_cols_config = settings.REFCELLS_SOILED_COLUMNS_TO_ANALYZE
        
        if ref_col not in df_ref_cells.columns:
            logger.error(f"Columna de referencia '{ref_col}' no encontrada. Columnas disponibles: {df_ref_cells.columns.tolist()}")
            return False
        
        valid_soiled_cols = [col for col in soiled_cols_config if col in df_ref_cells.columns]
        if not valid_soiled_cols:
            logger.error(f"Ninguna de las columnas 'sucias' especificadas {soiled_cols_config} se encontró. Columnas disponibles: {df_ref_cells.columns.tolist()}")
            return False
        
        logger.info(f"Columna de referencia: {ref_col}. Columnas 'sucias' válidas para análisis: {valid_soiled_cols}")
        
        # Convertir a numérico
        cols_to_convert = [ref_col] + valid_soiled_cols + settings.REFCELLS_IRRADIANCE_COLUMNS_TO_PLOT
        for col in list(set(cols_to_convert)):
            if col in df_ref_cells.columns:
                df_ref_cells[col] = pd.to_numeric(df_ref_cells[col], errors='coerce')
        df_ref_cells.dropna(subset=[ref_col] + valid_soiled_cols, how='any', inplace=True)
        
        if df_ref_cells.empty:
            logger.warning(f"No hay datos después de convertir a numérico.")
            return True
        
        # --- 4. Cálculo de Soiling Ratios (SR) ---
        min_irradiance_threshold = settings.MIN_IRRADIANCE
        
        sr_df = pd.DataFrame(index=df_ref_cells.index)
        mask_irradiance_valid = df_ref_cells[ref_col] >= min_irradiance_threshold
        n_filtered_by_irradiance = (~mask_irradiance_valid).sum()
        if n_filtered_by_irradiance > 0:
            logger.info(f"Filtrados {n_filtered_by_irradiance} datos con irradiancia < {min_irradiance_threshold} W/m²")
        
        for soiled_col in valid_soiled_cols:
            mask_denom_valid = mask_irradiance_valid & (df_ref_cells[ref_col] > 0)
            sr_df[soiled_col] = np.nan
            sr_df.loc[mask_denom_valid, soiled_col] = (df_ref_cells.loc[mask_denom_valid, soiled_col] / df_ref_cells.loc[mask_denom_valid, ref_col]) * 100
        
        logger.info("Soiling Ratios calculados.")
        sr_df.dropna(how='all', inplace=True)
        if sr_df.empty:
            logger.warning("DataFrame de SR vacío después del cálculo.")
            return True
        
        # --- 4.5. Análisis de Propagación de Incertidumbre de SR (Mediodía Solar) ---
        logger.info("Iniciando análisis de propagación de incertidumbre de SR (mediodía solar)...")
        try:
            from analysis.sr_uncertainty_propagation import run_uncertainty_propagation_analysis
            # Los datos ya están filtrados por mediodía solar
            # Para el análisis de incertidumbre, usar la primera columna sucia y la columna de referencia
            # (el módulo de incertidumbre espera una columna sucia y una limpia)
            soiled_col_for_uncertainty = valid_soiled_cols[0] if valid_soiled_cols else None
            if soiled_col_for_uncertainty:
                uncertainty_success = run_uncertainty_propagation_analysis(
                    df_ref_cells,
                    soiled_col=soiled_col_for_uncertainty,
                    clean_col=ref_col,
                    output_dir=paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR  # Usar directorio de mediodía solar
                )
            else:
                logger.warning("No hay columnas sucias válidas para análisis de incertidumbre")
                uncertainty_success = False
            if uncertainty_success:
                logger.info("✅ Análisis de propagación de incertidumbre completado (mediodía solar).")
            else:
                logger.warning("⚠️  El análisis de propagación de incertidumbre no se completó exitosamente.")
        except ImportError as e:
            logger.error(f"No se pudo importar el módulo 'sr_uncertainty_propagation': {e}")
        except Exception as e:
            logger.error(f"Error al ejecutar el análisis de propagación de incertidumbre: {e}", exc_info=True)
        
        # --- 5. Filtrado de SR ---
        sr_min_val = settings.REFCELLS_SR_MIN_FILTER * 100
        sr_max_val = settings.REFCELLS_SR_MAX_FILTER * 100
        
        sr_filtered_df = sr_df.copy()
        for col in valid_soiled_cols:
            if col in sr_filtered_df.columns:
                condition = (sr_filtered_df[col] >= sr_min_val) & (sr_filtered_df[col] <= sr_max_val)
                sr_filtered_df[col] = sr_filtered_df[col].where(condition)
        
        sr_filtered_df.dropna(how='all', inplace=True)
        logger.info(f"Soiling Ratios filtrados ({sr_min_val}% - {sr_max_val}%). Filas restantes: {len(sr_filtered_df)}")
        if sr_filtered_df.empty:
            logger.warning("DataFrame de SR filtrado está vacío.")
            return True
        
        # --- 6. Preparar DataFrames para CSV (Minutal, Diario Q25, Semanal Q25) ---
        # Guardar SRs filtrados (minutales)
        sr_minute_filename = os.path.join(paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_CSV, 'sr_minute_filtered.csv')
        sr_filtered_df.to_csv(sr_minute_filename)
        logger.info(f"SRs minutales filtrados guardados en: {sr_minute_filename}")
        
        # Agregación diaria (Q25) - solo columnas válidas
        df_daily_sr_q25 = sr_filtered_df[valid_soiled_cols].resample('D').quantile(0.25).dropna(how='all')
        daily_sr_q25_filename = os.path.join(paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_CSV, 'sr_daily_q25.csv')
        if not df_daily_sr_q25.empty:
            df_daily_sr_q25.to_csv(daily_sr_q25_filename)
            logger.info(f"SRs diarios Q25 guardados en: {daily_sr_q25_filename}")
        
        # Agregación semanal (Q25) - solo columnas válidas
        df_weekly_sr_q25 = sr_filtered_df[valid_soiled_cols].resample('W-SUN').quantile(0.25).dropna(how='all')
        weekly_sr_q25_filename = os.path.join(paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_CSV, 'sr_weekly_q25.csv')
        if not df_weekly_sr_q25.empty:
            df_weekly_sr_q25.to_csv(weekly_sr_q25_filename)
            logger.info(f"SRs semanales Q25 guardados en: {weekly_sr_q25_filename}")
        
        # --- 7. Generación de Gráficos (usando rutas de mediodía solar) ---
        logger.info("Generando gráficos para Celdas de Referencia (Mediodía Solar)...")
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Guardar rutas originales temporalmente
        original_csv_dir = paths.REFCELLS_OUTPUT_SUBDIR_CSV
        original_graph_dir = paths.REFCELLS_OUTPUT_SUBDIR_GRAPH
        
        # Usar rutas de mediodía solar temporalmente
        paths.REFCELLS_OUTPUT_SUBDIR_CSV = paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_CSV
        paths.REFCELLS_OUTPUT_SUBDIR_GRAPH = paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_GRAPH
        
        try:
            # --- 7.1 Gráfico de Irradiancias ---
            irr_cols_to_plot = settings.REFCELLS_IRRADIANCE_COLUMNS_TO_PLOT
            actual_irr_cols_to_plot = [col for col in irr_cols_to_plot if col in df_ref_cells.columns]
            
            if actual_irr_cols_to_plot:
                fig_irr, ax_irr = plt.subplots(figsize=(12, 6))
                plotted_irradiance = False
                for col_irr in actual_irr_cols_to_plot:
                    serie_irr_daily = df_ref_cells[col_irr].resample('D').mean().dropna()
                    if not serie_irr_daily.empty:
                        serie_irr_daily.plot(ax=ax_irr, style='-', alpha=0.8, label=f'{col_irr} (Media Diaria)')
                        plotted_irradiance = True
                
                if plotted_irradiance:
                    ax_irr.legend(loc='best')
                    ax_irr.set_ylabel('Irradiance [W/m²]')
                    ax_irr.set_xlabel('Day')
                    ax_irr.set_title('Daily Irradiance - Selected Reference Cells (Solar Noon)')
                    ax_irr.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    if settings.SAVE_FIGURES:
                        save_plot_matplotlib(fig_irr, 'refcells_irradiancia_seleccionada_diaria.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
                    if settings.SHOW_FIGURES: plt.show()
                    elif settings.SAVE_FIGURES: plt.close(fig_irr)
                    if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES: plt.close(fig_irr)
            
            # --- 7.2 Gráficos SR Semanal y Diario Combinados ---
            # Nota: df_daily_sr_q25 y df_weekly_sr_q25 ya fueron calculados arriba
            if not sr_filtered_df.empty and not df_weekly_sr_q25.empty:
                # Gráfico SR Semanal Combinado
                if not df_weekly_sr_q25.empty:
                    fig_w, ax_w = plt.subplots(figsize=(12, 6))
                    plotted_weekly_sr = False
                    for col in valid_soiled_cols:
                        if col in df_weekly_sr_q25.columns and not df_weekly_sr_q25[col].dropna().empty:
                            serie_w = df_weekly_sr_q25[col].copy()
                            if settings.REFCELLS_ADJUST_TO_100_FLAG:
                                serie_w_adj = _adjust_series_start_to_100(serie_w, f"{col} Semanal Q25")
                            else:
                                serie_w_adj = serie_w
                            if not serie_w_adj.dropna().empty:
                                serie_w_adj.plot(ax=ax_w, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)
                                plotted_weekly_sr = True
                    
                    if plotted_weekly_sr:
                        ax_w.legend(loc='best')
                        ax_w.set_ylabel(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
                        ax_w.set_xlabel('Week')
                        ax_w.set_title(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} - Reference Cells (Weekly Q25, Solar Noon)')
                        ax_w.set_ylim([50, 110])
                        ax_w.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        if settings.SAVE_FIGURES:
                            save_plot_matplotlib(fig_w, 'refcells_sr_combinado_semanal.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
                        if settings.SHOW_FIGURES: plt.show()
                        elif settings.SAVE_FIGURES: plt.close(fig_w)
                        if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES: plt.close(fig_w)
                
                # Gráfico SR Diario Combinado
                # Cargar datos de incertidumbre diaria para mediodía solar
                uncertainty_data_daily_solar_noon = None
                try:
                    # Para mediodía solar, los datos están en el subdirectorio mediodia_solar
                    uncertainty_file_solar_noon = os.path.join(paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR, "sr_daily_abs_with_U.csv")
                    if os.path.exists(uncertainty_file_solar_noon):
                        df_uncertainty = pd.read_csv(uncertainty_file_solar_noon, index_col='timestamp', parse_dates=True)
                        if df_uncertainty.index.tz is None:
                            df_uncertainty.index = df_uncertainty.index.tz_localize('UTC')
                        uncertainty_data_daily_solar_noon = df_uncertainty
                except Exception as e:
                    logger.warning(f"No se pudieron cargar datos de incertidumbre diaria para gráfico combinado (mediodía solar): {e}")
                
                if not df_daily_sr_q25.empty:
                    fig_d, ax_d = plt.subplots(figsize=(12, 6))
                    plotted_daily_sr = False
                    for col in valid_soiled_cols:
                        if col in df_daily_sr_q25.columns and not df_daily_sr_q25[col].dropna().empty:
                            serie_d = df_daily_sr_q25[col].copy()
                            if settings.REFCELLS_ADJUST_TO_100_FLAG:
                                serie_d_adj = _adjust_series_start_to_100(serie_d, f"{col} Diario Q25")
                            else:
                                serie_d_adj = serie_d
                            if not serie_d_adj.dropna().empty:
                                # Intentar agregar barras de error si hay datos de incertidumbre
                                if uncertainty_data_daily_solar_noon is not None and 'U_rel_k2' in uncertainty_data_daily_solar_noon.columns:
                                    uncertainty_index = uncertainty_data_daily_solar_noon.index
                                    yerr = []
                                    for date in serie_d_adj.index:
                                        sr_val = serie_d_adj.loc[date]
                                        if pd.notna(sr_val):
                                            if date in uncertainty_index:
                                                u_rel = uncertainty_data_daily_solar_noon.loc[date, 'U_rel_k2']
                                            else:
                                                time_diffs = abs(uncertainty_index - date)
                                                closest_idx = time_diffs.argmin()
                                                if time_diffs[closest_idx] <= pd.Timedelta(days=1):
                                                    u_rel = uncertainty_data_daily_solar_noon.iloc[closest_idx]['U_rel_k2']
                                                else:
                                                    u_rel = np.nan
                                            
                                            if pd.notna(u_rel):
                                                yerr.append(u_rel * sr_val / 100.0)
                                            else:
                                                yerr.append(0)
                                        else:
                                            yerr.append(0)
                                    
                                    ax_d.errorbar(serie_d_adj.index, serie_d_adj.values, yerr=yerr, fmt='o-', 
                                                 alpha=0.85, label=f'SR Photocells', linewidth=1, capsize=3, 
                                                 capthick=1.5, elinewidth=1.5, ecolor='lightblue')
                                else:
                                    serie_d_adj.plot(ax=ax_d, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)
                                plotted_daily_sr = True
                    
                    if plotted_daily_sr:
                        ax_d.legend(loc='best')
                        ax_d.set_ylabel(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
                        ax_d.set_xlabel('Day')
                        ax_d.set_title(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} - Reference Cells (Daily Q25, Solar Noon)')
                        ax_d.set_ylim([50, 110])
                        ax_d.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        if settings.SAVE_FIGURES:
                            save_plot_matplotlib(fig_d, 'refcells_sr_combinado_diario.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
                        if settings.SHOW_FIGURES: plt.show()
                        elif settings.SAVE_FIGURES: plt.close(fig_d)
                        if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES: plt.close(fig_d)
                
                # Generar gráficos individuales usando las funciones existentes
                # Estas funciones usarán las rutas temporales que configuramos
                for col in valid_soiled_cols:
                    if col in df_weekly_sr_q25.columns and not df_weekly_sr_q25[col].dropna().empty:
                        _generate_specific_cell_plot(
                            df_weekly_sr_q25,
                            col,
                            settings.ANALYSIS_START_DATE_GENERAL_STR,
                            settings.ANALYSIS_END_DATE_GENERAL_STR,
                            sr_min_val,
                            sr_max_val
                        )
                        _generate_first_3_months_plot(
                            df_weekly_sr_q25,
                            col,
                            sr_min_val,
                            sr_max_val
                        )
                        _generate_first_3_months_weekly_plot(
                            df_weekly_sr_q25,
                            col,
                            sr_min_val,
                            sr_max_val
                        )
                    
                    if col in df_daily_sr_q25.columns and not df_daily_sr_q25[col].dropna().empty:
                        # Usar archivo de incertidumbre de mediodía solar
                        uncertainty_file_solar_noon = os.path.join(paths.PROPAGACION_ERRORES_SOLAR_NOON_DIR, "sr_daily_abs_with_U.csv")
                        _generate_daily_q25_trend_plot(
                            df_daily_sr_q25,
                            col,
                            sr_min_val,
                            sr_max_val,
                            uncertainty_file=uncertainty_file_solar_noon
                        )
        finally:
            # Restaurar rutas originales
            paths.REFCELLS_OUTPUT_SUBDIR_CSV = original_csv_dir
            paths.REFCELLS_OUTPUT_SUBDIR_GRAPH = original_graph_dir
        
        logger.info("✅ Análisis de celdas de referencia (mediodía solar) completado exitosamente.")
        print(f"\nLos resultados de mediodía solar se guardaron en:")
        print(f"  - CSVs: {paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_CSV}")
        print(f"  - Gráficos: {paths.REFCELLS_SOLAR_NOON_OUTPUT_SUBDIR_GRAPH}\n")
        return True
        
    except Exception as e:
        logger.error(f"Error inesperado en análisis de mediodía solar: {e}", exc_info=True)
        return False


def _generate_specific_cell_plot(df_weekly_sr_q25: pd.DataFrame, cell_name: str, start_date: str, end_date: str, sr_min_val: float, sr_max_val: float) -> None:
    """
    Genera un gráfico específico de SR semanal para una celda particular con fechas personalizadas.
    Incluye línea de tendencia lineal, pendiente y R², y barras de error de incertidumbre.
    """
    if cell_name not in df_weekly_sr_q25.columns:
        logger.warning(f"La celda {cell_name} no se encuentra en los datos semanales.")
        return

    # Filtrar por fechas
    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True)
    df_filtered = df_weekly_sr_q25.loc[start:end]

    if df_filtered.empty:
        logger.warning(f"No hay datos para {cell_name} en el rango de fechas especificado.")
        return

    # Cargar datos de incertidumbre semanal
    uncertainty_data = None
    try:
        uncertainty_file = paths.SR_WEEKLY_ABS_WITH_U_FILE
        if os.path.exists(uncertainty_file):
            df_uncertainty = pd.read_csv(uncertainty_file, index_col='timestamp', parse_dates=True)
            if df_uncertainty.index.tz is None:
                df_uncertainty.index = df_uncertainty.index.tz_localize('UTC')
            # Filtrar por el mismo rango de fechas
            uncertainty_data = df_uncertainty.loc[start:end]
        else:
            logger.warning(f"Archivo de incertidumbre no encontrado: {uncertainty_file}")
    except Exception as e:
        logger.warning(f"No se pudieron cargar datos de incertidumbre: {e}")

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    serie = df_filtered[cell_name].copy()

    if settings.REFCELLS_ADJUST_TO_100_FLAG:
        serie = _adjust_series_start_to_100(serie, f"{cell_name} Semanal Q25")

    # Plotear datos originales con barras de error si están disponibles
    if uncertainty_data is not None and 'U_rel_k2' in uncertainty_data.columns and len(uncertainty_data) > 0:
        # Calcular barras de error: U_rel_k2 está en porcentaje, convertirlo a valor absoluto
        # U_rel_k2 es la incertidumbre relativa k=2, así que la barra de error es U_rel_k2 * SR / 100
        uncertainty_index = uncertainty_data.index
        yerr = []
        for date in serie.index:
            sr_val = serie.loc[date]
            if pd.notna(sr_val):
                # Buscar fecha más cercana en el archivo de incertidumbre
                if date in uncertainty_index:
                    u_rel = uncertainty_data.loc[date, 'U_rel_k2']
                else:
                    # Encontrar la fecha más cercana (dentro de 3 días)
                    time_diffs = abs(uncertainty_index - date)
                    closest_idx = time_diffs.argmin()
                    if time_diffs[closest_idx] <= pd.Timedelta(days=3):
                        u_rel = uncertainty_data.iloc[closest_idx]['U_rel_k2']
                    else:
                        u_rel = np.nan
                
                if pd.notna(u_rel):
                    # Incertidumbre absoluta = incertidumbre relativa * valor
                    yerr.append(u_rel * sr_val / 100.0)
                else:
                    yerr.append(0)
            else:
                yerr.append(0)
        
        # Graficar con barras de error
        ax.errorbar(serie.index, serie.values, yerr=yerr, fmt='o-', alpha=0.85, 
                   label=f'SR Photocells', linewidth=1, capsize=3, capthick=1.5,
                   elinewidth=1.5, color='blue', ecolor='lightblue')
    else:
        # Graficar sin barras de error
        serie.plot(ax=ax, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)

    # Calcular línea de tendencia usando fechas reales
    # Convertir fechas a días desde el primer punto para que la pendiente tenga sentido
    y = serie.values
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:  # Necesitamos al menos 2 puntos para una línea
        # Obtener fechas válidas
        valid_dates = serie.index[mask]
        # Convertir a días desde la primera fecha válida
        first_date = valid_dates[0]
        x_days = (valid_dates - first_date).total_seconds() / 86400.0  # Convertir a días
        y_valid = y[mask]
        
        # Calcular regresión usando días
        slope_days, intercept, r_value, p_value, std_err = stats.linregress(x_days, y_valid)
        r2 = r_value ** 2
        
        # Convertir pendiente a %/semana (multiplicar por 7 días)
        slope_weeks = slope_days * 7
        
        # Calcular valores de tendencia para todas las fechas
        all_dates = serie.index
        x_all_days = (all_dates - first_date).total_seconds() / 86400.0
        y_trend = slope_days * x_all_days + intercept
        
        # Graficar la tendencia sobre las mismas fechas con el mismo color que la curva
        ax.plot(serie.index, y_trend, '--', linewidth=2, alpha=0.7,
                label=f'Trend: {slope_weeks:.3f}%/Week, R²: {r2:.3f}', color=ax.lines[-1].get_color())

    ax.set_ylabel(f'Soiling Ratio [%]')
    ax.set_xlabel('Week')
    ax.set_title(f'SR Photocells')
    ax.grid(True, which='both', linestyle='--')
    ax.legend(loc='best')
    ax.set_ylim([50, 110])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    if settings.SAVE_FIGURES:
        safe_cell_name = re.sub(r'[^a-zA-Z0-9_]', '', cell_name)
        
        # Detectar si es para los primeros 3 meses
        first_date_data = df_weekly_sr_q25.index.min()
        end_3_months = first_date_data + pd.DateOffset(months=3)
        
        # Si las fechas coinciden con los primeros 3 meses, usar nombre específico
        if (start.date() == first_date_data.date() and 
            abs((end - end_3_months).days) <= 7):  # Tolerancia de 7 días
            filename = f'refcell_{safe_cell_name}_sr_semanal_3meses.png'
        else:
            filename = f'refcell_{safe_cell_name}_sr_semanal_periodo_especifico.png'
            
        save_plot_matplotlib(fig, filename, paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
    if settings.SHOW_FIGURES:
        plt.show()
    elif settings.SAVE_FIGURES:
        plt.close(fig)
    if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES:
        plt.close(fig)

def _generate_first_3_months_plot(df_weekly_sr_q25: pd.DataFrame, cell_name: str, sr_min_val: float, sr_max_val: float) -> None:
    """
    Genera un gráfico de SR mensual (promedio) para los primeros 3 meses de una celda, con eje X: Month 1, Month 2, Month 3.
    """
    if cell_name not in df_weekly_sr_q25.columns:
        logger.warning(f"La celda {cell_name} no se encuentra en los datos semanales.")
        return

    # Obtener el primer índice y calcular el rango de 3 meses
    first_date = df_weekly_sr_q25.index.min()
    third_month = (first_date + pd.DateOffset(months=3))
    df_filtered = df_weekly_sr_q25.loc[first_date:third_month]

    if df_filtered.empty:
        logger.warning(f"No hay datos para {cell_name} en los primeros 3 meses.")
        return

    # Agrupar por mes y calcular el promedio
    serie = df_filtered[cell_name].copy()
    serie_monthly = serie.resample('M').mean().iloc[:3]  # Solo los primeros 3 meses
    if settings.REFCELLS_ADJUST_TO_100_FLAG:
        serie_monthly = _adjust_series_start_to_100(serie_monthly, f"{cell_name} Mensual Q25")

    # Eje X: Month 1, Month 2, Month 3
    x = np.arange(1, len(serie_monthly) + 1)
    labels = [f"Month {i}" for i in x]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, serie_monthly.values, 'o-', alpha=0.85, label='SR Photocells', linewidth=1)

    # Calcular línea de tendencia
    y = serie_monthly.values
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        r2 = r_value ** 2
        y_trend = slope * x + intercept
        ax.plot(x, y_trend, '--', linewidth=2, alpha=0.7,
                label=f'Trend: {slope:.3f}%/Month, R²: {r2:.3f}', color=ax.lines[-1].get_color())

    ax.set_ylabel('Soiling Ratio [%]')
    ax.set_xlabel('Month')
    ax.set_title('SR Photocells (First 3 Months)')
    ax.grid(True, which='both', linestyle='--')
    ax.set_ylim([50, 110])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best')
    plt.tight_layout()

    if settings.SAVE_FIGURES:
        safe_cell_name = re.sub(r'[^a-zA-Z0-9_]', '', cell_name)
        save_plot_matplotlib(fig, f'refcell_{safe_cell_name}_sr_3meses.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
    if settings.SHOW_FIGURES:
        plt.show()
    elif settings.SAVE_FIGURES:
        plt.close(fig)
    if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES:
        plt.close(fig)

def _generate_first_3_months_weekly_plot(df_weekly_sr_q25: pd.DataFrame, cell_name: str, sr_min_val: float, sr_max_val: float) -> None:
    """
    Genera un gráfico de los primeros 3 meses usando datos semanales pero con eje X como Month 1, Month 2, Month 3.
    """
    if cell_name not in df_weekly_sr_q25.columns:
        logger.warning(f"La celda {cell_name} no se encuentra en los datos semanales.")
        return

    # Obtener el primer índice y calcular el rango de 3 meses
    first_date = df_weekly_sr_q25.index.min()
    third_month = (first_date + pd.DateOffset(months=3))
    df_filtered = df_weekly_sr_q25.loc[first_date:third_month]

    if df_filtered.empty:
        logger.warning(f"No hay datos para {cell_name} en los primeros 3 meses.")
        return

    # Tomar datos semanales pero mostrarlos con eje mensual
    serie = df_filtered[cell_name].copy()
    if settings.REFCELLS_ADJUST_TO_100_FLAG:
        serie = _adjust_series_start_to_100(serie, f"{cell_name} Semanal Q25")

    # Crear posiciones X basadas en semanas pero escaladas a meses
    x_positions = []
    y_values = []
    
    for i, (date, value) in enumerate(serie.items()):
        if pd.notna(value):
            # Calcular posición empezando desde 0 (primer punto en el eje Y)
            days_from_start = (date - first_date).days
            month_position = (days_from_start / 30.44)  # 30.44 días promedio por mes
            month_position = min(month_position, 2.99)  # Limitar a máximo 2.99
            
            x_positions.append(month_position)
            y_values.append(value)

    if x_positions:
        # Cargar datos de incertidumbre semanal para agregar barras de error
        uncertainty_data = None
        try:
            uncertainty_file = paths.SR_WEEKLY_ABS_WITH_U_FILE
            if os.path.exists(uncertainty_file):
                df_uncertainty = pd.read_csv(uncertainty_file, index_col='timestamp', parse_dates=True)
                if df_uncertainty.index.tz is None:
                    df_uncertainty.index = df_uncertainty.index.tz_localize('UTC')
                # Filtrar por el mismo rango de fechas
                uncertainty_data = df_uncertainty.loc[first_date:third_month]
        except Exception as e:
            logger.warning(f"No se pudieron cargar datos de incertidumbre: {e}")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Preparar barras de error si están disponibles
        yerr_values = []
        has_uncertainty = False
        if uncertainty_data is not None and 'U_rel_k2' in uncertainty_data.columns and len(uncertainty_data) > 0:
            # Mapear fechas a posiciones X y calcular barras de error
            # Usar la fecha más cercana si no hay coincidencia exacta
            uncertainty_index = uncertainty_data.index
            for date, value in serie.items():
                if pd.notna(value):
                    # Buscar fecha más cercana en el archivo de incertidumbre
                    if date in uncertainty_index:
                        u_rel = uncertainty_data.loc[date, 'U_rel_k2']
                    else:
                        # Encontrar la fecha más cercana (dentro de 3 días)
                        time_diffs = abs(uncertainty_index - date)
                        closest_idx = time_diffs.argmin()
                        if time_diffs[closest_idx] <= pd.Timedelta(days=3):
                            u_rel = uncertainty_data.iloc[closest_idx]['U_rel_k2']
                        else:
                            u_rel = np.nan
                    
                    if pd.notna(u_rel):
                        # Incertidumbre absoluta = incertidumbre relativa * valor
                        yerr_values.append(u_rel * value / 100.0)
                        has_uncertainty = True
                    else:
                        yerr_values.append(0)
                else:
                    yerr_values.append(0)
            
            if has_uncertainty:
                logger.info(f"Agregando barras de error a gráfico de primeros 3 meses semanal ({len([v for v in yerr_values if v > 0])} puntos con incertidumbre)")
                # Plotear datos semanales con barras de error
                ax.errorbar(x_positions, y_values, yerr=yerr_values, fmt='o-', alpha=0.85, 
                           label='SR Photocells', linewidth=1, capsize=3, capthick=1.5,
                           elinewidth=1.5, color='blue', ecolor='lightblue')
            else:
                logger.warning("No se encontraron datos de incertidumbre para las fechas del gráfico")
                # Plotear datos semanales sin barras de error
                ax.plot(x_positions, y_values, 'o-', alpha=0.85, label='SR Photocells', linewidth=1)
        else:
            logger.warning(f"No se pudieron cargar datos de incertidumbre para el gráfico de primeros 3 meses")
            # Plotear datos semanales sin barras de error
            ax.plot(x_positions, y_values, 'o-', alpha=0.85, label='SR Photocells', linewidth=1)

        # Calcular línea de tendencia
        if len(x_positions) > 1:
            x_array = np.array(x_positions)
            y_array = np.array(y_values)
            mask = ~np.isnan(y_array)
            if np.sum(mask) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_array[mask], y_array[mask])
                r2 = r_value ** 2
                y_trend = slope * x_array + intercept
                ax.plot(x_array, y_trend, '--', linewidth=2, alpha=0.7,
                        label=f'Trend: {slope:.3f}%/Month, R²: {r2:.3f}', color=ax.lines[-1].get_color())

        ax.set_ylabel('Soiling Ratio [%]')
        ax.set_xlabel('Week')
        ax.set_title('SR Photocells (First 3 Months - Weekly)')
        ax.grid(True, which='both', linestyle='--')
        ax.set_ylim([50, 110])
        
        # Configurar eje X con números de semana
        # Calcular las semanas que tenemos datos
        week_numbers = []
        for date, value in serie.items():
            if pd.notna(value):
                # Calcular número de semana desde el inicio
                weeks_from_start = int((date - first_date).days / 7) + 1
                week_numbers.append(weeks_from_start)
        
        # Crear etiquetas para algunas semanas clave
        if week_numbers:
            min_week = min(week_numbers)
            max_week = max(week_numbers)
            # Mostrar etiquetas cada 2-3 semanas para no saturar
            tick_weeks = list(range(min_week, max_week + 1, 2))
            if max_week not in tick_weeks:
                tick_weeks.append(max_week)
            
            # Convertir números de semana a posiciones X
            tick_positions = [(w - 1) * (2.99 / (max_week - 1)) for w in tick_weeks]
            tick_labels = [f'W{w}' for w in tick_weeks]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
        
        ax.set_xlim([-0.1, 3.1])  # Empezar desde cerca del eje Y
        ax.legend(loc='best')
        plt.tight_layout()

        if settings.SAVE_FIGURES:
            safe_cell_name = re.sub(r'[^a-zA-Z0-9_]', '', cell_name)
            save_plot_matplotlib(fig, f'refcell_{safe_cell_name}_sr_semanal_3meses.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
        if settings.SHOW_FIGURES:
            plt.show()
        elif settings.SAVE_FIGURES:
            plt.close(fig)
        if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES:
            plt.close(fig)
    else:
        logger.info(f"No hay datos válidos para graficar la celda {cell_name}.")

def _generate_daily_q25_trend_plot(df_daily_sr_q25: pd.DataFrame, cell_name: str, sr_min_val: float, sr_max_val: float, uncertainty_file: Optional[str] = None) -> None:
    """
    Genera un gráfico específico de SR diario Q25 para la celda 411 con línea de tendencia.
    Incluye barras de error de incertidumbre.
    
    Args:
        df_daily_sr_q25: DataFrame con datos diarios Q25
        cell_name: Nombre de la celda a graficar
        sr_min_val: Valor mínimo de SR para filtrado
        sr_max_val: Valor máximo de SR para filtrado
        uncertainty_file: Ruta opcional al archivo de incertidumbre. Si no se proporciona, usa el archivo por defecto.
    """
    if cell_name not in df_daily_sr_q25.columns:
        logger.warning(f"La celda {cell_name} no se encuentra en los datos diarios.")
        return

    if df_daily_sr_q25[cell_name].dropna().empty:
        logger.warning(f"No hay datos para {cell_name} en los datos diarios Q25.")
        return

    # Cargar datos de incertidumbre diaria
    uncertainty_data = None
    try:
        if uncertainty_file is None:
            uncertainty_file = paths.SR_DAILY_ABS_WITH_U_FILE
        if os.path.exists(uncertainty_file):
            df_uncertainty = pd.read_csv(uncertainty_file, index_col='timestamp', parse_dates=True)
            if df_uncertainty.index.tz is None:
                df_uncertainty.index = df_uncertainty.index.tz_localize('UTC')
            # Filtrar por el mismo rango de fechas que los datos diarios
            start_date = df_daily_sr_q25.index.min()
            end_date = df_daily_sr_q25.index.max()
            uncertainty_data = df_uncertainty.loc[start_date:end_date]
        else:
            logger.warning(f"Archivo de incertidumbre diaria no encontrado: {uncertainty_file}")
    except Exception as e:
        logger.warning(f"No se pudieron cargar datos de incertidumbre diaria: {e}")

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    serie = df_daily_sr_q25[cell_name].copy()

    if settings.REFCELLS_ADJUST_TO_100_FLAG:
        serie = _adjust_series_start_to_100(serie, f"{cell_name} Diario Q25")

    # Plotear datos originales con barras de error si están disponibles
    if uncertainty_data is not None and 'U_rel_k2' in uncertainty_data.columns and len(uncertainty_data) > 0:
        # Calcular barras de error: U_rel_k2 está en porcentaje, convertirlo a valor absoluto
        uncertainty_index = uncertainty_data.index
        yerr = []
        for date in serie.index:
            sr_val = serie.loc[date]
            if pd.notna(sr_val):
                # Buscar fecha exacta o más cercana en el archivo de incertidumbre
                if date in uncertainty_index:
                    u_rel = uncertainty_data.loc[date, 'U_rel_k2']
                else:
                    # Encontrar la fecha más cercana (dentro de 1 día para datos diarios)
                    time_diffs = abs(uncertainty_index - date)
                    closest_idx = time_diffs.argmin()
                    if time_diffs[closest_idx] <= pd.Timedelta(days=1):
                        u_rel = uncertainty_data.iloc[closest_idx]['U_rel_k2']
                    else:
                        u_rel = np.nan
                
                if pd.notna(u_rel):
                    # Incertidumbre absoluta = incertidumbre relativa * valor
                    yerr.append(u_rel * sr_val / 100.0)
                else:
                    yerr.append(0)
            else:
                yerr.append(0)
        
        # Graficar con barras de error
        ax.errorbar(serie.index, serie.values, yerr=yerr, fmt='o-', alpha=0.85, 
                   label=f'SR Photocells (Diario Q25)', linewidth=1, capsize=3, capthick=1.5,
                   elinewidth=1.5, color='blue', ecolor='lightblue')
    else:
        # Graficar sin barras de error
        serie.plot(ax=ax, style='o-', alpha=0.85, label=f'SR Photocells (Diario Q25)', linewidth=1)

    # Calcular línea de tendencia usando fechas reales
    # Convertir fechas a días desde el primer punto para que la pendiente tenga sentido
    y = serie.values
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:  # Necesitamos al menos 2 puntos para una línea
        # Obtener fechas válidas
        valid_dates = serie.index[mask]
        # Convertir a días desde la primera fecha válida
        first_date = valid_dates[0]
        x_days = (valid_dates - first_date).total_seconds() / 86400.0  # Convertir a días
        y_valid = y[mask]
        
        # Calcular regresión usando días
        slope_days, intercept, r_value, p_value, std_err = stats.linregress(x_days, y_valid)
        r2 = r_value ** 2
        
        # Calcular valores de tendencia para todas las fechas
        all_dates = serie.index
        x_all_days = (all_dates - first_date).total_seconds() / 86400.0
        y_trend = slope_days * x_all_days + intercept
        
        # Graficar la tendencia sobre las mismas fechas con el mismo color que la curva
        ax.plot(serie.index, y_trend, '--', linewidth=2, alpha=0.7,
                label=f'Trend: {slope_days:.3f}%/day, R²: {r2:.3f}', color=ax.lines[-1].get_color())

    ax.set_ylabel(f'Soiling Ratio{" Adjusted" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
    ax.set_xlabel('Day')
    ax.set_title(f'Daily SR Q25 with Trend - {cell_name}')
    ax.grid(True, which='both', linestyle='--')
    ax.legend(loc='best')
    ax.set_ylim([50, 110])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if settings.SAVE_FIGURES:
        safe_cell_name = re.sub(r'[^a-zA-Z0-9_]', '', cell_name)
        filename = f'refcell_{safe_cell_name}_sr_diario_q25_tendencia.png'
        save_plot_matplotlib(fig, filename, paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
    if settings.SHOW_FIGURES:
        plt.show()
    elif settings.SAVE_FIGURES:
        plt.close(fig)
    if not settings.SHOW_FIGURES and not settings.SAVE_FIGURES:
        plt.close(fig)

def _analyze_cloudy_days_solar_noon(df_ref_cells: pd.DataFrame, ref_col: str, valid_soiled_cols: list) -> None:
    """
    Analiza días nublados durante el mediodía solar basándose únicamente en la variabilidad 
    de irradiancia de la celda de referencia (celda limpia).
    
    Args:
        df_ref_cells: DataFrame con datos de celdas de referencia
        ref_col: Nombre de la columna de referencia (irradiancia de celda limpia)
        valid_soiled_cols: Lista de columnas de celdas sucias válidas (solo para calcular SR)
    """
    logger.info("Iniciando análisis de días nublados durante mediodía solar...")
    logger.info(f"Análisis basado únicamente en celda de referencia (limpia): {ref_col}")
    
    try:
        from utils.solar_time import UtilsMedioDiaSolar
        
        # Verificar que la columna de referencia existe
        if ref_col not in df_ref_cells.columns:
            logger.error(f"Columna de referencia {ref_col} no encontrada en los datos.")
            return
        
        # Filtrar datos válidos solo para la celda de referencia
        df_ref_clean = df_ref_cells[[ref_col]].dropna()
        
        if df_ref_clean.empty:
            logger.warning("No hay datos válidos en la celda de referencia para análisis de nubosidad.")
            return
        
        # Obtener rango de fechas
        start_date = df_ref_clean.index.min().date()
        end_date = df_ref_clean.index.max().date()
        
        # Calcular intervalos de mediodía solar (±2.5 horas)
        solar_utils = UtilsMedioDiaSolar(
            datei=start_date,
            datef=end_date,
            freq='D',
            inter=300,  # 5 horas (±2.5h)
            tz_local_str=settings.DUSTIQ_LOCAL_TIMEZONE_STR,
            lat=settings.SITE_LATITUDE,
            lon=settings.SITE_LONGITUDE,
            alt=settings.SITE_ALTITUDE
        )
        
        solar_intervals_df = solar_utils.msd()
        
        if solar_intervals_df.empty:
            logger.warning("No se pudieron calcular intervalos de mediodía solar.")
            return
        
        # Filtrar datos de la celda de referencia por mediodía solar
        solar_noon_ref_data = []
        
        for _, row in solar_intervals_df.iterrows():
            start_time = pd.Timestamp(row[0], tz='UTC')
            end_time = pd.Timestamp(row[1], tz='UTC')
            
            # Filtrar solo datos de celda de referencia en este intervalo
            mask = (df_ref_clean.index >= start_time) & (df_ref_clean.index <= end_time)
            interval_data = df_ref_clean[mask].copy()
            
            if not interval_data.empty:
                interval_data['date'] = start_time.date()
                solar_noon_ref_data.append(interval_data)
        
        if not solar_noon_ref_data:
            logger.warning("No hay datos de celda de referencia durante intervalos de mediodía solar.")
            return
        
        # Combinar todos los datos de mediodía solar (solo celda de referencia)
        df_solar_noon_ref = pd.concat(solar_noon_ref_data, ignore_index=False)
        logger.info(f"Datos de celda de referencia filtrados por mediodía solar: {len(df_solar_noon_ref)} puntos")
        
        # Analizar variabilidad diaria de irradiancia SOLO de la celda de referencia
        daily_stats = df_solar_noon_ref.groupby('date').agg({
            ref_col: ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
        
        daily_stats.columns = ['Irr_Mean', 'Irr_Std', 'Irr_Min', 'Irr_Max', 'Irr_Count']
        
        # Calcular indicadores de nubosidad
        daily_stats['CV_Percent'] = (daily_stats['Irr_Std'] / daily_stats['Irr_Mean'] * 100).round(1)
        daily_stats['Range_W_m2'] = daily_stats['Irr_Max'] - daily_stats['Irr_Min']
        daily_stats['Variability_Index'] = (daily_stats['Range_W_m2'] / daily_stats['Irr_Mean']).round(3)
        
        # Clasificar días según nubosidad
        # Criterios: CV > 30% o Variability_Index > 0.5 = Nublado
        # CV > 15% o Variability_Index > 0.3 = Parcialmente nublado
        conditions = [
            (daily_stats['CV_Percent'] > 30) | (daily_stats['Variability_Index'] > 0.5),
            (daily_stats['CV_Percent'] > 15) | (daily_stats['Variability_Index'] > 0.3),
        ]
        choices = ['Nublado', 'Parcialmente_Nublado']
        daily_stats['Clasificacion_Clima'] = np.select(conditions, choices, default='Despejado')
        
        # Estadísticas por clasificación
        weather_summary = daily_stats['Clasificacion_Clima'].value_counts()
        logger.info(f"Clasificación climática basada en celda de referencia: {weather_summary.to_dict()}")
        logger.info("METODOLOGÍA: Días clasificados según variabilidad de irradiancia de celda limpia únicamente")
        
        # Calcular SR para cada día clasificado
        # Nota: La clasificación se basa SOLO en la celda de referencia, pero el SR se calcula con todas las celdas
        sr_by_weather = {}
        for weather_type in ['Despejado', 'Parcialmente_Nublado', 'Nublado']:
            weather_dates = daily_stats[daily_stats['Clasificacion_Clima'] == weather_type].index
            
            if len(weather_dates) > 0:
                # Para calcular SR, necesitamos volver a los datos originales completos
                # pero solo para los días clasificados según la celda de referencia
                weather_data_list = []
                
                for _, row in solar_intervals_df.iterrows():
                    start_time = pd.Timestamp(row[0], tz='UTC')
                    end_time = pd.Timestamp(row[1], tz='UTC')
                    date_key = start_time.date()
                    
                    if date_key in weather_dates:
                        # Filtrar datos completos (todas las celdas) para este día
                        mask = (df_ref_cells.index >= start_time) & (df_ref_cells.index <= end_time)
                        interval_data = df_ref_cells[mask].copy()
                        
                        if not interval_data.empty:
                            weather_data_list.append(interval_data)
                
                if weather_data_list:
                    weather_data = pd.concat(weather_data_list, ignore_index=False)
                    
                    # Calcular SR promedio para este tipo de clima
                    sr_weather = {}
                    for soiled_col in valid_soiled_cols:
                        if soiled_col in weather_data.columns and ref_col in weather_data.columns:
                            mask_valid = weather_data[ref_col] > 0
                            if mask_valid.sum() > 0:
                                sr_values = (weather_data.loc[mask_valid, soiled_col] / 
                                           weather_data.loc[mask_valid, ref_col] * 100)
                                sr_weather[f'SR_{soiled_col}_Mean'] = sr_values.mean()
                                sr_weather[f'SR_{soiled_col}_Std'] = sr_values.std()
                                sr_weather[f'SR_{soiled_col}_Count'] = len(sr_values)
                    
                    if sr_weather:  # Solo agregar si hay datos de SR
                        sr_by_weather[weather_type] = sr_weather
        
        # Guardar análisis de días nublados
        os.makedirs(paths.REFCELLS_OUTPUT_SUBDIR_CSV, exist_ok=True)
        
        # Guardar clasificación diaria
        daily_stats_filename = os.path.join(paths.REFCELLS_OUTPUT_SUBDIR_CSV, 'analisis_dias_nublados_solar_noon.csv')
        daily_stats.to_csv(daily_stats_filename)
        logger.info(f"Análisis de días nublados guardado en: {daily_stats_filename}")
        
        # Guardar resumen por tipo de clima
        if sr_by_weather:
            weather_summary_df = pd.DataFrame(sr_by_weather).T
            weather_summary_filename = os.path.join(paths.REFCELLS_OUTPUT_SUBDIR_CSV, 'resumen_sr_por_clima_solar_noon.csv')
            weather_summary_df.to_csv(weather_summary_filename)
            logger.info(f"Resumen SR por clima guardado en: {weather_summary_filename}")
        
        # --- Generar gráficos ---
        if settings.SAVE_FIGURES or settings.SHOW_FIGURES:
            
            # 1. Gráfico de clasificación temporal
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
            # Usar fuente por defecto de matplotlib (evita warning de Times New Roman no encontrada)
            # plt.rcParams["font.family"] = "Times New Roman"  # Comentado: fuente no disponible en el sistema
            # Usar fuente serif genérica o dejar la por defecto
            try:
                plt.rcParams["font.family"] = "serif"
            except:
                pass  # Usar fuente por defecto si serif no está disponible
            # Irradiancia promedio diaria
            dates = pd.to_datetime(daily_stats.index)
            colors_map = {'Despejado': '#2ca02c', 'Parcialmente_Nublado': '#ff7f0e', 'Nublado': '#d62728'}
            
            for weather_type in ['Despejado', 'Parcialmente_Nublado', 'Nublado']:
                mask = daily_stats['Clasificacion_Clima'] == weather_type
                if mask.sum() > 0:
                    ax1.scatter(dates[mask], daily_stats.loc[mask, 'Irr_Mean'], 
                              c=colors_map[weather_type], label=weather_type, alpha=0.7, s=30)
            
            ax1.set_title('Weather Classification - Solar Noon (Based on Clean Cell)', fontsize=14)
            ax1.set_ylabel('Average Clean Cell Irradiance [W/m²]', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Coeficiente de variación
            ax2.plot(dates, daily_stats['CV_Percent'], 'o-', color='#1f77b4', alpha=0.7, markersize=4)
            ax2.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Umbral Nublado (30%)')
            ax2.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Umbral Parcial (15%)')
            ax2.set_title('Coeficiente de Variación de Irradiancia', fontsize=14)
            ax2.set_ylabel('CV [%]', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Índice de variabilidad
            ax3.plot(dates, daily_stats['Variability_Index'], 's-', color='#ff7f0e', alpha=0.7, markersize=4)
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Umbral Nublado (0.5)')
            ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Umbral Parcial (0.3)')
            ax3.set_title('Índice de Variabilidad de Irradiancia', fontsize=14)
            ax3.set_ylabel('Índice de Variabilidad', fontsize=12)
            ax3.set_xlabel('Fecha', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Formatear fechas
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            if settings.SAVE_FIGURES:
                save_plot_matplotlib(fig, 'analisis_dias_nublados_solar_noon.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
            if settings.SHOW_FIGURES:
                plt.show()
            elif settings.SAVE_FIGURES:
                plt.close(fig)
            
            # 2. Gráfico de distribución de SR por clima
            if sr_by_weather and len(valid_soiled_cols) > 0:
                fig, axes = plt.subplots(1, len(valid_soiled_cols), figsize=(5*len(valid_soiled_cols), 6))
                if len(valid_soiled_cols) == 1:
                    axes = [axes]
                
                for i, soiled_col in enumerate(valid_soiled_cols):
                    weather_types = []
                    sr_means = []
                    sr_stds = []
                    
                    for weather_type in ['Despejado', 'Parcialmente_Nublado', 'Nublado']:
                        if weather_type in sr_by_weather:
                            sr_mean_key = f'SR_{soiled_col}_Mean'
                            sr_std_key = f'SR_{soiled_col}_Std'
                            if sr_mean_key in sr_by_weather[weather_type]:
                                weather_types.append(weather_type)
                                sr_means.append(sr_by_weather[weather_type][sr_mean_key])
                                sr_stds.append(sr_by_weather[weather_type][sr_std_key])
                    
                    if weather_types:
                        colors = [colors_map[wt] for wt in weather_types]
                        bars = axes[i].bar(weather_types, sr_means, yerr=sr_stds, 
                                         color=colors, alpha=0.7, capsize=5)
                        
                        axes[i].set_title(f'SR por Clima - {soiled_col}', fontsize=12)
                        axes[i].set_ylabel('Soiling Ratio [%]', fontsize=10)
                        axes[i].grid(True, alpha=0.3)
                        
                        # Agregar valores en las barras
                        for bar, mean_val in zip(bars, sr_means):
                            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                       f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=9)
                
                plt.suptitle('Comparación de Soiling Ratio por Condiciones Climáticas', fontsize=14)
                plt.tight_layout()
                
                if settings.SAVE_FIGURES:
                    save_plot_matplotlib(fig, 'sr_por_condiciones_climaticas_solar_noon.png', paths.REFCELLS_OUTPUT_SUBDIR_GRAPH)
                if settings.SHOW_FIGURES:
                    plt.show()
                elif settings.SAVE_FIGURES:
                    plt.close(fig)
        
        logger.info("Análisis de días nublados completado exitosamente.")
        
    except Exception as e:
        logger.error(f"Error en análisis de días nublados: {e}", exc_info=True)

def run_analysis():
    """
    Función estándar para ejecutar el análisis de Celdas de Referencia.
    Usa la configuración centralizada para rutas y parámetros.
    """
    raw_data_filepath = paths.REFCELLS_RAW_DATA_FILE
    return analyze_ref_cells_data(raw_data_filepath)

def run_analysis_solar_noon(hours_window: float = 2.5):
    """
    Función de entrada principal para ejecutar el análisis de celdas de referencia filtrado por mediodía solar.
    
    Args:
        hours_window: Ventana en horas alrededor del medio día solar (default: 2.5 horas)
    
    Returns:
        bool: True si el análisis fue exitoso, False en caso contrario
    """
    raw_data_filepath = paths.REFCELLS_RAW_DATA_FILE
    return analyze_ref_cells_data_solar_noon(raw_data_filepath, hours_window)

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando análisis de celdas de referencia desde main...")
    raw_data_path = paths.REFCELLS_RAW_DATA_FILE
    analyze_ref_cells_data(raw_data_path) 