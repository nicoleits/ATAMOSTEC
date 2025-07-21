# analysis/ref_cells_analyzer.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import re
from datetime import timezone
import scipy.stats as stats

from config.logging_config import logger
from config import paths, settings
from utils.helpers import save_plot_matplotlib

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
        sr_df = pd.DataFrame(index=df_ref_cells.index)
        for soiled_col in valid_soiled_cols:
            # Asegurar que el denominador no sea cero
            mask_denom_valid = df_ref_cells[ref_col] > 0
            sr_df[soiled_col] = np.nan # Inicializar
            sr_df.loc[mask_denom_valid, soiled_col] = (df_ref_cells.loc[mask_denom_valid, soiled_col] / df_ref_cells.loc[mask_denom_valid, ref_col]) * 100
        
        logger.info("Soiling Ratios calculados.")
        sr_df.dropna(how='all', inplace=True)
        if sr_df.empty:
            logger.warning("DataFrame de SR vacío después del cálculo y dropna.")
            print(f"\nLos gráficos de celdas de referencia se guardaron en: {paths.REFCELLS_OUTPUT_SUBDIR_GRAPH}\n")
            return True

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
                ax_irr.set_ylabel('Irradiancia [W/m²]')
                ax_irr.set_xlabel('Día')
                ax_irr.set_title('Irradiancia Diaria - Celdas de Referencia Seleccionadas')
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
            ax_w.set_ylabel(f'Soiling Ratio{" Ajustado" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
            ax_w.set_xlabel('Semana')
            ax_w.set_title(f'Soiling Ratio{" Ajustado" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} - Celdas de Referencia (Semanal Q25)')
            ax_w.set_ylim([90, 110])
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
                    serie_d_adj.plot(ax=ax_d, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)
                    plotted_daily_sr = True

        if plotted_daily_sr:
            ax_d.legend(loc='best')
            ax_d.set_ylabel(f'Soiling Ratio{" Ajustado" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
            ax_d.set_xlabel('Día')
            ax_d.set_title(f'Soiling Ratio{" Ajustado" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} - Celdas de Referencia (Diario Q25)')
            ax_d.set_ylim([90, 110])
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
                    ax_w_ind.set_ylabel(f'Soiling Ratio{" Ajustado" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
                    ax_w_ind.set_xlabel('Semana')
                    ax_w_ind.set_title('Soiling Ratio con Celdas de Referencia')
                    ax_w_ind.grid(True, which='both', linestyle='--'); ax_w_ind.legend(loc='best')
                    ax_w_ind.set_ylim([90, 110])
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
                    ax_d_ind.set_ylabel(f'Soiling Ratio{" Ajustado" if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} [%]')
                    ax_d_ind.set_xlabel('Día')
                    ax_d_ind.set_title(f'SR{" Adj." if settings.REFCELLS_ADJUST_TO_100_FLAG else ""} - {col} (Diario Q25)')
                    ax_d_ind.grid(True, which='both', linestyle='--'); ax_d_ind.legend(loc='best')
                    ax_d_ind.set_ylim([90, 110])
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

def _generate_specific_cell_plot(df_weekly_sr_q25: pd.DataFrame, cell_name: str, start_date: str, end_date: str, sr_min_val: float, sr_max_val: float) -> None:
    """
    Genera un gráfico específico de SR semanal para una celda particular con fechas personalizadas.
    Incluye línea de tendencia lineal, pendiente y R².
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

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    serie = df_filtered[cell_name].copy()

    if settings.REFCELLS_ADJUST_TO_100_FLAG:
        serie = _adjust_series_start_to_100(serie, f"{cell_name} Semanal Q25")

    # Plotear datos originales
    serie.plot(ax=ax, style='o-', alpha=0.85, label=f'SR Photocells', linewidth=1)

    # Calcular línea de tendencia
    x = np.arange(len(serie))
    y = serie.values
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:  # Necesitamos al menos 2 puntos para una línea
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        r2 = r_value ** 2
        y_trend = slope * x + intercept
        # Graficar la tendencia sobre las mismas fechas con el mismo color que la curva
        ax.plot(serie.index, y_trend, '--', linewidth=2, alpha=0.7,
                label=f'Trend: {slope:.3f}%/Week, R²: {r2:.3f})', color=ax.lines[-1].get_color())

    ax.set_ylabel(f'Soiling Ratio [%]')
    ax.set_xlabel('Week')
    ax.set_title(f'SR Photocells')
    ax.grid(True, which='both', linestyle='--')
    ax.legend(loc='best')
    ax.set_ylim([90, 110])
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
    ax.set_ylim([90, 110])
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
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plotear datos semanales
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
        ax.set_ylim([90, 110])
        
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
            
            # Irradiancia promedio diaria
            dates = pd.to_datetime(daily_stats.index)
            colors_map = {'Despejado': '#2ca02c', 'Parcialmente_Nublado': '#ff7f0e', 'Nublado': '#d62728'}
            
            for weather_type in ['Despejado', 'Parcialmente_Nublado', 'Nublado']:
                mask = daily_stats['Clasificacion_Clima'] == weather_type
                if mask.sum() > 0:
                    ax1.scatter(dates[mask], daily_stats.loc[mask, 'Irr_Mean'], 
                              c=colors_map[weather_type], label=weather_type, alpha=0.7, s=30)
            
            ax1.set_title('Clasificación Climática - Mediodía Solar (Basada en Celda Limpia)', fontsize=14)
            ax1.set_ylabel('Irradiancia Promedio Celda Limpia [W/m²]', fontsize=12)
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
    raw_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.REFCELLS_RAW_DATA_FILENAME)
    return analyze_ref_cells_data(raw_data_filepath)

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando análisis de celdas de referencia desde main...")
    raw_data_path = os.path.join(paths.BASE_INPUT_DIR, paths.REFCELLS_RAW_DATA_FILENAME)
    analyze_ref_cells_data(raw_data_path) 