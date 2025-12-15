# analysis/soiling_kit_analyzer.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
from datetime import timezone # Necesario para df.index.tz != timezone.utc
from scipy import stats  # Para calcular la tendencia

from config.logging_config import logger
from config import paths, settings
from utils.helpers import save_plot_matplotlib

def _calculate_trend(series: pd.Series) -> tuple:
    """
    Calcula la tendencia lineal de una serie temporal.
    Retorna la pendiente, el intercepto y los valores ajustados.
    """
    if series.empty or series.dropna().empty:
        return None, None, None
    
    # Convertir fechas a números para el ajuste lineal
    x = np.arange(len(series))
    y = series.values
    
    # Calcular regresión lineal
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calcular valores ajustados
    trend_values = slope * x + intercept
    
    return slope, intercept, trend_values

def analyze_soiling_kit_data(raw_data_filepath: str) -> bool:
    """
    Analiza los datos del Soiling Kit basándose en la lógica del notebook.
    Realiza corrección de temperatura, calcula SR (Protegido/Expuesto),
    filtra datos, guarda resultados en CSV y genera gráficos.
    """
    logger.info(f"--- Iniciando Análisis de Soiling Kit desde: {raw_data_filepath} (Lógica Notebook) ---")
    
    os.makedirs(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, exist_ok=True)
    os.makedirs(paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    date_fmt_daily = mdates.DateFormatter('%Y-%m-%d')
    # date_fmt_hourly = mdates.DateFormatter('%Y-%m-%d %H:%M') # Definido por si se necesita

    try:
        # --- 1. Carga de Datos ---
        df_sk_raw = pd.read_csv(raw_data_filepath)
        logger.info(f"Datos de Soiling Kit cargados inicialmente: {len(df_sk_raw)} filas.")
        if df_sk_raw.empty:
            logger.warning("El archivo de datos de Soiling Kit está vacío.")
            return False

        # --- 2. Preprocesamiento del Índice y Timestamps ---
        df_sk = df_sk_raw.copy()
        time_col = settings.SOILING_KIT_TIME_COLUMN
        time_fmt = settings.SOILING_KIT_TIME_FORMAT

        if time_col not in df_sk.columns:
            logger.error(f"Columna de tiempo '{time_col}' no encontrada. Columnas: {df_sk.columns.tolist()}")
            return False

        df_sk.rename(columns={time_col: 'Original_Timestamp_Col'}, inplace=True)
        
        # Log a sample of the timestamp column before conversion
        if not df_sk.empty and 'Original_Timestamp_Col' in df_sk.columns:
            logger.info(f"Sample of 'Original_Timestamp_Col' before parsing: {df_sk['Original_Timestamp_Col'].head().tolist()}")
        else:
            logger.warning("'Original_Timestamp_Col' not available or df_sk is empty before parsing.")

        df_sk.index = pd.to_datetime(df_sk['Original_Timestamp_Col'], format=time_fmt, errors='coerce')
        
        # Log a sample of the index after conversion
        if not df_sk.empty:
            logger.info(f"Sample of DatetimeIndex after parsing: {df_sk.index[:5].tolist()}")
        else:
            logger.warning("df_sk is empty after attempting to_datetime conversion.")

        rows_before_nat_drop = len(df_sk)
        df_sk.dropna(axis=0, subset=[df_sk.index.name], inplace=True) # Eliminar filas con NaT en el índice
        if len(df_sk) < rows_before_nat_drop:
            logger.info(f"{rows_before_nat_drop - len(df_sk)} filas eliminadas debido a NaT en el índice.")

        if not isinstance(df_sk.index, pd.DatetimeIndex):
            logger.error("La conversión del índice a DatetimeIndex falló.")
            return False
            
        if df_sk.index.tz is None:
            logger.info("Índice es DatetimeIndex pero naive. Localizando a UTC...")
            df_sk.index = df_sk.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
        elif df_sk.index.tz != timezone.utc:
            logger.info(f"Índice con zona horaria {df_sk.index.tz}. Convirtiendo a UTC...")
            df_sk.index = df_sk.index.tz_convert('UTC')
        else:
            logger.info("Índice ya es DatetimeIndex y está en UTC.")
        
        df_sk.sort_index(inplace=True)
        logger.info(f"Índice procesado a UTC y ordenado. Rango: {df_sk.index.min()} a {df_sk.index.max() if not df_sk.empty else 'N/A'}")

        # --- 3. Filtrado por Fecha y Hora General ---
        # Usar fecha de inicio específica para Soiling Kit si está definida, sino la general
        specific_start_date_str = getattr(settings, 'SOILING_KIT_ANALYSIS_START_DATE_STR', None)
        if specific_start_date_str:
            start_date_analysis_str = specific_start_date_str
            logger.info(f"Usando fecha de inicio específica para Soiling Kit: {start_date_analysis_str}")
        else:
            start_date_analysis_str = settings.ANALYSIS_START_DATE_GENERAL_STR
            logger.info(f"Usando fecha de inicio general para Soiling Kit: {start_date_analysis_str}")

        start_date_analysis = pd.to_datetime(start_date_analysis_str, utc=True, errors='coerce')
        
        # Usar fecha de fin específica para Soiling Kit si está definida, sino la general
        specific_end_date_str = getattr(settings, 'SOILING_KIT_ANALYSIS_END_DATE_STR', None)
        if specific_end_date_str:
            end_date_analysis_str = specific_end_date_str
            logger.info(f"Usando fecha de fin específica para Soiling Kit: {end_date_analysis_str}")
        else:
            end_date_analysis_str = settings.ANALYSIS_END_DATE_GENERAL_STR
            logger.info(f"Usando fecha de fin general para Soiling Kit: {end_date_analysis_str}")

        end_date_analysis = pd.to_datetime(end_date_analysis_str, utc=True, errors='coerce')

        if pd.notna(end_date_analysis) and end_date_analysis.hour == 0: # Hacer inclusivo el día final
             end_date_analysis = end_date_analysis.replace(hour=23, minute=59, second=59, microsecond=999999)

        original_rows = len(df_sk)
        if pd.notna(start_date_analysis):
            df_sk = df_sk[df_sk.index >= start_date_analysis]
        if pd.notna(end_date_analysis):
            df_sk = df_sk[df_sk.index <= end_date_analysis]
        if len(df_sk) < original_rows:
            logger.info(f"Filtrado por fecha ({start_date_analysis_str} - {end_date_analysis_str}). Filas: {len(df_sk)}")

        if settings.SOILING_KIT_FILTER_START_TIME and settings.SOILING_KIT_FILTER_END_TIME:
            original_rows_time_filter = len(df_sk)
            try:
                df_sk = df_sk.between_time(settings.SOILING_KIT_FILTER_START_TIME, settings.SOILING_KIT_FILTER_END_TIME)
                logger.info(f"Filtrado por horario UTC ({settings.SOILING_KIT_FILTER_START_TIME}-{settings.SOILING_KIT_FILTER_END_TIME}). Filas antes: {original_rows_time_filter}, después: {len(df_sk)}")
            except Exception as e_time_filter:
                logger.error(f"Error en filtrado between_time: {e_time_filter}")
        
        if df_sk.empty:
            logger.warning("No hay datos del Soiling Kit después del filtrado por fecha/hora.")
            return True

        # --- 4. Verificación y Conversión de Columnas Necesarias ---
        isc_soiled_col = settings.SOILING_KIT_ISC_SOILED_COL
        isc_ref_col = settings.SOILING_KIT_ISC_REF_COL
        temp_soiled_col = settings.SOILING_KIT_TEMP_SOILED_COL
        temp_ref_col = settings.SOILING_KIT_TEMP_REF_COL
        
        required_cols = [isc_soiled_col, isc_ref_col, temp_soiled_col, temp_ref_col]
        missing_cols = [col for col in required_cols if col not in df_sk.columns]
        if missing_cols:
            logger.error(f"Faltan columnas requeridas: {missing_cols}. Columnas disponibles: {df_sk.columns.tolist()}")
            return False
        
        for col in required_cols: # Convertir a numérico
            df_sk[col] = pd.to_numeric(df_sk[col], errors='coerce')
        
        # Eliminar filas si alguna de las columnas Isc o Temp es NaN después de la conversión
        df_sk.dropna(subset=required_cols, how='any', inplace=True)
        logger.info(f"Conversión a numérico y dropna en columnas requeridas. Filas restantes: {len(df_sk)}")
        if df_sk.empty:
            logger.warning("No hay datos después de conversión/dropna de columnas Isc/Temp.")
            return True

        # --- 5. Corrección de Temperatura para Isc ---
        alpha_isc = settings.SOILING_KIT_ALPHA_ISC_CORR
        temp_ref = settings.SOILING_KIT_TEMP_REF_C
        
        df_sk['Isc_Soiled_Corrected'] = df_sk[isc_soiled_col] * (1 + (alpha_isc * (temp_ref - df_sk[temp_soiled_col])))
        df_sk['Isc_Ref_Corrected'] = df_sk[isc_ref_col] * (1 + (alpha_isc * (temp_ref - df_sk[temp_ref_col])))
        logger.info("Corrección de temperatura aplicada a Isc.")

        # --- 6. Cálculo de Soiling Ratio (SR) ---
        # SR = (Protegido / Expuesto) * 100, según notebook
        # Asegurar que el denominador no sea cero o muy pequeño para evitar inf/muy grandes números
        # Y que el numerador sea positivo
        mask_sr_calc = (df_sk[isc_soiled_col].abs() > 1e-6) & (df_sk[isc_ref_col] > 0)
        df_sk['SR_Raw'] = np.nan
        df_sk.loc[mask_sr_calc, 'SR_Raw'] = (df_sk.loc[mask_sr_calc, isc_ref_col] / df_sk.loc[mask_sr_calc, isc_soiled_col]) * 100
        
        mask_sr_c_calc = (df_sk['Isc_Soiled_Corrected'].abs() > 1e-6) & (df_sk['Isc_Ref_Corrected'] > 0)
        df_sk['SR_TempCorrected'] = np.nan
        df_sk.loc[mask_sr_c_calc, 'SR_TempCorrected'] = (df_sk.loc[mask_sr_c_calc, 'Isc_Ref_Corrected'] / df_sk.loc[mask_sr_c_calc, 'Isc_Soiled_Corrected']) * 100
        logger.info("Soiling Ratios (Raw y Corregido por Temp) calculados como (Protegido/Expuesto)*100.")

        # --- 7. Filtrado de SR ---
        sr_threshold = settings.SOILING_KIT_SR_LOWER_THRESHOLD
        
        df_sk_sr_filtered = df_sk.copy()
        if 'SR_Raw' in df_sk_sr_filtered.columns:
             df_sk_sr_filtered['SR_Raw_Filtered'] = df_sk_sr_filtered['SR_Raw'][df_sk_sr_filtered['SR_Raw'] > sr_threshold]
        if 'SR_TempCorrected' in df_sk_sr_filtered.columns:
            df_sk_sr_filtered['SR_TempCorrected_Filtered'] = df_sk_sr_filtered['SR_TempCorrected'][df_sk_sr_filtered['SR_TempCorrected'] > sr_threshold]
        
        logger.info(f"Soiling Ratios filtrados (SR > {sr_threshold}%).")

        # --- 7.5. Análisis de Propagación de Incertidumbre de SR ---
        logger.info("Iniciando análisis de propagación de incertidumbre de SR (Soiling Kit)...")
        try:
            from analysis.sr_uncertainty_soiling_kit import run_uncertainty_propagation_analysis
            # Ejecutar análisis con corrección de temperatura (más común)
            uncertainty_success = run_uncertainty_propagation_analysis(
                df_sk,
                use_temp_correction=True
            )
            if uncertainty_success:
                logger.info("✅ Análisis de propagación de incertidumbre completado exitosamente (SR con corrección de temperatura).")
            else:
                logger.warning("⚠️  El análisis de propagación de incertidumbre no se completó exitosamente.")
        except ImportError as e:
            logger.error(f"No se pudo importar el módulo 'sr_uncertainty_soiling_kit': {e}")
        except Exception as e:
            logger.error(f"Error al ejecutar el análisis de propagación de incertidumbre: {e}", exc_info=True)
        # Continuar con el resto del análisis aunque falle la incertidumbre

        # --- 8. Guardar CSVs ---
        # Datos procesados (con Isc corregidas y SRs calculados antes de filtrar SR)
        df_sk[['Original_Timestamp_Col', isc_soiled_col, isc_ref_col, temp_soiled_col, temp_ref_col, 
               'Isc_Soiled_Corrected', 'Isc_Ref_Corrected', 'SR_Raw', 'SR_TempCorrected']].to_csv(
            os.path.join(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, 'soiling_kit_data_processed_and_sr.csv')
        )
        logger.info(f"Datos procesados y SRs (sin filtrar SR) guardados.")

        # Ratios de SR filtrados (minutales)
        if 'SR_Raw_Filtered' in df_sk_sr_filtered.columns and 'SR_TempCorrected_Filtered' in df_sk_sr_filtered.columns:
            df_sk_sr_filtered[['SR_Raw_Filtered', 'SR_TempCorrected_Filtered']].dropna(how='all').to_csv(
                os.path.join(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, 'soiling_kit_sr_minutal_filtered.csv')
            )
            logger.info(f"SRs minutales filtrados guardados.")
        
        # SR Diario (Media)
        sr_daily_raw_mean = df_sk_sr_filtered['SR_Raw_Filtered'].resample('D').mean().dropna()
        sr_daily_corrected_mean = df_sk_sr_filtered['SR_TempCorrected_Filtered'].resample('D').mean().dropna()
        if not sr_daily_raw_mean.empty:
            sr_daily_raw_mean.to_csv(os.path.join(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, 'soiling_kit_sr_raw_daily_mean.csv'), header=True)
        if not sr_daily_corrected_mean.empty:
            sr_daily_corrected_mean.to_csv(os.path.join(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, 'soiling_kit_sr_corrected_daily_mean.csv'), header=True)

        # SR Semanal (Q25)
        sr_weekly_raw_q25 = df_sk_sr_filtered['SR_Raw_Filtered'].resample('W').quantile(0.25).dropna()
        sr_weekly_corrected_q25 = df_sk_sr_filtered['SR_TempCorrected_Filtered'].resample('W').quantile(0.25).dropna()
        if not sr_weekly_raw_q25.empty:
            sr_weekly_raw_q25.to_csv(os.path.join(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, 'soiling_kit_sr_raw_weekly_q25.csv'), header=True)
        if not sr_weekly_corrected_q25.empty:
            sr_weekly_corrected_q25.to_csv(os.path.join(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, 'soiling_kit_sr_corrected_weekly_q25.csv'), header=True)
        
        # SR Mensual (Q25)
        sr_monthly_raw_q25 = df_sk_sr_filtered['SR_Raw_Filtered'].resample('ME').quantile(0.25).dropna()
        sr_monthly_corrected_q25 = df_sk_sr_filtered['SR_TempCorrected_Filtered'].resample('ME').quantile(0.25).dropna()
        if not sr_monthly_raw_q25.empty:
            sr_monthly_raw_q25.to_csv(os.path.join(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, 'soiling_kit_sr_raw_monthly_q25.csv'), header=True)
        if not sr_monthly_corrected_q25.empty:
            sr_monthly_corrected_q25.to_csv(os.path.join(paths.SOILING_KIT_OUTPUT_SUBDIR_CSV, 'soiling_kit_sr_corrected_monthly_q25.csv'), header=True)
        
        logger.info("SRs agregados (diario mean, semanal q25, mensual q25) guardados.")

        # --- 8.5. Cargar datos de incertidumbre para agregar barras de error a los gráficos ---
        uncertainty_data_daily = None
        uncertainty_data_weekly = None
        uncertainty_data_monthly = None
        
        try:
            if os.path.exists(paths.SOILING_KIT_SR_DAILY_ABS_WITH_U_FILE):
                uncertainty_data_daily = pd.read_csv(paths.SOILING_KIT_SR_DAILY_ABS_WITH_U_FILE, index_col=0, parse_dates=True)
                logger.info(f"✅ Datos de incertidumbre diarios cargados: {len(uncertainty_data_daily)} puntos")
            else:
                logger.warning(f"⚠️  Archivo de incertidumbre diario no encontrado en: {paths.SOILING_KIT_SR_DAILY_ABS_WITH_U_FILE}")
                logger.info("   Los gráficos se generarán sin barras de error.")
        except Exception as e:
            logger.warning(f"⚠️  Error al cargar datos de incertidumbre diarios: {e}")
        
        try:
            if os.path.exists(paths.SOILING_KIT_SR_WEEKLY_ABS_WITH_U_FILE):
                uncertainty_data_weekly = pd.read_csv(paths.SOILING_KIT_SR_WEEKLY_ABS_WITH_U_FILE, index_col=0, parse_dates=True)
                logger.info(f"✅ Datos de incertidumbre semanales cargados: {len(uncertainty_data_weekly)} puntos")
            else:
                logger.warning(f"⚠️  Archivo de incertidumbre semanal no encontrado en: {paths.SOILING_KIT_SR_WEEKLY_ABS_WITH_U_FILE}")
        except Exception as e:
            logger.warning(f"⚠️  Error al cargar datos de incertidumbre semanales: {e}")
        
        try:
            # Intentar cargar datos de incertidumbre mensuales (si existen)
            if os.path.exists(paths.SOILING_KIT_SR_MONTHLY_ABS_WITH_U_FILE):
                uncertainty_data_monthly = pd.read_csv(paths.SOILING_KIT_SR_MONTHLY_ABS_WITH_U_FILE, index_col=0, parse_dates=True)
                logger.info(f"✅ Datos de incertidumbre mensuales cargados: {len(uncertainty_data_monthly)} puntos")
            else:
                logger.info(f"   Archivo de incertidumbre mensual no encontrado en: {paths.SOILING_KIT_SR_MONTHLY_ABS_WITH_U_FILE}")
                logger.info("   El gráfico se generará sin barras de error.")
        except Exception as e:
            logger.warning(f"⚠️  Error al cargar datos de incertidumbre mensuales: {e}")
        
        # Función auxiliar para obtener barras de error de incertidumbre
        def get_error_bars(sr_series, uncertainty_data):
            """
            Obtiene las barras de error (yerr) para una serie de SR usando datos de incertidumbre.
            
            Args:
                sr_series: Serie de pandas con valores de SR
                uncertainty_data: DataFrame con datos de incertidumbre (debe tener índice de fechas y columna 'U_rel_k2')
            
            Returns:
                Lista de valores de error (yerr) para usar con errorbar() o None si no hay datos
            """
            if uncertainty_data is None or 'U_rel_k2' not in uncertainty_data.columns:
                return None
            
            yerr = []
            uncertainty_index = uncertainty_data.index
            
            # Determinar tolerancia basada en la frecuencia de los datos
            # Si hay menos de 20 puntos, probablemente es mensual o semanal
            if len(sr_series) < 20:
                max_tolerance = pd.Timedelta(days=15)  # Mayor tolerancia para mensual
            elif len(sr_series) < 100:
                max_tolerance = pd.Timedelta(days=3)   # Tolerancia para semanal
            else:
                max_tolerance = pd.Timedelta(days=1)   # Tolerancia para diario
            
            for date in sr_series.index:
                sr_val = sr_series.loc[date]
                if pd.notna(sr_val):
                    # Buscar fecha exacta o más cercana
                    if date in uncertainty_index:
                        u_rel = uncertainty_data.loc[date, 'U_rel_k2']
                    else:
                        # Encontrar la fecha más cercana dentro de la tolerancia
                        time_diffs = abs(uncertainty_index - date)
                        closest_idx = time_diffs.argmin()
                        if time_diffs[closest_idx] <= max_tolerance:
                            u_rel = uncertainty_data.iloc[closest_idx]['U_rel_k2']
                        else:
                            u_rel = np.nan
                    
                    if pd.notna(u_rel):
                        # Incertidumbre absoluta = incertidumbre relativa * valor
                        # u_rel está en porcentaje (%), convertir a valor absoluto
                        yerr.append(u_rel * sr_val / 100.0)
                    else:
                        yerr.append(0)
                else:
                    yerr.append(0)
            
            return yerr if any(err > 0 for err in yerr) else None

        # --- 9. Generación de Gráficos ---
        logger.info("Generando gráficos para Soiling Kit...")

        # Isc Raw (Media Diaria)
        fig1, ax1 = plt.subplots(figsize=(15, 7))
        if not df_sk[isc_soiled_col].dropna().empty: df_sk[isc_soiled_col].resample('D').mean().plot(ax=ax1, label=f'{isc_soiled_col} (Exposed)')
        if not df_sk[isc_ref_col].dropna().empty: df_sk[isc_ref_col].resample('D').mean().plot(ax=ax1, label=f'{isc_ref_col} (Protected)')
        ax1.set_ylabel('Current [A]', fontsize=16)
        ax1.set_xlabel('Time', fontsize=14)
        ax1.grid(True)
        ax1.set_title('Soiling Kit - Original Isc (Daily Average)', fontsize=16)
        if ax1.has_data(): ax1.legend(fontsize=12, frameon=True)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.xaxis.set_major_formatter(date_fmt_daily)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        save_plot_matplotlib(fig1, 'sk_isc_raw_daily.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
        print("Mostrando figura 1: Isc Raw (Media Diaria)")
        plt.show()
        plt.close(fig1)

        # Isc Corregida (Media Diaria)
        fig2, ax2 = plt.subplots(figsize=(15, 7))
        if not df_sk['Isc_Soiled_Corrected'].dropna().empty: df_sk['Isc_Soiled_Corrected'].resample('D').mean().plot(ax=ax2, label='Temperature Corrected Exposed Isc')
        if not df_sk['Isc_Ref_Corrected'].dropna().empty: df_sk['Isc_Ref_Corrected'].resample('D').mean().plot(ax=ax2, label='Temperature Corrected Protected Isc')
        ax2.set_ylabel('Temperature Corrected Isc [A]', fontsize=16)
        ax2.set_xlabel('Time', fontsize=14)
        ax2.grid(True)
        ax2.set_title('Soiling Kit - Temperature Corrected Isc (Daily Average)', fontsize=16)
        if ax2.has_data(): ax2.legend(fontsize=12)
        ax2.xaxis.set_major_formatter(date_fmt_daily)
        ax2.tick_params(axis='both', labelsize=12)
        save_plot_matplotlib(fig2, 'sk_isc_corrected_daily.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
        print("Mostrando figura 2: Isc Corregida (Media Diaria)")
        plt.show()
        plt.close(fig2)

        # SR Raw (Media Diaria)
        if not sr_daily_raw_mean.empty:
            fig3, ax3 = plt.subplots(figsize=(15, 7))
            
            # Intentar obtener barras de error
            yerr_daily_raw = get_error_bars(sr_daily_raw_mean, uncertainty_data_daily)
            
            if yerr_daily_raw is not None:
                # Graficar con barras de error
                avg_error = np.mean([e for e in yerr_daily_raw if e > 0])
                logger.info(f"   📊 Agregando barras de error para SR Raw diario (error promedio: {avg_error:.2f}%)")
                ax3.errorbar(sr_daily_raw_mean.index, sr_daily_raw_mean.values, yerr=yerr_daily_raw, 
                            fmt='-*', alpha=0.75, label='Raw SR (Daily Average)', 
                            markersize=6, capsize=4, capthick=2.0, elinewidth=2.0, ecolor='blue')
            else:
                # Graficar sin barras de error
                sr_daily_raw_mean.plot(ax=ax3, label='Raw SR (Daily Average)', style='-*')
            
            # Calcular y graficar tendencia
            slope, intercept, trend_values = _calculate_trend(sr_daily_raw_mean)
            if slope is not None:
                trend_label = f'Trend ({slope:.2f}%/day)'
                ax3.plot(sr_daily_raw_mean.index, trend_values, 'r--', label=trend_label)
            
            ax3.set_ylabel('Soiling Ratio (P/E) [%]', fontsize=16)
            ax3.set_xlabel('Time', fontsize=14)
            ax3.grid(True)
            ax3.set_title(f'Soiling Kit - Raw SR (Daily Average, SR > {sr_threshold}%)', fontsize=16)
            ax3.set_ylim(50, 115)
            if ax3.has_data(): ax3.legend(fontsize=12)
            ax3.tick_params(axis='both', labelsize=12)
            ax3.xaxis.set_major_formatter(date_fmt_daily)
            save_plot_matplotlib(fig3, 'sk_sr_raw_daily.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
            print("Mostrando figura 3: SR Raw (Media Diaria)")
            plt.show()
            plt.close(fig3)

        # SR Corregido (Media Diaria)
        if not sr_daily_corrected_mean.empty:
            fig4, ax4 = plt.subplots(figsize=(15, 7))
            
            # Intentar obtener barras de error
            yerr_daily_corrected = get_error_bars(sr_daily_corrected_mean, uncertainty_data_daily)
            
            if yerr_daily_corrected is not None:
                # Graficar con barras de error
                avg_error = np.mean([e for e in yerr_daily_corrected if e > 0])
                logger.info(f"   📊 Agregando barras de error para SR Corregido diario (error promedio: {avg_error:.2f}%)")
                ax4.errorbar(sr_daily_corrected_mean.index, sr_daily_corrected_mean.values, yerr=yerr_daily_corrected, 
                            fmt='-o', alpha=0.75, label='Temperature Corrected SR (Daily Average)', 
                            markersize=4, capsize=4, capthick=2.0, elinewidth=2.0, ecolor='blue')
            else:
                # Graficar sin barras de error
                sr_daily_corrected_mean.plot(ax=ax4, label='Temperature Corrected SR (Daily Average)')
            
            # Calcular y graficar tendencia
            slope, intercept, trend_values = _calculate_trend(sr_daily_corrected_mean)
            if slope is not None:
                trend_label = f'Trend ({slope:.2f}%/day)'
                ax4.plot(sr_daily_corrected_mean.index, trend_values, 'r--', label=trend_label)
            
            ax4.set_ylabel('Temperature Corrected Soiling Ratio (P/E) [%]', fontsize=16)
            ax4.set_xlabel('Time', fontsize=14)
            ax4.grid(True)
            ax4.set_title(f'Soiling Kit - Temperature Corrected SR (Daily Average, SR > {sr_threshold}%)', fontsize=16)
            ax4.set_ylim(50, 115)
            if ax4.has_data(): ax4.legend(fontsize=12)
            ax4.tick_params(axis='both', labelsize=12)
            ax4.xaxis.set_major_formatter(date_fmt_daily)
            save_plot_matplotlib(fig4, 'sk_sr_corrected_daily.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
            print("Mostrando figura 4: SR Corregido (Media Diaria)")
            plt.show()
            plt.close(fig4)

        # SR Raw por Franjas Horarias (Media Diaria)
        if 'SR_Raw_Filtered' in df_sk_sr_filtered.columns and not df_sk_sr_filtered['SR_Raw_Filtered'].dropna().empty:
            fig5, ax5 = plt.subplots(figsize=(15, 7))
            plotted_franjas = False
            for start_h, end_h in [('11:00', '13:00'), ('13:00', '15:00'), ('15:00', '17:00')]: # UTC
                sr_slice = df_sk_sr_filtered['SR_Raw_Filtered'].between_time(start_h, end_h)
                if not sr_slice.dropna().empty:
                    sr_slice.resample('D').mean().plot(ax=ax5, label=f'{start_h}-{end_h} UTC')
                    plotted_franjas = True
            if plotted_franjas: 
                ax5.set_ylabel('Soiling Ratio (P/E) [%]', fontsize=16)
                ax5.set_xlabel('Time', fontsize=14)
                ax5.grid(True)
                ax5.set_title(f'Soiling Kit - Raw SR by Time Slot (Daily Average, SR > {sr_threshold}%)', fontsize=16)
                ax5.set_ylim(50, 115)
                ax5.legend(fontsize=12)
                ax5.tick_params(axis='both', labelsize=12)
                ax5.xaxis.set_major_formatter(date_fmt_daily)
                save_plot_matplotlib(fig5, 'sk_sr_raw_franjas_horarias_daily.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
                print("Mostrando figura 5: SR Raw por Franjas Horarias")
                plt.show()
            plt.close(fig5)
        
        # SR Corregido (Minutal y Media Diaria)
        if 'SR_TempCorrected_Filtered' in df_sk_sr_filtered.columns and not df_sk_sr_filtered['SR_TempCorrected_Filtered'].dropna().empty:
            fig6, ax6 = plt.subplots(figsize=(15, 7))
            df_sk_sr_filtered['SR_TempCorrected_Filtered'].plot(ax=ax6, style='.', alpha=0.3, label='Temperature Corrected SR (hourly data)', markersize=2)
            if not sr_daily_corrected_mean.empty:
                sr_daily_corrected_mean.plot(ax=ax6, style='-', linewidth=2, label='Temperature Corrected SR (Daily Average)')
            ax6.set_ylabel('Temperature Corrected Soiling Ratio (P/E) [%]', fontsize=16)
            ax6.set_xlabel('Time', fontsize=14)
            ax6.grid(True)
            ax6.set_title(f'Soiling Kit - Temperature Corrected SR (Minute and Daily Average, SR > {sr_threshold}%)', fontsize=16)
            ax6.set_ylim(50, 115)
            if ax6.has_data(): ax6.legend(fontsize=12)
            ax6.tick_params(axis='both', labelsize=12)
            ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')) # Usar formato con hora
            fig6.autofmt_xdate()
            save_plot_matplotlib(fig6, 'sk_sr_corrected_minutal_daily_mean.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
            print("Mostrando figura 6: SR Corregido Minutal y Media Diaria")
            plt.show()
            plt.close(fig6)

        # SR Semanal Q25 (Raw y Corregido)
        if not sr_weekly_raw_q25.empty or not sr_weekly_corrected_q25.empty:
            fig7, ax7 = plt.subplots(figsize=(15, 7))
            plotted_q25 = False

            # Procesar SR Raw
            temp_sr_weekly_raw = pd.Series(dtype=float)
            x_ticks_labels_raw = []
            x_ticks_positions_raw = []
            color_raw = '#1f77b4'  # Azul
            color_corr = '#ff7f0e'  # Naranja

            if not sr_weekly_raw_q25.empty:
                _series_to_plot_raw = sr_weekly_raw_q25.copy()
                if not _series_to_plot_raw.dropna().empty:
                    temp_sr_weekly_raw = _series_to_plot_raw.dropna()
                    
                    # Obtener barras de error ANTES de cambiar los índices
                    yerr_weekly_raw = get_error_bars(temp_sr_weekly_raw, uncertainty_data_weekly)
                    
                    x_ticks_labels_raw = [date.strftime('%Y-%m-%d') for date in temp_sr_weekly_raw.index.to_pydatetime()]
                    x_ticks_positions_raw = list(range(len(temp_sr_weekly_raw)))
                    
                    if yerr_weekly_raw is not None:
                        # Graficar con barras de error
                        avg_error = np.mean([e for e in yerr_weekly_raw if e > 0])
                        logger.info(f"   📊 Agregando barras de error para SR Raw semanal (error promedio: {avg_error:.2f}%)")
                        ax7.errorbar(x_ticks_positions_raw, temp_sr_weekly_raw.values, yerr=yerr_weekly_raw,
                                    fmt='o-', label='Raw SR (Weekly Q25)', 
                                    color=color_raw, markersize=6, linewidth=2, capsize=4, capthick=2.0, elinewidth=2.0, ecolor=color_raw)
                    else:
                        # Graficar sin barras de error
                        temp_sr_weekly_raw.index = x_ticks_positions_raw
                        temp_sr_weekly_raw.plot(ax=ax7, style='-o', label='Raw SR (Weekly Q25)', color=color_raw)
                    
                    # Calcular y graficar tendencia (usar índices numéricos)
                    temp_sr_weekly_raw_for_trend = temp_sr_weekly_raw.copy()
                    if temp_sr_weekly_raw_for_trend.index.dtype != np.int64:
                        temp_sr_weekly_raw_for_trend.index = x_ticks_positions_raw
                    slope, intercept, trend_values = _calculate_trend(temp_sr_weekly_raw_for_trend)
                    if slope is not None:
                        trend_label = f'Raw Trend ({slope:.2f}%/week)'
                        ax7.plot(x_ticks_positions_raw, trend_values, '--', color=color_raw, label=trend_label)
                    
                    # Guardar índices numéricos para usar en el resto del código
                    temp_sr_weekly_raw.index = x_ticks_positions_raw
                    plotted_q25 = True
            
            # Procesar SR Corregido
            temp_sr_weekly_corrected = pd.Series(dtype=float)
            x_ticks_labels_corrected = []
            x_ticks_positions_corrected = []

            if not sr_weekly_corrected_q25.empty:
                _series_to_plot_corrected = sr_weekly_corrected_q25.copy()
                if not _series_to_plot_corrected.dropna().empty:
                    temp_sr_weekly_corrected = _series_to_plot_corrected.dropna()
                    
                    # Obtener barras de error ANTES de cambiar los índices
                    yerr_weekly_corrected = get_error_bars(temp_sr_weekly_corrected, uncertainty_data_weekly)
                    
                    x_ticks_labels_corrected = [date.strftime('%Y-%m-%d') for date in temp_sr_weekly_corrected.index.to_pydatetime()]
                    x_ticks_positions_corrected = list(range(len(temp_sr_weekly_corrected)))
                    
                    plot_style_corrected = '-s' if plotted_q25 and not sr_weekly_raw_q25.empty else '-o'
                    
                    if yerr_weekly_corrected is not None:
                        # Graficar con barras de error
                        avg_error = np.mean([e for e in yerr_weekly_corrected if e > 0])
                        logger.info(f"   📊 Agregando barras de error para SR Corregido semanal (error promedio: {avg_error:.2f}%)")
                        # Determinar el formato correcto para errorbar
                        if plotted_q25 and not sr_weekly_raw_q25.empty:
                            fmt_corrected = 's-'
                        else:
                            fmt_corrected = 'o-'
                        ax7.errorbar(x_ticks_positions_corrected, temp_sr_weekly_corrected.values, yerr=yerr_weekly_corrected,
                                    fmt=fmt_corrected, label='Temperature Corrected SR (Weekly Q25)', 
                                    color=color_corr, markersize=6, linewidth=2, capsize=4, capthick=2.0, elinewidth=2.0, ecolor=color_corr)
                    else:
                        # Graficar sin barras de error
                        temp_sr_weekly_corrected.index = x_ticks_positions_corrected
                        temp_sr_weekly_corrected.plot(ax=ax7, style=plot_style_corrected, label='Temperature Corrected SR (Weekly Q25)', color=color_corr)
                        # Restaurar índices originales para calcular tendencia
                        temp_sr_weekly_corrected.index = _series_to_plot_corrected.dropna().index
                    
                    # Calcular y graficar tendencia (usar índices numéricos para la tendencia)
                    temp_sr_weekly_corrected_for_trend = temp_sr_weekly_corrected.copy()
                    temp_sr_weekly_corrected_for_trend.index = x_ticks_positions_corrected
                    slope, intercept, trend_values = _calculate_trend(temp_sr_weekly_corrected_for_trend)
                    if slope is not None:
                        trend_label = f'Corrected Trend ({slope:.2f}%/week)'
                        ax7.plot(x_ticks_positions_corrected, trend_values, '--', color=color_corr, label=trend_label)
                    
                    # Guardar índices numéricos para usar en el resto del código
                    temp_sr_weekly_corrected.index = x_ticks_positions_corrected
                    plotted_q25 = True

            if plotted_q25:
                ax7.set_ylabel('Soiling Ratio [%]', fontsize=14)
                ax7.set_xlabel('Date', fontsize=14)
                ax7.grid(True)
                ax7.set_title(f'Soiling Kit - Weekly SR ', fontsize=16)
                ax7.legend(loc='best', frameon=True, fontsize=12)
                ax7.set_ylim(50, 115)
                final_tick_positions = []
                final_tick_labels = []
                # Priorizar etiquetas y posiciones de la serie raw si existe, sino de la corregida.
                if x_ticks_positions_raw:
                    final_tick_positions = x_ticks_positions_raw
                    final_tick_labels = x_ticks_labels_raw
                elif x_ticks_positions_corrected:
                    final_tick_positions = x_ticks_positions_corrected
                    final_tick_labels = x_ticks_labels_corrected
                if final_tick_positions:
                    num_ticks_to_show = 10
                    tick_spacing = max(1, len(final_tick_positions) // num_ticks_to_show)
                    indices_to_show = list(range(0, len(final_tick_positions), tick_spacing))
                    if final_tick_positions and (len(final_tick_positions) - 1) not in indices_to_show:
                        if not indices_to_show or (len(final_tick_positions) - 1 - indices_to_show[-1]) >= tick_spacing // 2:
                            indices_to_show.append(len(final_tick_positions) - 1)
                    selected_positions = [final_tick_positions[i] for i in indices_to_show if i < len(final_tick_positions)]
                    selected_labels = [final_tick_labels[i] for i in indices_to_show if i < len(final_tick_labels)]
                    ax7.set_xticks(selected_positions)
                    ax7.set_xticklabels(selected_labels, rotation=30, ha='right')
                fig7.tight_layout()
                save_plot_matplotlib(fig7, 'sk_sr_q25_semanal.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
                print("Mostrando figura 7: SR Semanal Q25 (Raw y Corregido)")
                plt.show()
                plt.close(fig7)
            else:
                logger.info("No hay datos Q25 semanales para graficar.")
                if 'fig7' in locals() and plt.fignum_exists(fig7.number): plt.close(fig7)

        # SR Mensual Q25 (Raw y Corregido)
        if not sr_monthly_raw_q25.empty or not sr_monthly_corrected_q25.empty:
            fig8, ax8 = plt.subplots(figsize=(15, 7))
            plotted_monthly_q25 = False

            # Procesar SR Raw
            temp_sr_monthly_raw = pd.Series(dtype=float)
            x_ticks_labels_raw_monthly = []
            x_ticks_positions_raw_monthly = []
            color_raw = '#1f77b4'  # Azul
            color_corr = '#ff7f0e'  # Naranja

            if not sr_monthly_raw_q25.empty:
                _series_to_plot_raw_monthly = sr_monthly_raw_q25.copy()
                if not _series_to_plot_raw_monthly.dropna().empty:
                    temp_sr_monthly_raw = _series_to_plot_raw_monthly.dropna()
                    
                    # Obtener barras de error ANTES de cambiar los índices
                    yerr_monthly_raw = get_error_bars(temp_sr_monthly_raw, uncertainty_data_monthly)
                    
                    x_ticks_labels_raw_monthly = [date.strftime('%Y-%m') for date in temp_sr_monthly_raw.index.to_pydatetime()]
                    x_ticks_positions_raw_monthly = list(range(len(temp_sr_monthly_raw)))
                    
                    if yerr_monthly_raw is not None:
                        # Graficar con barras de error
                        avg_error = np.mean([e for e in yerr_monthly_raw if e > 0])
                        logger.info(f"   📊 Agregando barras de error para SR Raw mensual (error promedio: {avg_error:.2f}%)")
                        ax8.errorbar(x_ticks_positions_raw_monthly, temp_sr_monthly_raw.values, yerr=yerr_monthly_raw,
                                    fmt='o-', label='Raw SR (Monthly Q25)', 
                                    color=color_raw, markersize=8, linewidth=2, capsize=4, capthick=2.0, elinewidth=2.0, ecolor=color_raw)
                    else:
                        # Graficar sin barras de error
                        temp_sr_monthly_raw.index = x_ticks_positions_raw_monthly
                        temp_sr_monthly_raw.plot(ax=ax8, style='-o', label='Raw SR (Monthly Q25)', color=color_raw, markersize=8)
                    
                    # Calcular y graficar tendencia (usar índices numéricos)
                    temp_sr_monthly_raw_for_trend = temp_sr_monthly_raw.copy()
                    if temp_sr_monthly_raw_for_trend.index.dtype != np.int64:
                        temp_sr_monthly_raw_for_trend.index = x_ticks_positions_raw_monthly
                    slope, intercept, trend_values = _calculate_trend(temp_sr_monthly_raw_for_trend)
                    if slope is not None:
                        trend_label = f'Raw Trend ({slope:.2f}%/month)'
                        ax8.plot(x_ticks_positions_raw_monthly, trend_values, '--', color=color_raw, label=trend_label)
                    
                    # Guardar índices numéricos para usar en el resto del código
                    temp_sr_monthly_raw.index = x_ticks_positions_raw_monthly
                    plotted_monthly_q25 = True
            
            # Procesar SR Corregido
            temp_sr_monthly_corrected = pd.Series(dtype=float)
            x_ticks_labels_corrected_monthly = []
            x_ticks_positions_corrected_monthly = []

            if not sr_monthly_corrected_q25.empty:
                _series_to_plot_corrected_monthly = sr_monthly_corrected_q25.copy()
                if not _series_to_plot_corrected_monthly.dropna().empty:
                    temp_sr_monthly_corrected = _series_to_plot_corrected_monthly.dropna()
                    
                    # Obtener barras de error ANTES de cambiar los índices
                    yerr_monthly_corrected = get_error_bars(temp_sr_monthly_corrected, uncertainty_data_monthly)
                    
                    x_ticks_labels_corrected_monthly = [date.strftime('%Y-%m') for date in temp_sr_monthly_corrected.index.to_pydatetime()]
                    x_ticks_positions_corrected_monthly = list(range(len(temp_sr_monthly_corrected)))
                    
                    plot_style_corrected_monthly = '-s' if plotted_monthly_q25 and not sr_monthly_raw_q25.empty else '-o'
                    
                    if yerr_monthly_corrected is not None:
                        # Graficar con barras de error
                        avg_error = np.mean([e for e in yerr_monthly_corrected if e > 0])
                        logger.info(f"   📊 Agregando barras de error para SR Corregido mensual (error promedio: {avg_error:.2f}%)")
                        # Determinar el formato correcto para errorbar
                        if plotted_monthly_q25 and not sr_monthly_raw_q25.empty:
                            fmt_corrected_monthly = 's-'
                        else:
                            fmt_corrected_monthly = 'o-'
                        ax8.errorbar(x_ticks_positions_corrected_monthly, temp_sr_monthly_corrected.values, yerr=yerr_monthly_corrected,
                                    fmt=fmt_corrected_monthly, label='Temperature Corrected SR (Monthly Q25)', 
                                    color=color_corr, markersize=8, linewidth=2, capsize=4, capthick=2.0, elinewidth=2.0, ecolor=color_corr)
                    else:
                        # Graficar sin barras de error
                        temp_sr_monthly_corrected.index = x_ticks_positions_corrected_monthly
                        temp_sr_monthly_corrected.plot(ax=ax8, style=plot_style_corrected_monthly, label='Temperature Corrected SR (Monthly Q25)', color=color_corr, markersize=8)
                        # Restaurar índices originales para calcular tendencia
                        temp_sr_monthly_corrected.index = _series_to_plot_corrected_monthly.dropna().index
                    
                    # Calcular y graficar tendencia (usar índices numéricos para la tendencia)
                    temp_sr_monthly_corrected_for_trend = temp_sr_monthly_corrected.copy()
                    temp_sr_monthly_corrected_for_trend.index = x_ticks_positions_corrected_monthly
                    slope, intercept, trend_values = _calculate_trend(temp_sr_monthly_corrected_for_trend)
                    if slope is not None:
                        trend_label = f'Corrected Trend ({slope:.2f}%/month)'
                        ax8.plot(x_ticks_positions_corrected_monthly, trend_values, '--', color=color_corr, label=trend_label)
                    
                    # Guardar índices numéricos para usar en el resto del código
                    temp_sr_monthly_corrected.index = x_ticks_positions_corrected_monthly
                    plotted_monthly_q25 = True

            if plotted_monthly_q25:
                ax8.set_ylabel('Soiling Ratio [%]', fontsize=14)
                ax8.set_xlabel('Date', fontsize=14)
                ax8.grid(True)
                ax8.set_title(f'Soiling Kit - Monthly SR (Q25)', fontsize=16)
                ax8.legend(loc='best', frameon=True, fontsize=12)
                ax8.set_ylim(50, 115)
                final_tick_positions_monthly = []
                final_tick_labels_monthly = []
                # Priorizar etiquetas y posiciones de la serie raw si existe, sino de la corregida.
                if x_ticks_positions_raw_monthly:
                    final_tick_positions_monthly = x_ticks_positions_raw_monthly
                    final_tick_labels_monthly = x_ticks_labels_raw_monthly
                elif x_ticks_positions_corrected_monthly:
                    final_tick_positions_monthly = x_ticks_positions_corrected_monthly
                    final_tick_labels_monthly = x_ticks_labels_corrected_monthly
                if final_tick_positions_monthly:
                    # Para mensual, mostrar todas las etiquetas
                    ax8.set_xticks(final_tick_positions_monthly)
                    ax8.set_xticklabels(final_tick_labels_monthly, rotation=30, ha='right')
                fig8.tight_layout()
                save_plot_matplotlib(fig8, 'sk_sr_q25_mensual.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
                print("Mostrando figura 8: SR Mensual Q25 (Raw y Corregido)")
                plt.show()
                plt.close(fig8)
            else:
                logger.info("No hay datos Q25 mensuales para graficar.")
                if 'fig8' in locals() and plt.fignum_exists(fig8.number): plt.close(fig8)

        # Temperaturas Módulos (Media Diaria)
        fig9, ax9 = plt.subplots(figsize=(15, 7))
        if not df_sk[temp_soiled_col].dropna().empty: df_sk[temp_soiled_col].resample('D').mean().plot(ax=ax9, label=f'{temp_soiled_col} (Exposed)', style='*')
        if not df_sk[temp_ref_col].dropna().empty: df_sk[temp_ref_col].resample('D').mean().plot(ax=ax9, label=f'{temp_ref_col} (Protected)', style='*')
        ax9.set_ylabel('Temperature [°C]', fontsize=16)
        ax9.set_xlabel('Time', fontsize=14)
        ax9.set_xlim(pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-08-11 23:59:59', tz='UTC'))
        ax9.grid(True)
        ax9.set_title('Soiling Kit - Module Temperatures (Daily Average)', fontsize=16)
        if ax9.has_data(): ax9.legend(fontsize=12)
        ax9.tick_params(axis='both', labelsize=12)
        
        # Corregir el problema de etiquetas sobrepuestas en el eje X
        ax9.xaxis.set_major_formatter(date_fmt_daily)
        
        # Ajustar el espaciado de las etiquetas del eje X para evitar sobreposición
        # Obtener las posiciones actuales de los ticks
        tick_positions = ax9.get_xticks()
        tick_labels = ax9.get_xticklabels()
        
        if len(tick_positions) > 10:  # Si hay muchas etiquetas, espaciarlas
            # Mostrar solo algunas etiquetas para evitar sobreposición
            num_ticks_to_show = 10
            tick_spacing = max(1, len(tick_positions) // num_ticks_to_show)
            indices_to_show = list(range(0, len(tick_positions), tick_spacing))
            
            # Asegurar que se muestre la última etiqueta
            if len(tick_positions) > 0 and (len(tick_positions) - 1) not in indices_to_show:
                if not indices_to_show or (len(tick_positions) - 1 - indices_to_show[-1]) >= tick_spacing // 2:
                    indices_to_show.append(len(tick_positions) - 1)
            
            selected_positions = [tick_positions[i] for i in indices_to_show if i < len(tick_positions)]
            selected_labels = [tick_labels[i] for i in indices_to_show if i < len(tick_labels)]
            ax9.set_xticks(selected_positions)
            ax9.set_xticklabels(selected_labels, rotation=45, ha='right')
        else:
            # Si hay pocas etiquetas, solo rotarlas
            ax9.tick_params(axis='x', rotation=45)
        
        # Ajustar el layout para evitar cortes
        fig9.tight_layout()
        
        save_plot_matplotlib(fig9, 'sk_temperaturas_modulos_daily.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
        print("Mostrando figura 9: Temperaturas de Módulos (Media Diaria)")
        plt.show()
        plt.close(fig9)

        # Temperaturas Módulos (Datos en Bruto)
        fig9b, ax9b = plt.subplots(figsize=(15, 7))
        if not df_sk[temp_soiled_col].dropna().empty:
            df_sk[temp_soiled_col].plot(ax=ax9b, label=f'{temp_soiled_col} (Exposed)', alpha=0.7)
        if not df_sk[temp_ref_col].dropna().empty:
            df_sk[temp_ref_col].plot(ax=ax9b, label=f'{temp_ref_col} (Protected)', alpha=0.7)
        ax9b.set_ylabel('Temperature [°C]', fontsize=16)
        ax9b.set_xlabel('Time', fontsize=14)
        ax9b.grid(True)
        ax9b.set_title('Soiling Kit - Module Temperatures (Raw Data)', fontsize=16)
        if ax9b.has_data(): ax9b.legend(fontsize=12)
        ax9b.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        save_plot_matplotlib(fig9b, 'sk_temperaturas_modulos_raw.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
        print("Mostrando figura 9b: Temperaturas de Módulos (Datos en Bruto)")
        plt.show()
        plt.close(fig9b)

        # Temperaturas Módulos (Datos en Bruto) 
        start_plot = pd.Timestamp('2025-07-01', tz='UTC')
        end_plot = pd.Timestamp('2025-08-11 23:59:59', tz='UTC')
        df_sk_plot = df_sk[(df_sk.index >= start_plot) & (df_sk.index <= end_plot)]
        fig9c, ax9c = plt.subplots(figsize=(15, 7))
        if not df_sk_plot[temp_soiled_col].dropna().empty:
            df_sk_plot[temp_soiled_col].plot(ax=ax9c, label='Te(C) (Exposed)', alpha=0.7, style='.')
        if not df_sk_plot[temp_ref_col].dropna().empty:
            df_sk_plot[temp_ref_col].plot(ax=ax9c, label='Tp(C) (Protected)', alpha=0.7, style='.')
        ax9c.set_ylabel('Temperature [°C]', fontsize=16)
        ax9c.set_xlabel('Time', fontsize=14)
        ax9c.grid(True)
        ax9c.set_title('Soiling Kit - Module Temperatures (Raw Data)', fontsize=16)
        if ax9c.has_data(): ax9c.legend(fontsize=12, loc='upper left')
        ax9c.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        save_plot_matplotlib(fig9c, 'sk_temperaturas_modulos_raw_abr_may_2025.png', paths.SOILING_KIT_OUTPUT_SUBDIR_GRAPH)
        print("Mostrando figura 9c: Temperaturas de Módulos (Datos en Bruto)")
        plt.show()
        plt.close(fig9c)

        logger.info("--- Fin Análisis de Datos de Soiling Kit (Lógica Notebook) ---")
        return True

    except FileNotFoundError:
        logger.error(f"Archivo de datos de Soiling Kit no encontrado: {raw_data_filepath}")
    except pd.errors.EmptyDataError:
        logger.error(f"El archivo de datos {raw_data_filepath} está vacío o no se pudo parsear.")
    except KeyError as e_key:
        logger.error(f"Error de clave (columna faltante o mal configurada) en Soiling Kit: {e_key}")
    except Exception as e:
        logger.error(f"Error inesperado procesando Soiling Kit: {e}", exc_info=True)
    
    return False 

def run_analysis():
    """
    Función estándar para ejecutar el análisis de Soiling Kit.
    Usa la configuración centralizada para rutas y parámetros.
    """
    raw_data_filepath = paths.SOILING_KIT_RAW_DATA_FILE
    return analyze_soiling_kit_data(raw_data_filepath)

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("Ejecutando análisis de Soiling Kit...")
    # Usar rutas centralizadas desde config/paths.py
    raw_data_filepath = paths.SOILING_KIT_RAW_DATA_FILE
    analyze_soiling_kit_data(raw_data_filepath) 