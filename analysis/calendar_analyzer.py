# analysis/calendar_analyzer.py

import pandas as pd
import os
from config.logging_config import logger
from config import paths # Para acceder a BASE_INPUT_DIR, etc.
from config import settings # Para acceder a CALENDAR_SHEET_NAME

def analyze_calendar_data(calendar_file_path: str, 
                          output_csv_dir: str, 
                          sheet_name: str = 'Hoja1') -> bool:
    """
    Analiza el archivo CSV del calendario de muestras de soiling.
    Realiza limpieza, renombrado de columnas, y genera dos CSVs:
    1. Un CSV con columnas seleccionadas del calendario.
    2. Un CSV con datos agrupados por Periodo para la Estructura 'Fija a RC'.

    Args:
        calendar_file_path: Ruta completa al archivo CSV del calendario.
        output_csv_dir: Directorio donde se guardarán los CSVs generados.
        sheet_name: No se usa para archivos CSV, se mantiene por compatibilidad.

    Returns:
        True si el análisis y guardado de CSVs fue exitoso, False en caso contrario.
    """
    logger.info("--- Iniciando Análisis de Calendario de Muestras ---")
    logger.info(f"Cargando datos del calendario desde: {calendar_file_path}")

    # Asegurar que el directorio de salida para CSVs generales exista
    # En el notebook, estos CSVs se guardaban en BASE_INPUT_DIR ('datos')
    # Para la versión .py, es mejor guardarlos en el directorio de salida procesado general.
    # Si se desea el comportamiento original, output_csv_dir debería ser paths.BASE_INPUT_DIR.
    os.makedirs(output_csv_dir, exist_ok=True)

    try:
        # Leer el archivo CSV en lugar de Excel
        df_calendario = pd.read_csv(calendar_file_path)
        logger.info("Datos del calendario cargados exitosamente.")

        df_calendario.columns = df_calendario.columns.str.strip()

        cols_fechas = ['Inicio Exposición', 'Fin Exposicion']
        for col in cols_fechas:
            if col in df_calendario.columns:
                df_calendario[col] = pd.to_datetime(df_calendario[col], errors='coerce')
            else:
                logger.warning(f"Columna de fecha esperada '{col}' no encontrada.")
        
        if 'Fecha medición' in df_calendario.columns:
            df_calendario.rename(columns={'Fecha medición': 'Fin Exposicion'}, inplace=True)
            logger.info("Columna 'Fecha medición' renombrada a 'Fin Exposicion'.")
        elif 'Fin Exposicion' not in df_calendario.columns:
            logger.warning("No se encontró 'Fecha medición' para renombrar, ni 'Fin Exposicion' existente.")

        # --- Guardado del DataFrame de Calendario con Columnas Seleccionadas ---
        cols_originales_a_mantener = [
            'Inicio Exposición', 'Fin Exposicion', 'Estructura', 'Exposición', 
            'Periodo', 'Masa A', 'Masa B', 'Masa C', 'Estado'
        ]
        cols_existentes_seleccionadas = [col for col in cols_originales_a_mantener if col in df_calendario.columns]

        path_csv_seleccionado_opcional = os.path.join(output_csv_dir, 'calendario_muestras_seleccionado.csv')
        if cols_existentes_seleccionadas:
            df_calendario_seleccionado_opcional = df_calendario[cols_existentes_seleccionadas].copy()
            try:
                df_calendario_seleccionado_opcional.to_csv(path_csv_seleccionado_opcional, index=False, date_format='%Y-%m-%d')
                logger.info(f"CSV de calendario seleccionado guardado en: {path_csv_seleccionado_opcional}")
            except Exception as e:
                logger.error(f"Error al guardar CSV opcional de calendario: {e}")
        else:
            logger.warning("No se seleccionaron columnas para el CSV opcional de calendario.")

        # --- Agrupar por Periodo para Estructura 'Fija a RC' ---
        logger.info("Agrupando por Periodo para Estructura 'Fija a RC'")
        required_cols_new_analysis = ['Estructura', 'Periodo', 'Fin Exposicion']
        missing_cols = [col for col in required_cols_new_analysis if col not in df_calendario.columns]

        path_csv_fija_rc_periodos = os.path.join(output_csv_dir, 'calendario_fija_rc_por_periodo.csv')
        if missing_cols:
            logger.error(f"Faltan columnas para análisis 'Fija a RC': {missing_cols}. No se generará {path_csv_fija_rc_periodos}")
            return False # Considerar este error como fallo parcial
        
        df_fija_rc = df_calendario[
            df_calendario['Estructura'].fillna('').astype(str).str.strip() == 'Fija a RC'
        ].copy()

        if df_fija_rc.empty:
            logger.warning("No se encontraron datos con Estructura 'Fija a RC'.")
            # Guardar CSV vacío para consistencia si se espera el archivo
            pd.DataFrame().to_csv(path_csv_fija_rc_periodos, index=False)
        else:
            df_fija_rc.dropna(subset=['Fin Exposicion', 'Periodo'], inplace=True)
            if df_fija_rc.empty:
                logger.warning("No hay datos para 'Fija a RC' con 'Periodo' y 'Fin Exposicion' válidos.")
                pd.DataFrame().to_csv(path_csv_fija_rc_periodos, index=False)
            else:
                df_resultado_fija_rc = df_fija_rc.groupby('Periodo')['Fin Exposicion'].apply(
                    lambda dates: sorted(list(dates.dropna().unique()))
                ).reset_index()
                df_resultado_fija_rc.rename(columns={'Fin Exposicion': 'Fechas Fin Exposicion (Fija a RC)'}, inplace=True)
                logger.info(f"Resultado del análisis 'Fija a RC':\n{df_resultado_fija_rc.to_string()}")
                
                try:
                    df_to_save = df_resultado_fija_rc.copy()
                    def format_dates_list(date_list):
                        if isinstance(date_list, list) and all(isinstance(d, pd.Timestamp) for d in date_list):
                            return [d.strftime('%Y-%m-%d') for d in date_list]
                        return date_list
                    df_to_save['Fechas Fin Exposicion (Fija a RC)'] = df_to_save['Fechas Fin Exposicion (Fija a RC)'].apply(format_dates_list).astype(str)
                    df_to_save.to_csv(path_csv_fija_rc_periodos, index=False)
                    logger.info(f"CSV agrupado por periodo para 'Fija a RC' guardado en: {path_csv_fija_rc_periodos}")
                except Exception as e:
                    logger.error(f"Error al guardar CSV de 'Fija a RC' por periodo: {e}")
                    return False
        
        logger.info("--- Fin Análisis de Calendario de Muestras ---")
        return True

    except FileNotFoundError:
        logger.error(f"Archivo de calendario no encontrado: {calendar_file_path}")
    except pd.errors.EmptyDataError: 
        logger.error(f"La hoja '{sheet_name}' en {calendar_file_path} está vacía o no se pudo leer.")
    except KeyError as e: 
        logger.error(f"Error de clave procesando '{sheet_name}' de {calendar_file_path}: {e}.")
    except Exception as e:
        logger.error(f"Error inesperado procesando calendario '{sheet_name}': {e}", exc_info=True)
    
    return False # Si ocurre cualquier excepción 

def run_analysis():
    """
    Función estándar para ejecutar el análisis de Calendario.
    Usa la configuración centralizada para rutas y parámetros.
    """
    calendar_file_path = os.path.join(paths.BASE_INPUT_DIR, paths.CALENDAR_RAW_DATA_FILENAME)
    output_csv_dir = paths.CALENDAR_OUTPUT_SUBDIR_CSV
    sheet_name = settings.CALENDAR_SHEET_NAME
    return analyze_calendar_data(calendar_file_path, output_csv_dir, sheet_name)

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando análisis de Calendario...")
    calendar_file_path = os.path.join(paths.BASE_INPUT_DIR, paths.CALENDAR_RAW_DATA_FILENAME)
    output_csv_dir = paths.CALENDAR_OUTPUT_SUBDIR_CSV
    sheet_name = settings.CALENDAR_SHEET_NAME
    analyze_calendar_data(calendar_file_path, output_csv_dir, sheet_name) 