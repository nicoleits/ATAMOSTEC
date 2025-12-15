"""
Análisis de Incertidumbre del Soiling Ratio (SR) para PV Glasses mediante Propagación de Errores (GUM)

Este módulo calcula la incertidumbre del SR usando propagación de errores según GUM:
- Incertidumbres del fabricante (Si-V-10TC-T): aditiva y de escala (MISMAS FOTOCELDAS QUE REF_CELLS)
- El SR se calcula como: SR = 100 × R_FCi_Avg / REF
- Donde REF = (R_FC1_Avg + R_FC2_Avg) / 2 (promedio de dos celdas de referencia)
- Propagación minuto a minuto usando derivadas parciales
- Agregación a escalas diarias, semanales y mensuales

DESCRIPCIÓN:
------------
El SR se calcula como: SR = 100 × R_FCi_Avg / REF
donde:
- R_FCi_Avg = irradiancia celda sucia i (i = 3, 4, 5)
- REF = (R_FC1_Avg + R_FC2_Avg) / 2 = promedio de dos celdas de referencia limpias

La incertidumbre se propaga usando:
- u(R_FCi)² = u_add² + (u_scale * R_FCi)² (para cada celda)
- u(REF)² = (1/4) * [u(R_FC1)² + u(R_FC2)²] (asumiendo independencia)
- Var(SR) = (∂SR/∂R_FCi)² * u(R_FCi)² + (∂SR/∂REF)² * u(REF)² + 2 * (∂SR/∂R_FCi) * (∂SR/∂REF) * Cov(R_FCi, REF)
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple, List
from datetime import datetime
import pytz
import config.settings as settings
import config.paths as paths

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE INCERTIDUMBRES DEL FABRICANTE
# ============================================================================

# Incertidumbres del fabricante (Si-V-10TC-T) a k=2
# MISMAS FOTOCELDAS QUE REF_CELLS
U_ADD_K2 = 5.0  # W/m² (aditiva, k=2)
U_SCALE_K2 = 0.025  # 2.5% (de escala, k=2)

# Convertir a k=1 (1σ)
U_ADD = U_ADD_K2 / 2.0  # 2.5 W/m²
U_SCALE = U_SCALE_K2 / 2.0  # 0.0125 (1.25%)

# Factor de cobertura para expandir al final
K_EXPAND = 2.0

# Columnas de datos
REF_COL1 = "R_FC1_Avg"  # Celda de referencia 1
REF_COL2 = "R_FC2_Avg"  # Celda de referencia 2
SOILED_COLS = ["R_FC3_Avg", "R_FC4_Avg", "R_FC5_Avg"]  # Celdas sucias

# Umbral mínimo de irradiancia (noche)
MIN_IRRADIANCE_THRESHOLD = 200.0  # W/m²

# Timezone para agregaciones
TZ_ANALYSIS = "America/Santiago"


def channel_u(I: pd.Series, u_add: float, u_scale: float) -> pd.Series:
    """
    Calcula la incertidumbre u(I) por canal usando el modelo combinado aditivo + escala.
    
    Fórmula: u(I)² = u_add² + (u_scale * I)²
    
    Args:
        I: Serie temporal de irradiancia (W/m²)
        u_add: Incertidumbre aditiva (W/m², k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(I) en W/m² (k=1)
    """
    u_squared = u_add**2 + (u_scale * I)**2
    u_I = np.sqrt(u_squared)
    return u_I


def ref_u(R_FC1: pd.Series, R_FC2: pd.Series, u_add: float, u_scale: float, rho: float = 0.0) -> pd.Series:
    """
    Calcula la incertidumbre de REF = (R_FC1 + R_FC2) / 2.
    
    Fórmulas:
    - REF = (R_FC1 + R_FC2) / 2
    - ∂REF/∂R_FC1 = 1/2
    - ∂REF/∂R_FC2 = 1/2
    - Var(REF) = (1/2)² * u(R_FC1)² + (1/2)² * u(R_FC2)² + 2 * (1/2) * (1/2) * Cov(R_FC1, R_FC2)
    - Si rho = 0 (independencia): Var(REF) = (1/4) * [u(R_FC1)² + u(R_FC2)²]
    
    Args:
        R_FC1: Serie temporal de irradiancia celda referencia 1 (W/m²)
        R_FC2: Serie temporal de irradiancia celda referencia 2 (W/m²)
        u_add: Incertidumbre aditiva (W/m², k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
        rho: Coeficiente de correlación entre R_FC1 y R_FC2 (default 0.0 = independencia)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(REF) en W/m² (k=1)
    """
    # Asegurar que tienen el mismo índice
    common_index = R_FC1.index.intersection(R_FC2.index)
    R_FC1_aligned = R_FC1.reindex(common_index)
    R_FC2_aligned = R_FC2.reindex(common_index)
    
    # Calcular incertidumbres individuales
    u_R_FC1 = channel_u(R_FC1_aligned, u_add, u_scale)
    u_R_FC2 = channel_u(R_FC2_aligned, u_add, u_scale)
    
    # Covarianza
    if rho != 0.0:
        cov_R1_R2 = rho * u_R_FC1 * u_R_FC2
    else:
        cov_R1_R2 = pd.Series(0.0, index=common_index)
    
    # Varianza de REF
    var_REF = (0.5**2) * (u_R_FC1**2) + (0.5**2) * (u_R_FC2**2) + 2 * 0.5 * 0.5 * cov_R1_R2
    
    # Incertidumbre estándar
    u_REF = np.sqrt(var_REF)
    
    return u_REF


def propagate_sr_minute_pv_glasses(
    R_FCi: pd.Series,
    R_FC1: pd.Series,
    R_FC2: pd.Series,
    u_add: float = U_ADD,
    u_scale: float = U_SCALE,
    k_expand: float = K_EXPAND,
    rho_ref: float = 0.0,
    rho_soiled_ref: float = 0.0
) -> pd.DataFrame:
    """
    Calcula SR y su incertidumbre minuto a minuto usando propagación de errores (GUM).
    
    Fórmulas:
    - REF = (R_FC1 + R_FC2) / 2
    - SR = 100 × R_FCi / REF
    - ∂SR/∂R_FCi = 100 / REF
    - ∂SR/∂REF = -100 × R_FCi / REF²
    - Var(SR) = (∂SR/∂R_FCi)² * u(R_FCi)² + (∂SR/∂REF)² * u(REF)² + 2 * (∂SR/∂R_FCi) * (∂SR/∂REF) * Cov(R_FCi, REF)
    
    Args:
        R_FCi: Serie temporal de irradiancia celda sucia i (W/m²)
        R_FC1: Serie temporal de irradiancia celda referencia 1 (W/m²)
        R_FC2: Serie temporal de irradiancia celda referencia 2 (W/m²)
        u_add: Incertidumbre aditiva (W/m², k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
        k_expand: Factor de cobertura para expandir (default 2.0)
        rho_ref: Coeficiente de correlación entre R_FC1 y R_FC2 (default 0.0)
        rho_soiled_ref: Coeficiente de correlación entre R_FCi y REF (default 0.0)
    
    Returns:
        pd.DataFrame con columnas:
        - SR: Soiling Ratio (%)
        - REF: Promedio de referencia (W/m²)
        - u_SR_k1_abs: Incertidumbre absoluta k=1 (%)
        - u_SR_k1_rel: Incertidumbre relativa k=1 (adimensional)
        - U_SR_k2_abs: Incertidumbre absoluta expandida k=2 (%)
        - U_SR_k2_rel: Incertidumbre relativa expandida k=2 (adimensional)
    """
    # Asegurar que todas tienen el mismo índice
    common_index = R_FCi.index.intersection(R_FC1.index).intersection(R_FC2.index)
    R_FCi_aligned = R_FCi.reindex(common_index)
    R_FC1_aligned = R_FC1.reindex(common_index)
    R_FC2_aligned = R_FC2.reindex(common_index)
    
    # Inicializar DataFrame de resultados
    result = pd.DataFrame(index=common_index)
    
    # Calcular REF
    REF = (R_FC1_aligned + R_FC2_aligned) / 2.0
    result['REF'] = REF
    
    # Calcular SR
    mask_valid = (REF > 0) & R_FCi_aligned.notna() & R_FC1_aligned.notna() & R_FC2_aligned.notna()
    result['SR'] = np.nan
    result.loc[mask_valid, 'SR'] = 100.0 * R_FCi_aligned[mask_valid] / REF[mask_valid]
    
    # Calcular incertidumbres
    u_R_FCi = channel_u(R_FCi_aligned, u_add, u_scale)
    u_REF = ref_u(R_FC1_aligned, R_FC2_aligned, u_add, u_scale, rho_ref)
    
    # Derivadas parciales
    dSR_dR_FCi = np.nan * np.ones_like(REF)
    dSR_dREF = np.nan * np.ones_like(REF)
    dSR_dR_FCi[mask_valid] = 100.0 / REF[mask_valid]
    dSR_dREF[mask_valid] = -100.0 * R_FCi_aligned[mask_valid] / (REF[mask_valid]**2)
    
    # Covarianza entre R_FCi y REF
    if rho_soiled_ref != 0.0:
        cov_Ri_REF = rho_soiled_ref * u_R_FCi * u_REF
    else:
        cov_Ri_REF = pd.Series(0.0, index=common_index)
    
    # Varianza propagada
    var_SR = (dSR_dR_FCi**2) * (u_R_FCi**2) + (dSR_dREF**2) * (u_REF**2) + 2 * dSR_dR_FCi * dSR_dREF * cov_Ri_REF
    
    # Incertidumbre estándar (k=1)
    u_SR_k1_abs = np.sqrt(var_SR)
    
    # Incertidumbre relativa k=1 (evitar división por cero)
    result['u_SR_k1_abs'] = u_SR_k1_abs
    mask_sr_valid = (result['SR'] > 0) & (result['SR'] < 1000)  # Filtrar SR extremos
    result['u_SR_k1_rel'] = np.nan
    result.loc[mask_sr_valid, 'u_SR_k1_rel'] = u_SR_k1_abs[mask_sr_valid] / result.loc[mask_sr_valid, 'SR']
    
    # Incertidumbre expandida k=2
    result['U_SR_k2_abs'] = k_expand * u_SR_k1_abs
    result['U_SR_k2_rel'] = k_expand * result['u_SR_k1_rel']
    
    return result


def process_campaign_uncertainty(
    df_pv_glasses: pd.DataFrame,
    ref_col1: str = REF_COL1,
    ref_col2: str = REF_COL2,
    soiled_cols: List[str] = None
) -> Optional[pd.DataFrame]:
    """
    Procesa la incertidumbre minuto a minuto sobre toda la campaña para todas las celdas sucias.
    
    Args:
        df_pv_glasses: DataFrame con datos de PV Glasses
                     Debe tener índice DatetimeIndex y columnas R_FC1_Avg, R_FC2_Avg, R_FC3_Avg, R_FC4_Avg, R_FC5_Avg
        ref_col1: Nombre de columna de celda referencia 1
        ref_col2: Nombre de columna de celda referencia 2
        soiled_cols: Lista de columnas de celdas sucias (default: R_FC3_Avg, R_FC4_Avg, R_FC5_Avg)
    
    Returns:
        pd.DataFrame con columnas: timestamp, REF, SR_R_FC3, SR_R_FC4, SR_R_FC5, y sus incertidumbres
        o None si hay error
    """
    if soiled_cols is None:
        soiled_cols = SOILED_COLS
    
    try:
        logger.info("Procesando incertidumbre de campaña minuto a minuto (PV Glasses)...")
        
        # Verificar que las columnas necesarias existen
        required_cols = [ref_col1, ref_col2] + soiled_cols
        missing_cols = [col for col in required_cols if col not in df_pv_glasses.columns]
        if missing_cols:
            logger.error(f"Columnas faltantes: {missing_cols}")
            return None
        
        # Convertir a numérico
        for col in required_cols:
            df_pv_glasses[col] = pd.to_numeric(df_pv_glasses[col], errors='coerce')
        
        # Filtrar datos válidos
        df_valid = df_pv_glasses[required_cols].copy()
        df_valid = df_valid.dropna(subset=[ref_col1, ref_col2])
        df_valid = df_valid[~df_valid.index.duplicated(keep='first')]
        df_valid = df_valid.sort_index()
        
        logger.info(f"Datos válidos: {len(df_valid)} puntos")
        
        # Inicializar DataFrame de resultados
        result = pd.DataFrame(index=df_valid.index)
        
        # Calcular REF una vez
        REF = (df_valid[ref_col1] + df_valid[ref_col2]) / 2.0
        result['REF'] = REF
        
        # Procesar cada celda sucia
        for soiled_col in soiled_cols:
            logger.info(f"Procesando {soiled_col}...")
            
            # Calcular SR e incertidumbre para esta celda
            df_sr = propagate_sr_minute_pv_glasses(
                df_valid[soiled_col],
                df_valid[ref_col1],
                df_valid[ref_col2]
            )
            
            # Agregar columnas al resultado
            sr_col_name = f"SR_{soiled_col.replace('_Avg', '')}"
            result[sr_col_name] = df_sr['SR']
            result[f'u_{sr_col_name}_k1_abs'] = df_sr['u_SR_k1_abs']
            result[f'u_{sr_col_name}_k1_rel'] = df_sr['u_SR_k1_rel']
            result[f'U_{sr_col_name}_k2_abs'] = df_sr['U_SR_k2_abs']
            result[f'U_{sr_col_name}_k2_rel'] = df_sr['U_SR_k2_rel']
        
        logger.info(f"Procesamiento completado. Columnas en resultado: {result.columns.tolist()}")
        return result
        
    except Exception as e:
        logger.error(f"Error al procesar incertidumbre de campaña: {e}", exc_info=True)
        return None


def calculate_campaign_uncertainty(df_result: pd.DataFrame) -> Optional[dict]:
    """
    Calcula estadísticas globales de incertidumbre para toda la campaña.
    
    Args:
        df_result: DataFrame con resultados de process_campaign_uncertainty
    
    Returns:
        dict con estadísticas de incertidumbre o None si hay error
    """
    try:
        stats = {}
        
        # Encontrar todas las columnas de SR
        sr_cols = [col for col in df_result.columns if col.startswith('SR_')]
        
        for sr_col in sr_cols:
            u_k1_rel_col = f'u_{sr_col}_k1_rel'
            U_k2_rel_col = f'U_{sr_col}_k2_rel'
            
            if u_k1_rel_col in df_result.columns:
                u_k1_rel = df_result[u_k1_rel_col].dropna()
                if len(u_k1_rel) > 0:
                    stats[f'{sr_col}_u_k1_rel_mean'] = u_k1_rel.mean()
                    stats[f'{sr_col}_u_k1_rel_std'] = u_k1_rel.std()
                    stats[f'{sr_col}_u_k1_rel_min'] = u_k1_rel.min()
                    stats[f'{sr_col}_u_k1_rel_max'] = u_k1_rel.max()
            
            if U_k2_rel_col in df_result.columns:
                U_k2_rel = df_result[U_k2_rel_col].dropna()
                if len(U_k2_rel) > 0:
                    stats[f'{sr_col}_U_k2_rel_mean'] = U_k2_rel.mean()
                    stats[f'{sr_col}_U_k2_rel_std'] = U_k2_rel.std()
                    stats[f'{sr_col}_U_k2_rel_min'] = U_k2_rel.min()
                    stats[f'{sr_col}_U_k2_rel_max'] = U_k2_rel.max()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error al calcular estadísticas de campaña: {e}", exc_info=True)
        return None


def aggregate_with_uncertainty(
    df_result: pd.DataFrame,
    freq: str = 'D',
    method: str = 'mean'
) -> Optional[pd.DataFrame]:
    """
    Agrega SR con incertidumbre a escalas temporales mayores.
    
    Args:
        df_result: DataFrame con resultados de process_campaign_uncertainty
        freq: Frecuencia de agregación ('D'=diario, 'W'=semanal, 'ME'=mensual)
        method: Método de agregación ('mean' o 'quantile')
    
    Returns:
        pd.DataFrame agregado con incertidumbre o None si hay error
    """
    try:
        # Encontrar todas las columnas de SR
        sr_cols = [col for col in df_result.columns if col.startswith('SR_')]
        
        if not sr_cols:
            logger.warning("No se encontraron columnas de SR para agregar")
            return None
        
        # Agregar cada SR
        df_agg = pd.DataFrame()
        
        for sr_col in sr_cols:
            # Agregar SR
            if method == 'mean':
                sr_agg = df_result[sr_col].resample(freq).mean()
            elif method == 'quantile':
                sr_agg = df_result[sr_col].resample(freq).quantile(0.25)
            else:
                logger.error(f"Método de agregación desconocido: {method}")
                return None
            
            df_agg[sr_col] = sr_agg
            
            # Agregar incertidumbres (usar promedio)
            u_k1_rel_col = f'u_{sr_col}_k1_rel'
            U_k2_rel_col = f'U_{sr_col}_k2_rel'
            
            if u_k1_rel_col in df_result.columns:
                df_agg[u_k1_rel_col] = df_result[u_k1_rel_col].resample(freq).mean()
            
            if U_k2_rel_col in df_result.columns:
                df_agg[U_k2_rel_col] = df_result[U_k2_rel_col].resample(freq).mean()
        
        return df_agg
        
    except Exception as e:
        logger.error(f"Error al agregar con incertidumbre: {e}", exc_info=True)
        return None


def run_uncertainty_propagation_analysis_pv_glasses(
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None
) -> bool:
    """
    Función principal para ejecutar el análisis de incertidumbre de PV Glasses.
    
    Args:
        input_file: Ruta al archivo CSV de entrada (default: paths.PV_GLASSES_RAW_DATA_FILE)
        output_dir: Directorio de salida (default: paths.PROPAGACION_ERRORES_PV_GLASSES_DIR)
    
    Returns:
        bool: True si el análisis fue exitoso, False en caso contrario
    """
    try:
        logger.info("=" * 80)
        logger.info("INICIANDO ANÁLISIS DE INCERTIDUMBRE DE PV GLASSES")
        logger.info("=" * 80)
        
        # Usar valores por defecto si no se proporcionan
        if input_file is None:
            input_file = paths.PV_GLASSES_RAW_DATA_FILE
        
        if output_dir is None:
            output_dir = paths.PROPAGACION_ERRORES_PV_GLASSES_DIR
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Verificar que el archivo de entrada existe
        if not os.path.exists(input_file):
            logger.error(f"Archivo de entrada no encontrado: {input_file}")
            return False
        
        logger.info(f"Archivo de entrada: {input_file}")
        logger.info(f"Directorio de salida: {output_dir}")
        
        # Cargar datos
        logger.info("Cargando datos de PV Glasses...")
        try:
            # Intentar leer con pandas (asumiendo formato similar a ref_cells)
            df_pv_glasses = pd.read_csv(input_file, index_col='_time', parse_dates=True)
            
            # Asegurar que el índice es DatetimeIndex y está en UTC
            if not isinstance(df_pv_glasses.index, pd.DatetimeIndex):
                logger.error("El índice no es un DatetimeIndex después de cargar el archivo")
                return False
            
            # Manejar timezone: asegurar UTC
            if df_pv_glasses.index.tz is None:
                logger.info("Índice es DatetimeIndex pero naive. Localizando a UTC...")
                df_pv_glasses.index = df_pv_glasses.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            elif df_pv_glasses.index.tz != pytz.UTC:
                logger.info(f"Índice con zona horaria {df_pv_glasses.index.tz}. Convirtiendo a UTC...")
                df_pv_glasses.index = df_pv_glasses.index.tz_convert('UTC')
            else:
                logger.info("Índice ya es DatetimeIndex y está en UTC.")
            
            df_pv_glasses.sort_index(inplace=True)
            logger.info(f"Índice procesado a UTC. Rango: {df_pv_glasses.index.min()} a {df_pv_glasses.index.max()}")
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}", exc_info=True)
            return False
        
        logger.info(f"Datos cargados: {len(df_pv_glasses)} filas, {len(df_pv_glasses.columns)} columnas")
        logger.info(f"Columnas disponibles: {df_pv_glasses.columns.tolist()}")
        
        # Procesar incertidumbre minuto a minuto
        logger.info("Procesando incertidumbre minuto a minuto...")
        df_result = process_campaign_uncertainty(df_pv_glasses)
        
        if df_result is None:
            logger.error("Error al procesar incertidumbre minuto a minuto")
            return False
        
        logger.info(f"Resultados procesados: {len(df_result)} puntos")
        
        # Guardar resultados minuto a minuto
        output_file_minute = paths.PV_GLASSES_SR_MINUTE_WITH_UNCERTAINTY_FILE
        df_result.to_csv(output_file_minute)
        logger.info(f"Resultados minuto a minuto guardados en: {output_file_minute}")
        
        # Calcular estadísticas de campaña
        logger.info("Calculando estadísticas de campaña...")
        stats = calculate_campaign_uncertainty(df_result)
        
        if stats:
            # Guardar resumen
            summary_file = paths.PV_GLASSES_SR_UNCERTAINTY_SUMMARY_FILE
            with open(summary_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("RESUMEN DE INCERTIDUMBRE DE PV GLASSES\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Archivo de entrada: {input_file}\n")
                f.write(f"Total de puntos: {len(df_result)}\n\n")
                
                f.write("ESTADÍSTICAS DE INCERTIDUMBRE:\n")
                f.write("-" * 80 + "\n")
                for key, value in sorted(stats.items()):
                    f.write(f"{key}: {value:.6f}\n")
            
            logger.info(f"Resumen guardado en: {summary_file}")
        
        # Agregar a escalas diarias, semanales y mensuales
        freq_mapping = {
            'D': ('diario', paths.PV_GLASSES_SR_DAILY_ABS_WITH_U_FILE),
            'W': ('semanal', paths.PV_GLASSES_SR_WEEKLY_ABS_WITH_U_FILE),
            'ME': ('mensual', paths.PV_GLASSES_SR_MONTHLY_ABS_WITH_U_FILE)
        }
        
        for freq, (freq_name, output_file_agg) in freq_mapping.items():
            logger.info(f"Agregando a escala {freq_name}...")
            df_agg = aggregate_with_uncertainty(df_result, freq=freq, method='mean')
            
            if df_agg is not None and len(df_agg) > 0:
                df_agg.to_csv(output_file_agg)
                logger.info(f"Resultados {freq_name} guardados en: {output_file_agg}")
        
        logger.info("=" * 80)
        logger.info("ANÁLISIS DE INCERTIDUMBRE DE PV GLASSES COMPLETADO")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en análisis de incertidumbre de PV Glasses: {e}", exc_info=True)
        return False

