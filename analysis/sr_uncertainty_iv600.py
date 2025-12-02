"""
Análisis de Incertidumbre del Soiling Ratio (SR) para IV600 mediante Propagación de Errores (GUM)

Este módulo calcula la incertidumbre del SR usando propagación de errores según GUM:
- Incertidumbres de medición de corriente (Isc) y potencia (Pmax) según certificado de calibración
- Propagación minuto a minuto usando derivadas parciales
- Agregación a escalas diarias, semanales y mensuales

DESCRIPCIÓN:
------------
El SR se calcula como:
- SR_Isc = 100 * Isc_soiled / Isc_reference
- SR_Pmax = 100 * Pmax_soiled / Pmax_reference

Módulos:
- 1MD434 (sucio) vs 1MD439 (referencia)
- 1MD440 (sucio) vs 1MD439 (referencia)

INCERTIDUMBRES DEL CERTIFICADO DE CALIBRACIÓN:
------------------------------------------------
Según certificado de calibración del IV600:
- Corriente (Isc): ±(0.2%Isc) - solo componente de escala
- Potencia (Pmax): ±(1.0%lectura + 6 dgt)
  - Componente de escala: 1.0%
  - Componente aditiva: 6 dígitos × resolución
    - Para rango 50-9999 W: resolución = 1 W, entonces 6 dgt = 6 W
    - Para rango 10k-59.99k W: resolución = 0.01k W = 10 W, entonces 6 dgt = 60 W
  - Se usa 6 W como valor conservador (rango más común)
- Condiciones: 23°C ± 5°C, <80%RH
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple, List
from datetime import datetime
import config.settings as settings
import config.paths as paths

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE INCERTIDUMBRES DEL CERTIFICADO DE CALIBRACIÓN
# ============================================================================

# Incertidumbres del IV600 según certificado de calibración (k=2)
# Formato: ±[%lectura + (núm. dgt*resolución)]

# Corriente (Isc): ±(0.2%Isc)
U_ISC_ADD_K2 = 0.0  # A (aditiva, k=2) - Sin componente aditiva
U_ISC_SCALE_K2 = 0.002  # 0.2% (de escala, k=2) - Valor del certificado

# Potencia (Pmax): ±(1.0%lectura + 6 dgt)
# Para rango 50-9999 W: resolución = 1 W, entonces 6 dgt = 6 W
# Para rango 10k-59.99k W: resolución = 0.01k W = 10 W, entonces 6 dgt = 60 W
# Se usa 6 W como valor conservador (rango más común)
U_PMAX_ADD_K2 = 6.0  # W (aditiva, k=2) - 6 dígitos × 1 W
U_PMAX_SCALE_K2 = 0.01  # 1.0% (de escala, k=2) - Valor del certificado

# Convertir a k=1 (1σ)
U_ISC_ADD = U_ISC_ADD_K2 / 2.0  # A (0.0)
U_ISC_SCALE = U_ISC_SCALE_K2 / 2.0  # 0.001 (0.1%)

U_PMAX_ADD = U_PMAX_ADD_K2 / 2.0  # W (3.0 W)
U_PMAX_SCALE = U_PMAX_SCALE_K2 / 2.0  # 0.005 (0.5%)

# Factor de cobertura
K_EXPAND = 2.0

# Módulos de interés
TEST_MODULES = ['1MD434', '1MD440']  # Módulos sucios
REF_MODULE = '1MD439'  # Módulo de referencia

# Umbrales mínimos
MIN_ISC_THRESHOLD = 0.2  # A (según certificado: Iscmin = 0.2A)
MIN_PMAX_THRESHOLD = 50.0  # W (según certificado: rango mínimo)
MIN_VOLTAGE_THRESHOLD = 15.0  # V (según certificado: VCC > 15V)


def isc_u(I: pd.Series, u_add: float, u_scale: float) -> pd.Series:
    """
    Calcula la incertidumbre u(Isc) usando el modelo combinado aditivo + escala.
    
    Fórmula: u(Isc)² = u_add² + (u_scale * Isc)²
    
    Args:
        I: Serie temporal de corriente de cortocircuito (A)
        u_add: Incertidumbre aditiva (A, k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(Isc) en A (k=1)
    """
    u_squared = u_add**2 + (u_scale * I)**2
    u_I = np.sqrt(u_squared)
    return u_I


def pmax_u(P: pd.Series, u_add: float, u_scale: float) -> pd.Series:
    """
    Calcula la incertidumbre u(Pmax) usando el modelo combinado aditivo + escala.
    
    Fórmula: u(Pmax)² = u_add² + (u_scale * Pmax)²
    
    Args:
        P: Serie temporal de potencia máxima (W)
        u_add: Incertidumbre aditiva (W, k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(Pmax) en W (k=1)
    """
    u_squared = u_add**2 + (u_scale * P)**2
    u_P = np.sqrt(u_squared)
    return u_P


def propagate_sr_isc_minute_iv600(
    Isc_soiled: pd.Series,
    Isc_ref: pd.Series,
    u_isc_add: float = U_ISC_ADD,
    u_isc_scale: float = U_ISC_SCALE,
    k_expand: float = K_EXPAND,
    rho: float = 0.0
) -> pd.DataFrame:
    """
    Calcula SR_Isc y su incertidumbre minuto a minuto usando propagación de errores (GUM).
    
    Fórmulas:
    - SR_Isc = 100 * Isc_soiled / Isc_ref
    - ∂SR_Isc/∂Isc_soiled = 100 / Isc_ref
    - ∂SR_Isc/∂Isc_ref = -100 * Isc_soiled / Isc_ref²
    - Var(SR_Isc) = (∂SR_Isc/∂Isc_soiled)² * u(Isc_soiled)² + (∂SR_Isc/∂Isc_ref)² * u(Isc_ref)² + 2 * (∂SR_Isc/∂Isc_soiled) * (∂SR_Isc/∂Isc_ref) * Cov(Isc_soiled, Isc_ref)
    
    Args:
        Isc_soiled: Serie temporal de corriente módulo sucio (A)
        Isc_ref: Serie temporal de corriente módulo referencia (A)
        u_isc_add: Incertidumbre aditiva de corriente (A, k=1)
        u_isc_scale: Incertidumbre de escala de corriente (adimensional, k=1)
        k_expand: Factor de cobertura para expandir (default 2.0)
        rho: Coeficiente de correlación entre Isc_soiled e Isc_ref (default 0.0)
    
    Returns:
        pd.DataFrame con columnas:
        - SR_Isc: Soiling Ratio basado en Isc (%)
        - u_SR_Isc_k1_abs: Incertidumbre absoluta k=1 (%)
        - u_SR_Isc_k1_rel: Incertidumbre relativa k=1 (adimensional)
        - U_SR_Isc_k2_abs: Incertidumbre absoluta expandida k=2 (%)
        - U_SR_Isc_k2_rel: Incertidumbre relativa expandida k=2 (adimensional)
    """
    # Asegurar que tienen el mismo índice
    common_index = Isc_soiled.index.intersection(Isc_ref.index)
    Isc_soiled_aligned = Isc_soiled.reindex(common_index)
    Isc_ref_aligned = Isc_ref.reindex(common_index)
    
    # Inicializar DataFrame de resultados
    result = pd.DataFrame(index=common_index)
    
    # Calcular SR_Isc
    mask_valid = (Isc_ref_aligned > 0) & Isc_soiled_aligned.notna() & Isc_ref_aligned.notna()
    result['SR_Isc'] = np.nan
    result.loc[mask_valid, 'SR_Isc'] = 100.0 * Isc_soiled_aligned[mask_valid] / Isc_ref_aligned[mask_valid]
    
    # Calcular incertidumbres
    u_Isc_soiled = isc_u(Isc_soiled_aligned, u_isc_add, u_isc_scale)
    u_Isc_ref = isc_u(Isc_ref_aligned, u_isc_add, u_isc_scale)
    
    # Derivadas parciales
    dSR_dIsc_soiled = np.nan * np.ones_like(Isc_ref_aligned)
    dSR_dIsc_ref = np.nan * np.ones_like(Isc_ref_aligned)
    dSR_dIsc_soiled[mask_valid] = 100.0 / Isc_ref_aligned[mask_valid]
    dSR_dIsc_ref[mask_valid] = -100.0 * Isc_soiled_aligned[mask_valid] / (Isc_ref_aligned[mask_valid]**2)
    
    # Covarianza
    if rho != 0.0:
        cov_Isc = rho * u_Isc_soiled * u_Isc_ref
    else:
        cov_Isc = pd.Series(0.0, index=common_index)
    
    # Varianza propagada
    var_SR_Isc = (dSR_dIsc_soiled**2) * (u_Isc_soiled**2) + (dSR_dIsc_ref**2) * (u_Isc_ref**2) + 2 * dSR_dIsc_soiled * dSR_dIsc_ref * cov_Isc
    
    # Incertidumbre estándar (k=1)
    u_SR_Isc_k1_abs = np.sqrt(var_SR_Isc)
    
    # Incertidumbre relativa k=1
    result['u_SR_Isc_k1_abs'] = u_SR_Isc_k1_abs
    mask_sr_valid = (result['SR_Isc'] > 0) & (result['SR_Isc'] < 1000)
    result['u_SR_Isc_k1_rel'] = np.nan
    result.loc[mask_sr_valid, 'u_SR_Isc_k1_rel'] = u_SR_Isc_k1_abs[mask_sr_valid] / result.loc[mask_sr_valid, 'SR_Isc']
    
    # Incertidumbre expandida k=2
    result['U_SR_Isc_k2_abs'] = k_expand * u_SR_Isc_k1_abs
    result['U_SR_Isc_k2_rel'] = k_expand * result['u_SR_Isc_k1_rel']
    
    return result


def propagate_sr_pmax_minute_iv600(
    Pmax_soiled: pd.Series,
    Pmax_ref: pd.Series,
    u_pmax_add: float = U_PMAX_ADD,
    u_pmax_scale: float = U_PMAX_SCALE,
    k_expand: float = K_EXPAND,
    rho: float = 0.0
) -> pd.DataFrame:
    """
    Calcula SR_Pmax y su incertidumbre minuto a minuto usando propagación de errores (GUM).
    
    Fórmulas:
    - SR_Pmax = 100 * Pmax_soiled / Pmax_ref
    - ∂SR_Pmax/∂Pmax_soiled = 100 / Pmax_ref
    - ∂SR_Pmax/∂Pmax_ref = -100 * Pmax_soiled / Pmax_ref²
    - Var(SR_Pmax) = (∂SR_Pmax/∂Pmax_soiled)² * u(Pmax_soiled)² + (∂SR_Pmax/∂Pmax_ref)² * u(Pmax_ref)² + 2 * (∂SR_Pmax/∂Pmax_soiled) * (∂SR_Pmax/∂Pmax_ref) * Cov(Pmax_soiled, Pmax_ref)
    
    Args:
        Pmax_soiled: Serie temporal de potencia máxima módulo sucio (W)
        Pmax_ref: Serie temporal de potencia máxima módulo referencia (W)
        u_pmax_add: Incertidumbre aditiva de potencia (W, k=1)
        u_pmax_scale: Incertidumbre de escala de potencia (adimensional, k=1)
        k_expand: Factor de cobertura para expandir (default 2.0)
        rho: Coeficiente de correlación entre Pmax_soiled y Pmax_ref (default 0.0)
    
    Returns:
        pd.DataFrame con columnas:
        - SR_Pmax: Soiling Ratio basado en Pmax (%)
        - u_SR_Pmax_k1_abs: Incertidumbre absoluta k=1 (%)
        - u_SR_Pmax_k1_rel: Incertidumbre relativa k=1 (adimensional)
        - U_SR_Pmax_k2_abs: Incertidumbre absoluta expandida k=2 (%)
        - U_SR_Pmax_k2_rel: Incertidumbre relativa expandida k=2 (adimensional)
    """
    # Asegurar que tienen el mismo índice
    common_index = Pmax_soiled.index.intersection(Pmax_ref.index)
    Pmax_soiled_aligned = Pmax_soiled.reindex(common_index)
    Pmax_ref_aligned = Pmax_ref.reindex(common_index)
    
    # Inicializar DataFrame de resultados
    result = pd.DataFrame(index=common_index)
    
    # Calcular SR_Pmax
    mask_valid = (Pmax_ref_aligned > 0) & Pmax_soiled_aligned.notna() & Pmax_ref_aligned.notna()
    result['SR_Pmax'] = np.nan
    result.loc[mask_valid, 'SR_Pmax'] = 100.0 * Pmax_soiled_aligned[mask_valid] / Pmax_ref_aligned[mask_valid]
    
    # Calcular incertidumbres
    u_Pmax_soiled = pmax_u(Pmax_soiled_aligned, u_pmax_add, u_pmax_scale)
    u_Pmax_ref = pmax_u(Pmax_ref_aligned, u_pmax_add, u_pmax_scale)
    
    # Derivadas parciales
    dSR_dPmax_soiled = np.nan * np.ones_like(Pmax_ref_aligned)
    dSR_dPmax_ref = np.nan * np.ones_like(Pmax_ref_aligned)
    dSR_dPmax_soiled[mask_valid] = 100.0 / Pmax_ref_aligned[mask_valid]
    dSR_dPmax_ref[mask_valid] = -100.0 * Pmax_soiled_aligned[mask_valid] / (Pmax_ref_aligned[mask_valid]**2)
    
    # Covarianza
    if rho != 0.0:
        cov_Pmax = rho * u_Pmax_soiled * u_Pmax_ref
    else:
        cov_Pmax = pd.Series(0.0, index=common_index)
    
    # Varianza propagada
    var_SR_Pmax = (dSR_dPmax_soiled**2) * (u_Pmax_soiled**2) + (dSR_dPmax_ref**2) * (u_Pmax_ref**2) + 2 * dSR_dPmax_soiled * dSR_dPmax_ref * cov_Pmax
    
    # Incertidumbre estándar (k=1)
    u_SR_Pmax_k1_abs = np.sqrt(var_SR_Pmax)
    
    # Incertidumbre relativa k=1
    result['u_SR_Pmax_k1_abs'] = u_SR_Pmax_k1_abs
    mask_sr_valid = (result['SR_Pmax'] > 0) & (result['SR_Pmax'] < 1000)
    result['u_SR_Pmax_k1_rel'] = np.nan
    result.loc[mask_sr_valid, 'u_SR_Pmax_k1_rel'] = u_SR_Pmax_k1_abs[mask_sr_valid] / result.loc[mask_sr_valid, 'SR_Pmax']
    
    # Incertidumbre expandida k=2
    result['U_SR_Pmax_k2_abs'] = k_expand * u_SR_Pmax_k1_abs
    result['U_SR_Pmax_k2_rel'] = k_expand * result['u_SR_Pmax_k1_rel']
    
    return result


def process_campaign_uncertainty(
    df_iv600: pd.DataFrame,
    test_modules: List[str] = None,
    ref_module: str = None
) -> Optional[pd.DataFrame]:
    """
    Procesa la incertidumbre minuto a minuto sobre toda la campaña para todos los módulos.
    
    Args:
        df_iv600: DataFrame con datos de IV600
                 Debe tener índice DatetimeIndex y columnas con MultiIndex:
                 - ('1MD434', 'Isc'), ('1MD434', 'Pmax')
                 - ('1MD439', 'Isc'), ('1MD439', 'Pmax')
                 - ('1MD440', 'Isc'), ('1MD440', 'Pmax')
        test_modules: Lista de módulos sucios (default: ['1MD434', '1MD440'])
        ref_module: Módulo de referencia (default: '1MD439')
    
    Returns:
        pd.DataFrame con columnas: timestamp, SR_Isc_434vs439, SR_Pmax_434vs439, SR_Isc_440vs439, SR_Pmax_440vs439, y sus incertidumbres
        o None si hay error
    """
    if test_modules is None:
        test_modules = TEST_MODULES
    if ref_module is None:
        ref_module = REF_MODULE
    
    try:
        logger.info("Procesando incertidumbre de campaña minuto a minuto (IV600)...")
        
        # Verificar estructura de columnas (MultiIndex)
        if not isinstance(df_iv600.columns, pd.MultiIndex):
            logger.error("Las columnas deben ser MultiIndex con (módulo, parámetro)")
            return None
        
        # Inicializar DataFrame de resultados
        result = pd.DataFrame(index=df_iv600.index)
        
        # Procesar cada módulo de prueba
        for test_mod in test_modules:
            logger.info(f"Procesando módulo {test_mod} vs {ref_module}...")
            
            # Verificar que las columnas necesarias existen
            test_isc_col = (test_mod, 'Isc')
            test_pmax_col = (test_mod, 'Pmax')
            ref_isc_col = (ref_module, 'Isc')
            ref_pmax_col = (ref_module, 'Pmax')
            
            required_cols = [test_isc_col, test_pmax_col, ref_isc_col, ref_pmax_col]
            missing_cols = [col for col in required_cols if col not in df_iv600.columns]
            if missing_cols:
                logger.warning(f"Columnas faltantes para {test_mod}: {missing_cols}")
                continue
            
            # Convertir a numérico
            for col in required_cols:
                df_iv600[col] = pd.to_numeric(df_iv600[col], errors='coerce')
            
            # Filtrar datos válidos
            mask_valid = (
                df_iv600[ref_isc_col].notna() & 
                df_iv600[ref_pmax_col].notna() &
                (df_iv600[ref_isc_col] >= MIN_ISC_THRESHOLD) &
                (df_iv600[ref_pmax_col] >= MIN_PMAX_THRESHOLD)
            )
            
            if not mask_valid.any():
                logger.warning(f"No hay datos válidos para {test_mod}")
                continue
            
            # Calcular SR_Isc e incertidumbre
            df_sr_isc = propagate_sr_isc_minute_iv600(
                df_iv600[test_isc_col],
                df_iv600[ref_isc_col]
            )
            
            # Calcular SR_Pmax e incertidumbre
            df_sr_pmax = propagate_sr_pmax_minute_iv600(
                df_iv600[test_pmax_col],
                df_iv600[ref_pmax_col]
            )
            
            # Agregar columnas al resultado
            sr_isc_col_name = f"SR_Isc_{test_mod}vs{ref_module}"
            sr_pmax_col_name = f"SR_Pmax_{test_mod}vs{ref_module}"
            
            result[sr_isc_col_name] = df_sr_isc['SR_Isc']
            result[f'u_{sr_isc_col_name}_k1_abs'] = df_sr_isc['u_SR_Isc_k1_abs']
            result[f'u_{sr_isc_col_name}_k1_rel'] = df_sr_isc['u_SR_Isc_k1_rel']
            result[f'U_{sr_isc_col_name}_k2_abs'] = df_sr_isc['U_SR_Isc_k2_abs']
            result[f'U_{sr_isc_col_name}_k2_rel'] = df_sr_isc['U_SR_Isc_k2_rel']
            
            result[sr_pmax_col_name] = df_sr_pmax['SR_Pmax']
            result[f'u_{sr_pmax_col_name}_k1_abs'] = df_sr_pmax['u_SR_Pmax_k1_abs']
            result[f'u_{sr_pmax_col_name}_k1_rel'] = df_sr_pmax['u_SR_Pmax_k1_rel']
            result[f'U_{sr_pmax_col_name}_k2_abs'] = df_sr_pmax['U_SR_Pmax_k2_abs']
            result[f'U_{sr_pmax_col_name}_k2_rel'] = df_sr_pmax['U_SR_Pmax_k2_rel']
        
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
        freq: Frecuencia de agregación ('D'=diario, 'W'=semanal, 'M'=mensual)
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


def run_uncertainty_propagation_analysis_iv600(
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None
) -> bool:
    """
    Función principal para ejecutar el análisis de incertidumbre de IV600.
    
    Args:
        input_file: Ruta al archivo CSV de entrada (default: paths.IV600_RAW_DATA_FILE)
        output_dir: Directorio de salida (default: paths.PROPAGACION_ERRORES_IV600_DIR)
    
    Returns:
        bool: True si el análisis fue exitoso, False en caso contrario
    """
    try:
        logger.info("=" * 80)
        logger.info("INICIANDO ANÁLISIS DE INCERTIDUMBRE DE IV600")
        logger.info("=" * 80)
        
        # Usar valores por defecto si no se proporcionan
        if input_file is None:
            # Intentar encontrar el archivo de datos IV600
            input_file = os.path.join(paths.BASE_INPUT_DIR, 'raw_iv600_data.csv')
        
        if output_dir is None:
            output_dir = paths.PROPAGACION_ERRORES_IV600_DIR
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Verificar que el archivo de entrada existe
        if not os.path.exists(input_file):
            logger.error(f"Archivo de entrada no encontrado: {input_file}")
            return False
        
        logger.info(f"Archivo de entrada: {input_file}")
        logger.info(f"Directorio de salida: {output_dir}")
        
        # Cargar datos
        logger.info("Cargando datos de IV600...")
        try:
            # Leer datos IV600 (formato específico con MultiIndex)
            df_iv600 = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
            
            # Verificar si necesita procesamiento adicional para crear MultiIndex
            # Esto depende del formato específico del archivo
            logger.info(f"Datos cargados: {len(df_iv600)} filas, {len(df_iv600.columns)} columnas")
            logger.info(f"Columnas disponibles: {df_iv600.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            return False
        
        # Procesar incertidumbre minuto a minuto
        logger.info("Procesando incertidumbre minuto a minuto...")
        df_result = process_campaign_uncertainty(df_iv600)
        
        if df_result is None:
            logger.error("Error al procesar incertidumbre minuto a minuto")
            return False
        
        logger.info(f"Resultados procesados: {len(df_result)} puntos")
        
        # Guardar resultados minuto a minuto
        output_file_minute = paths.IV600_SR_MINUTE_WITH_UNCERTAINTY_FILE
        df_result.to_csv(output_file_minute)
        logger.info(f"Resultados minuto a minuto guardados en: {output_file_minute}")
        
        # Calcular estadísticas de campaña
        logger.info("Calculando estadísticas de campaña...")
        stats = calculate_campaign_uncertainty(df_result)
        
        if stats:
            # Guardar resumen
            summary_file = paths.IV600_SR_UNCERTAINTY_SUMMARY_FILE
            with open(summary_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("RESUMEN DE INCERTIDUMBRE DE IV600\n")
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
            'D': ('diario', paths.IV600_SR_DAILY_ABS_WITH_U_FILE),
            'W': ('semanal', paths.IV600_SR_WEEKLY_ABS_WITH_U_FILE),
            'M': ('mensual', paths.IV600_SR_MONTHLY_ABS_WITH_U_FILE)
        }
        
        for freq, (freq_name, output_file_agg) in freq_mapping.items():
            logger.info(f"Agregando a escala {freq_name}...")
            df_agg = aggregate_with_uncertainty(df_result, freq=freq, method='mean')
            
            if df_agg is not None and len(df_agg) > 0:
                df_agg.to_csv(output_file_agg)
                logger.info(f"Resultados {freq_name} guardados en: {output_file_agg}")
        
        logger.info("=" * 80)
        logger.info("ANÁLISIS DE INCERTIDUMBRE DE IV600 COMPLETADO")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en análisis de incertidumbre de IV600: {e}", exc_info=True)
        return False

