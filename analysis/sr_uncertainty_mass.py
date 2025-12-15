"""
Análisis de Incertidumbre de Mediciones de Masa mediante Propagación de Errores (GUM)

Este módulo calcula la incertidumbre de las diferencias de masa (Δm = m_soiled - m_clean)
usando propagación de errores según GUM:
- Incertidumbres de la balanza analítica según especificaciones del fabricante o certificado
- Propagación de errores para diferencias de masa
- Cálculo de incertidumbre expandida (k=2)

DESCRIPCIÓN:
------------
Las diferencias de masa se calculan como:
- Δm = m_soiled - m_clean (en gramos)
- Δm_mg = Δm × 1000 (en miligramos)

La incertidumbre se propaga usando:
- u(m)² = u_add² + (u_scale × m)² (para cada medición de masa)
- u(Δm)² = u(m_soiled)² + u(m_clean)² (asumiendo independencia)
- Si hay correlación: u(Δm)² = u(m_soiled)² + u(m_clean)² - 2×rho×u(m_soiled)×u(m_clean)
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple
from datetime import datetime
import config.settings as settings
import config.paths as paths

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE INCERTIDUMBRES DE LA BALANZA ANALÍTICA
# ============================================================================

# Incertidumbres de la balanza analítica (k=2)
# NOTA: Estos valores deben ajustarse según el certificado de calibración de la balanza utilizada
# Valores típicos para balanzas analíticas:
# - Resolución: 0.1 mg (0.0001 g)
# - Incertidumbre expandida (k=2): típicamente ±0.1 mg a ±0.2 mg

# Incertidumbre aditiva (k=2) en gramos
# Valor por defecto: 0.0002 g = 0.2 mg (conservador para balanzas analíticas típicas)
U_MASS_ADD_K2 = 0.0002  # g (aditiva, k=2) - Valor estimado, ajustar según certificado

# Incertidumbre de escala (k=2) - generalmente despreciable para balanzas analíticas
# Si existe, típicamente es muy pequeña (< 0.01% del valor)
U_MASS_SCALE_K2 = 0.0  # Adimensional (de escala, k=2) - Generalmente despreciable

# Convertir a k=1 (1σ)
U_MASS_ADD = U_MASS_ADD_K2 / 2.0  # 0.0001 g = 0.1 mg
U_MASS_SCALE = U_MASS_SCALE_K2 / 2.0  # 0.0 (sin componente de escala)

# Factor de cobertura para expandir al final
K_EXPAND = 2.0

# Columnas de datos esperadas
MASS_COLS = ['Masa A', 'Masa B', 'Masa C']
MASS_SOILED_COLS = ['Masa_A_Soiled_g', 'Masa_B_Soiled_g', 'Masa_C_Soiled_g']
MASS_CLEAN_COLS = ['Masa_A_Clean_g', 'Masa_B_Clean_g', 'Masa_C_Clean_g']
MASS_DIFF_COLS = ['Diferencia_Masa_A_mg', 'Diferencia_Masa_B_mg', 'Diferencia_Masa_C_mg']


def mass_u(m: pd.Series, u_add: float, u_scale: float) -> pd.Series:
    """
    Calcula la incertidumbre u(m) de una medición de masa usando el modelo combinado aditivo + escala.
    
    Fórmula: u(m)² = u_add² + (u_scale × m)²
    
    Args:
        m: Serie temporal de masa (g)
        u_add: Incertidumbre aditiva (g, k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(m) en g (k=1)
    """
    u_squared = u_add**2 + (u_scale * m)**2
    u_m = np.sqrt(u_squared)
    return u_m


def propagate_mass_difference(
    m_soiled: pd.Series,
    m_clean: pd.Series,
    u_add: float = U_MASS_ADD,
    u_scale: float = U_MASS_SCALE,
    k_expand: float = K_EXPAND,
    rho: float = 0.0
) -> pd.DataFrame:
    """
    Calcula la diferencia de masa y su incertidumbre usando propagación de errores (GUM).
    
    Fórmulas:
    - Δm = m_soiled - m_clean
    - u(Δm)² = u(m_soiled)² + u(m_clean)² - 2×rho×u(m_soiled)×u(m_clean)
    - Si rho = 0 (independencia): u(Δm)² = u(m_soiled)² + u(m_clean)²
    
    Args:
        m_soiled: Serie temporal de masa soiled (g)
        m_clean: Serie temporal de masa clean (g)
        u_add: Incertidumbre aditiva (g, k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
        k_expand: Factor de cobertura para expandir (default 2.0)
        rho: Coeficiente de correlación entre mediciones (default 0.0 = independencia)
    
    Returns:
        pd.DataFrame con columnas:
        - m_soiled: Masa soiled (g)
        - m_clean: Masa clean (g)
        - delta_m_g: Diferencia de masa (g)
        - delta_m_mg: Diferencia de masa (mg)
        - u_delta_m_k1_g: Incertidumbre absoluta k=1 (g)
        - u_delta_m_k1_mg: Incertidumbre absoluta k=1 (mg)
        - u_delta_m_k1_rel: Incertidumbre relativa k=1 (adimensional)
        - U_delta_m_k2_g: Incertidumbre absoluta expandida k=2 (g)
        - U_delta_m_k2_mg: Incertidumbre absoluta expandida k=2 (mg)
        - U_delta_m_k2_rel: Incertidumbre relativa expandida k=2 (adimensional)
    """
    # Asegurar que tienen el mismo índice
    common_index = m_soiled.index.intersection(m_clean.index)
    m_soiled_aligned = m_soiled.reindex(common_index)
    m_clean_aligned = m_clean.reindex(common_index)
    
    # Inicializar DataFrame de resultados
    result = pd.DataFrame(index=common_index)
    result['m_soiled'] = m_soiled_aligned
    result['m_clean'] = m_clean_aligned
    
    # Calcular diferencia de masa
    mask_valid = m_soiled_aligned.notna() & m_clean_aligned.notna()
    result['delta_m_g'] = np.nan
    result.loc[mask_valid, 'delta_m_g'] = m_soiled_aligned[mask_valid] - m_clean_aligned[mask_valid]
    result['delta_m_mg'] = result['delta_m_g'] * 1000.0  # Convertir a mg
    
    # Calcular incertidumbres de las masas individuales
    u_m_soiled = mass_u(m_soiled_aligned, u_add, u_scale)
    u_m_clean = mass_u(m_clean_aligned, u_add, u_scale)
    
    # Covarianza entre mediciones
    if rho != 0.0:
        cov_m_soiled_clean = rho * u_m_soiled * u_m_clean
    else:
        cov_m_soiled_clean = pd.Series(0.0, index=common_index)
    
    # Varianza propagada de la diferencia
    var_delta_m = (u_m_soiled**2) + (u_m_clean**2) - 2 * cov_m_soiled_clean
    
    # Incertidumbre estándar (k=1)
    u_delta_m_k1_g = np.sqrt(var_delta_m)
    u_delta_m_k1_mg = u_delta_m_k1_g * 1000.0  # Convertir a mg
    
    result['u_delta_m_k1_g'] = u_delta_m_k1_g
    result['u_delta_m_k1_mg'] = u_delta_m_k1_mg
    
    # Incertidumbre relativa k=1 (evitar división por cero)
    mask_delta_valid = (result['delta_m_g'].abs() > 1e-10) & mask_valid
    result['u_delta_m_k1_rel'] = np.nan
    result.loc[mask_delta_valid, 'u_delta_m_k1_rel'] = (
        u_delta_m_k1_g[mask_delta_valid] / result.loc[mask_delta_valid, 'delta_m_g'].abs()
    )
    
    # Incertidumbre expandida k=2
    result['U_delta_m_k2_g'] = k_expand * u_delta_m_k1_g
    result['U_delta_m_k2_mg'] = k_expand * u_delta_m_k1_mg
    result['U_delta_m_k2_rel'] = k_expand * result['u_delta_m_k1_rel']
    
    return result


def process_mass_uncertainty(
    df_mass: pd.DataFrame,
    mass_soiled_cols: list = None,
    mass_clean_cols: list = None,
    mass_diff_cols: list = None
) -> Optional[pd.DataFrame]:
    """
    Procesa la incertidumbre de diferencias de masa para todas las muestras.
    
    Args:
        df_mass: DataFrame con datos de masas
                 Debe tener columnas con masas soiled, clean y diferencias
        mass_soiled_cols: Lista de columnas de masas soiled (g)
        mass_clean_cols: Lista de columnas de masas clean (g)
        mass_diff_cols: Lista de columnas de diferencias de masa (mg)
    
    Returns:
        pd.DataFrame con columnas de incertidumbre agregadas
        o None si hay error
    """
    try:
        logger.info("Procesando incertidumbre de diferencias de masa...")
        
        # Usar columnas por defecto si no se especifican
        if mass_soiled_cols is None:
            mass_soiled_cols = MASS_SOILED_COLS
        if mass_clean_cols is None:
            mass_clean_cols = MASS_CLEAN_COLS
        if mass_diff_cols is None:
            mass_diff_cols = MASS_DIFF_COLS
        
        # Crear copia del DataFrame
        result = df_mass.copy()
        
        # Procesar cada par de masas (A, B, C)
        for i, (soiled_col, clean_col, diff_col) in enumerate(zip(mass_soiled_cols, mass_clean_cols, mass_diff_cols)):
            if soiled_col not in result.columns or clean_col not in result.columns:
                logger.warning(f"Columnas {soiled_col} o {clean_col} no encontradas, saltando...")
                continue
            
            # Convertir a Series
            m_soiled = pd.to_numeric(result[soiled_col], errors='coerce')
            m_clean = pd.to_numeric(result[clean_col], errors='coerce')
            
            # Calcular propagación de errores
            df_uncertainty = propagate_mass_difference(
                m_soiled,
                m_clean,
                u_add=U_MASS_ADD,
                u_scale=U_MASS_SCALE,
                k_expand=K_EXPAND,
                rho=0.0  # Independencia entre mediciones
            )
            
            # Agregar columnas de incertidumbre al resultado
            suffix = diff_col.replace('Diferencia_Masa_', '').replace('_mg', '')
            result[f'u_delta_m_{suffix}_k1_mg'] = df_uncertainty['u_delta_m_k1_mg']
            result[f'u_delta_m_{suffix}_k1_rel'] = df_uncertainty['u_delta_m_k1_rel']
            result[f'U_delta_m_{suffix}_k2_mg'] = df_uncertainty['U_delta_m_k2_mg']
            result[f'U_delta_m_{suffix}_k2_rel'] = df_uncertainty['U_delta_m_k2_rel']
            
            logger.info(f"Incertidumbre calculada para {suffix}")
        
        # Contar muestras válidas
        n_valid = len(result[result[mass_diff_cols[0]].notna()])
        logger.info(f"Incertidumbre calculada para {n_valid} muestras válidas")
        
        return result
        
    except Exception as e:
        logger.error(f"Error procesando incertidumbre de masas: {e}", exc_info=True)
        return None


def calculate_campaign_uncertainty(df_mass_uncertainty: pd.DataFrame, mass_diff_cols: list = None) -> Tuple[float, float]:
    """
    Calcula la incertidumbre de campaña como promedio de las incertidumbres relativas.
    
    Args:
        df_mass_uncertainty: DataFrame con resultados de process_mass_uncertainty
        mass_diff_cols: Lista de columnas de diferencias de masa
    
    Returns:
        tuple: (u_campaign_k1_rel, U_campaign_k2_rel) en porcentaje
    """
    if mass_diff_cols is None:
        mass_diff_cols = MASS_DIFF_COLS
    
    # Recopilar todas las incertidumbres relativas k=2
    all_uncertainties = []
    for diff_col in mass_diff_cols:
        suffix = diff_col.replace('Diferencia_Masa_', '').replace('_mg', '')
        u_col = f'U_delta_m_{suffix}_k2_rel'
        if u_col in df_mass_uncertainty.columns:
            valid_data = df_mass_uncertainty[u_col].dropna()
            valid_data = valid_data[np.isfinite(valid_data)]
            valid_data = valid_data[(valid_data >= 0) & (valid_data <= 10.0)]  # 0% a 1000% (razonable)
            all_uncertainties.extend(valid_data.tolist())
    
    if len(all_uncertainties) == 0:
        logger.warning("No hay datos válidos para calcular incertidumbre de campaña")
        return np.nan, np.nan
    
    U_campaign_k2_rel = np.mean(all_uncertainties) * 100  # Convertir a porcentaje
    u_campaign_k1_rel = U_campaign_k2_rel / K_EXPAND
    
    logger.info(f"Incertidumbre de campaña (k=1): {u_campaign_k1_rel:.3f}%")
    logger.info(f"Incertidumbre de campaña (k=2): {U_campaign_k2_rel:.3f}%")
    
    return u_campaign_k1_rel, U_campaign_k2_rel


def run_uncertainty_propagation_analysis(
    df_mass: pd.DataFrame,
    output_file: Optional[str] = None
) -> bool:
    """
    Función principal para ejecutar el análisis completo de propagación de incertidumbre de masas.
    
    Args:
        df_mass: DataFrame con datos de masas (debe tener columnas de masas soiled, clean y diferencias)
        output_file: Ruta del archivo de salida (opcional)
    
    Returns:
        bool: True si el análisis fue exitoso, False en caso contrario
    """
    try:
        logger.info("="*80)
        logger.info("INICIANDO ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE DE MASAS")
        logger.info("="*80)
        
        # Crear directorio de salida si no existe
        os.makedirs(paths.PROPAGACION_ERRORES_MASS_DIR, exist_ok=True)
        
        # Procesar incertidumbre
        df_mass_uncertainty = process_mass_uncertainty(df_mass)
        
        if df_mass_uncertainty is None or df_mass_uncertainty.empty:
            logger.error("No se pudieron calcular incertidumbres de masas")
            return False
        
        # Guardar resultados
        if output_file is None:
            output_file = paths.MASS_DIFFERENCES_WITH_UNCERTAINTY_FILE
        
        df_mass_uncertainty.to_csv(output_file, index=False)
        logger.info(f"✅ Resultados guardados en: {output_file}")
        
        # Calcular incertidumbre de campaña
        u_campaign_k1_rel, U_campaign_k2_rel = calculate_campaign_uncertainty(df_mass_uncertainty)
        
        if np.isnan(u_campaign_k1_rel) or np.isnan(U_campaign_k2_rel):
            logger.warning("No se pudo calcular incertidumbre de campaña")
        else:
            logger.info(f"Incertidumbre de campaña calculada: {U_campaign_k2_rel:.3f}% (k=2)")
        
        logger.info("="*80)
        logger.info("✅ ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE COMPLETADO (MASAS)")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en análisis de propagación de incertidumbre: {e}", exc_info=True)
        return False



