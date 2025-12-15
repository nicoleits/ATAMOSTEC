"""
Análisis de Incertidumbre del Soiling Ratio (SR) para PVStand mediante Propagación de Errores (GUM)

Este módulo calcula la incertidumbre del SR usando propagación de errores según GUM:
- Incertidumbres de medición de corriente (Isc), potencia (Pmax) y temperatura (T)
- Propagación minuto a minuto usando derivadas parciales
- Considera corrección de temperatura para Isc y Pmax
- Agregación a escalas diarias, semanales y mensuales

DESCRIPCIÓN:
------------
El SR se calcula como:
- SR_Isc = 100 * Isc_soiled / Isc_reference
- SR_Pmax = 100 * Pmax_soiled / Pmax_reference

Con corrección de temperatura:
- Isc_corr = Isc / (1 + α_isc × (T - T_ref))
- Pmax_corr = Pmax / (1 + β_pmax × (T - T_ref))
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
# CONFIGURACIÓN DE INCERTIDUMBRES
# ============================================================================

# Incertidumbres del IV tracer (k=2) - VALORES DEL FABRICANTE
# Accuracy según especificaciones del fabricante:
# - Corriente (Isc): 0.2% of reading
# - Voltaje: 0.2% of reading
# - MPP (Pmax): 0.4% of reading
# Nota: Se asume que estos valores son expandidos a k=2 (95% confianza)
U_ISC_ADD_K2 = 0.0  # A (aditiva, k=2) - Sin componente aditiva mencionada
U_ISC_SCALE_K2 = 0.002  # 0.2% (de escala, k=2) - Valor del fabricante

U_PMAX_ADD_K2 = 0.0  # W (aditiva, k=2) - Sin componente aditiva mencionada
U_PMAX_SCALE_K2 = 0.004  # 0.4% (de escala, k=2) - Valor del fabricante

# Convertir a k=1 (1σ)
U_ISC_ADD = U_ISC_ADD_K2 / 2.0  # A (0.0 si no hay componente aditiva)
U_ISC_SCALE = U_ISC_SCALE_K2 / 2.0  # 0.001 (0.1%)

U_PMAX_ADD = U_PMAX_ADD_K2 / 2.0  # W (0.0 si no hay componente aditiva)
U_PMAX_SCALE = U_PMAX_SCALE_K2 / 2.0  # 0.002 (0.2%)

# Incertidumbres del sensor de temperatura (k=2) - VALORES ESTIMADOS
U_TEMP_ADD_K2 = 1.0  # °C (aditiva, k=2)
U_TEMP_ADD = U_TEMP_ADD_K2 / 2.0  # °C

# Incertidumbres de coeficientes de temperatura (k=1)
U_ALPHA_ISC = 0.0001  # 1/°C (k=1) - Valor estimado
U_BETA_PMAX = 0.0001  # 1/°C (k=1) - Valor estimado

# Factor de cobertura
K_EXPAND = 2.0

# Coeficientes de temperatura (desde settings)
ALPHA_ISC = settings.PVSTAND_ALPHA_ISC_CORR  # -0.0004 o 0.0004
BETA_PMAX = settings.PVSTAND_BETA_PMAX_CORR  # +0.0037 o -0.0037
TEMP_REF = settings.PVSTAND_TEMP_REF_CORRECTION_C  # 25.0

# Umbrales mínimos
MIN_ISC_THRESHOLD = 0.5  # A
MIN_PMAX_THRESHOLD = 170.0  # W


def isc_u(I: pd.Series, u_add: float, u_scale: float) -> pd.Series:
    """Calcula incertidumbre de corriente."""
    u_squared = u_add**2 + (u_scale * I)**2
    return np.sqrt(u_squared)


def pmax_u(P: pd.Series, u_add: float, u_scale: float) -> pd.Series:
    """Calcula incertidumbre de potencia."""
    u_squared = u_add**2 + (u_scale * P)**2
    return np.sqrt(u_squared)


def temp_u(T: pd.Series, u_add: float) -> pd.Series:
    """Calcula incertidumbre de temperatura."""
    return pd.Series(u_add, index=T.index)


def isc_corrected_u(
    Isc: pd.Series,
    T: pd.Series,
    u_Isc: pd.Series,
    u_T: pd.Series,
    alpha: float,
    u_alpha: float,
    T_ref: float
) -> pd.Series:
    """
    Calcula incertidumbre de Isc corregida por temperatura.
    
    Isc_corr = Isc / (1 + α × (T - T_ref))
    
    Derivadas parciales:
    - ∂Isc_corr/∂Isc = 1 / (1 + α × (T - T_ref))
    - ∂Isc_corr/∂T = -α × Isc / (1 + α × (T - T_ref))²
    - ∂Isc_corr/∂α = -Isc × (T - T_ref) / (1 + α × (T - T_ref))²
    """
    denom = 1 + alpha * (T - T_ref)
    
    dIsc_corr_dIsc = 1 / denom
    dIsc_corr_dT = -alpha * Isc / (denom**2)
    dIsc_corr_dalpha = -Isc * (T - T_ref) / (denom**2)
    
    var_Isc_corr = (
        (dIsc_corr_dIsc**2) * (u_Isc**2) +
        (dIsc_corr_dT**2) * (u_T**2) +
        (dIsc_corr_dalpha**2) * (u_alpha**2)
    )
    
    return np.sqrt(var_Isc_corr)


def pmax_corrected_u(
    Pmax: pd.Series,
    T: pd.Series,
    u_Pmax: pd.Series,
    u_T: pd.Series,
    beta: float,
    u_beta: float,
    T_ref: float
) -> pd.Series:
    """
    Calcula incertidumbre de Pmax corregida por temperatura.
    
    Pmax_corr = Pmax / (1 + β × (T - T_ref))
    """
    denom = 1 + beta * (T - T_ref)
    
    dPmax_corr_dPmax = 1 / denom
    dPmax_corr_dT = -beta * Pmax / (denom**2)
    dPmax_corr_dbeta = -Pmax * (T - T_ref) / (denom**2)
    
    var_Pmax_corr = (
        (dPmax_corr_dPmax**2) * (u_Pmax**2) +
        (dPmax_corr_dT**2) * (u_T**2) +
        (dPmax_corr_dbeta**2) * (u_beta**2)
    )
    
    return np.sqrt(var_Pmax_corr)


def propagate_sr_isc_minute(
    Isc_soiled: pd.Series,
    Isc_ref: pd.Series,
    T_soiled: pd.Series,
    T_ref: pd.Series,
    use_temp_correction: bool = True,
    u_isc_add: float = U_ISC_ADD,
    u_isc_scale: float = U_ISC_SCALE,
    u_temp_add: float = U_TEMP_ADD,
    u_alpha: float = U_ALPHA_ISC,
    alpha: float = ALPHA_ISC,
    T_ref_corr: float = TEMP_REF,
    k_expand: float = K_EXPAND,
    rho_isc: float = 0.0
) -> pd.DataFrame:
    """Propaga incertidumbre para SR_Isc."""
    common_index = Isc_soiled.index.intersection(Isc_ref.index).intersection(T_soiled.index).intersection(T_ref.index)
    Isc_s = Isc_soiled.reindex(common_index)
    Isc_r = Isc_ref.reindex(common_index)
    T_s = T_soiled.reindex(common_index)
    T_r = T_ref.reindex(common_index)
    
    result = pd.DataFrame(index=common_index)
    
    mask_valid = (
        (Isc_s > MIN_ISC_THRESHOLD) &
        (Isc_r > MIN_ISC_THRESHOLD) &
        Isc_s.notna() & Isc_r.notna() &
        T_s.notna() & T_r.notna()
    )
    
    if use_temp_correction:
        Isc_s_corr = Isc_s / (1 + alpha * (T_s - T_ref_corr))
        Isc_r_corr = Isc_r / (1 + alpha * (T_r - T_ref_corr))
        
        result['SR'] = np.nan
        result.loc[mask_valid, 'SR'] = 100.0 * Isc_s_corr[mask_valid] / Isc_r_corr[mask_valid]
        
        u_Isc_s = isc_u(Isc_s, u_isc_add, u_isc_scale)
        u_Isc_r = isc_u(Isc_r, u_isc_add, u_isc_scale)
        u_T_s = temp_u(T_s, u_temp_add)
        u_T_r = temp_u(T_r, u_temp_add)
        
        u_Isc_s_corr = isc_corrected_u(Isc_s, T_s, u_Isc_s, u_T_s, alpha, u_alpha, T_ref_corr)
        u_Isc_r_corr = isc_corrected_u(Isc_r, T_r, u_Isc_r, u_T_r, alpha, u_alpha, T_ref_corr)
        
        dSR_dIsc_s = np.nan * np.ones_like(Isc_s)
        dSR_dIsc_r = np.nan * np.ones_like(Isc_s)
        dSR_dIsc_s[mask_valid] = 100.0 / Isc_r_corr[mask_valid]
        dSR_dIsc_r[mask_valid] = -100.0 * Isc_s_corr[mask_valid] / (Isc_r_corr[mask_valid]**2)
        
        cov = pd.Series(0.0, index=common_index) if rho_isc == 0.0 else rho_isc * u_Isc_r_corr * u_Isc_s_corr
        
        var_SR = (
            (dSR_dIsc_s**2) * (u_Isc_s_corr**2) +
            (dSR_dIsc_r**2) * (u_Isc_r_corr**2) +
            2 * dSR_dIsc_s * dSR_dIsc_r * cov
        )
    else:
        result['SR'] = np.nan
        result.loc[mask_valid, 'SR'] = 100.0 * Isc_s[mask_valid] / Isc_r[mask_valid]
        
        u_Isc_s = isc_u(Isc_s, u_isc_add, u_isc_scale)
        u_Isc_r = isc_u(Isc_r, u_isc_add, u_isc_scale)
        
        dSR_dIsc_s = np.nan * np.ones_like(Isc_s)
        dSR_dIsc_r = np.nan * np.ones_like(Isc_s)
        dSR_dIsc_s[mask_valid] = 100.0 / Isc_r[mask_valid]
        dSR_dIsc_r[mask_valid] = -100.0 * Isc_s[mask_valid] / (Isc_r[mask_valid]**2)
        
        cov = pd.Series(0.0, index=common_index) if rho_isc == 0.0 else rho_isc * u_Isc_r * u_Isc_s
        
        var_SR = (
            (dSR_dIsc_s**2) * (u_Isc_s**2) +
            (dSR_dIsc_r**2) * (u_Isc_r**2) +
            2 * dSR_dIsc_s * dSR_dIsc_r * cov
        )
    
    u_SR_k1_abs = np.sqrt(var_SR)
    result['u_SR_k1_abs'] = u_SR_k1_abs
    mask_sr_valid = (result['SR'] > 0) & (result['SR'] < 200)
    result['u_SR_k1_rel'] = np.nan
    result.loc[mask_sr_valid, 'u_SR_k1_rel'] = u_SR_k1_abs[mask_sr_valid] / result.loc[mask_sr_valid, 'SR']
    result['U_SR_k2_abs'] = k_expand * u_SR_k1_abs
    result['U_SR_k2_rel'] = k_expand * result['u_SR_k1_rel']
    
    return result


def propagate_sr_pmax_minute(
    Pmax_soiled: pd.Series,
    Pmax_ref: pd.Series,
    T_soiled: pd.Series,
    T_ref: pd.Series,
    use_temp_correction: bool = True,
    u_pmax_add: float = U_PMAX_ADD,
    u_pmax_scale: float = U_PMAX_SCALE,
    u_temp_add: float = U_TEMP_ADD,
    u_beta: float = U_BETA_PMAX,
    beta: float = BETA_PMAX,
    T_ref_corr: float = TEMP_REF,
    k_expand: float = K_EXPAND,
    rho_pmax: float = 0.0
) -> pd.DataFrame:
    """Propaga incertidumbre para SR_Pmax."""
    common_index = Pmax_soiled.index.intersection(Pmax_ref.index).intersection(T_soiled.index).intersection(T_ref.index)
    P_s = Pmax_soiled.reindex(common_index)
    P_r = Pmax_ref.reindex(common_index)
    T_s = T_soiled.reindex(common_index)
    T_r = T_ref.reindex(common_index)
    
    result = pd.DataFrame(index=common_index)
    
    mask_valid = (
        (P_s > MIN_PMAX_THRESHOLD) &
        (P_r > MIN_PMAX_THRESHOLD) &
        P_s.notna() & P_r.notna() &
        T_s.notna() & T_r.notna()
    )
    
    if use_temp_correction:
        P_s_corr = P_s / (1 + beta * (T_s - T_ref_corr))
        P_r_corr = P_r / (1 + beta * (T_r - T_ref_corr))
        
        result['SR'] = np.nan
        result.loc[mask_valid, 'SR'] = 100.0 * P_s_corr[mask_valid] / P_r_corr[mask_valid]
        
        u_P_s = pmax_u(P_s, u_pmax_add, u_pmax_scale)
        u_P_r = pmax_u(P_r, u_pmax_add, u_pmax_scale)
        u_T_s = temp_u(T_s, u_temp_add)
        u_T_r = temp_u(T_r, u_temp_add)
        
        u_P_s_corr = pmax_corrected_u(P_s, T_s, u_P_s, u_T_s, beta, u_beta, T_ref_corr)
        u_P_r_corr = pmax_corrected_u(P_r, T_r, u_P_r, u_T_r, beta, u_beta, T_ref_corr)
        
        dSR_dP_s = np.nan * np.ones_like(P_s)
        dSR_dP_r = np.nan * np.ones_like(P_s)
        dSR_dP_s[mask_valid] = 100.0 / P_r_corr[mask_valid]
        dSR_dP_r[mask_valid] = -100.0 * P_s_corr[mask_valid] / (P_r_corr[mask_valid]**2)
        
        cov = pd.Series(0.0, index=common_index) if rho_pmax == 0.0 else rho_pmax * u_P_r_corr * u_P_s_corr
        
        var_SR = (
            (dSR_dP_s**2) * (u_P_s_corr**2) +
            (dSR_dP_r**2) * (u_P_r_corr**2) +
            2 * dSR_dP_s * dSR_dP_r * cov
        )
    else:
        result['SR'] = np.nan
        result.loc[mask_valid, 'SR'] = 100.0 * P_s[mask_valid] / P_r[mask_valid]
        
        u_P_s = pmax_u(P_s, u_pmax_add, u_pmax_scale)
        u_P_r = pmax_u(P_r, u_pmax_add, u_pmax_scale)
        
        dSR_dP_s = np.nan * np.ones_like(P_s)
        dSR_dP_r = np.nan * np.ones_like(P_s)
        dSR_dP_s[mask_valid] = 100.0 / P_r[mask_valid]
        dSR_dP_r[mask_valid] = -100.0 * P_s[mask_valid] / (P_r[mask_valid]**2)
        
        cov = pd.Series(0.0, index=common_index) if rho_pmax == 0.0 else rho_pmax * u_P_r * u_P_s
        
        var_SR = (
            (dSR_dP_s**2) * (u_P_s**2) +
            (dSR_dP_r**2) * (u_P_r**2) +
            2 * dSR_dP_s * dSR_dP_r * cov
        )
    
    u_SR_k1_abs = np.sqrt(var_SR)
    result['u_SR_k1_abs'] = u_SR_k1_abs
    mask_sr_valid = (result['SR'] > 0) & (result['SR'] < 200)
    result['u_SR_k1_rel'] = np.nan
    result.loc[mask_sr_valid, 'u_SR_k1_rel'] = u_SR_k1_abs[mask_sr_valid] / result.loc[mask_sr_valid, 'SR']
    result['U_SR_k2_abs'] = k_expand * u_SR_k1_abs
    result['U_SR_k2_rel'] = k_expand * result['u_SR_k1_rel']
    
    return result


def process_campaign_uncertainty(
    df_pvstand: pd.DataFrame,
    sr_type: str = 'Isc',  # 'Isc' o 'Pmax'
    use_temp_correction: bool = True
) -> Optional[pd.DataFrame]:
    """
    Procesa incertidumbre minuto a minuto para PVStand.
    
    Args:
        df_pvstand: DataFrame con datos de PVStand (debe tener columnas de Isc/Pmax y temperatura)
        sr_type: 'Isc' o 'Pmax'
        use_temp_correction: Si True, usa corrección de temperatura
    """
    try:
        logger.info(f"Procesando incertidumbre de campaña minuto a minuto (PVStand - {sr_type})...")
        
        # Log de columnas disponibles
        logger.info(f"Columnas disponibles en DataFrame: {df_pvstand.columns.tolist()}")
        
        # Identificar columnas según sr_type
        if sr_type == 'Isc':
            col_soiled = settings.PVSTAND_ISC_COLUMN  # Necesita estar en settings
            col_ref = settings.PVSTAND_ISC_REF_COLUMN
        else:  # Pmax
            col_soiled = settings.PVSTAND_POUT_COLUMN
            col_ref = settings.PVSTAND_PMAX_REF_COLUMN
        
        temp_soiled_col = settings.PVSTAND_TEMP_SENSOR_SOILED_COL
        temp_ref_col = settings.PVSTAND_TEMP_SENSOR_REFERENCE_COL
        
        logger.info(f"Buscando columnas: col_soiled='{col_soiled}', col_ref='{col_ref}', temp_soiled='{temp_soiled_col}', temp_ref='{temp_ref_col}'")
        
        # Verificar que las columnas existen
        missing_cols = []
        if col_soiled not in df_pvstand.columns:
            missing_cols.append(col_soiled)
        if col_ref not in df_pvstand.columns:
            missing_cols.append(col_ref)
        if temp_soiled_col not in df_pvstand.columns:
            missing_cols.append(temp_soiled_col)
        if temp_ref_col not in df_pvstand.columns:
            missing_cols.append(temp_ref_col)
        
        if missing_cols:
            logger.error(f"Columnas faltantes en DataFrame: {missing_cols}")
            logger.error(f"Columnas disponibles: {df_pvstand.columns.tolist()}")
            return None
        
        # Extraer series
        if sr_type == 'Isc':
            val_soiled = pd.to_numeric(df_pvstand[col_soiled], errors='coerce')
            val_ref = pd.to_numeric(df_pvstand[col_ref], errors='coerce')
        else:
            val_soiled = pd.to_numeric(df_pvstand[col_soiled], errors='coerce')
            val_ref = pd.to_numeric(df_pvstand[col_ref], errors='coerce')
        
        T_s = pd.to_numeric(df_pvstand[temp_soiled_col], errors='coerce')
        T_r = pd.to_numeric(df_pvstand[temp_ref_col], errors='coerce')
        
        # Alinear
        df_aligned = pd.DataFrame({
            'val_s': val_soiled,
            'val_r': val_ref,
            'T_s': T_s,
            'T_r': T_r
        })
        df_aligned = df_aligned.dropna(how='all')
        df_aligned = df_aligned[~df_aligned.index.duplicated(keep='first')]
        df_aligned = df_aligned.sort_index()
        
        logger.info(f"Datos alineados: {len(df_aligned)} puntos")
        
        # Calcular propagación
        if sr_type == 'Isc':
            result = propagate_sr_isc_minute(
                df_aligned['val_s'],
                df_aligned['val_r'],
                df_aligned['T_s'],
                df_aligned['T_r'],
                use_temp_correction=use_temp_correction
            )
        else:
            result = propagate_sr_pmax_minute(
                df_aligned['val_s'],
                df_aligned['val_r'],
                df_aligned['T_s'],
                df_aligned['T_r'],
                use_temp_correction=use_temp_correction
            )
        
        # Agregar columnas originales
        result[f'{sr_type}_soiled'] = df_aligned['val_s']
        result[f'{sr_type}_ref'] = df_aligned['val_r']
        result['T_soiled'] = df_aligned['T_s']
        result['T_ref'] = df_aligned['T_r']
        
        # Filtrar SR extremos
        mask_sr_reasonable = (result['SR'] >= 0) & (result['SR'] <= 200)
        n_sr_extreme = (~mask_sr_reasonable).sum()
        if n_sr_extreme > 0:
            logger.warning(f"Descartando {n_sr_extreme} minutos con SR extremo")
            result.loc[~mask_sr_reasonable, ['SR', 'u_SR_k1_rel', 'U_SR_k2_rel']] = np.nan
        
        # Reordenar
        result = result[[f'{sr_type}_soiled', f'{sr_type}_ref', 'T_soiled', 'T_ref', 'SR', 'u_SR_k1_rel', 'U_SR_k2_rel']]
        
        mask_final_valid = (
            result['SR'].notna() &
            (result['SR'] >= 0) &
            (result['SR'] <= 200) &
            result['U_SR_k2_rel'].notna() &
            np.isfinite(result['U_SR_k2_rel'])
        )
        n_valid_final = mask_final_valid.sum()
        logger.info(f"Incertidumbre calculada para {n_valid_final} minutos válidos")
        
        return result
        
    except Exception as e:
        logger.error(f"Error procesando incertidumbre de campaña: {e}", exc_info=True)
        return None


def calculate_campaign_uncertainty(df_sr_uncertainty: pd.DataFrame) -> Tuple[float, float]:
    """Calcula incertidumbre de campaña."""
    valid_data = df_sr_uncertainty['U_SR_k2_rel'].dropna()
    valid_data = valid_data[np.isfinite(valid_data)]
    valid_data = valid_data[(valid_data >= 0) & (valid_data <= 0.5)]
    
    if len(valid_data) == 0:
        logger.warning("No hay datos válidos para calcular incertidumbre de campaña")
        return np.nan, np.nan
    
    U_campaign_k2_rel = valid_data.mean() * 100
    u_campaign_k1_rel = U_campaign_k2_rel / K_EXPAND
    
    logger.info(f"Incertidumbre de campaña (k=1): {u_campaign_k1_rel:.3f}%")
    logger.info(f"Incertidumbre de campaña (k=2): {U_campaign_k2_rel:.3f}%")
    
    return u_campaign_k1_rel, U_campaign_k2_rel


def aggregate_with_uncertainty(
    sr_series: pd.Series,
    U_campaign_k2_rel: float,
    freq: str = 'D',
    quantile: float = 0.25,
    df_sr_uncertainty: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Agrega serie de SR y asigna incertidumbre."""
    if not np.isfinite(U_campaign_k2_rel) or U_campaign_k2_rel <= 0:
        logger.warning(f"U_campaign_k2_rel no es válido ({U_campaign_k2_rel}), usando valor por defecto 3.7%")
        U_campaign_k2_rel = 3.7
    
    sr_series_filtered = sr_series[(sr_series >= 0) & (sr_series <= 200)]
    
    if len(sr_series_filtered) == 0:
        logger.warning(f"No hay datos válidos de SR para agregar con frecuencia {freq}")
        return pd.DataFrame()
    
    try:
        sr_agg = sr_series_filtered.resample(freq, label='right', closed='right').quantile(quantile)
        n_minutes = sr_series_filtered.resample(freq, label='right', closed='right').count()
    except Exception as e:
        logger.warning(f"Error en resample: {e}. Convirtiendo a UTC...")
        sr_series_utc = sr_series_filtered.copy()
        if sr_series_utc.index.tz is not None:
            sr_series_utc.index = sr_series_utc.index.tz_convert('UTC')
        sr_agg = sr_series_utc.resample(freq, label='right', closed='right').quantile(quantile)
        n_minutes = sr_series_utc.resample(freq, label='right', closed='right').count()
    
    result = pd.DataFrame(index=sr_agg.index)
    result['SR_agg'] = sr_agg
    result['n_minutes'] = n_minutes.reindex(sr_agg.index, fill_value=0)
    
    if df_sr_uncertainty is not None and 'U_SR_k2_rel' in df_sr_uncertainty.columns:
        logger.info(f"Calculando incertidumbre LOCAL para cada agregado {freq}...")
        df_u_valid = df_sr_uncertainty[
            df_sr_uncertainty['U_SR_k2_rel'].notna() &
            np.isfinite(df_sr_uncertainty['U_SR_k2_rel']) &
            (df_sr_uncertainty['U_SR_k2_rel'] >= 0) &
            (df_sr_uncertainty['U_SR_k2_rel'] <= 0.5)
        ].copy()
        
        if len(df_u_valid) > 0:
            try:
                u_agg = df_u_valid['U_SR_k2_rel'].resample(freq, label='right', closed='right').mean()
                u_agg = u_agg * 100
                result['U_rel_k2'] = u_agg.reindex(sr_agg.index)
                
                mask_missing = result['U_rel_k2'].isna()
                if mask_missing.any():
                    result.loc[mask_missing, 'U_rel_k2'] = U_campaign_k2_rel
            except Exception as e:
                logger.warning(f"Error calculando incertidumbre local: {e}. Usando incertidumbre global.")
                result['U_rel_k2'] = U_campaign_k2_rel
        else:
            result['U_rel_k2'] = U_campaign_k2_rel
    else:
        result['U_rel_k2'] = U_campaign_k2_rel
    
    U_rel_k2_frac = result['U_rel_k2'] / 100.0
    result['CI95_lo'] = result['SR_agg'] * (1 - U_rel_k2_frac)
    result['CI95_hi'] = result['SR_agg'] * (1 + U_rel_k2_frac)
    
    result = result[result['SR_agg'].notna()]
    
    return result


def run_uncertainty_propagation_analysis(
    df_pvstand: pd.DataFrame,
    sr_type: str = 'Isc',  # 'Isc' o 'Pmax'
    use_temp_correction: bool = True
) -> bool:
    """Función principal para ejecutar análisis de propagación de incertidumbre."""
    try:
        logger.info("="*80)
        logger.info(f"INICIANDO ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE DE SR (PVSTAND - {sr_type})")
        logger.info("="*80)
        
        os.makedirs(paths.PROPAGACION_ERRORES_PVSTAND_DIR, exist_ok=True)
        
        df_sr_uncertainty = process_campaign_uncertainty(
            df_pvstand,
            sr_type=sr_type,
            use_temp_correction=use_temp_correction
        )
        
        if df_sr_uncertainty is None or df_sr_uncertainty.empty:
            logger.error("No se pudieron calcular incertidumbres minuto a minuto")
            return False
        
        # Guardar resultados minutales
        output_minute_file = os.path.join(
            paths.PROPAGACION_ERRORES_PVSTAND_DIR,
            f"sr_{sr_type.lower()}_minute_with_uncertainty.csv"
        )
        df_sr_uncertainty.to_csv(output_minute_file)
        logger.info(f"✅ Resultados minutales guardados en: {output_minute_file}")
        
        u_campaign_k1_rel, U_campaign_k2_rel = calculate_campaign_uncertainty(df_sr_uncertainty)
        
        if np.isnan(u_campaign_k1_rel) or np.isnan(U_campaign_k2_rel):
            logger.error("No se pudo calcular incertidumbre de campaña")
            return False
        
        # Agregaciones
        sr_series = df_sr_uncertainty['SR'].dropna()
        sr_series = sr_series[np.isfinite(sr_series)]
        sr_series = sr_series[(sr_series >= 0) & (sr_series <= 200)]
        
        if len(sr_series) == 0:
            logger.warning("No hay datos válidos de SR para agregar")
            return False
        
        if sr_series.index.tz is None:
            sr_series.index = sr_series.index.tz_localize('UTC')
        else:
            sr_series.index = sr_series.index.tz_convert('UTC')
        
        df_sr_uncertainty_utc = df_sr_uncertainty.copy()
        if df_sr_uncertainty_utc.index.tz is None:
            df_sr_uncertainty_utc.index = df_sr_uncertainty_utc.index.tz_localize('UTC')
        else:
            df_sr_uncertainty_utc.index = df_sr_uncertainty_utc.index.tz_convert('UTC')
        
        # Agregaciones diarias, semanales y mensuales
        for freq, suffix in [('D', 'daily'), ('W-SUN', 'weekly'), ('ME', 'monthly')]:
            df_agg = aggregate_with_uncertainty(
                sr_series, U_campaign_k2_rel, freq=freq, quantile=0.25,
                df_sr_uncertainty=df_sr_uncertainty_utc
            )
            if not df_agg.empty:
                output_file = os.path.join(
                    paths.PROPAGACION_ERRORES_PVSTAND_DIR,
                    f"sr_{sr_type.lower()}_{suffix}_abs_with_U.csv"
                )
                df_agg.to_csv(output_file)
                logger.info(f"✅ Resultados {suffix} guardados en: {output_file}")
        
        logger.info("="*80)
        logger.info(f"✅ ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE COMPLETADO (PVSTAND - {sr_type})")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en análisis de propagación de incertidumbre: {e}", exc_info=True)
        return False

