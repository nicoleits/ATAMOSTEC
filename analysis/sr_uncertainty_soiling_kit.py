"""
Análisis de Incertidumbre del Soiling Ratio (SR) para Soiling Kit mediante Propagación de Errores (GUM)

Este módulo calcula la incertidumbre del SR usando propagación de errores según GUM:
- Incertidumbres de medición de corriente (Isc) y temperatura (T)
- Propagación minuto a minuto usando derivadas parciales
- Considera corrección de temperatura: Isc_corr = Isc × (1 + α_isc × (T - T_ref))
- Agregación a escalas diarias, semanales y mensuales

DESCRIPCIÓN:
------------
El SR se calcula como: SR = 100 * Isc(p) / Isc(e)
donde Isc(p) = corriente protegida (referencia), Isc(e) = corriente expuesta (sucio)

Con corrección de temperatura:
- Isc_corr = Isc × (1 + α_isc × (T_ref - T))
- SR_corr = 100 * Isc_Ref_Corrected / Isc_Soiled_Corrected

La incertidumbre se propaga usando derivadas parciales según GUM.
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

# Incertidumbres del amperímetro/multímetro (k=2) - VALORES ESTIMADOS
# TODO: Reemplazar con valores reales del certificado de calibración
U_ISC_ADD_K2 = 0.01  # A (aditiva, k=2) - Valor estimado, ajustar según especificaciones
U_ISC_SCALE_K2 = 0.01  # 1% (de escala, k=2) - Valor estimado

# Convertir a k=1 (1σ)
U_ISC_ADD = U_ISC_ADD_K2 / 2.0  # A
U_ISC_SCALE = U_ISC_SCALE_K2 / 2.0  # 0.005 (0.5%)

# Incertidumbres del sensor de temperatura (k=2) - VALORES ESTIMADOS
# TODO: Reemplazar con valores reales del certificado de calibración
U_TEMP_ADD_K2 = 1.0  # °C (aditiva, k=2) - Valor estimado
U_TEMP_SCALE_K2 = 0.0  # Sin componente de escala para temperatura

# Convertir a k=1 (1σ)
U_TEMP_ADD = U_TEMP_ADD_K2 / 2.0  # °C

# Incertidumbre del coeficiente de temperatura α_isc (k=1)
# TODO: Reemplazar con valor real del datasheet o calibración
U_ALPHA_ISC = 0.0001  # 1/°C (k=1) - Valor estimado (10% de α_isc = 0.0004)

# Factor de cobertura para expandir al final
K_EXPAND = 2.0

# Columnas de datos (desde settings)
ISC_SOILED_COL = settings.SOILING_KIT_ISC_SOILED_COL  # 'Isc(e)'
ISC_REF_COL = settings.SOILING_KIT_ISC_REF_COL  # 'Isc(p)'
TEMP_SOILED_COL = settings.SOILING_KIT_TEMP_SOILED_COL  # 'Te(C)'
TEMP_REF_COL = settings.SOILING_KIT_TEMP_REF_COL  # 'Tp(C)'

# Coeficiente de temperatura
ALPHA_ISC = settings.SOILING_KIT_ALPHA_ISC_CORR  # 0.0004
TEMP_REF = settings.SOILING_KIT_TEMP_REF_C  # 25.0

# Umbral mínimo de corriente (para filtrar datos inválidos)
MIN_ISC_THRESHOLD = 0.1  # A


def isc_u(I: pd.Series, u_add: float, u_scale: float) -> pd.Series:
    """
    Calcula la incertidumbre u(Isc) usando el modelo combinado aditivo + escala.
    
    Fórmula: u(Isc)² = u_add² + (u_scale * Isc)²
    
    Args:
        I: Serie temporal de corriente (A)
        u_add: Incertidumbre aditiva (A, k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(Isc) en A (k=1)
    """
    u_squared = u_add**2 + (u_scale * I)**2
    u_I = np.sqrt(u_squared)
    return u_I


def temp_u(T: pd.Series, u_add: float) -> pd.Series:
    """
    Calcula la incertidumbre u(T) usando modelo aditivo.
    
    Fórmula: u(T) = u_add
    
    Args:
        T: Serie temporal de temperatura (°C)
        u_add: Incertidumbre aditiva (°C, k=1)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(T) en °C (k=1)
    """
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
    Calcula la incertidumbre de Isc corregida por temperatura.
    
    Isc_corr = Isc × (1 + α × (T_ref - T))
    
    Derivadas parciales:
    - ∂Isc_corr/∂Isc = 1 + α × (T_ref - T)
    - ∂Isc_corr/∂T = -α × Isc
    - ∂Isc_corr/∂α = Isc × (T_ref - T)
    
    Var(Isc_corr) = (∂Isc_corr/∂Isc)² × u(Isc)² + 
                    (∂Isc_corr/∂T)² × u(T)² + 
                    (∂Isc_corr/∂α)² × u(α)²
    
    Args:
        Isc: Serie temporal de corriente (A)
        T: Serie temporal de temperatura (°C)
        u_Isc: Incertidumbre de corriente (A, k=1)
        u_T: Incertidumbre de temperatura (°C, k=1)
        alpha: Coeficiente de temperatura (1/°C)
        u_alpha: Incertidumbre de alpha (1/°C, k=1)
        T_ref: Temperatura de referencia (°C)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(Isc_corr) en A (k=1)
    """
    # Derivadas parciales
    dIsc_corr_dIsc = 1 + alpha * (T_ref - T)
    dIsc_corr_dT = -alpha * Isc
    dIsc_corr_dalpha = Isc * (T_ref - T)
    
    # Varianza propagada
    var_Isc_corr = (
        (dIsc_corr_dIsc**2) * (u_Isc**2) +
        (dIsc_corr_dT**2) * (u_T**2) +
        (dIsc_corr_dalpha**2) * (u_alpha**2)
    )
    
    u_Isc_corr = np.sqrt(var_Isc_corr)
    return u_Isc_corr


def propagate_sr_minute(
    Isc_e: pd.Series,
    Isc_p: pd.Series,
    T_e: pd.Series,
    T_p: pd.Series,
    use_temp_correction: bool = True,
    u_isc_add: float = U_ISC_ADD,
    u_isc_scale: float = U_ISC_SCALE,
    u_temp_add: float = U_TEMP_ADD,
    u_alpha: float = U_ALPHA_ISC,
    alpha: float = ALPHA_ISC,
    T_ref: float = TEMP_REF,
    k_expand: float = K_EXPAND,
    rho_isc: float = 0.0,
    rho_temp: float = 0.0
) -> pd.DataFrame:
    """
    Calcula SR y su incertidumbre minuto a minuto usando propagación de errores (GUM).
    
    Args:
        Isc_e: Serie temporal de corriente expuesta (sucio) (A)
        Isc_p: Serie temporal de corriente protegida (referencia) (A)
        T_e: Serie temporal de temperatura expuesta (°C)
        T_p: Serie temporal de temperatura protegida (°C)
        use_temp_correction: Si True, usa corrección de temperatura
        u_isc_add: Incertidumbre aditiva de corriente (A, k=1)
        u_isc_scale: Incertidumbre de escala de corriente (adimensional, k=1)
        u_temp_add: Incertidumbre aditiva de temperatura (°C, k=1)
        u_alpha: Incertidumbre de coeficiente de temperatura (1/°C, k=1)
        alpha: Coeficiente de temperatura (1/°C)
        T_ref: Temperatura de referencia (°C)
        k_expand: Factor de cobertura para expandir (default 2.0)
        rho_isc: Coeficiente de correlación entre Isc_e e Isc_p (default 0.0)
        rho_temp: Coeficiente de correlación entre T_e y T_p (default 0.0)
    
    Returns:
        pd.DataFrame con columnas:
        - SR: Soiling Ratio (%)
        - u_SR_k1_abs: Incertidumbre absoluta k=1 (%)
        - u_SR_k1_rel: Incertidumbre relativa k=1 (adimensional)
        - U_SR_k2_abs: Incertidumbre absoluta expandida k=2 (%)
        - U_SR_k2_rel: Incertidumbre relativa expandida k=2 (adimensional)
    """
    # Asegurar que todas las series tienen el mismo índice
    common_index = Isc_e.index.intersection(Isc_p.index).intersection(T_e.index).intersection(T_p.index)
    Isc_e_aligned = Isc_e.reindex(common_index)
    Isc_p_aligned = Isc_p.reindex(common_index)
    T_e_aligned = T_e.reindex(common_index)
    T_p_aligned = T_p.reindex(common_index)
    
    # Inicializar DataFrame de resultados
    result = pd.DataFrame(index=common_index)
    
    # Máscara de datos válidos
    mask_valid = (
        (Isc_e_aligned > MIN_ISC_THRESHOLD) &
        (Isc_p_aligned > MIN_ISC_THRESHOLD) &
        Isc_e_aligned.notna() &
        Isc_p_aligned.notna() &
        T_e_aligned.notna() &
        T_p_aligned.notna()
    )
    
    if use_temp_correction:
        # Calcular Isc corregidas por temperatura
        Isc_e_corr = Isc_e_aligned * (1 + alpha * (T_ref - T_e_aligned))
        Isc_p_corr = Isc_p_aligned * (1 + alpha * (T_ref - T_p_aligned))
        
        # Calcular SR corregido
        result['SR'] = np.nan
        result.loc[mask_valid, 'SR'] = 100.0 * Isc_p_corr[mask_valid] / Isc_e_corr[mask_valid]
        
        # Calcular incertidumbres de Isc
        u_Isc_e = isc_u(Isc_e_aligned, u_isc_add, u_isc_scale)
        u_Isc_p = isc_u(Isc_p_aligned, u_isc_add, u_isc_scale)
        
        # Calcular incertidumbres de temperatura
        u_T_e = temp_u(T_e_aligned, u_temp_add)
        u_T_p = temp_u(T_p_aligned, u_temp_add)
        
        # Calcular incertidumbres de Isc corregidas
        u_Isc_e_corr = isc_corrected_u(Isc_e_aligned, T_e_aligned, u_Isc_e, u_T_e, alpha, u_alpha, T_ref)
        u_Isc_p_corr = isc_corrected_u(Isc_p_aligned, T_p_aligned, u_Isc_p, u_T_p, alpha, u_alpha, T_ref)
        
        # Derivadas parciales de SR respecto a Isc corregidas
        dSR_dIsc_p_corr = np.nan * np.ones_like(Isc_e_aligned)
        dSR_dIsc_e_corr = np.nan * np.ones_like(Isc_e_aligned)
        dSR_dIsc_p_corr[mask_valid] = 100.0 / Isc_e_corr[mask_valid]
        dSR_dIsc_e_corr[mask_valid] = -100.0 * Isc_p_corr[mask_valid] / (Isc_e_corr[mask_valid]**2)
        
        # Covarianza entre Isc corregidas (asumiendo independencia entre módulos)
        cov_Isc_corr = pd.Series(0.0, index=common_index) if rho_isc == 0.0 else rho_isc * u_Isc_p_corr * u_Isc_e_corr
        
        # Varianza propagada
        var_SR = (
            (dSR_dIsc_p_corr**2) * (u_Isc_p_corr**2) +
            (dSR_dIsc_e_corr**2) * (u_Isc_e_corr**2) +
            2 * dSR_dIsc_p_corr * dSR_dIsc_e_corr * cov_Isc_corr
        )
        
    else:
        # Sin corrección de temperatura
        result['SR'] = np.nan
        result.loc[mask_valid, 'SR'] = 100.0 * Isc_p_aligned[mask_valid] / Isc_e_aligned[mask_valid]
        
        # Calcular incertidumbres de Isc
        u_Isc_e = isc_u(Isc_e_aligned, u_isc_add, u_isc_scale)
        u_Isc_p = isc_u(Isc_p_aligned, u_isc_add, u_isc_scale)
        
        # Derivadas parciales
        dSR_dIsc_p = np.nan * np.ones_like(Isc_e_aligned)
        dSR_dIsc_e = np.nan * np.ones_like(Isc_e_aligned)
        dSR_dIsc_p[mask_valid] = 100.0 / Isc_e_aligned[mask_valid]
        dSR_dIsc_e[mask_valid] = -100.0 * Isc_p_aligned[mask_valid] / (Isc_e_aligned[mask_valid]**2)
        
        # Covarianza
        cov_Isc = pd.Series(0.0, index=common_index) if rho_isc == 0.0 else rho_isc * u_Isc_p * u_Isc_e
        
        # Varianza propagada
        var_SR = (
            (dSR_dIsc_p**2) * (u_Isc_p**2) +
            (dSR_dIsc_e**2) * (u_Isc_e**2) +
            2 * dSR_dIsc_p * dSR_dIsc_e * cov_Isc
        )
    
    # Incertidumbre estándar (k=1)
    u_SR_k1_abs = np.sqrt(var_SR)
    
    # Incertidumbre relativa k=1
    result['u_SR_k1_abs'] = u_SR_k1_abs
    mask_sr_valid = (result['SR'] > 0) & (result['SR'] < 200)
    result['u_SR_k1_rel'] = np.nan
    result.loc[mask_sr_valid, 'u_SR_k1_rel'] = u_SR_k1_abs[mask_sr_valid] / result.loc[mask_sr_valid, 'SR']
    
    # Incertidumbre expandida k=2
    result['U_SR_k2_abs'] = k_expand * u_SR_k1_abs
    result['U_SR_k2_rel'] = k_expand * result['u_SR_k1_rel']
    
    return result


def process_campaign_uncertainty(
    df_soiling_kit: pd.DataFrame,
    use_temp_correction: bool = True
) -> Optional[pd.DataFrame]:
    """
    Procesa la incertidumbre minuto a minuto sobre toda la campaña.
    
    Args:
        df_soiling_kit: DataFrame con datos de Soiling Kit
                       Debe tener índice DatetimeIndex y columnas: Isc(e), Isc(p), Te(C), Tp(C)
        use_temp_correction: Si True, usa SR con corrección de temperatura
    
    Returns:
        pd.DataFrame con columnas: timestamp, Isc_e, Isc_p, T_e, T_p, SR, u_SR_k1_rel, U_SR_k2_rel
        o None si hay error
    """
    try:
        logger.info("Procesando incertidumbre de campaña minuto a minuto (Soiling Kit)...")
        
        # Validar columnas
        required_cols = [ISC_SOILED_COL, ISC_REF_COL, TEMP_SOILED_COL, TEMP_REF_COL]
        missing_cols = [col for col in required_cols if col not in df_soiling_kit.columns]
        if missing_cols:
            logger.error(f"Columnas faltantes: {missing_cols}")
            return None
        
        # Extraer series
        Isc_e = pd.to_numeric(df_soiling_kit[ISC_SOILED_COL], errors='coerce')
        Isc_p = pd.to_numeric(df_soiling_kit[ISC_REF_COL], errors='coerce')
        T_e = pd.to_numeric(df_soiling_kit[TEMP_SOILED_COL], errors='coerce')
        T_p = pd.to_numeric(df_soiling_kit[TEMP_REF_COL], errors='coerce')
        
        # Alinear y limpiar
        df_aligned = pd.DataFrame({
            'Isc_e': Isc_e,
            'Isc_p': Isc_p,
            'T_e': T_e,
            'T_p': T_p
        })
        df_aligned = df_aligned.dropna(how='all')
        df_aligned = df_aligned[~df_aligned.index.duplicated(keep='first')]
        df_aligned = df_aligned.sort_index()
        
        logger.info(f"Datos alineados: {len(df_aligned)} puntos")
        
        # Calcular propagación de incertidumbre
        result = propagate_sr_minute(
            df_aligned['Isc_e'],
            df_aligned['Isc_p'],
            df_aligned['T_e'],
            df_aligned['T_p'],
            use_temp_correction=use_temp_correction,
            u_isc_add=U_ISC_ADD,
            u_isc_scale=U_ISC_SCALE,
            u_temp_add=U_TEMP_ADD,
            u_alpha=U_ALPHA_ISC,
            alpha=ALPHA_ISC,
            T_ref=TEMP_REF,
            k_expand=K_EXPAND,
            rho_isc=0.0,  # Independencia instrumental
            rho_temp=0.0
        )
        
        # Agregar columnas originales
        result['Isc_e'] = df_aligned['Isc_e']
        result['Isc_p'] = df_aligned['Isc_p']
        result['T_e'] = df_aligned['T_e']
        result['T_p'] = df_aligned['T_p']
        
        # Filtrar SR extremos
        mask_sr_reasonable = (result['SR'] >= 0) & (result['SR'] <= 200)
        n_sr_extreme = (~mask_sr_reasonable).sum()
        if n_sr_extreme > 0:
            logger.warning(f"Descartando {n_sr_extreme} minutos con SR extremo (fuera de [0%, 200%])")
            result.loc[~mask_sr_reasonable, ['SR', 'u_SR_k1_rel', 'U_SR_k2_rel']] = np.nan
        
        # Reordenar columnas
        result = result[['Isc_e', 'Isc_p', 'T_e', 'T_p', 'SR', 'u_SR_k1_rel', 'U_SR_k2_rel']]
        
        # Contar minutos válidos
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
    """
    Calcula la incertidumbre de campaña como promedio temporal de U_SR_k2_rel.
    
    Args:
        df_sr_uncertainty: DataFrame con resultados de process_campaign_uncertainty
    
    Returns:
        tuple: (u_campaign_k1_rel, U_campaign_k2_rel) en porcentaje
    """
    valid_data = df_sr_uncertainty['U_SR_k2_rel'].dropna()
    valid_data = valid_data[np.isfinite(valid_data)]
    valid_data = valid_data[(valid_data >= 0) & (valid_data <= 0.5)]  # 0% a 50%
    
    if len(valid_data) == 0:
        logger.warning("No hay datos válidos para calcular incertidumbre de campaña")
        return np.nan, np.nan
    
    U_campaign_k2_rel = valid_data.mean() * 100  # Convertir a porcentaje
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
    """
    Agrega serie de SR y asigna incertidumbre a cada agregado.
    
    Similar a la función en sr_uncertainty_propagation.py pero adaptada para Soiling Kit.
    """
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
    
    # Calcular incertidumbre local si se proporciona df_sr_uncertainty
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
    
    # Intervalos de confianza al 95%
    U_rel_k2_frac = result['U_rel_k2'] / 100.0
    result['CI95_lo'] = result['SR_agg'] * (1 - U_rel_k2_frac)
    result['CI95_hi'] = result['SR_agg'] * (1 + U_rel_k2_frac)
    
    result = result[result['SR_agg'].notna()]
    
    return result


def run_uncertainty_propagation_analysis(
    df_soiling_kit: pd.DataFrame,
    use_temp_correction: bool = True
) -> bool:
    """
    Función principal para ejecutar el análisis completo de propagación de incertidumbre.
    
    Args:
        df_soiling_kit: DataFrame con datos de Soiling Kit
        use_temp_correction: Si True, usa SR con corrección de temperatura
    
    Returns:
        bool: True si el análisis fue exitoso, False en caso contrario
    """
    try:
        logger.info("="*80)
        logger.info("INICIANDO ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE DE SR (SOILING KIT)")
        logger.info("="*80)
        
        # Crear directorio de salida
        os.makedirs(paths.PROPAGACION_ERRORES_SOILING_KIT_DIR, exist_ok=True)
        
        # Procesar incertidumbre minuto a minuto
        df_sr_uncertainty = process_campaign_uncertainty(
            df_soiling_kit,
            use_temp_correction=use_temp_correction
        )
        
        if df_sr_uncertainty is None or df_sr_uncertainty.empty:
            logger.error("No se pudieron calcular incertidumbres minuto a minuto")
            return False
        
        # Guardar resultados minutales
        output_minute_file = paths.SOILING_KIT_SR_MINUTE_WITH_UNCERTAINTY_FILE
        df_sr_uncertainty.to_csv(output_minute_file)
        logger.info(f"✅ Resultados minutales guardados en: {output_minute_file}")
        
        # Calcular incertidumbre de campaña
        u_campaign_k1_rel, U_campaign_k2_rel = calculate_campaign_uncertainty(df_sr_uncertainty)
        
        if np.isnan(u_campaign_k1_rel) or np.isnan(U_campaign_k2_rel):
            logger.error("No se pudo calcular incertidumbre de campaña")
            return False
        
        # Agregar a escalas diarias, semanales y mensuales
        sr_series = df_sr_uncertainty['SR'].dropna()
        sr_series = sr_series[np.isfinite(sr_series)]
        sr_series = sr_series[(sr_series >= 0) & (sr_series <= 200)]
        
        if len(sr_series) == 0:
            logger.warning("No hay datos válidos de SR para agregar")
            return False
        
        # Asegurar timezone UTC
        if sr_series.index.tz is None:
            sr_series.index = sr_series.index.tz_localize('UTC')
        else:
            sr_series.index = sr_series.index.tz_convert('UTC')
        
        df_sr_uncertainty_utc = df_sr_uncertainty.copy()
        if df_sr_uncertainty_utc.index.tz is None:
            df_sr_uncertainty_utc.index = df_sr_uncertainty_utc.index.tz_localize('UTC')
        else:
            df_sr_uncertainty_utc.index = df_sr_uncertainty_utc.index.tz_convert('UTC')
        
        # Agregación diaria (Q25)
        df_daily = aggregate_with_uncertainty(
            sr_series, U_campaign_k2_rel, freq='D', quantile=0.25,
            df_sr_uncertainty=df_sr_uncertainty_utc
        )
        if not df_daily.empty:
            output_daily_file = paths.SOILING_KIT_SR_DAILY_ABS_WITH_U_FILE
            df_daily.to_csv(output_daily_file)
            logger.info(f"✅ Resultados diarios guardados en: {output_daily_file}")
        
        # Agregación semanal (Q25)
        df_weekly = aggregate_with_uncertainty(
            sr_series, U_campaign_k2_rel, freq='W-SUN', quantile=0.25,
            df_sr_uncertainty=df_sr_uncertainty_utc
        )
        if not df_weekly.empty:
            output_weekly_file = paths.SOILING_KIT_SR_WEEKLY_ABS_WITH_U_FILE
            df_weekly.to_csv(output_weekly_file)
            logger.info(f"✅ Resultados semanales guardados en: {output_weekly_file}")
        
        # Agregación mensual (Q25)
        df_monthly = aggregate_with_uncertainty(
            sr_series, U_campaign_k2_rel, freq='M', quantile=0.25,
            df_sr_uncertainty=df_sr_uncertainty_utc
        )
        if not df_monthly.empty:
            output_monthly_file = paths.SOILING_KIT_SR_MONTHLY_ABS_WITH_U_FILE
            df_monthly.to_csv(output_monthly_file)
            logger.info(f"✅ Resultados mensuales guardados en: {output_monthly_file}")
        
        logger.info("="*80)
        logger.info("✅ ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE COMPLETADO (SOILING KIT)")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en análisis de propagación de incertidumbre: {e}", exc_info=True)
        return False

