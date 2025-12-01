"""
Análisis de Incertidumbre del Soiling Ratio (SR) mediante Propagación de Errores (GUM)

Este módulo calcula la incertidumbre del SR usando propagación de errores según GUM:
- Incertidumbres del fabricante (Si-V-10TC-T): aditiva y de escala
- Propagación minuto a minuto usando derivadas parciales
- Agregación a escalas diarias, semanales y mensuales

DESCRIPCIÓN:
------------
El SR se calcula como: SR = 100 * S / C
donde S = irradiancia celda sucia (1RC411), C = irradiancia celda limpia (1RC412)

La incertidumbre se propaga usando:
- u(S)² = u_add² + (u_scale * S)²
- u(C)² = u_add² + (u_scale * C)²
- Var(SR) = (∂SR/∂S)² * u(S)² + (∂SR/∂C)² * u(C)² + 2 * (∂SR/∂S) * (∂SR/∂C) * Cov(S,C)
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Tuple
from datetime import datetime
import pytz
import config.settings as settings
import config.paths as paths

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE INCERTIDUMBRES DEL FABRICANTE
# ============================================================================

# Incertidumbres del fabricante (Si-V-10TC-T) a k=2
U_ADD_K2 = 5.0  # W/m² (aditiva, k=2)
U_SCALE_K2 = 0.025  # 2.5% (de escala, k=2)

# Convertir a k=1 (1σ)
U_ADD = U_ADD_K2 / 2.0  # 2.5 W/m²
U_SCALE = U_SCALE_K2 / 2.0  # 0.0125 (1.25%)

# Factor de cobertura para expandir al final
K_EXPAND = 2.0

# Columnas de datos
SOILED_COL = "1RC411(w.m-2)"  # Celda sucia
CLEAN_COL = "1RC412(w.m-2)"   # Celda limpia

# Umbral mínimo de irradiancia (noche)
MIN_IRRADIANCE_THRESHOLD = 10.0  # W/m²

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
    # Calcular u(I)² = u_add² + (u_scale * I)²
    u_squared = u_add**2 + (u_scale * I)**2
    u_I = np.sqrt(u_squared)
    return u_I


def propagate_sr_minute(
    S: pd.Series,
    C: pd.Series,
    u_add: float = U_ADD,
    u_scale: float = U_SCALE,
    k_expand: float = K_EXPAND,
    rho: float = 0.0
) -> pd.DataFrame:
    """
    Calcula SR y su incertidumbre minuto a minuto usando propagación de errores (GUM).
    
    Fórmulas:
    - SR = 100 * S / C
    - ∂SR/∂S = 100 / C
    - ∂SR/∂C = -100 * S / C²
    - Var(SR) = (∂SR/∂S)² * u(S)² + (∂SR/∂C)² * u(C)² + 2 * (∂SR/∂S) * (∂SR/∂C) * Cov(S,C)
    - Cov(S,C) = rho * u(S) * u(C) si rho != 0, sino 0
    
    Args:
        S: Serie temporal de irradiancia celda sucia (W/m²)
        C: Serie temporal de irradiancia celda limpia (W/m²)
        u_add: Incertidumbre aditiva (W/m², k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
        k_expand: Factor de cobertura para expandir (default 2.0)
        rho: Coeficiente de correlación entre canales (default 0.0 = independencia)
    
    Returns:
        pd.DataFrame con columnas:
        - SR: Soiling Ratio (%)
        - u_SR_k1_abs: Incertidumbre absoluta k=1 (%)
        - u_SR_k1_rel: Incertidumbre relativa k=1 (adimensional)
        - U_SR_k2_abs: Incertidumbre absoluta expandida k=2 (%)
        - U_SR_k2_rel: Incertidumbre relativa expandida k=2 (adimensional)
    """
    # Asegurar que S y C tienen el mismo índice
    common_index = S.index.intersection(C.index)
    S_aligned = S.reindex(common_index)
    C_aligned = C.reindex(common_index)
    
    # Inicializar DataFrame de resultados
    result = pd.DataFrame(index=common_index)
    
    # Calcular SR
    mask_valid = (C_aligned > 0) & S_aligned.notna() & C_aligned.notna()
    result['SR'] = np.nan
    result.loc[mask_valid, 'SR'] = 100.0 * S_aligned[mask_valid] / C_aligned[mask_valid]
    
    # Calcular incertidumbres de los canales
    u_S = channel_u(S_aligned, u_add, u_scale)
    u_C = channel_u(C_aligned, u_add, u_scale)
    
    # Derivadas parciales
    dSR_dS = np.nan * np.ones_like(C_aligned)
    dSR_dC = np.nan * np.ones_like(C_aligned)
    dSR_dS[mask_valid] = 100.0 / C_aligned[mask_valid]
    dSR_dC[mask_valid] = -100.0 * S_aligned[mask_valid] / (C_aligned[mask_valid]**2)
    
    # Covarianza
    if rho != 0.0:
        cov_SC = rho * u_S * u_C
    else:
        cov_SC = pd.Series(0.0, index=common_index)
    
    # Varianza propagada
    var_SR = (dSR_dS**2) * (u_S**2) + (dSR_dC**2) * (u_C**2) + 2 * dSR_dS * dSR_dC * cov_SC
    
    # Incertidumbre estándar (k=1)
    u_SR_k1_abs = np.sqrt(var_SR)
    
    # Incertidumbre relativa k=1 (evitar división por cero)
    result['u_SR_k1_abs'] = u_SR_k1_abs
    # Solo calcular incertidumbre relativa donde SR > 0 y es razonable
    mask_sr_valid = (result['SR'] > 0) & (result['SR'] < 1000)  # Filtrar SR extremos
    result['u_SR_k1_rel'] = np.nan
    result.loc[mask_sr_valid, 'u_SR_k1_rel'] = u_SR_k1_abs[mask_sr_valid] / result.loc[mask_sr_valid, 'SR']
    
    # Incertidumbre expandida k=2
    result['U_SR_k2_abs'] = k_expand * u_SR_k1_abs
    result['U_SR_k2_rel'] = k_expand * result['u_SR_k1_rel']
    
    return result


def process_campaign_uncertainty(
    df_ref_cells: pd.DataFrame,
    soiled_col: str = SOILED_COL,
    clean_col: str = CLEAN_COL
) -> Optional[pd.DataFrame]:
    """
    Procesa la incertidumbre minuto a minuto sobre toda la campaña.
    
    Args:
        df_ref_cells: DataFrame con datos de celdas de referencia
                     Debe tener índice DatetimeIndex
        soiled_col: Nombre de columna de celda sucia
        clean_col: Nombre de columna de celda limpia
    
    Returns:
        pd.DataFrame con columnas: timestamp, S, C, SR, u_SR_k1_rel, U_SR_k2_rel
        o None si hay error
    """
    try:
        logger.info("Procesando incertidumbre de campaña minuto a minuto...")
        
        # Validar columnas
        if soiled_col not in df_ref_cells.columns:
            logger.error(f"Columna de celda sucia '{soiled_col}' no encontrada")
            return None
        
        if clean_col not in df_ref_cells.columns:
            logger.error(f"Columna de celda limpia '{clean_col}' no encontrada")
            return None
        
        # Extraer series S y C
        S = df_ref_cells[soiled_col].copy()
        C = df_ref_cells[clean_col].copy()
        
        # Convertir a numérico si es necesario
        S = pd.to_numeric(S, errors='coerce')
        C = pd.to_numeric(C, errors='coerce')
        
        # Alinear por timestamp, eliminar duplicados, ordenar
        df_aligned = pd.DataFrame({'S': S, 'C': C})
        df_aligned = df_aligned.dropna(subset=['S', 'C'], how='all')
        df_aligned = df_aligned[~df_aligned.index.duplicated(keep='first')]
        df_aligned = df_aligned.sort_index()
        
        S_aligned = df_aligned['S']
        C_aligned = df_aligned['C']
        
        logger.info(f"Datos alineados: {len(S_aligned)} puntos")
        
        # Enmascarar casos edge:
        # - C <= 0 (división por cero)
        # - C < umbral mínimo (noche)
        # - NaN en S o C
        # - Valores negativos o saturados
        mask_valid = (
            (C_aligned > MIN_IRRADIANCE_THRESHOLD) & 
            S_aligned.notna() & 
            C_aligned.notna() &
            (S_aligned >= 0) &  # No valores negativos
            (C_aligned >= 0) &  # No valores negativos
            (S_aligned < 2000) &  # Umbral de saturación razonable
            (C_aligned < 2000)   # Umbral de saturación razonable
        )
        n_discarded = (~mask_valid).sum()
        pct_discarded = (n_discarded / len(S_aligned)) * 100 if len(S_aligned) > 0 else 0
        
        # Contar razones de descarte
        n_c_zero = (C_aligned <= 0).sum()
        n_c_low = ((C_aligned > 0) & (C_aligned <= MIN_IRRADIANCE_THRESHOLD)).sum()
        n_nan = (S_aligned.isna() | C_aligned.isna()).sum()
        n_negative = ((S_aligned < 0) | (C_aligned < 0)).sum()
        n_saturated = ((S_aligned >= 2000) | (C_aligned >= 2000)).sum()
        
        logger.info(f"Minutos descartados: {n_discarded} ({pct_discarded:.2f}%)")
        logger.info(f"  - C <= 0: {n_c_zero}")
        logger.info(f"  - C < {MIN_IRRADIANCE_THRESHOLD} W/m² (noche): {n_c_low}")
        logger.info(f"  - NaN: {n_nan}")
        logger.info(f"  - Valores negativos: {n_negative}")
        logger.info(f"  - Valores saturados (>= 2000 W/m²): {n_saturated}")
        logger.info(f"Minutos válidos: {mask_valid.sum()}")
        
        # Calcular propagación de incertidumbre
        result = propagate_sr_minute(
            S_aligned,
            C_aligned,
            u_add=U_ADD,
            u_scale=U_SCALE,
            k_expand=K_EXPAND,
            rho=0.0  # Independencia instrumental
        )
        
        # Agregar columnas S y C al resultado
        result['S'] = S_aligned
        result['C'] = C_aligned
        
        # Filtrar SR extremos (valores fuera de rango razonable)
        # SR debería estar entre 0% y 200% típicamente
        mask_sr_reasonable = (result['SR'] >= 0) & (result['SR'] <= 200)
        n_sr_extreme = (~mask_sr_reasonable).sum()
        if n_sr_extreme > 0:
            logger.warning(f"Descartando {n_sr_extreme} minutos con SR extremo (fuera de [0%, 200%])")
            result.loc[~mask_sr_reasonable, ['SR', 'u_SR_k1_rel', 'U_SR_k2_rel']] = np.nan
        
        # Reordenar columnas
        result = result[['S', 'C', 'SR', 'u_SR_k1_rel', 'U_SR_k2_rel']]
        
        # Contar minutos válidos (con SR razonable y incertidumbre finita)
        mask_final_valid = (
            result['SR'].notna() & 
            (result['SR'] >= 0) & 
            (result['SR'] <= 200) &
            result['U_SR_k2_rel'].notna() &
            np.isfinite(result['U_SR_k2_rel'])
        )
        n_valid_final = mask_final_valid.sum()
        logger.info(f"Incertidumbre calculada para {n_valid_final} minutos válidos (SR razonable y U finita)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error procesando incertidumbre de campaña: {e}", exc_info=True)
        return None


def calculate_campaign_uncertainty(df_sr_uncertainty: pd.DataFrame) -> Tuple[float, float]:
    """
    Calcula la incertidumbre de campaña como promedio temporal de U_SR_k2_rel.
    
    NOTA: Esta función calcula una incertidumbre global que puede no ser representativa
    si el SR varía significativamente con el tiempo. Se recomienda usar 
    calculate_uncertainty_by_period() o calculate_uncertainty_by_sr_range() en su lugar.
    
    Args:
        df_sr_uncertainty: DataFrame con resultados de process_campaign_uncertainty
    
    Returns:
        tuple: (u_campaign_k1_rel, U_campaign_k2_rel) en porcentaje
    """
    # Filtrar datos válidos: no NaN, finitos, y en rango razonable
    valid_data = df_sr_uncertainty['U_SR_k2_rel'].dropna()
    valid_data = valid_data[np.isfinite(valid_data)]
    # Filtrar valores extremos (típicamente U debería estar entre 0% y 50%)
    valid_data = valid_data[(valid_data >= 0) & (valid_data <= 0.5)]  # 0% a 50% en fracción
    
    if len(valid_data) == 0:
        logger.warning("No hay datos válidos para calcular incertidumbre de campaña")
        return np.nan, np.nan
    
    U_campaign_k2_rel = valid_data.mean() * 100  # Convertir a porcentaje
    u_campaign_k1_rel = U_campaign_k2_rel / K_EXPAND
    
    logger.info(f"Incertidumbre de campaña GLOBAL (k=1): {u_campaign_k1_rel:.3f}%")
    logger.info(f"Incertidumbre de campaña GLOBAL (k=2): {U_campaign_k2_rel:.3f}%")
    logger.info(f"Datos usados para cálculo: {len(valid_data)} de {len(df_sr_uncertainty)} minutos")
    logger.warning("⚠️  NOTA: Incertidumbre global puede no ser representativa si SR varía con el tiempo")
    
    return u_campaign_k1_rel, U_campaign_k2_rel


def calculate_uncertainty_by_period(
    df_sr_uncertainty: pd.DataFrame,
    freq: str = 'M'
) -> pd.DataFrame:
    """
    Calcula la incertidumbre por período temporal (mensual, trimestral, etc.).
    
    Esto es más representativo que una incertidumbre global cuando el SR varía
    con el tiempo de exposición.
    
    Args:
        df_sr_uncertainty: DataFrame con resultados de process_campaign_uncertainty
        freq: Frecuencia de agrupación ('M' para mensual, 'Q' para trimestral)
    
    Returns:
        pd.DataFrame con columnas:
        - periodo: Período temporal
        - U_k2_rel_mean: Incertidumbre promedio del período (k=2, %)
        - U_k2_rel_std: Desviación estándar de incertidumbre del período (%)
        - n_minutes: Número de minutos en el período
        - sr_mean: SR promedio del período (%)
    """
    logger.info(f"Calculando incertidumbre por período ({freq})...")
    
    # Filtrar datos válidos
    df_valid = df_sr_uncertainty[
        df_sr_uncertainty['U_SR_k2_rel'].notna() &
        np.isfinite(df_sr_uncertainty['U_SR_k2_rel']) &
        (df_sr_uncertainty['U_SR_k2_rel'] >= 0) &
        (df_sr_uncertainty['U_SR_k2_rel'] <= 0.5) &
        df_sr_uncertainty['SR'].notna() &
        (df_sr_uncertainty['SR'] >= 0) &
        (df_sr_uncertainty['SR'] <= 200)
    ].copy()
    
    if len(df_valid) == 0:
        logger.warning("No hay datos válidos para calcular incertidumbre por período")
        return pd.DataFrame()
    
    # Agrupar por período
    grouped = df_valid.groupby(pd.Grouper(freq=freq))
    
    results = []
    for periodo, group in grouped:
        if len(group) == 0:
            continue
        
        valid_u = group['U_SR_k2_rel'].dropna()
        valid_sr = group['SR'].dropna()
        
        if len(valid_u) > 0:
            U_k2_rel_mean = valid_u.mean() * 100  # Convertir a porcentaje
            U_k2_rel_std = valid_u.std() * 100
            sr_mean = valid_sr.mean() if len(valid_sr) > 0 else np.nan
            n_minutes = len(group)
            
            results.append({
                'periodo': periodo,
                'U_k2_rel_mean': U_k2_rel_mean,
                'U_k2_rel_std': U_k2_rel_std,
                'sr_mean': sr_mean,
                'n_minutes': n_minutes
            })
    
    if len(results) == 0:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results.set_index('periodo', inplace=True)
    
    logger.info(f"Incertidumbre calculada para {len(df_results)} períodos")
    logger.info(f"Rango de incertidumbre por período: {df_results['U_k2_rel_mean'].min():.3f}% a {df_results['U_k2_rel_mean'].max():.3f}%")
    
    return df_results


def calculate_uncertainty_by_sr_range(
    df_sr_uncertainty: pd.DataFrame,
    sr_bins: list = [0, 80, 90, 95, 100, 105, 200]
) -> pd.DataFrame:
    """
    Calcula la incertidumbre agrupada por rangos de SR.
    
    Esto es útil para entender cómo varía la incertidumbre con el nivel de soiling.
    
    Args:
        df_sr_uncertainty: DataFrame con resultados de process_campaign_uncertainty
        sr_bins: Límites de los rangos de SR (en %)
    
    Returns:
        pd.DataFrame con columnas:
        - sr_range: Rango de SR (ej: "80-90%")
        - U_k2_rel_mean: Incertidumbre promedio del rango (k=2, %)
        - U_k2_rel_std: Desviación estándar de incertidumbre del rango (%)
        - n_minutes: Número de minutos en el rango
    """
    logger.info("Calculando incertidumbre por rangos de SR...")
    
    # Filtrar datos válidos
    df_valid = df_sr_uncertainty[
        df_sr_uncertainty['U_SR_k2_rel'].notna() &
        np.isfinite(df_sr_uncertainty['U_SR_k2_rel']) &
        (df_sr_uncertainty['U_SR_k2_rel'] >= 0) &
        (df_sr_uncertainty['U_SR_k2_rel'] <= 0.5) &
        df_sr_uncertainty['SR'].notna() &
        (df_sr_uncertainty['SR'] >= 0) &
        (df_sr_uncertainty['SR'] <= 200)
    ].copy()
    
    if len(df_valid) == 0:
        logger.warning("No hay datos válidos para calcular incertidumbre por rangos de SR")
        return pd.DataFrame()
    
    # Crear rangos de SR
    df_valid['sr_range'] = pd.cut(df_valid['SR'], bins=sr_bins, include_lowest=True)
    
    # Agrupar por rango
    grouped = df_valid.groupby('sr_range')
    
    results = []
    for sr_range, group in grouped:
        if len(group) == 0:
            continue
        
        valid_u = group['U_SR_k2_rel'].dropna()
        
        if len(valid_u) > 0:
            U_k2_rel_mean = valid_u.mean() * 100  # Convertir a porcentaje
            U_k2_rel_std = valid_u.std() * 100
            n_minutes = len(group)
            
            results.append({
                'sr_range': str(sr_range),
                'U_k2_rel_mean': U_k2_rel_mean,
                'U_k2_rel_std': U_k2_rel_std,
                'n_minutes': n_minutes
            })
    
    if len(results) == 0:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    
    logger.info(f"Incertidumbre calculada para {len(df_results)} rangos de SR")
    
    return df_results


def aggregate_with_uncertainty(
    sr_series: pd.Series,
    U_campaign_k2_rel: float,
    freq: str = 'D',
    quantile: float = 0.25,
    df_sr_uncertainty: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Agrega serie de SR y asigna incertidumbre a cada agregado.
    
    Si se proporciona df_sr_uncertainty, calcula la incertidumbre local para cada
    agregado en lugar de usar una incertidumbre global. Esto es más representativo
    cuando el SR varía con el tiempo.
    
    Args:
        sr_series: Serie temporal de SR (%)
        U_campaign_k2_rel: Incertidumbre relativa de campaña k=2 (en porcentaje, ej: 3.67)
                         Usado como fallback si no se proporciona df_sr_uncertainty
        freq: Frecuencia de agregación ('D', 'W-SUN', 'M')
        quantile: Cuantil para agregación (default 0.25 para Q25)
        df_sr_uncertainty: DataFrame con incertidumbre minuto a minuto (opcional)
    
    Returns:
        pd.DataFrame con columnas:
        - SR_agg: SR agregado (%)
        - U_rel_k2: Incertidumbre relativa k=2 (%)
        - CI95_lo: Límite inferior IC 95% (%)
        - CI95_hi: Límite superior IC 95% (%)
        - n_minutes: Número de minutos usados en la agregación
    """
    # Validar que U_campaign_k2_rel sea finito (para fallback)
    if not np.isfinite(U_campaign_k2_rel) or U_campaign_k2_rel <= 0:
        logger.warning(f"U_campaign_k2_rel no es válido ({U_campaign_k2_rel}), usando valor por defecto 3.7%")
        U_campaign_k2_rel = 3.7
    
    # Filtrar SR extremos antes de agregar
    sr_series_filtered = sr_series[(sr_series >= 0) & (sr_series <= 200)]
    
    if len(sr_series_filtered) == 0:
        logger.warning(f"No hay datos válidos de SR para agregar con frecuencia {freq}")
        return pd.DataFrame()
    
    # Agregar SR con manejo de timezone (evitar NonExistentTimeError)
    try:
        # Intentar resample normal
        sr_agg = sr_series_filtered.resample(freq, label='right', closed='right').quantile(quantile)
        n_minutes = sr_series_filtered.resample(freq, label='right', closed='right').count()
    except Exception as e:
        # Si hay error de timezone, convertir a UTC primero
        logger.warning(f"Error en resample con timezone local: {e}. Convirtiendo a UTC...")
        sr_series_utc = sr_series_filtered.copy()
        if sr_series_utc.index.tz is not None:
            sr_series_utc.index = sr_series_utc.index.tz_convert('UTC')
        sr_agg = sr_series_utc.resample(freq, label='right', closed='right').quantile(quantile)
        n_minutes = sr_series_utc.resample(freq, label='right', closed='right').count()
    
    # Crear DataFrame de resultados
    result = pd.DataFrame(index=sr_agg.index)
    result['SR_agg'] = sr_agg
    result['n_minutes'] = n_minutes.reindex(sr_agg.index, fill_value=0)
    
    # Calcular incertidumbre local si se proporciona df_sr_uncertainty
    if df_sr_uncertainty is not None and 'U_SR_k2_rel' in df_sr_uncertainty.columns:
        logger.info(f"Calculando incertidumbre LOCAL para cada agregado {freq}...")
        
        # Filtrar datos válidos de incertidumbre
        df_u_valid = df_sr_uncertainty[
            df_sr_uncertainty['U_SR_k2_rel'].notna() &
            np.isfinite(df_sr_uncertainty['U_SR_k2_rel']) &
            (df_sr_uncertainty['U_SR_k2_rel'] >= 0) &
            (df_sr_uncertainty['U_SR_k2_rel'] <= 0.5)
        ].copy()
        
        if len(df_u_valid) > 0:
            # Agregar incertidumbre por período
            try:
                u_agg = df_u_valid['U_SR_k2_rel'].resample(freq, label='right', closed='right').mean()
                u_agg = u_agg * 100  # Convertir a porcentaje
                result['U_rel_k2'] = u_agg.reindex(sr_agg.index)
                
                # Rellenar con valor global si falta algún período
                mask_missing = result['U_rel_k2'].isna()
                if mask_missing.any():
                    result.loc[mask_missing, 'U_rel_k2'] = U_campaign_k2_rel
                    logger.info(f"Rellenando {mask_missing.sum()} períodos sin datos con incertidumbre global")
            except Exception as e:
                logger.warning(f"Error calculando incertidumbre local: {e}. Usando incertidumbre global.")
                result['U_rel_k2'] = U_campaign_k2_rel
        else:
            logger.warning("No hay datos válidos de incertidumbre. Usando incertidumbre global.")
            result['U_rel_k2'] = U_campaign_k2_rel
    else:
        # Usar incertidumbre global
        result['U_rel_k2'] = U_campaign_k2_rel
    
    # Calcular intervalos de confianza al 95% usando incertidumbre local
    U_rel_k2_frac = result['U_rel_k2'] / 100.0
    result['CI95_lo'] = result['SR_agg'] * (1 - U_rel_k2_frac)
    result['CI95_hi'] = result['SR_agg'] * (1 + U_rel_k2_frac)
    
    # Eliminar filas sin datos
    result = result[result['SR_agg'].notna()]
    
    return result


def validate_and_log_results(df_sr_uncertainty: pd.DataFrame) -> dict:
    """
    Valida resultados y genera estadísticas de logging.
    
    Args:
        df_sr_uncertainty: DataFrame con resultados de process_campaign_uncertainty
    
    Returns:
        dict: Diccionario con estadísticas de validación
    """
    stats = {}
    
    # Datos válidos
    valid_sr = df_sr_uncertainty['SR'].dropna()
    valid_u = df_sr_uncertainty['U_SR_k2_rel'].dropna()
    
    stats['n_minutes_valid'] = len(valid_sr)
    stats['n_minutes_total'] = len(df_sr_uncertainty)
    
    if len(valid_sr) > 0:
        stats['sr_mean'] = valid_sr.mean()
        stats['sr_min'] = valid_sr.min()
        stats['sr_max'] = valid_sr.max()
        stats['sr_std'] = valid_sr.std()
    
    if len(valid_u) > 0:
        # Filtrar valores finitos y en rango razonable
        valid_u_finite = valid_u[np.isfinite(valid_u)]
        valid_u_finite = valid_u_finite[(valid_u_finite >= 0) & (valid_u_finite <= 0.5)]  # 0% a 50% en fracción
        
        if len(valid_u_finite) > 0:
            stats['U_k2_rel_mean'] = valid_u_finite.mean() * 100  # En porcentaje
            stats['U_k2_rel_std'] = valid_u_finite.std() * 100
            stats['U_k2_rel_p25'] = valid_u_finite.quantile(0.25) * 100
            stats['U_k2_rel_p50'] = valid_u_finite.quantile(0.50) * 100
            stats['U_k2_rel_p75'] = valid_u_finite.quantile(0.75) * 100
            
            # Validar rango esperado [3.5%, 3.9%]
            if not (3.5 <= stats['U_k2_rel_mean'] <= 3.9):
                logger.warning(
                    f"⚠️  ALERTA: U_campaign_k2_rel ({stats['U_k2_rel_mean']:.3f}%) "
                    f"fuera del rango esperado [3.5%, 3.9%]"
                )
                
                # Estadísticas de S y C para diagnóstico
                valid_s = df_sr_uncertainty['S'].dropna()
                valid_c = df_sr_uncertainty['C'].dropna()
                
                if len(valid_s) > 0 and len(valid_c) > 0:
                    logger.warning(f"  Estadísticas S: media={valid_s.mean():.1f} W/m², std={valid_s.std():.1f} W/m²")
                    logger.warning(f"  Estadísticas C: media={valid_c.mean():.1f} W/m², std={valid_c.std():.1f} W/m²")
        else:
            logger.warning("No hay valores finitos de U_SR_k2_rel para calcular estadísticas")
            stats['U_k2_rel_mean'] = np.nan
            stats['U_k2_rel_std'] = np.nan
            stats['U_k2_rel_p25'] = np.nan
            stats['U_k2_rel_p50'] = np.nan
            stats['U_k2_rel_p75'] = np.nan
    
    # Logging
    logger.info("="*80)
    logger.info("ESTADÍSTICAS DE VALIDACIÓN")
    logger.info("="*80)
    logger.info(f"Total de minutos: {stats.get('n_minutes_total', 0):,}")
    logger.info(f"Minutos válidos: {stats.get('n_minutes_valid', 0):,}")
    
    if 'sr_mean' in stats:
        logger.info(f"SR - Media: {stats['sr_mean']:.2f}%, Min: {stats['sr_min']:.2f}%, Max: {stats['sr_max']:.2f}%")
    
    if 'U_k2_rel_mean' in stats:
        logger.info(f"U_SR_k2_rel - Media: {stats['U_k2_rel_mean']:.3f}%")
        logger.info(f"U_SR_k2_rel - Std: {stats['U_k2_rel_std']:.3f}%")
        logger.info(f"U_SR_k2_rel - P25: {stats['U_k2_rel_p25']:.3f}%, P50: {stats['U_k2_rel_p50']:.3f}%, P75: {stats['U_k2_rel_p75']:.3f}%")
    
    logger.info("="*80)
    
    return stats


def run_uncertainty_propagation_analysis(
    df_ref_cells: pd.DataFrame,
    soiled_col: str = SOILED_COL,
    clean_col: str = CLEAN_COL
) -> bool:
    """
    Función principal para ejecutar el análisis completo de propagación de incertidumbre.
    
    Args:
        df_ref_cells: DataFrame con datos de celdas de referencia
        soiled_col: Nombre de columna de celda sucia
        clean_col: Nombre de columna de celda limpia
    
    Returns:
        bool: True si el análisis fue exitoso, False en caso contrario
    """
    try:
        logger.info("="*80)
        logger.info("INICIANDO ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE DE SR")
        logger.info("="*80)
        
        # Crear directorio de salida (ya está creado en paths.py, pero asegurar que existe)
        os.makedirs(paths.PROPAGACION_ERRORES_REF_CELL_DIR, exist_ok=True)
        
        # --- Paso 1: Procesar incertidumbre minuto a minuto ---
        df_sr_uncertainty = process_campaign_uncertainty(
            df_ref_cells,
            soiled_col=soiled_col,
            clean_col=clean_col
        )
        
        if df_sr_uncertainty is None or df_sr_uncertainty.empty:
            logger.error("No se pudieron calcular incertidumbres minuto a minuto")
            return False
        
        # Guardar resultados minutales
        output_minute_file = paths.SR_MINUTE_WITH_UNCERTAINTY_FILE
        df_sr_uncertainty.to_csv(output_minute_file)
        logger.info(f"✅ Resultados minutales guardados en: {output_minute_file}")
        
        # --- Paso 2: Calcular incertidumbre de campaña ---
        u_campaign_k1_rel, U_campaign_k2_rel = calculate_campaign_uncertainty(df_sr_uncertainty)
        
        if np.isnan(u_campaign_k1_rel) or np.isnan(U_campaign_k2_rel):
            logger.error("No se pudo calcular incertidumbre de campaña")
            return False
        
        # --- Paso 3: Validar y generar estadísticas ---
        stats = validate_and_log_results(df_sr_uncertainty)
        
        # --- Paso 4: Análisis de incertidumbre por período y rangos de SR ---
        logger.info("Calculando incertidumbre por período temporal (mensual)...")
        df_uncertainty_monthly = calculate_uncertainty_by_period(df_sr_uncertainty, freq='M')
        if not df_uncertainty_monthly.empty:
            output_unc_monthly_file = os.path.join(paths.PROPAGACION_ERRORES_REF_CELL_DIR, 'sr_uncertainty_by_month.csv')
            df_uncertainty_monthly.to_csv(output_unc_monthly_file)
            logger.info(f"✅ Incertidumbre mensual guardada en: {output_unc_monthly_file}")
        
        logger.info("Calculando incertidumbre por rangos de SR...")
        df_uncertainty_sr_range = calculate_uncertainty_by_sr_range(df_sr_uncertainty)
        if not df_uncertainty_sr_range.empty:
            output_unc_sr_file = os.path.join(paths.PROPAGACION_ERRORES_REF_CELL_DIR, 'sr_uncertainty_by_sr_range.csv')
            df_uncertainty_sr_range.to_csv(output_unc_sr_file)
            logger.info(f"✅ Incertidumbre por rangos de SR guardada en: {output_unc_sr_file}")
        
        # --- Paso 5: Agregar a escalas diarias, semanales y mensuales con incertidumbre LOCAL ---
        # Filtrar SR válidos (finitos y en rango razonable)
        sr_series = df_sr_uncertainty['SR'].dropna()
        sr_series = sr_series[np.isfinite(sr_series)]
        sr_series = sr_series[(sr_series >= 0) & (sr_series <= 200)]
        
        if len(sr_series) == 0:
            logger.warning("No hay datos válidos de SR para agregar")
            return False
        
        # Asegurar timezone para agregaciones (usar UTC para evitar problemas de DST)
        if sr_series.index.tz is None:
            # Asumir UTC si no tiene timezone
            sr_series.index = sr_series.index.tz_localize('UTC')
        else:
            # Convertir a UTC para evitar problemas de DST en Chile
            sr_series.index = sr_series.index.tz_convert('UTC')
        
        # También convertir df_sr_uncertainty a UTC para consistencia
        df_sr_uncertainty_utc = df_sr_uncertainty.copy()
        if df_sr_uncertainty_utc.index.tz is None:
            df_sr_uncertainty_utc.index = df_sr_uncertainty_utc.index.tz_localize('UTC')
        else:
            df_sr_uncertainty_utc.index = df_sr_uncertainty_utc.index.tz_convert('UTC')
        
        # Agregación diaria (Q25) con incertidumbre LOCAL
        df_daily = aggregate_with_uncertainty(
            sr_series, U_campaign_k2_rel, freq='D', quantile=0.25,
            df_sr_uncertainty=df_sr_uncertainty_utc
        )
        if not df_daily.empty:
            output_daily_file = paths.SR_DAILY_ABS_WITH_U_FILE
            df_daily.to_csv(output_daily_file)
            logger.info(f"✅ Resultados diarios (con incertidumbre LOCAL) guardados en: {output_daily_file}")
        
        # Agregación semanal (Q25) con incertidumbre LOCAL
        df_weekly = aggregate_with_uncertainty(
            sr_series, U_campaign_k2_rel, freq='W-SUN', quantile=0.25,
            df_sr_uncertainty=df_sr_uncertainty_utc
        )
        if not df_weekly.empty:
            output_weekly_file = paths.SR_WEEKLY_ABS_WITH_U_FILE
            df_weekly.to_csv(output_weekly_file)
            logger.info(f"✅ Resultados semanales (con incertidumbre LOCAL) guardados en: {output_weekly_file}")
        
        # Agregación mensual (Q25) con incertidumbre LOCAL
        df_monthly = aggregate_with_uncertainty(
            sr_series, U_campaign_k2_rel, freq='M', quantile=0.25,
            df_sr_uncertainty=df_sr_uncertainty_utc
        )
        if not df_monthly.empty:
            output_monthly_file = paths.SR_MONTHLY_ABS_WITH_U_FILE
            df_monthly.to_csv(output_monthly_file)
            logger.info(f"✅ Resultados mensuales (con incertidumbre LOCAL) guardados en: {output_monthly_file}")
        
        # --- Paso 6: Análisis de sensibilidad de correlación (opcional) ---
        try:
            rho_sensitivity = sensitivity_analysis_rho(df_sr_uncertainty, rho_values=[0.0, 0.3, 0.5])
        except Exception as e:
            logger.warning(f"Error en análisis de sensibilidad de rho: {e}")
            rho_sensitivity = None
        
        # --- Paso 7: Generar reporte final ---
        summary_file = paths.SR_UNCERTAINTY_SUMMARY_FILE
        generate_summary_report(
            summary_file,
            df_sr_uncertainty,
            u_campaign_k1_rel,
            U_campaign_k2_rel,
            stats,
            output_minute_file,
            output_daily_file if not df_daily.empty else None,
            output_weekly_file if not df_weekly.empty else None,
            output_monthly_file if not df_monthly.empty else None,
            rho_sensitivity=rho_sensitivity,
            df_uncertainty_monthly=df_uncertainty_monthly if not df_uncertainty_monthly.empty else None,
            df_uncertainty_sr_range=df_uncertainty_sr_range if not df_uncertainty_sr_range.empty else None
        )
        
        # --- Paso 8: Generar gráficos de análisis de incertidumbre ---
        logger.info("\n📊 Generando gráficos de análisis de incertidumbre...")
        try:
            from analysis.plot_uncertainty_analysis import generate_all_uncertainty_plots
            from analysis.plot_uncertainty_solar_noon import generate_all_solar_noon_plots
            
            # Generar gráficos generales
            plot_files = generate_all_uncertainty_plots()
            
            # Generar gráficos de mediodía solar
            logger.info("\n" + "="*80)
            logger.info("Generando gráficos de análisis de incertidumbre durante mediodía solar...")
            logger.info("="*80)
            solar_noon_plot_files = generate_all_solar_noon_plots(hours_window=2.5)
            if solar_noon_plot_files:
                logger.info(f"✅ Se generaron {len(solar_noon_plot_files)} gráficos de mediodía solar")
            if plot_files:
                logger.info(f"✅ Se generaron {len(plot_files)} gráficos de análisis de incertidumbre")
            else:
                logger.warning("⚠️  No se generaron gráficos de análisis de incertidumbre")
        except Exception as e:
            logger.warning(f"⚠️  Error generando gráficos de análisis: {e}")
            # Continuar aunque falle la generación de gráficos
        
        logger.info("="*80)
        logger.info("✅ ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE COMPLETADO")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en análisis de propagación de incertidumbre: {e}", exc_info=True)
        return False


def sensitivity_analysis_rho(
    df_sr_uncertainty: pd.DataFrame,
    rho_values: list = [0.0, 0.3, 0.5]
) -> list:
    """
    Análisis de sensibilidad para diferentes valores de correlación rho.
    
    Args:
        df_sr_uncertainty: DataFrame con resultados base (rho=0.0)
        rho_values: Lista de valores de rho a evaluar
    
    Returns:
        list: Lista de tuplas (rho, U_mean)
    """
    logger.info("Análisis de sensibilidad para correlación entre canales...")
    
    S = df_sr_uncertainty['S'].dropna()
    C = df_sr_uncertainty['C'].dropna()
    
    results = []
    for rho in rho_values:
        result_rho = propagate_sr_minute(S, C, rho=rho)
        U_mean = result_rho['U_SR_k2_rel'].dropna().mean() * 100
        results.append((rho, U_mean))
        logger.info(f"  rho = {rho:.1f}: U_campaign_k2_rel = {U_mean:.3f}%")
    
    return results


def generate_summary_report(
    output_file: str,
    df_sr_uncertainty: pd.DataFrame,
    u_campaign_k1_rel: float,
    U_campaign_k2_rel: float,
    stats: dict,
    minute_file: str,
    daily_file: Optional[str],
    weekly_file: Optional[str],
    monthly_file: Optional[str],
    rho_sensitivity: Optional[list] = None,
    df_uncertainty_monthly: Optional[pd.DataFrame] = None,
    df_uncertainty_sr_range: Optional[pd.DataFrame] = None
) -> None:
    """
    Genera reporte final en texto con fórmulas, parámetros y resultados.
    
    Args:
        output_file: Ruta del archivo de salida
        df_sr_uncertainty: DataFrame con resultados minutales
        u_campaign_k1_rel: Incertidumbre de campaña k=1 (%)
        U_campaign_k2_rel: Incertidumbre de campaña k=2 (%)
        stats: Diccionario con estadísticas
        minute_file: Ruta del archivo CSV minutal
        daily_file: Ruta del archivo CSV diario (opcional)
        weekly_file: Ruta del archivo CSV semanal (opcional)
        monthly_file: Ruta del archivo CSV mensual (opcional)
        rho_sensitivity: Lista de resultados de sensibilidad (opcional)
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE INCERTIDUMBRE DE SOILING RATIO (PROPAGACIÓN GUM)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONFIGURACIÓN:\n")
        f.write("-"*80 + "\n")
        f.write(f"Columna celda sucia (S): {SOILED_COL}\n")
        f.write(f"Columna celda limpia (C): {CLEAN_COL}\n")
        f.write(f"Intervalo de muestreo: 1 minuto\n")
        f.write(f"Timezone para agregaciones: {TZ_ANALYSIS}\n\n")
        
        f.write("INCERTIDUMBRES DEL FABRICANTE (Si-V-10TC-T, k=2):\n")
        f.write("-"*80 + "\n")
        f.write(f"Incertidumbre aditiva (k=2): U_add_k2 = {U_ADD_K2} W/m²\n")
        f.write(f"Incertidumbre de escala (k=2): U_scale_k2 = {U_SCALE_K2*100:.2f}%\n")
        f.write(f"Factor de cobertura: k_expand = {K_EXPAND}\n\n")
        
        f.write("CONVERSIÓN A k=1 (1σ):\n")
        f.write("-"*80 + "\n")
        f.write(f"u_add = U_add_k2 / 2 = {U_ADD} W/m²\n")
        f.write(f"u_scale = U_scale_k2 / 2 = {U_SCALE*100:.3f}% ({U_SCALE})\n\n")
        
        f.write("FÓRMULAS DE PROPAGACIÓN:\n")
        f.write("-"*80 + "\n")
        f.write("1. Incertidumbre por canal:\n")
        f.write("   u(I)² = u_add² + (u_scale * I)²\n\n")
        f.write("2. Soiling Ratio:\n")
        f.write("   SR = 100 * S / C\n\n")
        f.write("3. Derivadas parciales:\n")
        f.write("   ∂SR/∂S = 100 / C\n")
        f.write("   ∂SR/∂C = -100 * S / C²\n\n")
        f.write("4. Varianza propagada (GUM, primer orden):\n")
        f.write("   Var(SR) = (∂SR/∂S)² * u(S)² + (∂SR/∂C)² * u(C)² + 2 * (∂SR/∂S) * (∂SR/∂C) * Cov(S,C)\n")
        f.write("   donde Cov(S,C) = rho * u(S) * u(C)\n")
        f.write("   (Por defecto: rho = 0.0, independencia entre canales)\n\n")
        f.write("5. Incertidumbre expandida:\n")
        f.write("   U_SR_k2 = k_expand * u_SR_k1\n\n")
        
        f.write("RESULTADOS DE CAMPAÑA (GLOBAL):\n")
        f.write("-"*80 + "\n")
        f.write(f"Total de minutos válidos: {stats.get('n_minutes_valid', 0):,}\n")
        f.write(f"Incertidumbre de campaña GLOBAL (k=1): u_campaign_k1_rel = {u_campaign_k1_rel:.3f}%\n")
        f.write(f"Incertidumbre de campaña GLOBAL (k=2): U_campaign_k2_rel = {U_campaign_k2_rel:.3f}%\n")
        f.write("\n⚠️  NOTA IMPORTANTE:\n")
        f.write("   La incertidumbre global puede no ser representativa si el SR varía\n")
        f.write("   significativamente con el tiempo de exposición. Se recomienda usar\n")
        f.write("   la incertidumbre LOCAL calculada para cada período (ver secciones siguientes).\n\n")
        
        if 'sr_mean' in stats:
            f.write("ESTADÍSTICAS DE SR:\n")
            f.write("-"*80 + "\n")
            f.write(f"SR promedio: {stats['sr_mean']:.2f}%\n")
            f.write(f"SR mínimo: {stats['sr_min']:.2f}%\n")
            f.write(f"SR máximo: {stats['sr_max']:.2f}%\n")
            f.write(f"SR desviación estándar: {stats['sr_std']:.2f}%\n\n")
        
        if 'U_k2_rel_mean' in stats:
            f.write("ESTADÍSTICAS DE INCERTIDUMBRE:\n")
            f.write("-"*80 + "\n")
            f.write(f"U_SR_k2_rel promedio: {stats['U_k2_rel_mean']:.3f}%\n")
            f.write(f"U_SR_k2_rel desviación estándar: {stats['U_k2_rel_std']:.3f}%\n")
            f.write(f"U_SR_k2_rel - P25: {stats['U_k2_rel_p25']:.3f}%, P50: {stats['U_k2_rel_p50']:.3f}%, P75: {stats['U_k2_rel_p75']:.3f}%\n\n")
        
        f.write("NOTAS:\n")
        f.write("-"*80 + "\n")
        f.write("- Propagación calculada minuto a minuto sobre toda la campaña\n")
        f.write("- Independencia entre canales asumida (Cov=0, rho=0.0)\n")
        f.write("- Incertidumbre expandida a k=2 según GUM\n")
        f.write("- Agregaciones (diarias/semanales/mensuales) usan Q25\n")
        f.write("- ⭐ MEJORA: Las agregaciones usan incertidumbre LOCAL calculada para cada período\n")
        f.write("  en lugar de una incertidumbre global, lo cual es más representativo cuando\n")
        f.write("  el SR varía con el tiempo de exposición.\n")
        f.write("- Intervalos de confianza al 95%: CI = SR_agg * (1 ± U_local_k2_rel)\n\n")
        
        if df_uncertainty_monthly is not None and not df_uncertainty_monthly.empty:
            f.write("INCERTIDUMBRE POR PERÍODO TEMPORAL (MENSUAL):\n")
            f.write("-"*80 + "\n")
            f.write("Esta es más representativa que la incertidumbre global cuando el SR varía con el tiempo.\n\n")
            for idx, row in df_uncertainty_monthly.iterrows():
                f.write(f"  {idx.strftime('%Y-%m')}: U_k2_rel = {row['U_k2_rel_mean']:.3f}% ± {row['U_k2_rel_std']:.3f}%")
                f.write(f" (SR promedio: {row['sr_mean']:.2f}%, n={row['n_minutes']:,} minutos)\n")
            f.write("\n")
        
        if df_uncertainty_sr_range is not None and not df_uncertainty_sr_range.empty:
            f.write("INCERTIDUMBRE POR RANGOS DE SR:\n")
            f.write("-"*80 + "\n")
            f.write("Muestra cómo varía la incertidumbre con el nivel de soiling.\n\n")
            for _, row in df_uncertainty_sr_range.iterrows():
                f.write(f"  {row['sr_range']}: U_k2_rel = {row['U_k2_rel_mean']:.3f}% ± {row['U_k2_rel_std']:.3f}%")
                f.write(f" (n={row['n_minutes']:,} minutos)\n")
            f.write("\n")
        
        if rho_sensitivity:
            f.write("ANÁLISIS DE SENSIBILIDAD (Correlación entre canales):\n")
            f.write("-"*80 + "\n")
            for rho, U_mean in rho_sensitivity:
                f.write(f"  rho = {rho:.1f}: U_campaign_k2_rel = {U_mean:.3f}%\n")
            f.write("\n")
        
        f.write("ARCHIVOS GENERADOS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Minutal: {minute_file}\n")
        if daily_file:
            f.write(f"Diario: {daily_file}\n")
        if weekly_file:
            f.write(f"Semanal: {weekly_file}\n")
        if monthly_file:
            f.write(f"Mensual: {monthly_file}\n")
        f.write(f"Este resumen: {output_file}\n")
        f.write("="*80 + "\n")
    
    logger.info(f"✅ Reporte guardado en: {output_file}")

