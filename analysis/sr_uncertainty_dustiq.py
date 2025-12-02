"""
Análisis de Incertidumbre del Soiling Ratio (SR) para DustIQ mediante Propagación de Errores (GUM)

Este módulo calcula la incertidumbre del SR usando propagación de errores según GUM:
- Incertidumbres del sensor DustIQ según especificaciones del fabricante
- El SR se obtiene directamente del sensor (SR_C11_Avg o promedio de canales)
- Propagación minuto a minuto
- Agregación a escalas diarias, semanales y mensuales

DESCRIPCIÓN:
------------
El SR se obtiene directamente del sensor DustIQ:
- SR = SR_C11_Avg (o promedio de SR_C11_Avg y SR_C12_Avg)

La incertidumbre del sensor según especificaciones del fabricante:
- Accuracy: ±0.1% of reading ±1%
- Esto se traduce en:
  - Incertidumbre aditiva: U_add_k2 = 0.1%
  - Incertidumbre de escala: U_scale_k2 = 1% (0.01 en fracción)
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

# Incertidumbres del sensor DustIQ según especificaciones del fabricante (k=2)
# Accuracy: ±0.1% of reading ±1%
U_SR_ADD_K2 = 0.1  # % (aditiva, k=2) - Valor del fabricante
U_SR_SCALE_K2 = 0.01  # 1% (de escala, k=2) - Valor del fabricante (0.01 = 1% en fracción)

# Convertir a k=1 (1σ)
U_SR_ADD = U_SR_ADD_K2 / 2.0  # 0.05%
U_SR_SCALE = U_SR_SCALE_K2 / 2.0  # 0.005 (0.5%)

# Factor de cobertura para expandir al final
K_EXPAND = 2.0

# Columnas de datos
SR_COL = "SR_C11_Avg"  # Columna principal de SR
SR_COL_ALT = "SR_C12_Avg"  # Columna alternativa (si se usa promedio)

# Umbral mínimo de SR (para filtrar datos inválidos)
MIN_SR_THRESHOLD = 0.0  # % (típicamente se filtra > 70% en análisis)


def sr_u(SR: pd.Series, u_add: float, u_scale: float) -> pd.Series:
    """
    Calcula la incertidumbre u(SR) usando el modelo combinado aditivo + escala.
    
    Fórmula: u(SR)² = u_add² + (u_scale * SR)²
    
    Args:
        SR: Serie temporal de Soiling Ratio (%)
        u_add: Incertidumbre aditiva (%, k=1)
        u_scale: Incertidumbre de escala (adimensional, k=1)
    
    Returns:
        pd.Series: Incertidumbre absoluta u(SR) en % (k=1)
    """
    u_squared = u_add**2 + (u_scale * SR)**2
    u_SR = np.sqrt(u_squared)
    return u_SR


def process_campaign_uncertainty(
    df_dustiq: pd.DataFrame,
    sr_col: str = SR_COL,
    use_average: bool = False
) -> Optional[pd.DataFrame]:
    """
    Procesa la incertidumbre minuto a minuto sobre toda la campaña.
    
    Args:
        df_dustiq: DataFrame con datos de DustIQ
                   Debe tener índice DatetimeIndex y columna SR_C11_Avg (y opcionalmente SR_C12_Avg)
        sr_col: Nombre de columna de SR principal
        use_average: Si True, usa promedio de SR_C11_Avg y SR_C12_Avg si ambos están disponibles
    
    Returns:
        pd.DataFrame con columnas: timestamp, SR, u_SR_k1_abs, u_SR_k1_rel, U_SR_k2_rel
        o None si hay error
    """
    try:
        logger.info("Procesando incertidumbre de campaña minuto a minuto (DustIQ)...")
        
        # Determinar columna de SR a usar
        if use_average and 'SR_C11_Avg' in df_dustiq.columns and 'SR_C12_Avg' in df_dustiq.columns:
            # Usar promedio de ambos canales
            df_dustiq['SR'] = df_dustiq[['SR_C11_Avg', 'SR_C12_Avg']].mean(axis=1)
            logger.info("Usando promedio de SR_C11_Avg y SR_C12_Avg")
        elif sr_col in df_dustiq.columns:
            df_dustiq['SR'] = pd.to_numeric(df_dustiq[sr_col], errors='coerce')
        else:
            logger.error(f"Columna de SR '{sr_col}' no encontrada")
            return None
        
        # Filtrar datos válidos
        df_valid = df_dustiq[['SR']].copy()
        df_valid = df_valid.dropna(subset=['SR'])
        df_valid = df_valid[~df_valid.index.duplicated(keep='first')]
        df_valid = df_valid.sort_index()
        
        logger.info(f"Datos válidos: {len(df_valid)} puntos")
        
        # Calcular incertidumbre
        u_SR_k1_abs = sr_u(df_valid['SR'], U_SR_ADD, U_SR_SCALE)
        
        # Crear DataFrame de resultados
        result = pd.DataFrame(index=df_valid.index)
        result['SR'] = df_valid['SR']
        result['u_SR_k1_abs'] = u_SR_k1_abs
        
        # Incertidumbre relativa k=1
        mask_sr_valid = (result['SR'] > 0) & (result['SR'] < 200)
        result['u_SR_k1_rel'] = np.nan
        result.loc[mask_sr_valid, 'u_SR_k1_rel'] = u_SR_k1_abs[mask_sr_valid] / result.loc[mask_sr_valid, 'SR']
        
        # Incertidumbre expandida k=2
        result['U_SR_k2_abs'] = K_EXPAND * u_SR_k1_abs
        result['U_SR_k2_rel'] = K_EXPAND * result['u_SR_k1_rel']
        
        # Filtrar SR extremos
        mask_sr_reasonable = (result['SR'] >= 0) & (result['SR'] <= 200)
        n_sr_extreme = (~mask_sr_reasonable).sum()
        if n_sr_extreme > 0:
            logger.warning(f"Descartando {n_sr_extreme} minutos con SR extremo (fuera de [0%, 200%])")
            result.loc[~mask_sr_reasonable, ['SR', 'u_SR_k1_rel', 'U_SR_k2_rel']] = np.nan
        
        # Reordenar columnas
        result = result[['SR', 'u_SR_k1_rel', 'U_SR_k2_rel']]
        
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
    """
    if not np.isfinite(U_campaign_k2_rel) or U_campaign_k2_rel <= 0:
        logger.warning(f"U_campaign_k2_rel no es válido ({U_campaign_k2_rel}), usando valor por defecto 2.0%")
        U_campaign_k2_rel = 2.0
    
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
    df_dustiq: pd.DataFrame,
    sr_col: str = SR_COL,
    use_average: bool = False
) -> bool:
    """
    Función principal para ejecutar el análisis completo de propagación de incertidumbre.
    
    Args:
        df_dustiq: DataFrame con datos de DustIQ
        sr_col: Nombre de columna de SR principal
        use_average: Si True, usa promedio de SR_C11_Avg y SR_C12_Avg
    
    Returns:
        bool: True si el análisis fue exitoso, False en caso contrario
    """
    try:
        logger.info("="*80)
        logger.info("INICIANDO ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE DE SR (DUSTIQ)")
        logger.info("="*80)
        
        os.makedirs(paths.PROPAGACION_ERRORES_DUSTIQ_DIR, exist_ok=True)
        
        df_sr_uncertainty = process_campaign_uncertainty(
            df_dustiq,
            sr_col=sr_col,
            use_average=use_average
        )
        
        if df_sr_uncertainty is None or df_sr_uncertainty.empty:
            logger.error("No se pudieron calcular incertidumbres minuto a minuto")
            return False
        
        output_minute_file = paths.DUSTIQ_SR_MINUTE_WITH_UNCERTAINTY_FILE
        df_sr_uncertainty.to_csv(output_minute_file)
        logger.info(f"✅ Resultados minutales guardados en: {output_minute_file}")
        
        u_campaign_k1_rel, U_campaign_k2_rel = calculate_campaign_uncertainty(df_sr_uncertainty)
        
        if np.isnan(u_campaign_k1_rel) or np.isnan(U_campaign_k2_rel):
            logger.error("No se pudo calcular incertidumbre de campaña")
            return False
        
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
        
        for freq, suffix in [('D', 'daily'), ('W-SUN', 'weekly'), ('M', 'monthly')]:
            df_agg = aggregate_with_uncertainty(
                sr_series, U_campaign_k2_rel, freq=freq, quantile=0.25,
                df_sr_uncertainty=df_sr_uncertainty_utc
            )
            if not df_agg.empty:
                output_file = os.path.join(
                    paths.PROPAGACION_ERRORES_DUSTIQ_DIR,
                    f"sr_{suffix}_abs_with_U.csv"
                )
                df_agg.to_csv(output_file)
                logger.info(f"✅ Resultados {suffix} guardados en: {output_file}")
        
        logger.info("="*80)
        logger.info("✅ ANÁLISIS DE PROPAGACIÓN DE INCERTIDUMBRE COMPLETADO (DUSTIQ)")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en análisis de propagación de incertidumbre: {e}", exc_info=True)
        return False

