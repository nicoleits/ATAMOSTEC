#!/usr/bin/env python3
"""
Procesamiento simple de IV600 para análisis de desviación estándar diaria
Período: Octubre 2024 - Marzo 2025
Horario unificado: 13:00-18:00
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_iv600_simple():
    """Procesa datos IV600 de manera simple para obtener SR por minuto"""
    
    logger.info("=== PROCESANDO IV600 DE MANERA SIMPLE ===")
    
    # Cargar datos IV600 - ruta relativa al proyecto
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = SCRIPT_DIR
    csv_path = os.path.join(PROJECT_ROOT, 'datos', 'raw_iv600_data.csv')
    
    if not os.path.exists(csv_path):
        logger.error(f"Archivo no encontrado: {csv_path}")
        return pd.DataFrame()
    
    try:
        logger.info("Cargando datos IV600...")
        df_raw = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        
        logger.info(f"Datos IV600 cargados: {len(df_raw)} registros")
        logger.info(f"Columnas: {df_raw.columns.tolist()}")
        logger.info(f"Rango de fechas: {df_raw.index.min()} a {df_raw.index.max()}")
        
        # Procesar timezone
        if df_raw.index.tz is not None:
            logger.info("Convirtiendo timezone...")
            df_raw.index = df_raw.index.tz_convert(None)
        
        # Filtrar por período octubre-marzo
        start_date = '2024-10-01'
        end_date = '2025-03-31'
        
        df_period = df_raw[(df_raw.index >= start_date) & (df_raw.index <= end_date)]
        logger.info(f"Datos del período Oct-Mar: {len(df_period)} registros")
        
        if df_period.empty:
            logger.warning("No hay datos IV600 para el período especificado")
            return pd.DataFrame()
        
        # Aplicar filtro de horario unificado (13:00-18:00)
        df_time = df_period.between_time('13:00', '18:00')
        logger.info(f"Datos con horario unificado (13:00-18:00): {len(df_time)} registros")
        
        if df_time.empty:
            logger.warning("No hay datos IV600 después del filtro de horario")
            return pd.DataFrame()
        
        # Verificar módulos disponibles
        modules = df_time['module'].unique()
        logger.info(f"Módulos disponibles: {modules}")
        
        # Módulos de interés (igual que en el análisis original)
        target_modules = ['1MD434', '1MD439', '1MD440']
        available_modules = [mod for mod in target_modules if mod in modules]
        
        if not available_modules:
            logger.warning(f"Ninguno de los módulos objetivo {target_modules} está disponible")
            return pd.DataFrame()
        
        logger.info(f"Módulos objetivo disponibles: {available_modules}")
        
        # Crear DataFrame con SR por módulo
        sr_data = []
        
        for module in available_modules:
            df_module = df_time[df_time['module'] == module].copy()
            
            if df_module.empty:
                continue
            
            # Usar Pmax como indicador de SR (simplificado)
            # En un análisis real, aquí se calcularía el SR comparando con módulo de referencia
            df_module['SR_IV600'] = (df_module['pmp'] / df_module['pmp'].mean()) * 100
            
            # Filtrar valores razonables de SR
            df_module = df_module[
                (df_module['SR_IV600'] >= 90) & 
                (df_module['SR_IV600'] <= 110)
            ]
            
            if not df_module.empty:
                sr_data.append(df_module[['SR_IV600']])
                logger.info(f"  {module}: {len(df_module)} registros válidos")
        
        if not sr_data:
            logger.warning("No se pudieron calcular SR válidos para ningún módulo")
            return pd.DataFrame()
        
        # Combinar todos los módulos
        df_combined = pd.concat(sr_data)
        
        # Remuestrear a 5 minutos para tener datos por minuto
        df_5min = df_combined.resample('5min').mean()
        
        # Eliminar filas con NaN
        df_final = df_5min.dropna()
        
        logger.info(f"SR IV600 final: {len(df_final)} registros")
        logger.info(f"Rango de fechas: {df_final.index.min()} a {df_final.index.max()}")
        logger.info(f"Rango de SR: {df_final['SR_IV600'].min():.2f}% a {df_final['SR_IV600'].max():.2f}%")
        
        # Guardar datos procesados - ruta relativa al proyecto
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = SCRIPT_DIR
        output_dir = os.path.join(PROJECT_ROOT, 'datos_procesados_analisis_integrado_py', 'iv600')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'iv600_sr_unified_hours_oct_mar.csv')
        df_final.to_csv(output_path)
        
        logger.info(f"✅ Datos IV600 guardados en: {output_path}")
        
        return df_final
        
    except Exception as e:
        logger.error(f"Error procesando IV600: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def main():
    """Función principal"""
    result = process_iv600_simple()
    
    if not result.empty:
        logger.info("🎉 ¡Procesamiento IV600 completado exitosamente!")
        return True
    else:
        logger.error("❌ No se pudieron procesar los datos IV600")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
