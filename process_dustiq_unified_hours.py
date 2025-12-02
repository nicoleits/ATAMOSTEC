#!/usr/bin/env python3
"""
Script para procesar datos de DustIQ con filtro horario UNIFICADO
Usando el mismo rango horario que RefCells y PVStand (13:00-18:00)
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Agregar el directorio de análisis al path - rutas relativas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(os.path.join(PROJECT_ROOT, 'analysis'))
sys.path.append(PROJECT_ROOT)

from config import paths

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_dustiq_raw_data():
    """Carga los datos raw de DustIQ"""
    try:
        csv_path = os.path.join(paths.BASE_INPUT_DIR, "raw_dustiq_data.csv")
        
        if not os.path.exists(csv_path):
            logger.error(f"Archivo no encontrado: {csv_path}")
            return pd.DataFrame()
        
        logger.info(f"Cargando datos DustIQ desde: {csv_path}")
        logger.info("Archivo grande, usando lectura por chunks...")
        
        # Leer por chunks para manejar archivos grandes
        chunk_size = 50000
        chunks = []
        
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            logger.info(f"Procesando chunk {i+1}...")
            
            # Convertir timestamp a datetime
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            chunk.set_index('timestamp', inplace=True)
            
            # Calcular SR_DustIQ como promedio de C11 y C12
            if 'SR_C11_Avg' in chunk.columns and 'SR_C12_Avg' in chunk.columns:
                chunk['SR_DustIQ'] = chunk[['SR_C11_Avg', 'SR_C12_Avg']].mean(axis=1)
            elif 'SR_C11_Avg' in chunk.columns:
                chunk['SR_DustIQ'] = chunk['SR_C11_Avg']
            else:
                logger.warning("No se encontraron columnas SR_C11_Avg o SR_C12_Avg")
                continue
            
            # Mantener solo las columnas necesarias
            chunk = chunk[['SR_DustIQ']]
            chunks.append(chunk)
        
        # Concatenar todos los chunks
        df = pd.concat(chunks, ignore_index=False)
        df = df.sort_index()
        
        logger.info(f"Datos DustIQ cargados: {len(df)} registros")
        logger.info(f"Rango de fechas: {df.index.min()} a {df.index.max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error cargando datos DustIQ: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def process_dustiq_unified_hours():
    """Procesa datos DustIQ con filtro horario UNIFICADO (13:00-18:00)"""
    
    logger.info("=== PROCESANDO DUSTIQ CON FILTRO HORARIO UNIFICADO ===")
    logger.info("USANDO RANGO 13:00-18:00 (IGUAL QUE REFCELLS Y PVSTAND)")
    
    # 1. Cargar datos raw
    df_raw = load_dustiq_raw_data()
    if df_raw.empty:
        logger.error("No se pudieron cargar los datos raw de DustIQ")
        return False
    
    # 2. Filtrar por período octubre-marzo
    start_date = '2024-10-01'
    end_date = '2025-03-31'
    
    logger.info(f"Filtrando por período: {start_date} a {end_date}")
    df_period = df_raw[(df_raw.index >= start_date) & (df_raw.index <= end_date)]
    logger.info(f"Datos del período: {len(df_period)} registros")
    
    if df_period.empty:
        logger.error("No hay datos para el período especificado")
        return False
    
    # 3. Aplicar filtro de horario UNIFICADO (13:00-18:00 UTC, igual que RefCells y PVStand)
    logger.info("Aplicando filtro de horario UNIFICADO (13:00-18:00 UTC)...")
    df_time_filtered = df_period.between_time('13:00', '18:00')
    logger.info(f"Datos después del filtro de horario unificado: {len(df_time_filtered)} registros")
    
    if df_time_filtered.empty:
        logger.error("No hay datos después del filtro de horario unificado")
        return False
    
    # 4. Aplicar filtro de calidad (SR > 93%)
    logger.info("Aplicando filtro de calidad SR > 93%...")
    df_quality = df_time_filtered[df_time_filtered['SR_DustIQ'] > 93.0]
    logger.info(f"Datos después del filtro de calidad: {len(df_quality)} registros")
    
    if df_quality.empty:
        logger.error("No hay datos después del filtro de calidad")
        return False
    
    # 5. Guardar datos procesados - ruta relativa al proyecto
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = SCRIPT_DIR
    output_dir = os.path.join(PROJECT_ROOT, 'datos_procesados_analisis_integrado_py', 'dustiq')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'dustiq_sr_unified_hours_oct_mar.csv')
    
    try:
        # Convertir índice a naive para evitar problemas de timezone
        df_save = df_quality.copy()
        if df_save.index.tz is not None:
            df_save.index = df_save.index.tz_localize(None)
        
        df_save.to_csv(output_file)
        logger.info(f"✅ Datos guardados exitosamente en: {output_file}")
        
        # Estadísticas finales
        logger.info("=== ESTADÍSTICAS FINALES (HORARIO UNIFICADO) ===")
        logger.info(f"Total registros originales: {len(df_raw)}")
        logger.info(f"Registros período Oct-Mar: {len(df_period)}")
        logger.info(f"Registros horario unificado (13-18h): {len(df_time_filtered)}")
        logger.info(f"Registros finales (calidad): {len(df_quality)}")
        logger.info(f"Rango final: {df_quality.index.min()} a {df_quality.index.max()}")
        logger.info(f"SR promedio: {df_quality['SR_DustIQ'].mean():.3f}%")
        logger.info(f"SR mediana: {df_quality['SR_DustIQ'].median():.3f}%")
        logger.info(f"SR std: {df_quality['SR_DustIQ'].std():.3f}%")
        
        # Verificar rango horario
        logger.info(f"Rango horario final: {df_quality.index.hour.min()}:00 a {df_quality.index.hour.max()}:00")
        
        return True
        
    except Exception as e:
        logger.error(f"Error guardando archivo: {e}")
        return False

def main():
    """Función principal"""
    try:
        success = process_dustiq_unified_hours()
        if success:
            print("\n🎉 ¡PROCESAMIENTO CON HORARIO UNIFICADO COMPLETADO EXITOSAMENTE!")
            print("Los datos de DustIQ con filtro horario unificado (13:00-18:00)")
            print("están disponibles en: dustiq_sr_unified_hours_oct_mar.csv")
            print("\n✅ AHORA TODAS LAS METODOLOGÍAS USAN EL MISMO RANGO HORARIO")
        else:
            print("\n❌ Error en el procesamiento")
            return False
        return True
    except Exception as e:
        logger.error(f"Error en main: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
