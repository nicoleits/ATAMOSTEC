#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar el análisis del calendario de muestras de PV Glasses
"""

import os
import sys
import logging

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para ejecutar el análisis del calendario"""
    logger.info("=" * 80)
    logger.info("INICIANDO ANÁLISIS DEL CALENDARIO DE MUESTRAS")
    logger.info("=" * 80)
    
    # Definir rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calendario_file = os.path.join(base_dir, "datos", "calendario", "20241114 Calendario toma de muestras soiling.xlsx")
    output_csv_dir = os.path.join(base_dir, "datos_procesados_analisis_integrado_py")
    
    # Verificar que el archivo existe
    if not os.path.exists(calendario_file):
        logger.error(f"❌ Archivo no encontrado: {calendario_file}")
        return False
    
    logger.info(f"📄 Archivo de calendario: {calendario_file}")
    logger.info(f"📁 Directorio de salida: {output_csv_dir}")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_csv_dir, exist_ok=True)
    
    try:
        # Importar la función de análisis
        from analysis.pv_glasses_analyzer import analizar_calendario_muestras
        
        # Ejecutar análisis
        logger.info("\n🚀 Ejecutando análisis del calendario...")
        df_calendario, df_resultado_fija_rc = analizar_calendario_muestras(
            file_path=calendario_file,
            output_csv_dir=output_csv_dir,
            sheet_name="Hoja1"
        )
        
        if df_calendario is not None:
            logger.info("\n✅ ANÁLISIS DEL CALENDARIO COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            
            # Mostrar archivos generados
            logger.info("\n📄 Archivos CSV generados:")
            calendario_csv = os.path.join(output_csv_dir, "calendario_muestras_seleccionado.csv")
            fija_rc_csv = os.path.join(output_csv_dir, "calendario_fija_rc_por_periodo.csv")
            
            if os.path.exists(calendario_csv):
                logger.info(f"   ✅ {calendario_csv}")
            else:
                logger.warning(f"   ⚠️  No se generó: {calendario_csv}")
            
            if os.path.exists(fija_rc_csv):
                logger.info(f"   ✅ {fija_rc_csv}")
            else:
                logger.warning(f"   ⚠️  No se generó: {fija_rc_csv}")
            
            return True
        else:
            logger.error("\n❌ El análisis del calendario falló")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Error de importación: {e}")
        logger.error("Asegúrate de que el módulo 'analysis.pv_glasses_analyzer' esté disponible")
        return False
    except Exception as e:
        logger.error(f"❌ Error durante el análisis: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

