#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar el análisis de PV Stand
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
    """Función principal para ejecutar el análisis de PV Stand"""
    logger.info("=" * 80)
    logger.info("INICIANDO ANÁLISIS DE PV STAND")
    logger.info("=" * 80)
    
    try:
        # Importar la función de análisis y las rutas
        from analysis.pvstand_analyzer import run_analysis
        from config import paths
        
        # Verificar que los archivos necesarios existan
        pv_iv_file = paths.PVSTAND_IV_DATA_FILE
        temp_file = paths.PVSTAND_TEMP_DATA_FILE
        
        logger.info(f"📄 Archivo de datos IV: {pv_iv_file}")
        logger.info(f"🌡️  Archivo de datos de temperatura: {temp_file}")
        
        # Verificar existencia de archivos
        if not os.path.exists(pv_iv_file):
            logger.error(f"❌ Archivo de datos IV no encontrado: {pv_iv_file}")
            logger.error("   Por favor, asegúrate de que el archivo existe antes de ejecutar el análisis.")
            return False
        
        if not os.path.exists(temp_file):
            logger.error(f"❌ Archivo de datos de temperatura no encontrado: {temp_file}")
            logger.error("   Por favor, asegúrate de que el archivo existe antes de ejecutar el análisis.")
            logger.info("   Nota: Este archivo puede requerir preprocesamiento previo.")
            return False
        
        logger.info("✅ Todos los archivos de entrada encontrados")
        logger.info("\n🚀 Ejecutando análisis de PV Stand...")
        logger.info("   Esto puede tomar varios minutos...")
        
        # Ejecutar análisis
        success = run_analysis()
        
        if success:
            logger.info("\n✅ ANÁLISIS DE PV STAND COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            
            # Mostrar ubicación de archivos generados
            logger.info("\n📄 Archivos CSV generados en:")
            logger.info(f"   {paths.PVSTAND_OUTPUT_SUBDIR_CSV}")
            
            logger.info("\n🖼️  Gráficos generados en:")
            logger.info(f"   {paths.PVSTAND_OUTPUT_SUBDIR_GRAPH}")
            
            return True
        else:
            logger.error("\n❌ El análisis de PV Stand falló")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Error de importación: {e}")
        logger.error("Asegúrate de que el módulo 'analysis.pvstand_analyzer' esté disponible")
        return False
    except Exception as e:
        logger.error(f"❌ Error durante el análisis: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

