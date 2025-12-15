#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ejecutar el análisis de PV Glasses
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
    """Función principal para ejecutar el análisis de PV Glasses"""
    logger.info("=" * 80)
    logger.info("INICIANDO ANÁLISIS DE PV GLASSES")
    logger.info("=" * 80)
    
    try:
        # Importar la función de análisis
        from analysis.pv_glasses_analyzer import run_analysis
        
        logger.info("\n🚀 Ejecutando análisis de PV Glasses...")
        logger.info("   Esto puede tomar varios minutos...")
        
        # Ejecutar análisis
        result = run_analysis()
        
        if result:
            logger.info("\n✅ ANÁLISIS DE PV GLASSES COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            
            # Mostrar ubicación de archivos generados
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            logger.info("\n📄 Archivos CSV generados en:")
            logger.info(f"   {os.path.join(base_dir, 'datos_procesados_analisis_integrado_py', 'pv_glasses')}")
            
            logger.info("\n🖼️  Gráficos generados en:")
            logger.info(f"   {os.path.join(base_dir, 'graficos_analisis_integrado_py', 'pv_glasses')}")
            
            return True
        else:
            logger.error("\n❌ El análisis de PV Glasses falló")
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

