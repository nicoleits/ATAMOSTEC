#!/usr/bin/env python3
"""
Script de ejecución para análisis PV Glasses con Cuantil 25 (Q25)
================================================================

Este script ejecuta el análisis de PV Glasses usando cuantil 25 en lugar de promedios,
lo que lo hace más robusto ante outliers y datos anómalos como el problema del 2025-08-07.

Uso:
    python run_pv_glasses_q25.py

Características del análisis Q25:
- Usa cuantil 25 en lugar de promedios
- No aplica filtro IQR por defecto (más permisivo)
- Mantiene datos que serían eliminados como outliers
- Genera gráficos separados con sufijo Q25

Autor: Sistema de Análisis de Soiling
Fecha: 2025-01-13
"""

import os
import sys
import logging
from pathlib import Path

# Agregar el directorio de análisis al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pv_glasses_q25.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para ejecutar el análisis Q25."""
    logger.info("🌞 INICIANDO ANÁLISIS PV GLASSES CON CUANTIL 25 (Q25)")
    logger.info("=" * 60)
    
    try:
        # Importar el módulo de análisis Q25
        from analysis.pv_glasses_analyzer_q25 import ejecutar_analisis_pv_glasses_q25
        
        # Definir rutas
        base_dir = Path(__file__).parent
        
        rutas = {
            'raw_data': base_dir / "datos" / "raw_pv_glasses_data.csv",
            'calendario': base_dir / "datos" / "20241114 Calendario toma de muestras soiling.xlsx",
            'output_csv': base_dir / "datos_procesados_analisis_integrado_py",
            'output_graphs': base_dir / "graficos_analisis_integrado_py"
        }
        
        # Verificar archivos de entrada
        archivos_requeridos = ['raw_data', 'calendario']
        for nombre, ruta in rutas.items():
            if nombre in archivos_requeridos:
                if not ruta.exists():
                    logger.error(f"❌ Archivo requerido no encontrado: {ruta}")
                    return False
                else:
                    logger.info(f"✅ Archivo encontrado: {ruta.name}")
        
        # Crear directorios de salida
        for nombre, ruta in rutas.items():
            if nombre.startswith('output_'):
                ruta.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Directorio preparado: {ruta}")
        
        logger.info("\n🚀 Ejecutando análisis PV Glasses Q25...")
        logger.info("📊 Características del análisis:")
        logger.info("   • Usa cuantil 25 (Q25) en lugar de promedios")
        logger.info("   • No aplica filtro IQR (más permisivo con outliers)")
        logger.info("   • Incluye datos del 2025-08-07 que se perdían antes")
        logger.info("   • Genera gráficos separados con sufijo Q25")
        
        # Ejecutar análisis
        ejecutar_analisis_pv_glasses_q25(
            raw_data_path=str(rutas['raw_data']),
            calendario_path=str(rutas['calendario']),
            output_csv_dir=str(rutas['output_csv']),
            output_graph_dir=str(rutas['output_graphs'])
        )
        
        logger.info("\n✅ ANÁLISIS PV GLASSES Q25 COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        
        # Mostrar archivos generados
        q25_dir = rutas['output_csv'] / "pv_glasses_q25"
        graph_dir = rutas['output_graphs'] / "pv_glasses_q25"
        
        logger.info("📄 Archivos CSV generados:")
        if q25_dir.exists():
            for archivo in q25_dir.glob("*.csv"):
                logger.info(f"   • {archivo.name}")
        
        logger.info("🖼️  Gráficos generados:")
        if graph_dir.exists():
            for archivo in graph_dir.glob("*.png"):
                logger.info(f"   • {archivo.name}")
        
        logger.info("\n💡 Comparar con análisis tradicional:")
        logger.info("   • Gráficos tradicionales: graficos_analisis_integrado_py/pv_glasses/")
        logger.info("   • Gráficos Q25: graficos_analisis_integrado_py/pv_glasses_q25/")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Error de importación: {e}")
        logger.error("Asegúrate de que el módulo pv_glasses_analyzer_q25 esté disponible")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error durante el análisis: {e}", exc_info=True)
        return False

def mostrar_ayuda():
    """Muestra información de ayuda sobre el script."""
    print("""
🌞 ANÁLISIS PV GLASSES CON CUANTIL 25 (Q25)
==========================================

Este script implementa un análisis alternativo de PV Glasses usando cuantil 25
en lugar de promedios, lo que lo hace más robusto ante outliers.

VENTAJAS DEL ANÁLISIS Q25:
• Más resistente a outliers y datos anómalos
• No pierde datos como el 2025-08-07 que se eliminaban antes
• Proporciona una visión más conservadora del rendimiento
• Útil para análisis de peor caso (worst-case scenario)

ARCHIVOS REQUERIDOS:
• datos/raw_pv_glasses_data.csv
• datos/20241114 Calendario toma de muestras soiling.xlsx

SALIDAS GENERADAS:
• CSV procesados en: datos_procesados_analisis_integrado_py/pv_glasses_q25/
• Gráficos en: graficos_analisis_integrado_py/pv_glasses_q25/

USO:
    python run_pv_glasses_q25.py
    python run_pv_glasses_q25.py --help

COMPARACIÓN:
• Análisis tradicional (promedio + IQR): puede perder datos anómalos
• Análisis Q25: conserva más datos, visión más conservadora
    """)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        mostrar_ayuda()
    else:
        success = main()
        sys.exit(0 if success else 1)
