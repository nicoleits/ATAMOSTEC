#!/usr/bin/env python3
"""
Script para ejecutar el análisis de RefCells para mediodía solar.
Este script ejecuta el análisis de celdas de referencia filtrado por horario de mediodía solar.
"""

import sys
import os

# Agregar el directorio raíz del proyecto al path (subir un nivel desde analysis/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from analysis.ref_cells_analyzer import run_analysis_solar_noon

if __name__ == "__main__":
    print("=" * 60)
    print("Ejecutando Análisis de RefCells para Mediodía Solar")
    print("=" * 60)
    
    # Ejecutar con ventana de 2.5 horas (default)
    # Puedes cambiar este valor si necesitas una ventana diferente
    hours_window = 2.5
    
    print(f"\nVentana horaria: ±{hours_window} horas alrededor del mediodía solar")
    print("\nIniciando análisis...\n")
    
    success = run_analysis_solar_noon(hours_window=hours_window)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Análisis completado exitosamente")
        print("=" * 60)
        print("\nLos resultados se guardaron en:")
        print("  - CSV: datos_procesados_analisis_integrado_py/ref_cells/mediodia_solar/")
        print("  - Gráficos: graficos_analisis_integrado_py/ref_cells/mediodia_solar/")
    else:
        print("\n" + "=" * 60)
        print("❌ El análisis encontró errores. Revisa los logs arriba.")
        print("=" * 60)
        sys.exit(1)

