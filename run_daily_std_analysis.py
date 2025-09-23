#!/usr/bin/env python3
"""
Script de ejecución para el análisis de desviación estándar diaria de SR
======================================================================

Este script ejecuta el análisis de desviación estándar diaria de los Soiling Ratios
para todas las metodologías disponibles (DustIQ, RefCells, PVStand, Soiling Kit, IV600).

Uso:
    python run_daily_std_analysis.py

Autor: Asistente IA
Fecha: 2025-01-27
"""

import os
import sys

# Agregar el directorio de análisis al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

from daily_sr_std_analysis import main

if __name__ == "__main__":
    print("=== EJECUTANDO ANÁLISIS DE DESVIACIÓN ESTÁNDAR DIARIA DE SR ===")
    print("Este análisis calculará la desviación estándar diaria de los Soiling Ratios")
    print("para cada metodología y generará gráficos comparativos.\n")
    
    try:
        success = main()
        if success:
            print("\n✅ Análisis completado exitosamente!")
            print("Revisa los archivos generados en:")
            print("  - Gráficos: graficos_analisis_integrado_py/analisis_varianza_intraday/")
            print("  - Estadísticas: datos_procesados_analisis_integrado_py/statistical_deviations/")
        else:
            print("\n❌ El análisis no se pudo completar. Revisa los logs para más detalles.")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
