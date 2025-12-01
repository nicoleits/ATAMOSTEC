import sys
from typing import List, Callable, Dict, Any
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Agregar el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Importar configuraciones de rutas
from config.paths import (
    TEMP_DATA_COMBINED_PROCESSED_CSV,
    BASE_INPUT_DIR
)

# Importar el sistema de preprocesamiento
from data_processing.data_preprocessor import run_preprocessing

# Importar las funciones de análisis de cada módulo
from analysis.soiling_kit_analyzer import run_analysis as run_soiling_kit
from analysis.dustiq_analyzer import run_analysis as run_dustiq
from analysis.ref_cells_analyzer import run_analysis as run_ref_cells
from analysis.pvstand_analyzer import run_analysis as run_pvstand
from analysis.pvstand_analyzer_solar_noon import run_analysis_solar_noon as run_pvstand_solar_noon
from analysis.pv_glasses_analyzer import run_analysis as run_pv_glasses
from analysis.pv_glasses_analyzer_q25 import ejecutar_analisis_pv_glasses_q25
from analysis.calendar_analyzer import run_analysis as run_calendar
from analysis.analisis_iv600_fixed import run_analysis as run_iv600_filtrado
from analysis.consolidated_weekly_q25_plot import create_consolidated_weekly_q25_plot, create_synchronized_weekly_q25_plot, create_consolidated_weekly_q25_plot_oct_mar
from analysis.statistical_deviation_analyzer import run_analysis as run_statistical_deviation
# from analysis.sr_uncertainty import run_analysis as run_sr_uncertainty  # Módulo eliminado

def run_pv_glasses_q25():
    """Wrapper para el análisis PV Glasses Q25 compatible con el menú principal."""
    try:
        ejecutar_analisis_pv_glasses_q25()
        return True
    except Exception as e:
        logger.error(f"Error en análisis PV Glasses Q25: {e}")
        return False

# Diccionario de opciones: {número: (nombre, función)}
ANALYSIS_OPTIONS = {
    1: ("Soiling Kit", run_soiling_kit),
    2: ("DustIQ", run_dustiq),
    3: ("Celdas de Referencia", run_ref_cells),
    4: ("PVStand", run_pvstand),
    5: ("PVStand - Mediodía Solar", run_pvstand_solar_noon),
    6: ("PV Glasses", run_pv_glasses),
    14: ("PV Glasses Q25 (Cuantil 25)", run_pv_glasses_q25),
    7: ("Calendario", run_calendar),
    8: ("Análisis IV600 Filtrado (sin picos)", run_iv600_filtrado),
    9: ("Gráfico Consolidado Semanal Q25 (sin tendencia)", create_consolidated_weekly_q25_plot),
    13: ("Gráfico Consolidado Sincronizado Q25", create_synchronized_weekly_q25_plot),
    15: ("Gráfico Consolidado Octubre 2024 - Marzo 2025", create_consolidated_weekly_q25_plot_oct_mar),
    12: ("Análisis de Desviaciones Estadísticas", run_statistical_deviation)
    # 16: ("Análisis de Incertidumbre de SR", run_sr_uncertainty)  # Módulo eliminado
}

# Análisis que requieren preprocesamiento específico
ANALYSES_REQUIRING_PREPROCESSING = {
    4: ["temperature_data"],  # PVStand requiere datos de temperatura
    5: ["data_temp"],  # PVStand - Mediodía Solar requiere datos de temperatura
    8: ["iv600_data"]         # IV600 Filtrado requiere datos procesados
}

def print_menu():
    print("\n=== Menú de Análisis de Datos ===")
    for num, (name, _) in ANALYSIS_OPTIONS.items():
        print(f"{num}. {name}")
    print("10. Ejecutar preprocesamiento solamente")
    print("11. Ejecutar todos los análisis")
    print("0. Salir")

def get_user_selection() -> List[int]:
    while True:
        try:
            choice = input("\nSelecciona los números de los análisis a ejecutar (separados por coma, ej: 1,3,5): ")
            if choice.strip() == "0":
                return []
            elif choice.strip() == "11":
                return list(ANALYSIS_OPTIONS.keys())
            elif choice.strip() == "10":
                return [10]  # Código especial para solo preprocesamiento
            
            selected = [int(x.strip()) for x in choice.split(",")]
            valid_options = list(ANALYSIS_OPTIONS.keys()) + [10, 11]
            if all(x in valid_options for x in selected):
                return selected
            else:
                print("Error: Selecciona números válidos del menú.")
        except ValueError:
            print("Error: Ingresa números separados por coma.")

def check_and_run_preprocessing(selected_analyses: List[int], force_reprocess: bool = False) -> bool:
    """Verifica si se necesita preprocesamiento y lo ejecuta si es necesario."""
    
    # Verificar archivos de salida existentes usando rutas centralizadas
    preprocessing_needed = False
    temp_data_exists = os.path.exists(TEMP_DATA_COMBINED_PROCESSED_CSV)  # Archivo generado por notebook
    iv600_data_exists = os.path.exists(os.path.join(BASE_INPUT_DIR, "processed_iv600_data.csv"))
    
    for analysis in selected_analyses:
        if analysis in ANALYSES_REQUIRING_PREPROCESSING:
            required_data = ANALYSES_REQUIRING_PREPROCESSING[analysis]
            for data_type in required_data:
                if data_type == "temperature_data" and not temp_data_exists:
                    preprocessing_needed = True
                    print(f"⚠️  Análisis {ANALYSIS_OPTIONS[analysis][0]} requiere datos de temperatura procesados")
                    print("📋 INSTRUCCIONES:")
                    print("   1. Ejecuta el notebook 'download/download_temp_only.ipynb'")
                    print("   2. Esto generará el archivo data_temp.csv necesario")
                elif data_type == "iv600_data" and not iv600_data_exists:
                    preprocessing_needed = True
                    print(f"⚠️  Análisis {ANALYSIS_OPTIONS[analysis][0]} requiere datos IV600 procesados")
    
    if preprocessing_needed or force_reprocess:
        print("\n🔄 Ejecutando preprocesamiento de datos...")
        success = run_preprocessing(force_reprocess=force_reprocess)
        
        if success:
            print("✅ Preprocesamiento completado exitosamente")
            # Verificar nuevamente si los datos de temperatura están disponibles
            if not os.path.exists(TEMP_DATA_COMBINED_PROCESSED_CSV):
                print("⚠️ ADVERTENCIA: Datos de temperatura aún no disponibles")
                print("📋 Ejecuta el notebook 'download/download_temp_only.ipynb' para generar data_temp.csv")
            return True
        else:
            print("❌ Error en el preprocesamiento")
            user_choice = input("¿Deseas continuar con los análisis disponibles? (s/n): ").lower()
            return user_choice == 's'
    
    print("✅ Todos los datos necesarios están disponibles")
    return True

def run_selected_analyses(selected: List[int]) -> Dict[str, Any]:
    results = {}
    figures = []
    
    for num in selected:
        if num == 9:
            print("\n🔍 Generando gráfico consolidado semanal Q25 (sin tendencia)...")
            try:
                success = create_consolidated_weekly_q25_plot()
                if success:
                    print("✅ Gráfico consolidado generado exitosamente.")
                else:
                    print("❌ No se pudo generar el gráfico consolidado.")
            except Exception as e:
                print(f"❌ Error generando gráfico consolidado: {e}")
            continue
        elif num == 13:
            print("\n🔍 Generando gráfico consolidado sincronizado Q25...")
            try:
                success = create_synchronized_weekly_q25_plot()
                if success:
                    print("✅ Gráfico consolidado sincronizado generado exitosamente.")
                else:
                    print("❌ No se pudo generar el gráfico consolidado sincronizado.")
            except Exception as e:
                print(f"❌ Error generando gráfico consolidado sincronizado: {e}")
            continue
        elif num == 15:
            print("\n🔍 Generando gráfico consolidado Octubre 2024 - Marzo 2025...")
            try:
                success = create_consolidated_weekly_q25_plot_oct_mar()
                if success:
                    print("✅ Gráfico consolidado Octubre-Marzo generado exitosamente.")
                else:
                    print("❌ No se pudo generar el gráfico consolidado Octubre-Marzo.")
            except Exception as e:
                print(f"❌ Error generando gráfico consolidado Octubre-Marzo: {e}")
            continue
        name, func = ANALYSIS_OPTIONS[num]
        print(f"\n🔍 Ejecutando análisis: {name}")
        try:
            analysis_result = func()
            results[name] = analysis_result
            
            # Buscar gráficos generados por el análisis
            analysis_subdir = name.lower().replace(" ", "_").replace("á", "a").replace("é", "e")
            graph_dir = os.path.join("graficos_analisis_integrado_py", analysis_subdir)
            
            if os.path.exists(graph_dir):
                analysis_figures = [os.path.join(graph_dir, f) for f in os.listdir(graph_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
                figures.extend(analysis_figures)
            
            print(f"✅ Análisis '{name}' completado.")
        except Exception as e:
            print(f"❌ Error en '{name}': {e}")
            print(f"Detalles del error: {str(e)}")
    
    return results, figures

def main():
    print("🌞 Bienvenido al Sistema de Análisis de Soiling")
    print("=" * 50)
    
    while True:
        print_menu()
        selected = get_user_selection()
        
        if not selected:
            print("👋 Saliendo del programa.")
            break
        
        # Caso especial: solo preprocesamiento
        if selected == [10]:
            print("\n🔄 Ejecutando preprocesamiento completo...")
            force = input("¿Forzar reprocesamiento de archivos existentes? (s/n): ").lower() == 's'
            success = run_preprocessing(force_reprocess=force)
            
            if success:
                print("✅ Preprocesamiento completado exitosamente")
            else:
                print("❌ Preprocesamiento completado con errores")
            
            continue
        
        # Verificar y ejecutar preprocesamiento si es necesario
        if not check_and_run_preprocessing(selected):
            print("❌ No se puede continuar sin el preprocesamiento necesario.")
            continue
        
        # Ejecutar análisis seleccionados
        print(f"\n🚀 Iniciando {len(selected)} análisis seleccionado(s)...")
        results, figures = run_selected_analyses(selected)

        
        # Preguntar si continuar
        if input("\n🔄 ¿Deseas ejecutar otro análisis? (s/n): ").lower() != 's':
            print("👋 Saliendo del programa.")
            break

if __name__ == "__main__":
    main() 