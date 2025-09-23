import pandas as pd
import numpy as np

def calcular_diferencia(masa_soiled, masa_clean):
    """
    Calcula la diferencia entre masa soiled y clean.
    Si algún valor es 0 o 0.0, retorna 0.
    Multiplica por 100 para convertir de gramos a miligramos (mg).
    Redondea el resultado a 4 cifras decimales para evitar errores de precisión.
    """
    # Verificar si alguno de los valores es 0 o 0.0
    if masa_soiled == 0.0 or masa_clean == 0.0 or masa_soiled == 0 or masa_clean == 0:
        return 0.0
    
    diferencia = masa_soiled - masa_clean
    diferencia_mg = diferencia * 1000  # Convertir a miligramos (1 g = 1000 mg)
    return round(diferencia_mg, 6)  # 6 decimales para mantener al menos 3 cifras significativas

def procesar_masas():
    """
    Procesa el CSV de masas secuencialmente, tomando el primer registro soiled
    y emparejándolo con el siguiente registro clean que encuentre.
    El Período corresponde al de la masa soiled considerada en la sustracción.
    """
    # Cargar el CSV
    df = pd.read_csv('/home/nicole/SR/SOILING/datos/calendario_muestras_seleccionado.csv')
    
    # Convertir las columnas de masa a numéricas, reemplazando 0.0 por 0
    df['Masa A'] = pd.to_numeric(df['Masa A'], errors='coerce').fillna(0)
    df['Masa B'] = pd.to_numeric(df['Masa B'], errors='coerce').fillna(0) 
    df['Masa C'] = pd.to_numeric(df['Masa C'], errors='coerce').fillna(0)
    
    resultados = []
    i = 0
    
    while i < len(df):
        if df.iloc[i]['Estado'] == 'soiled':
            fila_soiled = df.iloc[i]
            
            # Buscar el siguiente registro clean (sin importar fechas o exposición)
            j = i + 1
            fila_clean = None
            
            while j < len(df):
                if df.iloc[j]['Estado'] == 'clean':
                    fila_clean = df.iloc[j]
                    break
                j += 1
            
            if fila_clean is not None:
                # Calcular las sustracciones
                masa_a_diff = calcular_diferencia(fila_soiled['Masa A'], fila_clean['Masa A'])
                masa_b_diff = calcular_diferencia(fila_soiled['Masa B'], fila_clean['Masa B'])
                masa_c_diff = calcular_diferencia(fila_soiled['Masa C'], fila_clean['Masa C'])
                
                resultados.append({
                    'Estructura': fila_soiled['Estructura'],  # Estructura de la masa soiled
                    'Inicio_Exposicion': fila_soiled['Inicio Exposición'],
                    'Fin_Exposicion': fila_soiled['Fin Exposicion'],
                    'Periodo': fila_soiled['Periodo'],  # PERÍODO DE LA MASA SOILED
                    'Exposicion_dias': fila_soiled['Exposición'],
                    'Diferencia_Masa_A_mg': masa_a_diff,
                    'Diferencia_Masa_B_mg': masa_b_diff,
                    'Diferencia_Masa_C_mg': masa_c_diff,
                    'Fila_Soiled': i + 2,  # +2 porque el índice empieza en 0 y la primera fila es header
                    'Fila_Clean': j + 2,
                    'Masa_A_Soiled_g': fila_soiled['Masa A'],
                    'Masa_A_Clean_g': fila_clean['Masa A'],
                    'Masa_B_Soiled_g': fila_soiled['Masa B'],
                    'Masa_B_Clean_g': fila_clean['Masa B'],
                    'Masa_C_Soiled_g': fila_soiled['Masa C'],
                    'Masa_C_Clean_g': fila_clean['Masa C']
                })
                
                # Saltar al índice después del clean procesado para evitar reutilizarlo
                i = j
        i += 1
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar resultados
    print("Resultados del procesamiento de masas:")
    print("=" * 60)
    for idx, row in df_resultados.iterrows():
        print(f"\nPar {idx + 1}:")
        print(f"  Estructura: {row['Estructura']} (de la muestra soiled)")
        print(f"  Período: {row['Periodo']} (de la muestra soiled)")
        print(f"  Exposición: {row['Exposicion_dias']} días")
        print(f"  Fechas: {row['Inicio_Exposicion']} a {row['Fin_Exposicion']}")
        print(f"  Filas procesadas: {row['Fila_Soiled']} (soiled) - {row['Fila_Clean']} (clean)")
        print(f"  Masas soiled (g): A={row['Masa_A_Soiled_g']:.4f}, B={row['Masa_B_Soiled_g']:.4f}, C={row['Masa_C_Soiled_g']:.4f}")
        print(f"  Masas clean (g):  A={row['Masa_A_Clean_g']:.4f}, B={row['Masa_B_Clean_g']:.4f}, C={row['Masa_C_Clean_g']:.4f}")
        print(f"  Diferencias de masa (mg):")
        
        # Mostrar si alguna diferencia es 0 por tener valores 0
        masa_a_zero = row['Masa_A_Soiled_g'] == 0.0 or row['Masa_A_Clean_g'] == 0.0
        masa_b_zero = row['Masa_B_Soiled_g'] == 0.0 or row['Masa_B_Clean_g'] == 0.0
        masa_c_zero = row['Masa_C_Soiled_g'] == 0.0 or row['Masa_C_Clean_g'] == 0.0
        
        print(f"    Masa A: {row['Diferencia_Masa_A_mg']:.6f} mg" + (" (=0 por valor 0)" if masa_a_zero else ""))
        print(f"    Masa B: {row['Diferencia_Masa_B_mg']:.6f} mg" + (" (=0 por valor 0)" if masa_b_zero else ""))
        print(f"    Masa C: {row['Diferencia_Masa_C_mg']:.6f} mg" + (" (=0 por valor 0)" if masa_c_zero else ""))
    
    # Guardar resultados en CSV con formato limpio
    archivo_salida = '/home/nicole/SR/SOILING/resultados_diferencias_masas.csv'
    
    # Redondear columnas de diferencias a 6 decimales y masas a 4 decimales
    columnas_diferencias = ['Diferencia_Masa_A_mg', 'Diferencia_Masa_B_mg', 'Diferencia_Masa_C_mg']
    columnas_masas = ['Masa_A_Soiled_g', 'Masa_A_Clean_g', 'Masa_B_Soiled_g', 'Masa_B_Clean_g',
                     'Masa_C_Soiled_g', 'Masa_C_Clean_g']
    
    # Redondear diferencias a 6 decimales para mantener cifras significativas
    for col in columnas_diferencias:
        if col in df_resultados.columns:
            df_resultados[col] = df_resultados[col].round(6)
    
    # Redondear masas a 4 decimales (son valores más grandes)
    for col in columnas_masas:
        if col in df_resultados.columns:
            df_resultados[col] = df_resultados[col].round(4)
    
    df_resultados.to_csv(archivo_salida, index=False)
    print(f"\nResultados guardados en: {archivo_salida}")
    
    # Mostrar resumen estadístico
    print(f"\nResumen:")
    print(f"Total de pares soiled/clean procesados: {len(df_resultados)}")
    print(f"Estructuras analizadas: {df_resultados['Estructura'].nunique()}")
    print(f"Períodos incluidos: {', '.join(df_resultados['Periodo'].unique())}")
    
    # Verificar que todos los períodos corresponden a las muestras soiled
    print(f"\nVerificación:")
    print("Todos los períodos mostrados corresponden a las muestras soiled consideradas en la sustracción.")
    
    return df_resultados

if __name__ == "__main__":
    print("Iniciando análisis de masas...")
    print("Recuerda activar el entorno virtual antes de ejecutar.")
    print("El Período mostrado corresponde SIEMPRE a la muestra soiled.")
    print("-" * 50)
    
    try:
        resultados = procesar_masas()
        print("\n¡Análisis completado exitosamente!")
    except FileNotFoundError:
        print("Error: No se pudo encontrar el archivo CSV.")
        print("Verifica que el archivo esté en: /home/nicole/SR/SOILING/datos/calendario_muestras_seleccionado.csv")
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")