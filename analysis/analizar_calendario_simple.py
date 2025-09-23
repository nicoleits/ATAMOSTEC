#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versión simplificada para analizar el calendario de soiling
Sin tanto logging, con salida clara y entendible
"""

import pandas as pd
import os
from datetime import datetime

def analizar_calendario_simple(file_path: str, output_dir: str):
    """
    Versión simplificada para analizar el calendario sin tanto logging
    """
    print(f"📊 Analizando archivo: {file_path}")
    print("-" * 60)
    
    # 1. Leer Excel
    try:
        df = pd.read_excel(file_path, sheet_name='Hoja1')
        print(f"✅ Archivo leído exitosamente:")
        print(f"   - Filas: {df.shape[0]}")
        print(f"   - Columnas: {df.shape[1]}")
        print()
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return None
    
    # 2. Mostrar columnas disponibles
    print(f"📋 Columnas encontradas:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    print()
    
    # 3. Mostrar primeras filas para entender la estructura
    print("📄 Primeras 5 filas del archivo:")
    print(df.head().to_string())
    print()
    
    # 4. Limpiar y procesar fechas
    df.columns = df.columns.str.strip()
    
    # Renombrar columnas comunes
    if 'Fecha medición' in df.columns:
        df.rename(columns={'Fecha medición': 'Fin Exposicion'}, inplace=True)
        print("🔄 Renombrado: 'Fecha medición' → 'Fin Exposicion'")
    
    # Convertir fechas
    cols_fechas = ['Inicio Exposición', 'Fin Exposicion']
    for col in cols_fechas:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"📅 Convertida a fecha: {col}")
    print()
    
    # 5. Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. Guardar CSV completo
    csv_path = os.path.join(output_dir, 'calendario_muestras_completo.csv')
    df.to_csv(csv_path, index=False, date_format='%Y-%m-%d')
    print(f"💾 Guardado CSV completo: {csv_path}")
    
    # 7. Analizar estructura "Estructura" si existe
    if 'Estructura' in df.columns:
        print("\n🏗️ Análisis por Estructura:")
        estructuras = df['Estructura'].fillna('Sin especificar').value_counts()
        for estructura, count in estructuras.items():
            print(f"   - {estructura}: {count} registros")
        
        # 8. Filtrar "Fija a RC"
        df_fija = df[df['Estructura'].fillna('').str.strip() == 'Fija a RC'].copy()
        if not df_fija.empty:
            print(f"\n🎯 Análisis específico de 'Fija a RC':")
            print(f"   - Total registros: {len(df_fija)}")
            
            # Mostrar algunas filas de ejemplo
            print("\n📋 Ejemplos de registros 'Fija a RC':")
            cols_importantes = ['Inicio Exposición', 'Fin Exposicion', 'Periodo', 'Masa A', 'Masa B', 'Masa C']
            cols_existentes = [col for col in cols_importantes if col in df_fija.columns]
            print(df_fija[cols_existentes].head().to_string())
            
            # 9. Agrupar por periodo si existe
            if 'Periodo' in df_fija.columns:
                print(f"\n📊 Análisis por Periodo:")
                periodos = df_fija['Periodo'].fillna('Sin periodo').value_counts()
                for periodo, count in periodos.items():
                    print(f"   - {periodo}: {count} registros")
                
                # Agrupar fechas por periodo
                if 'Fin Exposicion' in df_fija.columns:
                    print(f"\n📅 Fechas de fin de exposición por periodo:")
                    df_fija_clean = df_fija.dropna(subset=['Periodo', 'Fin Exposicion'])
                    
                    if not df_fija_clean.empty:
                        resultado = df_fija_clean.groupby('Periodo')['Fin Exposicion'].apply(
                            lambda dates: sorted(list(dates.dropna().unique()))
                        ).reset_index()
                        resultado.rename(columns={'Fin Exposicion': 'Fechas_Fin_Exposicion'}, inplace=True)
                        
                        for _, row in resultado.iterrows():
                            periodo = row['Periodo']
                            fechas = row['Fechas_Fin_Exposicion']
                            print(f"   {periodo}:")
                            for fecha in fechas:
                                print(f"     - {fecha.strftime('%Y-%m-%d')}")
                        
                        # Guardar resultado
                        csv_fija_path = os.path.join(output_dir, 'calendario_fija_rc_por_periodo.csv')
                        resultado_save = resultado.copy()
                        resultado_save['Fechas_Fin_Exposicion'] = resultado_save['Fechas_Fin_Exposicion'].apply(
                            lambda fechas: [f.strftime('%Y-%m-%d') for f in fechas]
                        ).astype(str)
                        resultado_save.to_csv(csv_fija_path, index=False)
                        print(f"\n💾 Guardado análisis 'Fija a RC': {csv_fija_path}")
        else:
            print("\n⚠️ No se encontraron registros con Estructura 'Fija a RC'")
    else:
        print("\n⚠️ No se encontró la columna 'Estructura'")
    
    # 10. Resumen final
    print(f"\n" + "="*60)
    print("📊 RESUMEN FINAL:")
    print(f"   - Total registros procesados: {len(df)}")
    print(f"   - Archivos generados en: {output_dir}")
    print(f"   - Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return df

def main():
    """Función principal"""
    file_path = "/home/nicole/SR/SOILING/datos/20241114 Calendario toma de muestras soiling.xlsx"
    output_dir = "/home/nicole/SR/SOILING/datos_procesados_analisis_integrado_py/calendario"
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"❌ Error: No se encuentra el archivo {file_path}")
        return
    
    # Ejecutar análisis
    df = analizar_calendario_simple(file_path, output_dir)
    
    if df is not None:
        print("\n🎉 Análisis completado exitosamente!")
    else:
        print("\n❌ Error en el análisis")

if __name__ == "__main__":
    main()
