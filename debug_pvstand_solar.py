#!/usr/bin/env python3
"""
Script de diagnóstico para el problema del filtro de medio día solar en PVStand
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Agregar el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import paths, settings
from utils.solar_time import UtilsMedioDiaSolar

def debug_pvstand_solar_filter():
    """Diagnóstico del filtro de medio día solar para PVStand"""
    
    print("=== DIAGNÓSTICO FILTRO MEDIO DÍA SOLAR PVSTAND ===")
    
    # 1. Verificar archivos de datos
    pv_iv_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_IV_DATA_FILENAME)
    temperature_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_TEMP_DATA_FILENAME)
    
    print(f"\n1. VERIFICACIÓN DE ARCHIVOS:")
    print(f"   PVStand IV: {pv_iv_data_filepath} - {'✅ Existe' if os.path.exists(pv_iv_data_filepath) else '❌ No existe'}")
    print(f"   Temperatura: {temperature_data_filepath} - {'✅ Existe' if os.path.exists(temperature_data_filepath) else '❌ No existe'}")
    
    # 2. Cargar datos PVStand
    print(f"\n2. CARGA DE DATOS PVSTAND:")
    if os.path.exists(pv_iv_data_filepath):
        try:
            use_cols_iv = ['timestamp', '_measurement', 'Imax', 'Pmax', 'Umax']
            df_pvstand_raw_data = pd.read_csv(
                pv_iv_data_filepath, 
                usecols=lambda c: c in use_cols_iv or c.startswith('timestamp')
            )
            
            time_col_actual = [col for col in df_pvstand_raw_data.columns if col.startswith('timestamp')]
            if time_col_actual[0] != 'timestamp':
                df_pvstand_raw_data.rename(columns={time_col_actual[0]: 'timestamp'}, inplace=True)

            df_pvstand_raw_data['timestamp'] = pd.to_datetime(
                df_pvstand_raw_data['timestamp'], 
                errors='coerce'
            )
            df_pvstand_raw_data.dropna(subset=['timestamp'], inplace=True)
            df_pvstand_raw_data.set_index('timestamp', inplace=True)
            
            print(f"   Datos cargados: {len(df_pvstand_raw_data)} filas")
            print(f"   Rango de fechas: {df_pvstand_raw_data.index.min()} a {df_pvstand_raw_data.index.max()}")
            print(f"   Zona horaria: {df_pvstand_raw_data.index.tz}")
            
            # Asegurar zona horaria UTC
            if df_pvstand_raw_data.index.tz is None:
                df_pvstand_raw_data.index = df_pvstand_raw_data.index.tz_localize('UTC')
            elif df_pvstand_raw_data.index.tz != timezone.utc:
                df_pvstand_raw_data.index = df_pvstand_raw_data.index.tz_convert('UTC')
            
            print(f"   Zona horaria después de corrección: {df_pvstand_raw_data.index.tz}")
            
        except Exception as e:
            print(f"   ❌ Error cargando datos: {e}")
            return
    
    # 3. Verificar configuración de fechas
    print(f"\n3. CONFIGURACIÓN DE FECHAS:")
    filter_start_date_str = settings.PVSTAND_ANALYSIS_START_DATE_STR
    filter_end_date_str = settings.PVSTAND_ANALYSIS_END_DATE_STR
    
    print(f"   Fecha inicio análisis: {filter_start_date_str}")
    print(f"   Fecha fin análisis: {filter_end_date_str}")
    
    try:
        start_date_dt = pd.Timestamp(filter_start_date_str, tz='UTC')
        end_date_dt = pd.Timestamp(filter_end_date_str, tz='UTC')
        print(f"   Fechas parseadas correctamente")
    except Exception as e:
        print(f"   ❌ Error parseando fechas: {e}")
        return
    
    # 4. Filtrar por rango de fechas
    print(f"\n4. FILTRADO POR RANGO DE FECHAS:")
    start_date_only = pd.Timestamp(start_date_dt.date(), tz='UTC')
    end_date_only = pd.Timestamp(end_date_dt.date(), tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    df_filtered_dates = df_pvstand_raw_data[(df_pvstand_raw_data.index >= start_date_only) & (df_pvstand_raw_data.index <= end_date_only)]
    print(f"   Datos después de filtro de fechas: {len(df_filtered_dates)} filas")
    print(f"   Rango filtrado: {df_filtered_dates.index.min()} a {df_filtered_dates.index.max()}")
    
    if df_filtered_dates.empty:
        print(f"   ❌ No hay datos en el rango de fechas especificado")
        return
    
    # 5. Verificar configuración solar
    print(f"\n5. CONFIGURACIÓN SOLAR:")
    print(f"   Latitud: {settings.SITE_LATITUDE}°")
    print(f"   Longitud: {settings.SITE_LONGITUDE}°")
    print(f"   Altitud: {settings.SITE_ALTITUDE} m")
    print(f"   Zona horaria local: {settings.DUSTIQ_LOCAL_TIMEZONE_STR}")
    
    # 6. Calcular intervalos de medio día solar
    print(f"\n6. CÁLCULO DE INTERVALOS SOLARES:")
    try:
        # Obtener rango de fechas del DataFrame filtrado
        start_date = df_filtered_dates.index.min().date()
        end_date = df_filtered_dates.index.max().date()
        
        print(f"   Rango para cálculo solar: {start_date} a {end_date}")
        
        # Inicializar utilidades de medio día solar
        solar_utils = UtilsMedioDiaSolar(
            datei=start_date,
            datef=end_date,
            freq='D',
            inter=int(2.5 * 2 * 60),  # ±2.5 horas = 300 minutos
            tz_local_str=settings.DUSTIQ_LOCAL_TIMEZONE_STR,
            lat=settings.SITE_LATITUDE,
            lon=settings.SITE_LONGITUDE,
            alt=settings.SITE_ALTITUDE
        )
        
        # Obtener intervalos de medio día solar
        solar_intervals_df = solar_utils.msd()
        
        print(f"   Intervalos solares calculados: {len(solar_intervals_df)} días")
        
        if solar_intervals_df.empty:
            print(f"   ❌ No se pudieron calcular intervalos solares")
            return
        
        # Mostrar algunos intervalos de ejemplo
        print(f"   Primeros 3 intervalos:")
        for i, (_, row) in enumerate(solar_intervals_df.head(3).iterrows()):
            start_time = pd.Timestamp(row[0], tz='UTC')
            end_time = pd.Timestamp(row[1], tz='UTC')
            print(f"     Día {i+1}: {start_time} - {end_time}")
        
        # 7. Aplicar filtro de medio día solar
        print(f"\n7. APLICACIÓN DEL FILTRO SOLAR:")
        
        # Crear máscara para filtrar por medio día solar
        mask = pd.Series(False, index=df_filtered_dates.index)
        
        # Aplicar cada intervalo de medio día solar
        for _, row in solar_intervals_df.iterrows():
            start_time = pd.Timestamp(row[0], tz='UTC')
            end_time = pd.Timestamp(row[1], tz='UTC')
            
            # Aplicar máscara para este intervalo
            interval_mask = (df_filtered_dates.index >= start_time) & (df_filtered_dates.index <= end_time)
            mask = mask | interval_mask
        
        filtered_df = df_filtered_dates[mask]
        print(f"   Datos después de filtro solar: {len(filtered_df)} filas")
        print(f"   Porcentaje retenido: {len(filtered_df)/len(df_filtered_dates)*100:.1f}%")
        
        if filtered_df.empty:
            print(f"   ❌ PROBLEMA: DataFrame vacío después del filtro solar")
            print(f"   Posibles causas:")
            print(f"     - Los datos no coinciden con los intervalos solares")
            print(f"     - Problema de zona horaria")
            print(f"     - Los datos están en un rango horario diferente")
        else:
            print(f"   ✅ Filtro solar aplicado correctamente")
            print(f"   Rango final: {filtered_df.index.min()} a {filtered_df.index.max()}")
        
    except Exception as e:
        print(f"   ❌ Error en cálculo solar: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== FIN DEL DIAGNÓSTICO ===")

if __name__ == "__main__":
    debug_pvstand_solar_filter()