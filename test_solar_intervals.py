#!/usr/bin/env python3
"""
Script de prueba para verificar el cálculo de intervalos solares
"""

import os
import sys
import pandas as pd
from datetime import datetime, timezone

# Agregar el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import settings
from utils.solar_time import UtilsMedioDiaSolar

def test_solar_intervals():
    """Prueba el cálculo de intervalos solares"""
    
    print("=== PRUEBA DE CÁLCULO DE INTERVALOS SOLARES ===")
    
    # Configuración
    start_date = "2024-08-01"
    end_date = "2024-08-05"  # Solo 5 días para prueba
    interval_minutes = 300  # ±2.5 horas = 300 minutos
    
    print(f"Rango de fechas: {start_date} a {end_date}")
    print(f"Intervalo: ±{interval_minutes/2} minutos")
    print(f"Latitud: {settings.SITE_LATITUDE}°")
    print(f"Longitud: {settings.SITE_LONGITUDE}°")
    print(f"Altitud: {settings.SITE_ALTITUDE} m")
    print(f"Zona horaria local: {settings.DUSTIQ_LOCAL_TIMEZONE_STR}")
    
    try:
        # Crear instancia de UtilsMedioDiaSolar
        solar_utils = UtilsMedioDiaSolar(
            datei=start_date,
            datef=end_date,
            freq='D',
            inter=interval_minutes,
            tz_local_str=settings.DUSTIQ_LOCAL_TIMEZONE_STR,
            lat=settings.SITE_LATITUDE,
            lon=settings.SITE_LONGITUDE,
            alt=settings.SITE_ALTITUDE
        )
        
        print(f"\n✅ UtilsMedioDiaSolar creado exitosamente")
        
        # Calcular intervalos
        solar_intervals_df = solar_utils.msd()
        
        print(f"\n📊 RESULTADOS:")
        print(f"Intervalos calculados: {len(solar_intervals_df)}")
        
        if solar_intervals_df.empty:
            print("❌ No se calcularon intervalos")
            return False
        
        print(f"\n📅 INTERVALOS CALCULADOS:")
        for i, (_, row) in enumerate(solar_intervals_df.iterrows()):
            start_time = pd.Timestamp(row[0], tz='UTC')
            end_time = pd.Timestamp(row[1], tz='UTC')
            print(f"  Día {i+1}: {start_time} - {end_time}")
        
        # Verificar que los intervalos son razonables
        print(f"\n🔍 VERIFICACIÓN DE INTERVALOS:")
        for i, (_, row) in enumerate(solar_intervals_df.iterrows()):
            start_time = pd.Timestamp(row[0], tz='UTC')
            end_time = pd.Timestamp(row[1], tz='UTC')
            
            # Verificar que el intervalo es de aproximadamente 5 horas (±2.5)
            interval_duration = end_time - start_time
            expected_duration = pd.Timedelta(minutes=interval_minutes)
            
            print(f"  Día {i+1}:")
            print(f"    Duración: {interval_duration}")
            print(f"    Esperado: {expected_duration}")
            print(f"    Diferencia: {abs(interval_duration - expected_duration)}")
            
            # Verificar que el intervalo está en horario de día
            start_hour = start_time.hour
            end_hour = end_time.hour
            
            print(f"    Horario: {start_hour:02d}:00 - {end_hour:02d}:00")
            
            if 8 <= start_hour <= 18 and 8 <= end_hour <= 18:
                print(f"    ✅ Horario razonable")
            else:
                print(f"    ⚠️ Horario fuera del rango esperado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en el cálculo: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_overlap():
    """Prueba si los datos de PVStand se solapan con los intervalos solares"""
    
    print(f"\n=== PRUEBA DE SOLAPAMIENTO CON DATOS PVSTAND ===")
    
    # Cargar algunos datos de PVStand
    pv_data_file = os.path.join("datos", "raw_pvstand_iv_data.csv")
    
    if not os.path.exists(pv_data_file):
        print(f"❌ Archivo de datos no encontrado: {pv_data_file}")
        return False
    
    # Cargar datos de ejemplo
    df_sample = pd.read_csv(pv_data_file, nrows=1000)
    
    # Procesar timestamps
    time_col = [col for col in df_sample.columns if col.startswith('timestamp')][0]
    df_sample[time_col] = pd.to_datetime(df_sample[time_col])
    
    print(f"Datos de ejemplo cargados: {len(df_sample)} filas")
    print(f"Rango de datos: {df_sample[time_col].min()} a {df_sample[time_col].max()}")
    
    # Verificar distribución horaria
    df_sample['hour'] = df_sample[time_col].dt.hour
    hour_distribution = df_sample['hour'].value_counts().sort_index()
    
    print(f"\n📊 DISTRIBUCIÓN HORARIA DE DATOS:")
    for hour, count in hour_distribution.items():
        print(f"  {hour:02d}:00 - {count} registros")
    
    # Verificar si hay datos en horario de día (8-18h)
    daytime_data = df_sample[df_sample['hour'].between(8, 18)]
    print(f"\n📈 Datos en horario de día (8-18h): {len(daytime_data)} registros")
    
    if len(daytime_data) > 0:
        print(f"✅ Hay datos durante el día")
        return True
    else:
        print(f"❌ No hay datos durante el día")
        return False

if __name__ == "__main__":
    print("Iniciando pruebas de intervalos solares...")
    
    # Prueba 1: Cálculo de intervalos
    success1 = test_solar_intervals()
    
    # Prueba 2: Solapamiento con datos
    success2 = test_data_overlap()
    
    print(f"\n=== RESUMEN ===")
    print(f"Prueba intervalos solares: {'✅' if success1 else '❌'}")
    print(f"Prueba solapamiento datos: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print("✅ Todas las pruebas pasaron")
    else:
        print("❌ Algunas pruebas fallaron")