#!/usr/bin/env python3
"""
Script de prueba para diagnosticar problemas de conexión con ClickHouse
"""

import clickhouse_connect
import pandas as pd

# Configuración de ClickHouse
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}

def test_clickhouse_connection():
    """Prueba la conexión básica a ClickHouse"""
    try:
        print("🔄 Intentando conectar a ClickHouse...")
        print(f"Host: {CLICKHOUSE_CONFIG['host']}")
        print(f"Port: {CLICKHOUSE_CONFIG['port']}")
        print(f"User: {CLICKHOUSE_CONFIG['user']}")
        
        # Conectar a ClickHouse
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=CLICKHOUSE_CONFIG['port'],
            user=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password'],
            connect_timeout=10,
            send_receive_timeout=30
        )
        
        print("✅ Conexión exitosa a ClickHouse")
        
        # Probar una consulta simple
        print("🔄 Probando consulta simple...")
        result = client.query("SELECT 1 as test")
        print(f"✅ Consulta simple exitosa: {result.result_set}")
        
        # Probar listar bases de datos
        print("🔄 Listando bases de datos...")
        result = client.query("SHOW DATABASES")
        print("✅ Bases de datos disponibles:")
        for row in result.result_set:
            print(f"  - {row[0]}")
        
        # Probar listar tablas de ref_data
        print("🔄 Listando tablas de ref_data...")
        try:
            result = client.query("SHOW TABLES FROM ref_data")
            print("✅ Tablas en ref_data:")
            for row in result.result_set:
                print(f"  - {row[0]}")
        except Exception as e:
            print(f"❌ Error al listar tablas de ref_data: {e}")
        
        # Explorar estructura de tabla DustIQ
        print("🔄 Explorando estructura de tabla PSDA.dustiq...")
        try:
            # Ver estructura de la tabla
            result = client.query("DESCRIBE PSDA.dustiq")
            print("✅ Estructura de PSDA.dustiq:")
            for row in result.result_set:
                print(f"  - {row[0]} ({row[1]})")
            
            # Ver algunas filas de ejemplo
            print("🔄 Mostrando algunas filas de ejemplo...")
            result = client.query("SELECT * FROM PSDA.dustiq LIMIT 5")
            print("✅ Filas de ejemplo:")
            for row in result.result_set:
                print(f"  {row}")
                
        except Exception as e:
            print(f"❌ Error explorando tabla DustIQ: {e}")
        
        client.close()
        print("✅ Conexión cerrada correctamente")
        
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        print(f"Tipo de error: {type(e).__name__}")

if __name__ == "__main__":
    test_clickhouse_connection() 