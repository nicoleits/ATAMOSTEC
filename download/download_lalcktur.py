"""
Script para descargar datos de radiación solar (Lalcktur) desde ClickHouse

DESCRIPCIÓN:
------------
Este script descarga datos de radiación solar desde ClickHouse, específicamente:
- Esquema: lalcktur
- Tabla: cr1000x
- Atributos: DHI_Avg, DNI_Avg, GHI_Avg

Los datos se descargan en formato largo (con Attribute y Measure) y se convierten
a formato ancho con columnas: fecha hora, GHI, DHI, DNI.

USO:
----
Ejecutar el script desde la línea de comandos:
    python download_lalcktur.py

El script guiará al usuario a través de:
1. Configuración de fechas de inicio y fin
2. Descarga automática de los datos
3. Guardado en CSV en el directorio configurado

Los datos descargados se guardan en CSV en el directorio configurado:
    /home/atamos/atamostec/ATAMOSTEC/ATAMOSTEC/datos/lalcktur/

NOTAS:
------
- Requiere conexión al servidor ClickHouse configurado
- Las fechas por defecto son: 01/07/2024 - 31/12/2025
"""

# ============================================================================
# SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ============================================================================

# Importar librerías necesarias
import pandas as pd
import numpy as np
import os
import logging
import clickhouse_connect

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración de Clickhouse
CLICKHOUSE_CONFIG = {
    'host': "146.83.153.212",  # "172.24.61.95"
    'port': "30091",
    'user': "default",
    'password': "Psda2020"
}

# Configuración de fechas por defecto
DEFAULT_START_DATE = pd.to_datetime('01/07/2024', dayfirst=True).tz_localize('UTC')
DEFAULT_END_DATE = pd.to_datetime('31/12/2025', dayfirst=True).tz_localize('UTC')

# Directorio de salida
OUTPUT_DIR = "/home/atamos/atamostec/ATAMOSTEC/ATAMOSTEC/datos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Atributos a descargar
ATTRIBUTES = ['DHI_Avg', 'DNI_Avg', 'GHI_Avg']


# ============================================================================
# FUNCIÓN PARA CONFIGURAR FECHAS DE FORMA INTERACTIVA
# ============================================================================

def configurar_fechas():
    """
    Permite configurar las fechas de inicio y fin de forma interactiva.
    Si se presiona Enter, se usan las fechas por defecto.
    
    Returns:
        tuple: (start_date, end_date) como objetos datetime con timezone UTC
    """
    print("\n" + "="*60)
    print("CONFIGURACIÓN DE FECHAS")
    print("="*60)
    
    # Fechas por defecto
    default_start_str = DEFAULT_START_DATE.strftime('%d/%m/%Y')
    default_end_str = DEFAULT_END_DATE.strftime('%d/%m/%Y')
    
    print(f"\n📅 Fechas por defecto:")
    print(f"   Inicio: {default_start_str}")
    print(f"   Fin:    {default_end_str}")
    print(f"\n💡 Presiona Enter para usar las fechas por defecto")
    print(f"   O ingresa nuevas fechas en formato DD/MM/YYYY")
    print("-"*60)
    
    # Solicitar fecha de inicio
    while True:
        start_input = input(f"\n📌 Fecha de inicio [{default_start_str}]: ").strip()
        
        if start_input == "":
            # Usar fecha por defecto
            start_date = DEFAULT_START_DATE
            print(f"✅ Usando fecha por defecto: {default_start_str}")
            break
        else:
            # Intentar parsear la fecha ingresada
            try:
                start_date = pd.to_datetime(start_input, dayfirst=True)
                # Asegurar que tenga timezone UTC
                if start_date.tz is None:
                    start_date = start_date.tz_localize('UTC')
                else:
                    start_date = start_date.tz_convert('UTC')
                print(f"✅ Fecha de inicio configurada: {start_date.strftime('%d/%m/%Y %H:%M:%S UTC')}")
                break
            except ValueError:
                print("❌ Error: Formato de fecha inválido. Usa DD/MM/YYYY (ej: 01/07/2024)")
    
    # Solicitar fecha de fin
    while True:
        end_input = input(f"\n📌 Fecha de fin [{default_end_str}]: ").strip()
        
        if end_input == "":
            # Usar fecha por defecto
            end_date = DEFAULT_END_DATE
            print(f"✅ Usando fecha por defecto: {default_end_str}")
            break
        else:
            # Intentar parsear la fecha ingresada
            try:
                end_date = pd.to_datetime(end_input, dayfirst=True)
                # Asegurar que tenga timezone UTC
                if end_date.tz is None:
                    end_date = end_date.tz_localize('UTC')
                else:
                    end_date = end_date.tz_convert('UTC')
                print(f"✅ Fecha de fin configurada: {end_date.strftime('%d/%m/%Y %H:%M:%S UTC')}")
                break
            except ValueError:
                print("❌ Error: Formato de fecha inválido. Usa DD/MM/YYYY (ej: 31/12/2025)")
    
    # Validar que la fecha de inicio sea anterior a la de fin
    if start_date > end_date:
        print("\n⚠️  ADVERTENCIA: La fecha de inicio es posterior a la fecha de fin.")
        print("   Se intercambiarán automáticamente.")
        start_date, end_date = end_date, start_date
    
    print("\n" + "="*60)
    print(f"✅ Rango de fechas configurado:")
    print(f"   Desde: {start_date.strftime('%d/%m/%Y %H:%M:%S UTC')}")
    print(f"   Hasta: {end_date.strftime('%d/%m/%Y %H:%M:%S UTC')}")
    print("="*60 + "\n")
    
    return start_date, end_date


# ============================================================================
# FUNCIÓN DE DESCARGA DESDE CLICKHOUSE
# ============================================================================

def download_lalcktur(start_date, end_date, output_dir):
    """
    Descarga y procesa datos de radiación solar (Lalcktur) desde ClickHouse.
    
    Esta función:
    - Esquema: "lalcktur"
    - Tabla: "cr1000x"
    - Atributos: DHI_Avg, DNI_Avg, GHI_Avg
    - Convierte datos de formato largo a formato ancho (pivot)
    - Renombra columnas: DHI_Avg -> DHI, DNI_Avg -> DNI, GHI_Avg -> GHI
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("☀️  Iniciando descarga de datos Lalcktur desde ClickHouse...")
    client = None
    
    try:
        # Conectar a ClickHouse
        logger.info("Conectando a ClickHouse...")
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=int(CLICKHOUSE_CONFIG['port']),
            username=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password']
        )
        logger.info("✅ Conexión a ClickHouse establecida")
        
        # Convertir fechas al formato correcto para ClickHouse (asegurar timezone UTC)
        if isinstance(start_date, pd.Timestamp):
            start_date_utc = pd.Timestamp(start_date)
        else:
            start_date_utc = pd.to_datetime(start_date)
            
        if isinstance(end_date, pd.Timestamp):
            end_date_utc = pd.Timestamp(end_date)
        else:
            end_date_utc = pd.to_datetime(end_date)
        
        # Asegurar timezone UTC
        if start_date_utc.tz is None:
            start_date_utc = start_date_utc.tz_localize('UTC')
        else:
            start_date_utc = start_date_utc.tz_convert('UTC')
            
        if end_date_utc.tz is None:
            end_date_utc = end_date_utc.tz_localize('UTC')
        else:
            end_date_utc = end_date_utc.tz_convert('UTC')
        
        start_str = start_date_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date_utc.strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"📅 Consultando datos desde {start_str} hasta {end_str}")
        
        # Construir la lista de atributos para la consulta SQL
        attributes_str = "', '".join(ATTRIBUTES)
        
        # Consultar datos de lalcktur desde el esquema lalcktur
        logger.info("Consultando datos Lalcktur desde ClickHouse...")
        query = f"""
        SELECT 
            StampTime,
            Attribute,
            Measure
        FROM lalcktur.cr1000x 
        WHERE StampTime >= '{start_str}' AND StampTime <= '{end_str}'
        AND Attribute IN ('{attributes_str}')
        ORDER BY StampTime, Attribute
        """
        
        logger.info("Ejecutando consulta ClickHouse...")
        logger.info(f"Consulta: {query[:200]}...")
        
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se encontraron datos de Lalcktur en ClickHouse")
            return False
            
        logger.info(f"📊 Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame
        logger.info("Procesando datos...")
        df_lalcktur = pd.DataFrame(result.result_set, columns=['StampTime', 'Attribute', 'Measure'])
        
        # Convertir StampTime a datetime y asegurar que esté en UTC
        df_lalcktur['StampTime'] = pd.to_datetime(df_lalcktur['StampTime'])
        if df_lalcktur['StampTime'].dt.tz is None:
            df_lalcktur['StampTime'] = df_lalcktur['StampTime'].dt.tz_localize('UTC')
        else:
            df_lalcktur['StampTime'] = df_lalcktur['StampTime'].dt.tz_convert('UTC')

        # Pivotar los datos para convertir de long format a wide format
        logger.info("Pivotando datos de long format a wide format...")

        # Primero, manejar duplicados agregando por promedio
        logger.info("Manejando duplicados agrupando por promedio...")
        df_lalcktur_grouped = df_lalcktur.groupby(['StampTime', 'Attribute'])['Measure'].mean().reset_index()

        # Ahora hacer el pivot sin duplicados
        df_lalcktur_pivot = df_lalcktur_grouped.pivot(index='StampTime', columns='Attribute', values='Measure')

        # Renombrar el índice a 'fecha hora'
        df_lalcktur_pivot.index.name = 'fecha hora'
        
        # Renombrar las columnas: DHI_Avg -> DHI, DNI_Avg -> DNI, GHI_Avg -> GHI
        column_rename_map = {
            'DHI_Avg': 'DHI',
            'DNI_Avg': 'DNI',
            'GHI_Avg': 'GHI'
        }
        
        # Renombrar solo las columnas que existen
        existing_columns = {k: v for k, v in column_rename_map.items() if k in df_lalcktur_pivot.columns}
        df_lalcktur_pivot.rename(columns=existing_columns, inplace=True)
        
        # Reordenar columnas en el orden especificado: fecha hora, GHI, DHI, DNI
        # Asegurar que todas las columnas existan
        ordered_columns = ['GHI', 'DHI', 'DNI']
        available_columns = [col for col in ordered_columns if col in df_lalcktur_pivot.columns]
        
        # Si hay columnas adicionales, agregarlas al final
        other_columns = [col for col in df_lalcktur_pivot.columns if col not in ordered_columns]
        final_column_order = available_columns + other_columns
        
        df_lalcktur_pivot = df_lalcktur_pivot[final_column_order]
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_lalcktur_pivot.index.min()}")
        logger.info(f"   Fecha más reciente: {df_lalcktur_pivot.index.max()}")

        # Verificar que hay datos en el rango especificado
        if len(df_lalcktur_pivot) == 0:
            logger.warning("⚠️  No se encontraron datos en el rango de fechas especificado.")
            return False

        # Crear carpeta específica para Lalcktur
        section_dir = os.path.join(output_dir, 'lalcktur')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'raw_lalcktur_data.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_lalcktur_pivot.to_csv(output_filepath)

        logger.info(f"✅ Datos Lalcktur desde ClickHouse guardados exitosamente")
        logger.info(f"📊 Total de registros: {len(df_lalcktur_pivot)}")
        logger.info(f"📅 Rango de fechas: {df_lalcktur_pivot.index.min()} a {df_lalcktur_pivot.index.max()}")
        logger.info(f"📊 Columnas: {', '.join(df_lalcktur_pivot.columns.tolist())}")

        # Mostrar estadísticas básicas
        logger.info("📊 Estadísticas de los datos:")
        for col in ['GHI', 'DHI', 'DNI']:
            if col in df_lalcktur_pivot.columns:
                logger.info(f"   {col} - Rango: {df_lalcktur_pivot[col].min():.3f} a {df_lalcktur_pivot[col].max():.3f}")
                logger.info(f"   {col} - Promedio: {df_lalcktur_pivot[col].mean():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos Lalcktur: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


# ============================================================================
# CONFIGURACIÓN INICIAL AL EJECUTAR EL SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DESCARGA DE DATOS LALCKTUR")
    print("="*60)
    print("\nEste script descarga datos de radiación solar desde ClickHouse")
    print("Esquema: lalcktur | Tabla: cr1000x")
    print("Atributos: DHI_Avg, DNI_Avg, GHI_Avg")
    print("="*60)
    
    # Configurar fechas de forma interactiva
    START_DATE, END_DATE = configurar_fechas()
    
    # Mostrar configuración final
    logger.info(f"Rango de fechas: {START_DATE.strftime('%Y-%m-%d')} a {END_DATE.strftime('%Y-%m-%d')}")
    logger.info(f"Directorio de salida: {OUTPUT_DIR}")
    
    # Ejecutar descarga
    print("\n" + "="*60)
    print("INICIANDO DESCARGA")
    print("="*60)
    
    resultado = download_lalcktur(START_DATE, END_DATE, OUTPUT_DIR)
    
    # Mostrar resumen final
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    if resultado:
        print("✅ Descarga completada exitosamente")
    else:
        print("❌ La descarga falló. Revisa los logs para más detalles.")
    print("="*60 + "\n")

