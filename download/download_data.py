"""
Script para descargar datos de sistemas fotovoltaicos desde ClickHouse e InfluxDB

DESCRIPCIÓN:
------------
Este script proporciona una interfaz interactiva para descargar y procesar datos de 
diferentes sensores y sistemas de monitoreo fotovoltaico desde bases de datos ClickHouse 
e InfluxDB. Permite configurar rangos de fechas, horarios y seleccionar qué tipos de 
datos descargar.

FUNCIONALIDADES PRINCIPALES:
----------------------------
1. IV600: Descarga de curvas IV del trazador manual con cálculo de parámetros eléctricos
   (PMP, ISC, VOC, IMP, VMP) o descarga de curvas completas para gráficos I-V y P-V

2. PV Glasses: Descarga de datos de fotoceldas (RFC1-RFC5) con selección de tipo de 
   estadística (Avg, Max, Min, Std) y filtrado por horario (13:00-18:00)

3. DustIQ: Descarga de datos del sensor de polvo (SR_C11_Avg)

4. Soiling Kit: Descarga de datos del kit de ensuciamiento con corrientes de cortocircuito
   y temperaturas de celdas limpias y sucias

5. PVStand: Descarga de datos de módulos PVStand (perc1fixed y perc2fixed) con parámetros
   de potencia, corriente y voltaje máximos

6. Solys2: Descarga de datos de radiación solar (GHI, DHI, DNI) desde PSDA.meteo6857

CARACTERÍSTICAS:
----------------
- Interfaz interactiva con menú de opciones
- Configuración flexible de rangos de fechas y horarios
- Selección dinámica de fotoceldas disponibles en la base de datos
- Manejo automático de timezones (UTC)
- Logging detallado de operaciones
- Validación de datos y manejo de errores
- Organización automática de archivos en subdirectorios

USO:
----
Ejecutar el script desde la línea de comandos:
    python download_data.py

El script guiará al usuario a través de:
1. Configuración de fechas de inicio y fin
2. Selección del tipo de datos a descargar
3. Configuración adicional según el tipo seleccionado (fotoceldas, horarios, etc.)

Los datos descargados se guardan en CSV en el directorio configurado:
datos/ (relativo al directorio raíz del proyecto)

NOTAS:
------
- Convertido desde download_notebook.ipynb
- Requiere conexión a los servidores ClickHouse e InfluxDB configurados
- Las fechas por defecto son: 01/07/2024 - 31/12/2025
"""

# ============================================================================
# SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN INICIAL
# ============================================================================

# Importar librerías necesarias - Config. InfluxDB y Clickhouse
import pandas as pd
import numpy as np
import os
import sys
import logging
import re
from datetime import datetime
import clickhouse_connect
from influxdb_client import InfluxDBClient
import gc

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración de InfluxDB
INFLUX_CONFIG = {
    'url': "http://146.83.153.212:27017",  # "http://172.24.61.95:27017"
    'token': "piDbFR_bfRWO5Epu1IS96WbkNpSZZCYgwZZR29PcwUsxXwKdIyLMhVAhU4-5ohWeXIsX7Dp_X-WiPIDx0beafg==",
    'org': "atamostec",
    'timeout': 300000
}

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

# Directorio de salida - ruta relativa al directorio del proyecto
# Obtener el directorio del script y construir la ruta relativa
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Subir un nivel desde download/ para llegar a ATAMOSTEC/ (PROJECT_ROOT)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datos")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# FUNCIÓN PARA CONFIGURAR FECHAS DE FORMA INTERACTIVA
# ============================================================================

def configurar_rango_horario():
    """
    Permite configurar un rango horario de forma interactiva.
    Si se presiona Enter, se usa el rango completo del día (00:00-23:59).
    
    Returns:
        tuple: (hora_inicio, hora_fin) como strings en formato HH:MM
    """
    print("\n" + "="*60)
    print("CONFIGURACIÓN DE RANGO HORARIO")
    print("="*60)
    
    hora_inicio_default = "00:00"
    hora_fin_default = "23:59"
    
    print(f"\n🕐 Rango horario por defecto:")
    print(f"   Inicio: {hora_inicio_default}")
    print(f"   Fin:    {hora_fin_default}")
    print(f"\n💡 Presiona Enter para usar el rango completo del día")
    print(f"   O ingresa horas en formato HH:MM (ej: 09:00, 18:00)")
    print("-"*60)
    
    # Solicitar hora de inicio
    while True:
        hora_inicio_input = input(f"\n📌 Hora de inicio [{hora_inicio_default}]: ").strip()
        
        if hora_inicio_input == "":
            hora_inicio = hora_inicio_default
            print(f"✅ Usando hora por defecto: {hora_inicio}")
            break
        else:
            # Validar formato HH:MM
            try:
                hora, minuto = map(int, hora_inicio_input.split(':'))
                if 0 <= hora <= 23 and 0 <= minuto <= 59:
                    hora_inicio = hora_inicio_input
                    print(f"✅ Hora de inicio configurada: {hora_inicio}")
                    break
                else:
                    print("❌ Error: Hora inválida. Usa formato HH:MM con horas 0-23 y minutos 0-59")
            except ValueError:
                print("❌ Error: Formato inválido. Usa HH:MM (ej: 09:00)")
    
    # Solicitar hora de fin
    while True:
        hora_fin_input = input(f"\n📌 Hora de fin [{hora_fin_default}]: ").strip()
        
        if hora_fin_input == "":
            hora_fin = hora_fin_default
            print(f"✅ Usando hora por defecto: {hora_fin}")
            break
        else:
            # Validar formato HH:MM
            try:
                hora, minuto = map(int, hora_fin_input.split(':'))
                if 0 <= hora <= 23 and 0 <= minuto <= 59:
                    hora_fin = hora_fin_input
                    print(f"✅ Hora de fin configurada: {hora_fin}")
                    break
                else:
                    print("❌ Error: Hora inválida. Usa formato HH:MM con horas 0-23 y minutos 0-59")
            except ValueError:
                print("❌ Error: Formato inválido. Usa HH:MM (ej: 18:00)")
    
    # Validar que la hora de inicio sea anterior a la de fin
    hora_inicio_minutos = int(hora_inicio.split(':')[0]) * 60 + int(hora_inicio.split(':')[1])
    hora_fin_minutos = int(hora_fin.split(':')[0]) * 60 + int(hora_fin.split(':')[1])
    
    if hora_inicio_minutos >= hora_fin_minutos:
        print("\n⚠️  ADVERTENCIA: La hora de inicio es posterior o igual a la hora de fin.")
        print("   Se intercambiarán automáticamente.")
        hora_inicio, hora_fin = hora_fin, hora_inicio
    
    print("\n" + "="*60)
    print(f"✅ Rango horario configurado:")
    print(f"   Desde: {hora_inicio}")
    print(f"   Hasta: {hora_fin}")
    print("="*60 + "\n")
    
    return hora_inicio, hora_fin


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
# SECCIÓN 2: CLASE PARA MANEJAR CONEXIONES A INFLUXDB
# ============================================================================

class InfluxDBManager:
    """
    Clase para gestionar conexiones y consultas a InfluxDB.
    
    Esta clase proporciona métodos para:
    - Conectar y desconectar de InfluxDB
    - Ejecutar consultas Flux y obtener resultados como DataFrames
    """
    
    def __init__(self, config):
        """
        Inicializa el manager con la configuración de InfluxDB.
        
        Args:
            config (dict): Diccionario con configuración de InfluxDB
                - url: URL del servidor InfluxDB
                - token: Token de autenticación
                - org: Organización
                - timeout: Timeout en milisegundos
        """
        self.config = config
        self.client = None
        self.query_api = None
        
    def connect(self):
        """
        Establece conexión con InfluxDB y configura el query_api.
        
        Returns:
            bool: True si la conexión fue exitosa, False en caso contrario
        """
        try:
            self.client = InfluxDBClient(
                url=self.config['url'],
                token=self.config['token'],
                org=self.config['org'],
                timeout=self.config['timeout']
            )
            self.query_api = self.client.query_api()
            logger.info("✅ Cliente InfluxDB y query_api inicializados.")
            return True
        except Exception as e:
            logger.error(f"❌ Error al conectar con InfluxDB: {e}")
            return False
            
    def disconnect(self):
        """Cierra la conexión con InfluxDB si está abierta."""
        if self.client:
            self.client.close()
            logger.info("✅ Conexión a InfluxDB cerrada.")
            
    def query_influxdb(self, bucket, tables, attributes, start_date, stop_date):
        """
        Ejecuta una consulta Flux en InfluxDB y retorna los resultados como DataFrame.
        
        Args:
            bucket (str): Nombre del bucket de InfluxDB
            tables (list): Lista de nombres de mediciones (measurements) a consultar
            attributes (list): Lista de campos (fields) a recuperar
            start_date (datetime): Fecha de inicio del rango (con timezone)
            stop_date (datetime): Fecha de fin del rango (con timezone)
            
        Returns:
            pd.DataFrame o None: DataFrame con los resultados o None si hay error o no hay datos
        """
        try:
            # Convertir fechas al formato correcto para InfluxDB (RFC3339)
            # Asegurar que las fechas tengan timezone UTC
            if isinstance(start_date, pd.Timestamp):
                start_date_utc = pd.Timestamp(start_date)
            else:
                start_date_utc = pd.to_datetime(start_date)
                
            if isinstance(stop_date, pd.Timestamp):
                stop_date_utc = pd.Timestamp(stop_date)
            else:
                stop_date_utc = pd.to_datetime(stop_date)
            
            # Asegurar timezone UTC
            if start_date_utc.tz is None:
                start_date_utc = start_date_utc.tz_localize('UTC')
            else:
                start_date_utc = start_date_utc.tz_convert('UTC')
                
            if stop_date_utc.tz is None:
                stop_date_utc = stop_date_utc.tz_localize('UTC')
            else:
                stop_date_utc = stop_date_utc.tz_convert('UTC')
            
            # Formatear fechas en formato RFC3339 para InfluxDB
            start_str = start_date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            stop_str = stop_date_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Construir la lista de atributos en formato correcto para Flux
            attributes_str = " or ".join([f'r["_field"] == "{attr}"' for attr in attributes])
            
            # Construir la lista de tablas (measurements) en formato correcto para Flux
            tables_str = " or ".join([f'r["_measurement"] == "{table}"' for table in tables])

            # Construir la consulta Flux
            query = f'''
            from(bucket: "{bucket}")
                |> range(start: {start_str}, stop: {stop_str})
                |> filter(fn: (r) => {tables_str})
                |> filter(fn: (r) => {attributes_str})
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            logger.info(f"📊 Consultando InfluxDB:")
            logger.info(f"   Bucket: {bucket}")
            logger.info(f"   Tables: {tables}")
            logger.info(f"   Attributes: {attributes}")
            logger.info(f"   Rango: {start_str} a {stop_str}")
            
            # Ejecutar la consulta
            result = self.query_api.query_data_frame(query)
            
            if result.empty:
                logger.warning("⚠️  No se encontraron datos en la consulta.")
                return None
                
            logger.info(f"✅ Datos obtenidos: {len(result)} registros")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en la consulta a InfluxDB: {e}")
            import traceback
            logger.error(f"Detalles del error:\n{traceback.format_exc()}")
            return None


# ============================================================================
# SECCIÓN 3: FUNCIONES DE DESCARGA DESDE CLICKHOUSE
# ============================================================================

# Fotoceldas por defecto
DEFAULT_PHOTODIODES = ["R_FC1_Avg", "R_FC2_Avg", "R_FC3_Avg", "R_FC4_Avg", "R_FC5_Avg"]

# Tipos de estadística disponibles
STAT_TYPES = ["Avg", "Max", "Min", "Std"]

# Mapeo de atributos a columnas de ClickHouse
# Formato: "R_FC{N}_{StatType}" -> "RFC{N}{StatType}"
ATTRIBUTE_TO_COLUMN = {
    "R_FC1_Avg": "RFC1Avg",
    "R_FC2_Avg": "RFC2Avg",
    "R_FC3_Avg": "RFC3Avg",
    "R_FC4_Avg": "RFC4Avg",
    "R_FC5_Avg": "RFC5Avg",
    "R_FC1_Max": "RFC1Max",
    "R_FC2_Max": "RFC2Max",
    "R_FC3_Max": "RFC3Max",
    "R_FC4_Max": "RFC4Max",
    "R_FC5_Max": "RFC5Max",
    "R_FC1_Min": "RFC1Min",
    "R_FC2_Min": "RFC2Min",
    "R_FC3_Min": "RFC3Min",
    "R_FC4_Min": "RFC4Min",
    "R_FC5_Min": "RFC5Min",
    "R_FC1_Std": "RFC1Std",
    "R_FC2_Std": "RFC2Std",
    "R_FC3_Std": "RFC3Std",
    "R_FC4_Std": "RFC4Std",
    "R_FC5_Std": "RFC5Std",
}


def obtener_fotoceldas_disponibles():
    """
    Obtiene las columnas disponibles (fotoceldas) de la tabla ClickHouse.
    Organiza las fotoceldas por número (RFC1, RFC2, etc.) y tipo de estadística.
    
    Returns:
        dict: Diccionario con estructura {numero_fotocelda: {stat_type: columna_clickhouse}}
              o None si hay error
    """
    client = None
    try:
        logger.info("Conectando a ClickHouse para obtener fotoceldas disponibles...")
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=int(CLICKHOUSE_CONFIG['port']),
            username=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password']
        )
        
        # Consultar estructura de la tabla
        schema = "PSDA"
        table = "ftc6852"
        query = f"DESCRIBE TABLE {schema}.{table}"
        
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se pudieron obtener las columnas de la tabla")
            return None
        
        # Organizar columnas por número de fotocelda y tipo de estadística
        fotoceldas_organizadas = {}
        for row in result.result_set:
            column_name = row[0]  # Nombre de la columna
            column_upper = column_name.upper()
            
            # Filtrar solo columnas que parezcan fotoceldas (RFC*)
            if 'RFC' in column_upper:
                # Extraer número de fotocelda (ej: RFC1Avg -> 1)
                match = re.search(r'RFC(\d+)', column_upper)
                if match:
                    num_fotocelda = int(match.group(1))
                    
                    # Determinar tipo de estadística
                    stat_type = None
                    for stat in STAT_TYPES:
                        if stat.upper() in column_upper:
                            stat_type = stat
                            break
                    
                    if stat_type:
                        if num_fotocelda not in fotoceldas_organizadas:
                            fotoceldas_organizadas[num_fotocelda] = {}
                        fotoceldas_organizadas[num_fotocelda][stat_type] = column_name
        
        logger.info(f"✅ Fotoceldas disponibles encontradas: {len(fotoceldas_organizadas)} fotoceldas")
        return fotoceldas_organizadas
        
    except Exception as e:
        logger.error(f"❌ Error al obtener fotoceldas disponibles: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return None
    finally:
        if client:
            client.close()


def seleccionar_fotoceldas():
    """
    Permite seleccionar las fotoceldas y tipo de estadística de forma interactiva.
    Si se presiona Enter, se usan las fotoceldas por defecto (Avg).
    
    Returns:
        list: Lista de nombres de atributos seleccionados (ej: ["R_FC1_Avg", "R_FC2_Max"])
    """
    print("\n" + "="*60)
    print("SELECCIÓN DE FOTOCELDAS Y TIPO DE ESTADÍSTICA")
    print("="*60)
    
    # Obtener fotoceldas disponibles
    print("\n📡 Obteniendo fotoceldas disponibles desde ClickHouse...")
    fotoceldas_organizadas = obtener_fotoceldas_disponibles()
    
    if fotoceldas_organizadas is None or len(fotoceldas_organizadas) == 0:
        print("⚠️  No se pudieron obtener las fotoceldas disponibles.")
        print("   Usando fotoceldas por defecto.")
        return DEFAULT_PHOTODIODES
    
    # Paso 1: Seleccionar números de fotoceldas
    print(f"\n📊 Fotoceldas disponibles:")
    numeros_fotoceldas = sorted(fotoceldas_organizadas.keys())
    for num in numeros_fotoceldas:
        stats_disponibles = list(fotoceldas_organizadas[num].keys())
        stats_str = ", ".join(stats_disponibles)
        es_default = num <= 5  # RFC1 a RFC5 son las por defecto
        default_mark = " (por defecto)" if es_default else ""
        print(f"  {num}. RFC{num} - Tipos disponibles: {stats_str}{default_mark}")
    
    print(f"\n💡 Fotoceldas por defecto: RFC1-RFC5 con tipo Avg")
    print(f"   (Números: {', '.join(map(str, [n for n in numeros_fotoceldas if n <= 5]))})")
    print(f"\n💡 Presiona Enter para usar las fotoceldas por defecto (RFC1-RFC5, Avg)")
    print(f"   O ingresa los números de las fotoceldas separados por comas (ej: 1,2,3)")
    print("-"*60)
    
    seleccion_fotoceldas_input = input(f"\n📌 Selecciona números de fotoceldas: ").strip()
    
    if seleccion_fotoceldas_input == "":
        # Usar fotoceldas por defecto
        print(f"✅ Usando fotoceldas por defecto: RFC1-RFC5 con tipo Avg")
        return DEFAULT_PHOTODIODES
    
    # Procesar selección de fotoceldas
    try:
        numeros_seleccionados = [int(x.strip()) for x in seleccion_fotoceldas_input.split(',')]
        numeros_validos = [n for n in numeros_seleccionados if n in numeros_fotoceldas]
        
        if len(numeros_validos) == 0:
            print("❌ No se seleccionaron fotoceldas válidas. Usando fotoceldas por defecto.")
            return DEFAULT_PHOTODIODES
        
        print(f"✅ Fotoceldas seleccionadas: RFC{', RFC'.join(map(str, numeros_validos))}")
        
    except ValueError:
        print("❌ Error: Formato inválido. Usa números separados por comas (ej: 1,2,3)")
        print("   Usando fotoceldas por defecto.")
        return DEFAULT_PHOTODIODES
    
    # Paso 2: Seleccionar tipo de estadística
    print(f"\n📊 Tipos de estadística disponibles:")
    for idx, stat_type in enumerate(STAT_TYPES, 1):
        es_default = stat_type == "Avg"
        default_mark = " (por defecto)" if es_default else ""
        print(f"  {idx}. {stat_type}{default_mark}")
    
    print(f"\n💡 Tipo por defecto: Avg")
    print(f"💡 Presiona Enter para usar Avg")
    print(f"   O ingresa el número del tipo de estadística (1=Avg, 2=Max, 3=Min, 4=Std)")
    print("-"*60)
    
    seleccion_stat_input = input(f"\n📌 Selecciona tipo de estadística: ").strip()
    
    if seleccion_stat_input == "":
        stat_type_seleccionado = "Avg"
    else:
        try:
            stat_num = int(seleccion_stat_input)
            if 1 <= stat_num <= len(STAT_TYPES):
                stat_type_seleccionado = STAT_TYPES[stat_num - 1]
            else:
                print(f"⚠️  Número fuera de rango. Usando Avg por defecto.")
                stat_type_seleccionado = "Avg"
        except ValueError:
            print("❌ Error: Formato inválido. Usando Avg por defecto.")
            stat_type_seleccionado = "Avg"
    
    print(f"✅ Tipo de estadística seleccionado: {stat_type_seleccionado}")
    
    # Construir lista de atributos seleccionados
    atributos_seleccionados = []
    for num_fotocelda in numeros_validos:
        if num_fotocelda in fotoceldas_organizadas:
            if stat_type_seleccionado in fotoceldas_organizadas[num_fotocelda]:
                # Construir nombre de atributo: R_FC{N}_{StatType}
                attr_name = f"R_FC{num_fotocelda}_{stat_type_seleccionado}"
                atributos_seleccionados.append(attr_name)
            else:
                print(f"⚠️  RFC{num_fotocelda} no tiene tipo {stat_type_seleccionado}. Se omitirá.")
    
    if len(atributos_seleccionados) == 0:
        print("❌ No se pudieron construir atributos válidos. Usando fotoceldas por defecto.")
        return DEFAULT_PHOTODIODES
    
    print(f"✅ Atributos seleccionados: {', '.join(atributos_seleccionados)}")
    print("="*60 + "\n")
    
    return atributos_seleccionados

def download_iv600(start_date, end_date, output_dir):
    """
    Descarga y procesa datos de IV600 desde ClickHouse.
    
    Esta función:
    - Conecta a ClickHouse
    - Consulta datos de curvas IV del trazador manual
    - Calcula parámetros eléctricos (PMP, ISC, VOC, IMP, VMP)
    - Filtra por rango de fechas
    - Guarda los datos en CSV
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("🔋 Iniciando descarga de datos IV600 desde ClickHouse...")
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
        
        # Consultar datos
        logger.info("Consultando datos IV600...")
        query = "SELECT * FROM ref_data.iv_curves_trazador_manual"
        data_iv_curves = client.query(query)
        logger.info(f"📊 Datos obtenidos: {len(data_iv_curves.result_set)} registros")
        
        # Procesar datos
        logger.info("Procesando datos...")
        curves_list = []
        for curve in data_iv_curves.result_set:
            currents = curve[4]
            voltages = curve[3]
            powers = [currents[i] * voltages[i] for i in range(len(currents))]
            timestamp = curve[0]
            module = curve[2]
            pmp = max(powers)
            isc = max(currents)
            voc = max(voltages)
            imp = currents[np.argmax(powers)]
            vmp = voltages[np.argmax(powers)]
            curves_list.append([timestamp, module, pmp, isc, voc, imp, vmp])

        # Crear DataFrame
        logger.info("Creando DataFrame...")
        column_names = ["timestamp", "module", "pmp", "isc", "voc", "imp", "vmp"]
        df_curves = pd.DataFrame(curves_list, columns=column_names)
        
        # Convertir timestamp a datetime y asegurar que esté en UTC
        df_curves['timestamp'] = pd.to_datetime(df_curves['timestamp'])
        if df_curves['timestamp'].dt.tz is None:
            df_curves['timestamp'] = df_curves['timestamp'].dt.tz_localize('UTC')
        else:
            df_curves['timestamp'] = df_curves['timestamp'].dt.tz_convert('UTC')
        
        df_curves.set_index('timestamp', inplace=True)
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_curves.index.min()}")
        logger.info(f"   Fecha más reciente: {df_curves.index.max()}")
        
        # Filtrar por fecha usando query para mayor flexibilidad
        logger.info(f"Filtrando datos entre {start_date} y {end_date}...")
        df_curves = df_curves.query('@start_date <= index <= @end_date')
        
        if len(df_curves) == 0:
            logger.warning("⚠️  No se encontraron datos en el rango de fechas especificado.")
            logger.info("Ajustando el rango de fechas al rango disponible en los datos...")
            df_curves = df_curves.sort_index()
        else:
            logger.info(f"✅ Se encontraron {len(df_curves)} registros en el rango especificado.")

        # Crear carpeta específica para IV600
        section_dir = os.path.join(output_dir, 'iv600')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'raw_iv600_data.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_curves.to_csv(output_filepath)
        logger.info(f"✅ Datos guardados exitosamente. Total de registros: {len(df_curves)}")
        logger.info(f"📅 Rango de fechas: {df_curves.index.min()} a {df_curves.index.max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos IV600: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


def download_iv600_curves_complete(start_date, end_date, output_dir, hora_inicio="00:00", hora_fin="23:59"):
    """
    Descarga los datos completos de las curvas IV desde ClickHouse.
    
    Esta función:
    - Conecta a ClickHouse
    - Consulta datos de curvas IV del trazador manual
    - Filtra por rango de fechas y horario
    - Expande cada curva en múltiples filas (una por cada punto)
    - Guarda los datos completos en CSV para gráficos I-V y P-V
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        hora_inicio (str): Hora de inicio del rango horario (formato HH:MM)
        hora_fin (str): Hora de fin del rango horario (formato HH:MM)
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("🔋 Iniciando descarga de curvas IV completas desde ClickHouse...")
    logger.info(f"🕐 Rango horario: {hora_inicio} - {hora_fin}")
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
        
        # Consultar datos
        logger.info("Consultando datos IV600...")
        query = "SELECT * FROM ref_data.iv_curves_trazador_manual"
        data_iv_curves = client.query(query)
        logger.info(f"📊 Datos obtenidos: {len(data_iv_curves.result_set)} curvas")
        
        # Procesar datos y expandir curvas
        logger.info("Procesando y expandiendo curvas...")
        curves_data = []
        
        for curve in data_iv_curves.result_set:
            timestamp = curve[0]
            module = curve[2]
            voltages = curve[3]  # Array de voltajes
            currents = curve[4]  # Array de corrientes
            
            # Convertir timestamp a datetime
            if isinstance(timestamp, str):
                timestamp_dt = pd.to_datetime(timestamp)
            else:
                timestamp_dt = pd.to_datetime(timestamp)
            
            # Asegurar timezone UTC
            if timestamp_dt.tz is None:
                timestamp_dt = timestamp_dt.tz_localize('UTC')
            else:
                timestamp_dt = timestamp_dt.tz_convert('UTC')
            
            # Filtrar por rango de fechas
            if start_date <= timestamp_dt <= end_date:
                # Filtrar por rango horario
                hora_actual = timestamp_dt.strftime('%H:%M')
                hora_actual_minutos = int(hora_actual.split(':')[0]) * 60 + int(hora_actual.split(':')[1])
                hora_inicio_minutos = int(hora_inicio.split(':')[0]) * 60 + int(hora_inicio.split(':')[1])
                hora_fin_minutos = int(hora_fin.split(':')[0]) * 60 + int(hora_fin.split(':')[1])
                
                if hora_inicio_minutos <= hora_actual_minutos <= hora_fin_minutos:
                    # Expandir curva: crear una fila por cada punto
                    if len(voltages) == len(currents):
                        for i in range(len(voltages)):
                            voltage = voltages[i]
                            current = currents[i]
                            power = voltage * current
                            
                            curves_data.append({
                                'timestamp': timestamp_dt,
                                'module': module,
                                'voltage': voltage,
                                'current': current,
                                'power': power
                            })
        
        if len(curves_data) == 0:
            logger.warning("⚠️  No se encontraron curvas en el rango de fechas y horario especificado.")
            return False
        
        # Crear DataFrame
        logger.info("Creando DataFrame...")
        df_curves_complete = pd.DataFrame(curves_data)
        
        # Ordenar por timestamp y módulo
        df_curves_complete = df_curves_complete.sort_values(['timestamp', 'module', 'voltage'])
        
        logger.info(f"📊 Total de puntos de curva: {len(df_curves_complete)}")
        logger.info(f"📊 Total de curvas: {df_curves_complete.groupby(['timestamp', 'module']).ngroups}")
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_curves_complete['timestamp'].min()}")
        logger.info(f"   Fecha más reciente: {df_curves_complete['timestamp'].max()}")
        logger.info(f"   Módulos únicos: {df_curves_complete['module'].nunique()}")
        
        # Crear carpeta específica para IV600 Curves
        section_dir = os.path.join(output_dir, 'iv600_curves')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'raw_iv600_curves.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_curves_complete.to_csv(output_filepath, index=False)
        
        logger.info(f"✅ Datos guardados exitosamente")
        logger.info(f"📊 Total de puntos: {len(df_curves_complete):,}")
        logger.info(f"📊 Columnas: timestamp, module, voltage, current, power")
        logger.info(f"📅 Rango de fechas: {df_curves_complete['timestamp'].min()} a {df_curves_complete['timestamp'].max()}")
        logger.info(f"🕐 Rango horario aplicado: {hora_inicio} - {hora_fin}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de curvas IV completas: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


def download_pv_glasses(start_date, end_date, output_dir, attributes=None):
    """
    Descarga y procesa datos de PV Glasses desde ClickHouse.
    
    Esta función:
    - Esquema: "PSDA"
    - Tabla: "ftc6852"
    - Columnas: Se pueden seleccionar dinámicamente o usar las por defecto
    
    Filtra por horario (13:00-18:00) y guarda en CSV.
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        attributes (list, optional): Lista de atributos/fotoceldas a descargar.
                                     Si es None, usa las fotoceldas por defecto.
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("🔋 Iniciando descarga de datos PV Glasses desde ClickHouse...")
    
    # Usar fotoceldas por defecto si no se especifican
    if attributes is None:
        attributes = DEFAULT_PHOTODIODES
    
    client = None
    
    try:
        # Configuración de la consulta
        schema = "PSDA"  # Esquema/base de datos en ClickHouse
        table = "ftc6852"  # Tabla en ClickHouse
        
        logger.info(f"📊 Configuración:")
        logger.info(f"   Esquema: {schema}")
        logger.info(f"   Tabla: {table}")
        logger.info(f"   Attributes: {attributes}")
        
        # Conectar a ClickHouse
        logger.info("Conectando a ClickHouse...")
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_CONFIG['host'],
            port=int(CLICKHOUSE_CONFIG['port']),
            username=CLICKHOUSE_CONFIG['user'],
            password=CLICKHOUSE_CONFIG['password']
        )
        logger.info("✅ Conexión a ClickHouse establecida")
        
        # Convertir fechas al formato para ClickHouse (asegurar timezone UTC)
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
        
        # Mapear nombres de atributos a columnas de ClickHouse
        # Formato esperado: "R_FC{N}_{StatType}" -> "RFC{N}{StatType}"
        columns_clickhouse = ["timestamp"]
        for attr in attributes:
            if attr in ATTRIBUTE_TO_COLUMN:
                # Usar mapeo conocido
                columns_clickhouse.append(ATTRIBUTE_TO_COLUMN[attr])
            else:
                # Intentar construir el nombre de columna dinámicamente
                # Patrón: R_FC{N}_{StatType} -> RFC{N}{StatType}
                match = re.match(r'R_FC(\d+)_(\w+)', attr)
                if match:
                    num_fotocelda = match.group(1)
                    stat_type = match.group(2)
                    columna_clickhouse = f"RFC{num_fotocelda}{stat_type}"
                    columns_clickhouse.append(columna_clickhouse)
                else:
                    # Si no coincide el patrón, usar el nombre del atributo directamente
                    columns_clickhouse.append(attr)
        
        columns_str = ", ".join(columns_clickhouse)
        
        query = f"""
        SELECT 
            {columns_str}
        FROM {schema}.{table}
        WHERE timestamp >= '{start_str}' 
        AND timestamp <= '{end_str}'
        ORDER BY timestamp
        """
        
        logger.info("Ejecutando consulta ClickHouse...")
        logger.info(f"Consulta: {query}")
        
        # Ejecutar consulta
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se obtuvieron datos de PV Glasses desde ClickHouse")
            return False
        
        logger.info(f"📊 Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame usando los nombres de columnas de ClickHouse
        df_glasses = pd.DataFrame(result.result_set, columns=columns_clickhouse)
        
        # Renombrar columnas para mantener compatibilidad con el código existente
        # timestamp → _time
        # RFC1Avg → R_FC1_Avg, etc.
        column_rename_map = {'timestamp': '_time'}
        for attr in attributes:
            if attr in ATTRIBUTE_TO_COLUMN:
                # Si el atributo está en el mapeo, renombrar la columna de ClickHouse al atributo
                column_clickhouse = ATTRIBUTE_TO_COLUMN[attr]
                column_rename_map[column_clickhouse] = attr
            else:
                # Intentar construir el nombre de columna dinámicamente
                match = re.match(r'R_FC(\d+)_(\w+)', attr)
                if match:
                    num_fotocelda = match.group(1)
                    stat_type = match.group(2)
                    column_clickhouse = f"RFC{num_fotocelda}{stat_type}"
                    column_rename_map[column_clickhouse] = attr
                # Si no coincide el patrón, el atributo es el mismo que la columna en ClickHouse
                # No necesitamos renombrar
        
        df_glasses.rename(columns=column_rename_map, inplace=True)
        
        # Convertir timestamp a datetime y asegurar timezone UTC
        df_glasses['_time'] = pd.to_datetime(df_glasses['_time'])
        if df_glasses['_time'].dt.tz is None:
            df_glasses['_time'] = df_glasses['_time'].dt.tz_localize('UTC')
        else:
            df_glasses['_time'] = df_glasses['_time'].dt.tz_convert('UTC')
        
        # Establecer índice de tiempo
        df_glasses.set_index('_time', inplace=True)
        
        logger.info(f"📅 Rango de fechas obtenido: {df_glasses.index.min()} a {df_glasses.index.max()}")
        
        # Filtrar por horario (13:00 a 18:00)
        df_glasses = df_glasses.between_time('13:00', '18:00')
        logger.info(f"🕐 Después del filtro horario (13:00-18:00): {len(df_glasses)} registros")
        
        # Seleccionar solo las columnas numéricas para el cálculo
        numeric_columns = df_glasses.select_dtypes(include=[np.number]).columns
        df_glasses_numeric = df_glasses[numeric_columns]
        
        # Calcular referencia (promedio de la primera fotocelda disponible)
        if len(numeric_columns) > 0:
            primera_fotocelda = numeric_columns[0]
            df_glasses['Ref'] = df_glasses_numeric[primera_fotocelda].mean()
            logger.info(f"📊 Referencia calculada usando: {primera_fotocelda}")
        
        # Crear carpeta específica para PV Glasses
        section_dir = os.path.join(output_dir, 'pv_glasses')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'raw_pv_glasses_data.csv')
        df_glasses.to_csv(output_filepath)
        
        logger.info(f"✅ Datos PV Glasses guardados exitosamente")
        logger.info(f"📊 Total de registros: {len(df_glasses)}")
        logger.info(f"📅 Rango de fechas: {df_glasses.index.min()} a {df_glasses.index.max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos PV Glasses: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    
    finally:
        # Cerrar conexión ClickHouse si existe
        if client:
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


def download_dustiq_clickhouse(start_date, end_date, output_dir):
    """
    Descarga y procesa datos de DustIQ desde ClickHouse.
    
    Esta función:
    - Esquema: "PSDA"
    - Tabla: "dustiq"
    - Atributo: "SR_C11_Avg"
    - Convierte datos de formato largo a formato ancho (pivot)
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("🌪️  Iniciando descarga de datos DustIQ desde ClickHouse...")
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
        
        # Consultar datos de dustiq desde el esquema PSDA
        logger.info("Consultando datos DustIQ desde ClickHouse...")
        query = f"""
        SELECT 
            Stamptime,
            Attribute,
            Measure
        FROM PSDA.dustiq 
        WHERE Stamptime >= '{start_str}' AND Stamptime <= '{end_str}'
        AND Attribute = 'SR_C11_Avg'
        ORDER BY Stamptime, Attribute
        """
        
        logger.info("Ejecutando consulta ClickHouse...")
        logger.info(f"Consulta: {query[:200]}...")
        
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se encontraron datos de DustIQ en ClickHouse")
            return False
            
        logger.info(f"📊 Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame
        logger.info("Procesando datos...")
        df_dustiq = pd.DataFrame(result.result_set, columns=['Stamptime', 'Attribute', 'Measure'])
        
        # Convertir Stamptime a datetime y asegurar que esté en UTC
        df_dustiq['Stamptime'] = pd.to_datetime(df_dustiq['Stamptime'])
        if df_dustiq['Stamptime'].dt.tz is None:
            df_dustiq['Stamptime'] = df_dustiq['Stamptime'].dt.tz_localize('UTC')
        else:
            df_dustiq['Stamptime'] = df_dustiq['Stamptime'].dt.tz_convert('UTC')

        # Pivotar los datos para convertir de long format a wide format
        logger.info("Pivotando datos de long format a wide format...")

        # Primero, manejar duplicados agregando por promedio
        logger.info("Manejando duplicados agrupando por promedio...")
        df_dustiq_grouped = df_dustiq.groupby(['Stamptime', 'Attribute'])['Measure'].mean().reset_index()

        # Ahora hacer el pivot sin duplicados
        df_dustiq_pivot = df_dustiq_grouped.pivot(index='Stamptime', columns='Attribute', values='Measure')

        # Renombrar el índice
        df_dustiq_pivot.index.name = 'timestamp'
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_dustiq_pivot.index.min()}")
        logger.info(f"   Fecha más reciente: {df_dustiq_pivot.index.max()}")

        # Verificar que hay datos en el rango especificado
        if len(df_dustiq_pivot) == 0:
            logger.warning("⚠️  No se encontraron datos en el rango de fechas especificado.")
            return False

        # Crear carpeta específica para DustIQ
        section_dir = os.path.join(output_dir, 'dustiq')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'raw_dustiq_data.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_dustiq_pivot.to_csv(output_filepath)

        logger.info(f"✅ Datos DustIQ desde ClickHouse guardados exitosamente")
        logger.info(f"📊 Total de registros: {len(df_dustiq_pivot)}")
        logger.info(f"📅 Rango de fechas: {df_dustiq_pivot.index.min()} a {df_dustiq_pivot.index.max()}")

        # Mostrar estadísticas básicas
        logger.info("📊 Estadísticas de los datos:")
        if 'SR_C11_Avg' in df_dustiq_pivot.columns:
            logger.info(f"   SR_C11_Avg - Rango: {df_dustiq_pivot['SR_C11_Avg'].min():.3f} a {df_dustiq_pivot['SR_C11_Avg'].max():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos DustIQ desde ClickHouse: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


def download_soiling_kit_clickhouse(start_date, end_date, output_dir):
    """
    Descarga y procesa datos del Soiling Kit desde ClickHouse.
    
    Esta función:
    - Esquema: "PSDA"
    - Tabla: "soiling_kit"
    - Columnas: "stamptime", "Isc(p)", "Isc(e)", "Tp(C)", "Te(C)"
    - La tabla ya tiene las columnas en formato ancho (wide format)
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("🌪️  Iniciando descarga de datos del Soiling Kit desde ClickHouse...")
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
        
        # Consultar datos del Soiling Kit desde PSDA.soiling_kit
        # La tabla ya tiene las columnas en formato ancho (wide format)
        logger.info("Consultando datos del Soiling Kit desde ClickHouse...")
        query = f"""
        SELECT 
            stamptime,
            `Isc(p)`,
            `Isc(e)`,
            `Tp(C)`,
            `Te(C)`
        FROM PSDA.soiling_kit 
        WHERE stamptime >= '{start_str}' AND stamptime <= '{end_str}'
        ORDER BY stamptime
        """
        
        logger.info("Ejecutando consulta ClickHouse...")
        logger.info(f"Consulta: {query[:200]}...")
        
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se encontraron datos del Soiling Kit en ClickHouse")
            return False
            
        logger.info(f"📊 Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame
        logger.info("Procesando datos...")
        df_soilingkit = pd.DataFrame(result.result_set, columns=['stamptime', 'Isc(p)', 'Isc(e)', 'Tp(C)', 'Te(C)'])
        
        # Convertir stamptime a datetime y asegurar que esté en UTC
        df_soilingkit['stamptime'] = pd.to_datetime(df_soilingkit['stamptime'])
        if df_soilingkit['stamptime'].dt.tz is None:
            df_soilingkit['stamptime'] = df_soilingkit['stamptime'].dt.tz_localize('UTC')
        else:
            df_soilingkit['stamptime'] = df_soilingkit['stamptime'].dt.tz_convert('UTC')
        
        # Reordenar columnas según el orden correcto de la base de datos: stamptime, Isc(p), Isc(e), Tp(C), Te(C)
        column_order = ['stamptime', 'Isc(p)', 'Isc(e)', 'Tp(C)', 'Te(C)']
        # Seleccionar solo las columnas en el orden correcto
        df_soilingkit = df_soilingkit[column_order]
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_soilingkit['stamptime'].min()}")
        logger.info(f"   Fecha más reciente: {df_soilingkit['stamptime'].max()}")

        # Verificar que hay datos en el rango especificado
        if len(df_soilingkit) == 0:
            logger.warning("⚠️  No se encontraron datos en el rango de fechas especificado.")
            return False

        # Crear carpeta específica para Soiling Kit
        section_dir = os.path.join(output_dir, 'soiling_kit')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'soiling_kit_raw_data.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_soilingkit.to_csv(output_filepath, index=False)

        logger.info(f"✅ Datos del Soiling Kit desde ClickHouse guardados exitosamente")
        logger.info(f"📊 Total de registros: {len(df_soilingkit)}")
        logger.info(f"📅 Rango de fechas: {df_soilingkit['stamptime'].min()} a {df_soilingkit['stamptime'].max()}")

        # Mostrar estadísticas básicas
        logger.info("📊 Estadísticas de los datos:")
        if 'Isc(e)' in df_soilingkit.columns:
            logger.info(f"   Isc(e) - Rango: {df_soilingkit['Isc(e)'].min():.3f} a {df_soilingkit['Isc(e)'].max():.3f}")
        if 'Isc(p)' in df_soilingkit.columns:
            logger.info(f"   Isc(p) - Rango: {df_soilingkit['Isc(p)'].min():.3f} a {df_soilingkit['Isc(p)'].max():.3f}")
        if 'Te(C)' in df_soilingkit.columns:
            logger.info(f"   Te(C) - Rango: {df_soilingkit['Te(C)'].min():.1f} a {df_soilingkit['Te(C)'].max():.1f}")
        if 'Tp(C)' in df_soilingkit.columns:
            logger.info(f"   Tp(C) - Rango: {df_soilingkit['Tp(C)'].min():.1f} a {df_soilingkit['Tp(C)'].max():.1f}")
        
        # Mostrar información sobre la estructura de datos
        logger.info("📋 Estructura de datos del Soiling Kit:")
        logger.info(f"   - Isc(e): Corriente de cortocircuito de la celda limpia (referencia)")
        logger.info(f"   - Isc(p): Corriente de cortocircuito de la celda sucia (panel)")
        logger.info(f"   - Te(C): Temperatura de la celda limpia en Celsius")
        logger.info(f"   - Tp(C): Temperatura de la celda sucia en Celsius")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos del Soiling Kit desde ClickHouse: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


def download_pvstand_clickhouse(start_date, end_date, output_dir):
    """
    Descarga y procesa datos de PVStand desde ClickHouse.
    
    Esta función:
    - Esquema: "PSDA"
    - Tablas: "perc1fixed" y "perc2fixed"
    - Columnas: "timestamp", "module", "pmax", "imax", "umax"
    - Combina datos de ambas tablas usando UNION ALL
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("🔋 Iniciando descarga de datos PVStand desde ClickHouse...")
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
        
        # Consultar datos de PVStand desde las tablas perc1fixed y perc2fixed
        logger.info("Consultando datos PVStand desde ClickHouse...")
        query = f"""
        SELECT 
            timestamp,
            'perc1fixed' as module,
            pmax,
            imax,
            umax
        FROM PSDA.perc1fixed 
        WHERE timestamp >= '{start_str}' AND timestamp <= '{end_str}'
        
        UNION ALL
        
        SELECT 
            timestamp,
            'perc2fixed' as module,
            pmax,
            imax,
            umax
        FROM PSDA.perc2fixed 
        WHERE timestamp >= '{start_str}' AND timestamp <= '{end_str}'
        
        ORDER BY timestamp
        """
        
        logger.info("Ejecutando consulta ClickHouse...")
        logger.info(f"Consulta: {query[:200]}...")
        
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se encontraron datos de PVStand en ClickHouse")
            return False
            
        logger.info(f"📊 Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame
        logger.info("Procesando datos...")
        df_pvstand = pd.DataFrame(result.result_set, columns=['timestamp', 'module', 'pmax', 'imax', 'umax'])
        
        # Convertir timestamp a datetime y asegurar que esté en UTC
        df_pvstand['timestamp'] = pd.to_datetime(df_pvstand['timestamp'])
        if df_pvstand['timestamp'].dt.tz is None:
            df_pvstand['timestamp'] = df_pvstand['timestamp'].dt.tz_localize('UTC')
        else:
            df_pvstand['timestamp'] = df_pvstand['timestamp'].dt.tz_convert('UTC')

        # Establecer timestamp como índice
        df_pvstand.set_index('timestamp', inplace=True)
        
        # Ordenar por timestamp (importante para series temporales)
        logger.info("Ordenando datos por timestamp...")
        df_pvstand = df_pvstand.sort_index()
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_pvstand.index.min()}")
        logger.info(f"   Fecha más reciente: {df_pvstand.index.max()}")

        # Verificar que hay datos en el rango especificado
        if len(df_pvstand) == 0:
            logger.warning("⚠️  No se encontraron datos en el rango de fechas especificado.")
            return False

        # Mostrar distribución por módulo
        module_counts = df_pvstand['module'].value_counts()
        logger.info("📊 Distribución por módulo:")
        for module, count in module_counts.items():
            logger.info(f"   - {module}: {count} registros")

        # Crear carpeta específica para PVStand
        section_dir = os.path.join(output_dir, 'pvstand')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'raw_pvstand_iv_data.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_pvstand.to_csv(output_filepath)

        logger.info(f"✅ Datos PVStand desde ClickHouse guardados exitosamente")
        logger.info(f"📊 Total de registros: {len(df_pvstand)}")
        logger.info(f"📅 Rango de fechas: {df_pvstand.index.min()} a {df_pvstand.index.max()}")

        # Mostrar estadísticas básicas por módulo
        logger.info("📊 Estadísticas de los datos por módulo:")
        for module in ['perc1fixed', 'perc2fixed']:
            if module in df_pvstand['module'].values:
                module_data = df_pvstand[df_pvstand['module'] == module]
                logger.info(f"\n{module}:")
                logger.info(f"   pmax - Rango: {module_data['pmax'].min():.3f} a {module_data['pmax'].max():.3f}")
                logger.info(f"   imax - Rango: {module_data['imax'].min():.3f} a {module_data['imax'].max():.3f}")
                logger.info(f"   umax - Rango: {module_data['umax'].min():.3f} a {module_data['umax'].max():.3f}")
        
        # Mostrar información sobre la estructura de datos
        logger.info("\n📋 Estructura de datos del PVStand:")
        logger.info(f"   - module: Identificador del módulo (perc1fixed/perc2fixed)")
        logger.info(f"   - pmax: Potencia máxima del módulo")
        logger.info(f"   - imax: Corriente máxima del módulo")
        logger.info(f"   - umax: Voltaje máximo del módulo")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos PVStand desde ClickHouse: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


def download_pvstand_temperature_clickhouse(start_date, end_date, output_dir):
    """
    Descarga y procesa datos de temperatura de módulos PVStand desde ClickHouse.
    
    Esta función:
    - Esquema: "PSDA"
    - Tabla: "fixed_plant_atamo_1" (misma tabla que RefCells)
    - Columnas: "timestamp", "1TE416(C)", "1TE418(C)"
    - Filtra por horario (13:00-18:00) como en el análisis de PVStand
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("🌡️  Iniciando descarga de datos de temperatura PVStand desde ClickHouse...")
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
        
        # Consultar datos de temperatura desde ClickHouse
        # Los datos están en la misma tabla que RefCells: PSDA.fixed_plant_atamo_1
        logger.info("Consultando datos de temperatura PVStand desde ClickHouse...")
        query = f"""
        SELECT 
            timestamp,
            `1TE416(C)`,
            `1TE418(C)`
        FROM PSDA.fixed_plant_atamo_1 
        WHERE timestamp >= '{start_str}' AND timestamp <= '{end_str}'
        ORDER BY timestamp
        """
        
        logger.info("Ejecutando consulta ClickHouse...")
        logger.info(f"Consulta: {query[:200]}...")
        
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se encontraron datos de temperatura PVStand en ClickHouse")
            logger.info("💡 Verificando tabla: PSDA.fixed_plant_atamo_1")
            return False
            
        logger.info(f"📊 Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame
        logger.info("Procesando datos...")
        df_temp = pd.DataFrame(result.result_set, columns=['timestamp', '1TE416(C)', '1TE418(C)'])
        
        # Convertir timestamp a datetime y asegurar que esté en UTC
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
        if df_temp['timestamp'].dt.tz is None:
            df_temp['timestamp'] = df_temp['timestamp'].dt.tz_localize('UTC')
        else:
            df_temp['timestamp'] = df_temp['timestamp'].dt.tz_convert('UTC')
        
        # Establecer timestamp como índice
        df_temp.set_index('timestamp', inplace=True)
        
        # Ordenar por timestamp
        logger.info("Ordenando datos por timestamp...")
        df_temp = df_temp.sort_index()
        
        # Filtrar por horario (13:00-18:00) como en el análisis de PVStand
        logger.info("Aplicando filtro horario (13:00-18:00)...")
        df_temp_filtered = df_temp.between_time('13:00', '18:00')
        logger.info(f"📊 Registros después del filtro horario: {len(df_temp_filtered)}")
        
        # Renombrar índice a TIMESTAMP para compatibilidad con el código existente
        df_temp_filtered.index.name = 'TIMESTAMP'
        
        # Resetear índice para guardar con TIMESTAMP como columna
        df_temp_filtered = df_temp_filtered.reset_index()
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_temp_filtered['TIMESTAMP'].min()}")
        logger.info(f"   Fecha más reciente: {df_temp_filtered['TIMESTAMP'].max()}")
        
        # Verificar que hay datos en el rango especificado
        if len(df_temp_filtered) == 0:
            logger.warning("⚠️  No se encontraron datos en el rango de fechas y horario especificado.")
            return False
        
        # Crear carpeta específica para PVStand temperatura
        section_dir = os.path.join(output_dir, 'pvstand')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'data_temp.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_temp_filtered.to_csv(output_filepath, index=False)
        
        logger.info(f"✅ Datos de temperatura PVStand desde ClickHouse guardados exitosamente")
        logger.info(f"📊 Total de registros: {len(df_temp_filtered)}")
        logger.info(f"📅 Rango de fechas: {df_temp_filtered['TIMESTAMP'].min()} a {df_temp_filtered['TIMESTAMP'].max()}")
        
        # Mostrar estadísticas básicas
        logger.info("📊 Estadísticas de los datos:")
        if '1TE416(C)' in df_temp_filtered.columns:
            logger.info(f"   1TE416(C) - Rango: {df_temp_filtered['1TE416(C)'].min():.2f} a {df_temp_filtered['1TE416(C)'].max():.2f} °C")
            logger.info(f"   1TE416(C) - Promedio: {df_temp_filtered['1TE416(C)'].mean():.2f} °C")
        if '1TE418(C)' in df_temp_filtered.columns:
            logger.info(f"   1TE418(C) - Rango: {df_temp_filtered['1TE418(C)'].min():.2f} a {df_temp_filtered['1TE418(C)'].max():.2f} °C")
            logger.info(f"   1TE418(C) - Promedio: {df_temp_filtered['1TE418(C)'].mean():.2f} °C")
        
        # Mostrar información sobre la estructura de datos
        logger.info("\n📋 Estructura de datos de temperatura PVStand:")
        logger.info(f"   - TIMESTAMP: Fecha y hora de la medición")
        logger.info(f"   - 1TE416(C): Temperatura del módulo sucio (perc1fixed)")
        logger.info(f"   - 1TE418(C): Temperatura del módulo de referencia (perc2fixed)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos de temperatura PVStand desde ClickHouse: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


def download_solys2_clickhouse(start_date, end_date, output_dir):
    """
    Descarga y procesa datos de radiación solar (Solys2) desde ClickHouse.
    
    Esta función:
    - Esquema: "PSDA"
    - Tabla: "meteo6857"
    - Columnas: timestamp, GHIAvg, DHIAvg, DNIAvg
    - Renombra columnas: GHIAvg -> GHI, DHIAvg -> DHI, DNIAvg -> DNI
    - Renombra timestamp -> fecha hora
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("☀️  Iniciando descarga de datos Solys2 desde ClickHouse...")
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
        
        # Consultar datos de solys2 desde el esquema PSDA
        # La tabla ya tiene las columnas en formato ancho
        logger.info("Consultando datos Solys2 desde ClickHouse...")
        query = f"""
        SELECT 
            timestamp,
            GHIAvg,
            DHIAvg,
            DNIAvg
        FROM PSDA.meteo6857 
        WHERE timestamp >= '{start_str}' AND timestamp <= '{end_str}'
        ORDER BY timestamp
        """
        
        logger.info("Ejecutando consulta ClickHouse...")
        logger.info(f"Consulta: {query[:200]}...")
        
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se encontraron datos de Solys2 en ClickHouse")
            return False
            
        logger.info(f"📊 Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame
        logger.info("Procesando datos...")
        df_solys2 = pd.DataFrame(result.result_set, columns=['timestamp', 'GHIAvg', 'DHIAvg', 'DNIAvg'])
        
        # Convertir timestamp a datetime y asegurar que esté en UTC
        df_solys2['timestamp'] = pd.to_datetime(df_solys2['timestamp'])
        if df_solys2['timestamp'].dt.tz is None:
            df_solys2['timestamp'] = df_solys2['timestamp'].dt.tz_localize('UTC')
        else:
            df_solys2['timestamp'] = df_solys2['timestamp'].dt.tz_convert('UTC')

        # Establecer timestamp como índice
        df_solys2.set_index('timestamp', inplace=True)
        
        # Renombrar el índice a 'fecha hora'
        df_solys2.index.name = 'fecha hora'
        
        # Renombrar las columnas: GHIAvg -> GHI, DHIAvg -> DHI, DNIAvg -> DNI
        column_rename_map = {
            'GHIAvg': 'GHI',
            'DHIAvg': 'DHI',
            'DNIAvg': 'DNI'
        }
        
        df_solys2.rename(columns=column_rename_map, inplace=True)
        
        # Reordenar columnas en el orden especificado: fecha hora, GHI, DHI, DNI
        ordered_columns = ['GHI', 'DHI', 'DNI']
        available_columns = [col for col in ordered_columns if col in df_solys2.columns]
        
        # Si hay columnas adicionales, agregarlas al final
        other_columns = [col for col in df_solys2.columns if col not in ordered_columns]
        final_column_order = available_columns + other_columns
        
        df_solys2 = df_solys2[final_column_order]
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_solys2.index.min()}")
        logger.info(f"   Fecha más reciente: {df_solys2.index.max()}")

        # Verificar que hay datos en el rango especificado
        if len(df_solys2) == 0:
            logger.warning("⚠️  No se encontraron datos en el rango de fechas especificado.")
            return False

        # Crear carpeta específica para Solys2
        section_dir = os.path.join(output_dir, 'solys2')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'raw_solys2_data.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_solys2.to_csv(output_filepath)

        logger.info(f"✅ Datos Solys2 desde ClickHouse guardados exitosamente")
        logger.info(f"📊 Total de registros: {len(df_solys2)}")
        logger.info(f"📅 Rango de fechas: {df_solys2.index.min()} a {df_solys2.index.max()}")
        logger.info(f"📊 Columnas: {', '.join(df_solys2.columns.tolist())}")

        # Mostrar estadísticas básicas
        logger.info("📊 Estadísticas de los datos:")
        for col in ['GHI', 'DHI', 'DNI']:
            if col in df_solys2.columns:
                logger.info(f"   {col} - Rango: {df_solys2[col].min():.3f} a {df_solys2[col].max():.3f}")
                logger.info(f"   {col} - Promedio: {df_solys2[col].mean():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos Solys2: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


def download_refcells_clickhouse(start_date, end_date, output_dir):
    """
    Descarga y procesa datos de celdas de referencia desde ClickHouse.
    
    Esta función descarga desde ClickHouse:
    - Esquema: PSDA
    - Tabla: fixed_plant_atamo_1
    - Columnas: timestamp, 1RC411(w.m-2), 1RC412(w.m-2)
    
    Args:
        start_date (datetime): Fecha de inicio del rango (con timezone)
        end_date (datetime): Fecha de fin del rango (con timezone)
        output_dir (str): Directorio donde guardar los archivos
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    logger.info("🔋 Iniciando descarga de datos de celdas de referencia desde ClickHouse...")
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
        
        # Convertir fechas al formato para ClickHouse
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
        
        # Configuración de la consulta
        schema = "PSDA"
        table = "fixed_plant_atamo_1"
        columns_to_query = ["timestamp", "1RC411(w.m-2)", "1RC412(w.m-2)"]
        
        logger.info(f"📊 Configuración:")
        logger.info(f"   Esquema: {schema}")
        logger.info(f"   Tabla: {table}")
        logger.info(f"   Columnas: {', '.join(columns_to_query)}")
        
        # Consultar datos
        query = f"""
        SELECT 
            timestamp,
            `1RC411(w.m-2)`,
            `1RC412(w.m-2)`
        FROM {schema}.{table}
        WHERE timestamp >= '{start_str}' AND timestamp <= '{end_str}'
        ORDER BY timestamp
        """
        
        logger.info("Ejecutando consulta ClickHouse...")
        logger.info(f"Consulta: {query[:200]}...")
        
        result = client.query(query)
        
        if not result.result_set:
            logger.warning("⚠️  No se encontraron datos de ref cells en ClickHouse")
            return False
        
        logger.info(f"📊 Datos obtenidos: {len(result.result_set)} registros")
        
        # Convertir a DataFrame
        logger.info("Procesando datos...")
        df_refcells = pd.DataFrame(result.result_set, columns=columns_to_query)
        
        # Convertir timestamp a datetime
        df_refcells['timestamp'] = pd.to_datetime(df_refcells['timestamp'])
        if df_refcells['timestamp'].dt.tz is None:
            df_refcells['timestamp'] = df_refcells['timestamp'].dt.tz_localize('UTC')
        else:
            df_refcells['timestamp'] = df_refcells['timestamp'].dt.tz_convert('UTC')
        
        # Establecer timestamp como índice para resample
        df_refcells.set_index('timestamp', inplace=True)
        df_refcells = df_refcells.sort_index()
        
        # Aplicar resample a 1 minuto (promedio) si es necesario
        # Verificar si ya está en resolución de 1 minuto
        if len(df_refcells) > 1:
            time_diff = df_refcells.index.to_series().diff().median()
            if time_diff > pd.Timedelta(minutes=1.5):
                logger.info("Aplicando resample a 1 minuto...")
                df_refcells = df_refcells.resample('1min').mean().dropna(how='all')
                logger.info(f"📊 Registros después del resample: {len(df_refcells)}")
        
        # Resetear índice para guardar
        df_refcells = df_refcells.reset_index()
        
        # Mostrar información sobre el rango de fechas en los datos
        logger.info(f"📅 Rango de fechas en los datos:")
        logger.info(f"   Fecha más antigua: {df_refcells['timestamp'].min()}")
        logger.info(f"   Fecha más reciente: {df_refcells['timestamp'].max()}")
        
        # Verificar que hay datos en el rango especificado
        if len(df_refcells) == 0:
            logger.warning("⚠️  No se encontraron datos en el rango de fechas especificado.")
            return False
        
        # Crear carpeta específica para RefCells
        section_dir = os.path.join(output_dir, 'refcells')
        os.makedirs(section_dir, exist_ok=True)
        logger.info(f"📁 Carpeta de sección: {section_dir}")
        
        # Guardar datos
        output_filepath = os.path.join(section_dir, 'refcells_data.csv')
        logger.info(f"💾 Guardando datos en: {output_filepath}")
        df_refcells.to_csv(output_filepath, index=False)
        
        logger.info(f"✅ Datos de celdas de referencia desde ClickHouse guardados exitosamente")
        logger.info(f"📊 Total de registros: {len(df_refcells)}")
        logger.info(f"📊 Columnas: {', '.join(df_refcells.columns.tolist())}")
        logger.info(f"📅 Rango de fechas: {df_refcells['timestamp'].min()} a {df_refcells['timestamp'].max()}")
        
        # Mostrar estadísticas básicas
        logger.info("📊 Estadísticas de los datos:")
        for col in ['1RC411(w.m-2)', '1RC412(w.m-2)']:
            if col in df_refcells.columns:
                logger.info(f"   {col} - Rango: {df_refcells[col].min():.3f} a {df_refcells[col].max():.3f} W/m²")
                logger.info(f"   {col} - Promedio: {df_refcells[col].mean():.3f} W/m²")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en la descarga de datos de celdas de referencia desde ClickHouse: {e}")
        import traceback
        logger.error(f"Detalles del error:\n{traceback.format_exc()}")
        return False
    finally:
        if client:
            logger.info("Cerrando conexión a ClickHouse...")
            client.close()
            logger.info("✅ Conexión a ClickHouse cerrada")


# ============================================================================
# MENÚ INTERACTIVO PARA SELECCIONAR QUÉ DESCARGAR
# ============================================================================

def mostrar_menu():
    """Muestra el menú de opciones de descarga."""
    print("\n" + "="*60)
    print("MENÚ DE DESCARGA DE DATOS")
    print("="*60)
    print("\nOpciones disponibles desde ClickHouse:")
    print("  1. IV600 (puntos característicos)")
    print("  2. Curvas IV600")
    print("  3. Ambas (IV600 y PV Glasses)")
    print("  4. PV Glasses")
    print("  5. DustIQ")
    print("  6. Soiling Kit")
    print("  7. PVStand")
    print("  8. PVStand Temperatura")
    print("  9. Solys2 (Radiación solar)")
    print(" 10. RefCells (Celdas de referencia)")
    print("  0. Salir")
    print("-"*60)


def ejecutar_descargas(start_date, end_date, opcion, fotoceldas_seleccionadas=None):
    """
    Ejecuta las descargas según la opción seleccionada.
    
    Args:
        start_date (datetime): Fecha de inicio
        end_date (datetime): Fecha de fin
        opcion (str): Opción seleccionada ('1', '2', '3', o '0')
        fotoceldas_seleccionadas (list, optional): Lista de fotoceldas seleccionadas para PV Glasses
    """
    resultados = {}
    
    if opcion == '1':
        # Solo IV600 desde ClickHouse
        print("\n🔋 Iniciando descarga de IV600 desde ClickHouse...")
        resultados['iv600'] = download_iv600(start_date, end_date, OUTPUT_DIR)

    elif opcion == '2':
        # Curvas IV completas desde ClickHouse
        print("\n🔋 Iniciando descarga de curvas IV completas desde ClickHouse...")
        # Solicitar rango horario
        hora_inicio, hora_fin = configurar_rango_horario()
        resultados['iv600_curves_complete'] = download_iv600_curves_complete(
            start_date, end_date, OUTPUT_DIR, hora_inicio, hora_fin
        )

    elif opcion == '3':
        # Ambas: IV600 y PV Glasses desde ClickHouse
        print("\n🔋 Iniciando descarga de IV600 desde ClickHouse...")
        resultados['iv600'] = download_iv600(start_date, end_date, OUTPUT_DIR)
        
        # Limpiar memoria y dar tiempo para cerrar conexiones
        gc.collect()
        
        print("\n🔋 Iniciando descarga de PV Glasses desde ClickHouse...")
        if fotoceldas_seleccionadas is None:
            fotoceldas_seleccionadas = DEFAULT_PHOTODIODES
        resultados['pv_glasses'] = download_pv_glasses(start_date, end_date, OUTPUT_DIR, fotoceldas_seleccionadas)
        
        # Limpiar memoria después de ambas descargas
        gc.collect()
        
    elif opcion == '4':
        # PV Glasses desde ClickHouse
        print("\n🔋 Iniciando descarga de PV Glasses desde ClickHouse...")
        if fotoceldas_seleccionadas is None:
            fotoceldas_seleccionadas = DEFAULT_PHOTODIODES
        resultados['pv_glasses'] = download_pv_glasses(start_date, end_date, OUTPUT_DIR, fotoceldas_seleccionadas)
        
    elif opcion == '5':
        # DustIQ desde ClickHouse
        print("\n🌪️  Iniciando descarga de DustIQ desde ClickHouse...")
        resultados['dustiq'] = download_dustiq_clickhouse(start_date, end_date, OUTPUT_DIR)
        
    elif opcion == '6':
        # Soiling Kit desde ClickHouse
        print("\n🌪️  Iniciando descarga del Soiling Kit desde ClickHouse...")
        resultados['soiling_kit'] = download_soiling_kit_clickhouse(start_date, end_date, OUTPUT_DIR)
        
    elif opcion == '7':
        # PVStand desde ClickHouse
        print("\n🔋 Iniciando descarga de PVStand desde ClickHouse...")
        resultados['pvstand'] = download_pvstand_clickhouse(start_date, end_date, OUTPUT_DIR)
        
    elif opcion == '8':
        # PVStand Temperatura desde ClickHouse
        print("\n🌡️  Iniciando descarga de temperatura PVStand desde ClickHouse...")
        resultados['pvstand_temp'] = download_pvstand_temperature_clickhouse(start_date, end_date, OUTPUT_DIR)
        
    elif opcion == '9':
        # Solys2 desde ClickHouse
        print("\n☀️  Iniciando descarga de Solys2 desde ClickHouse...")
        resultados['solys2'] = download_solys2_clickhouse(start_date, end_date, OUTPUT_DIR)
        
    elif opcion == '10':
        # RefCells desde ClickHouse
        print("\n🔋 Iniciando descarga de RefCells desde ClickHouse...")
        resultados['refcells'] = download_refcells_clickhouse(start_date, end_date, OUTPUT_DIR)
        
    elif opcion == '0':
        print("\n👋 Saliendo...")
        return
    else:
        print("\n❌ Opción inválida. Por favor selecciona una opción válida.")
        return
    
    # Mostrar resumen de resultados
    print("\n" + "="*60)
    print("RESUMEN DE DESCARGAS")
    print("="*60)
    
    if 'iv600' in resultados:
        estado = "✅ Exitoso" if resultados['iv600'] else "❌ Fallido"
        print(f"  IV600: {estado}")
    
    if 'pv_glasses' in resultados:
        estado = "✅ Exitoso" if resultados['pv_glasses'] else "❌ Fallido"
        print(f"  PV Glasses: {estado}")
    
    if 'iv600_curves_complete' in resultados:
        estado = "✅ Exitoso" if resultados['iv600_curves_complete'] else "❌ Fallido"
        print(f"  Curvas IV Completas: {estado}")
    
    if 'dustiq' in resultados:
        estado = "✅ Exitoso" if resultados['dustiq'] else "❌ Fallido"
        print(f"  DustIQ: {estado}")
    
    if 'soiling_kit' in resultados:
        estado = "✅ Exitoso" if resultados['soiling_kit'] else "❌ Fallido"
        print(f"  Soiling Kit: {estado}")
    
    if 'pvstand' in resultados:
        estado = "✅ Exitoso" if resultados['pvstand'] else "❌ Fallido"
        print(f"  PVStand: {estado}")
    
    if 'pvstand_temp' in resultados:
        estado = "✅ Exitoso" if resultados['pvstand_temp'] else "❌ Fallido"
        print(f"  PVStand Temperatura: {estado}")
    
    if 'solys2' in resultados:
        estado = "✅ Exitoso" if resultados['solys2'] else "❌ Fallido"
        print(f"  Solys2: {estado}")
    
    if 'refcells' in resultados:
        estado = "✅ Exitoso" if resultados['refcells'] else "❌ Fallido"
        print(f"  RefCells: {estado}")
    
    print("="*60 + "\n")


# ============================================================================
# CONFIGURACIÓN INICIAL AL EJECUTAR EL SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Configurar fechas de forma interactiva
    START_DATE, END_DATE = configurar_fechas()
    
    # Mostrar configuración final
    logger.info(f"Rango de fechas: {START_DATE.strftime('%Y-%m-%d')} a {END_DATE.strftime('%Y-%m-%d')}")
    logger.info(f"Directorio de salida: {OUTPUT_DIR}")
    
    # Mostrar menú y ejecutar descargas
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ").strip()
        
        if opcion == '0':
            print("\n👋 ¡Hasta luego!")
            break
        
        if opcion in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            # Si la opción incluye PV Glasses, permitir seleccionar fotoceldas
            fotoceldas_seleccionadas = None
            if opcion in ['3', '4']:
                fotoceldas_seleccionadas = seleccionar_fotoceldas()
            
            ejecutar_descargas(START_DATE, END_DATE, opcion, fotoceldas_seleccionadas)
            
            # Preguntar si quiere hacer otra descarga
            continuar = input("\n¿Deseas realizar otra descarga? (s/n): ").strip().lower()
            if continuar != 's':
                print("\n👋 ¡Hasta luego!")
                break
        else:
            print("\n❌ Opción inválida. Por favor selecciona 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 o 10.")

