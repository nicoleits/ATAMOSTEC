# Guía de Configuración del Entorno

Este documento explica cómo configurar el entorno de desarrollo para ejecutar el script `download_data.py`.

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Opción 1: Configuración Automática (Recomendada)

Ejecuta el script de configuración automática:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

Este script:
- Crea un entorno virtual llamado `venv`
- Instala todas las dependencias necesarias
- Configura el entorno listo para usar

## Opción 2: Configuración Manual

### Paso 1: Crear entorno virtual

```bash
python3 -m venv venv
```

### Paso 2: Activar el entorno virtual

**En Linux/Mac:**
```bash
source venv/bin/activate
```

**En Windows:**
```bash
venv\Scripts\activate
```

### Paso 3: Actualizar pip

```bash
pip install --upgrade pip
```

### Paso 4: Instalar dependencias

```bash
pip install -r requirements.txt
```

## Dependencias Instaladas

El archivo `requirements.txt` incluye las siguientes librerías:

- **pandas** (>=2.0.0): Manipulación y análisis de datos
- **numpy** (>=1.24.0): Operaciones numéricas
- **clickhouse-connect** (>=0.6.0): Cliente para conectar con ClickHouse
- **influxdb-client** (>=1.38.0): Cliente para conectar con InfluxDB
- **streamlit** (>=1.28.0): Framework para aplicaciones web (si se usa)
- **plotly** (>=5.15.0): Visualización de datos (si se usa)
- **python-dateutil** (>=2.8.0): Utilidades para manejo de fechas

## Verificar la Instalación

Para verificar que todo está instalado correctamente:

```bash
python3 -c "import pandas, numpy, clickhouse_connect, influxdb_client; print('✅ Todas las dependencias están instaladas')"
```

## Ejecutar el Script

Una vez configurado el entorno:

```bash
# Asegúrate de que el entorno virtual esté activado
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Ejecuta el script
python download_data.py
```

## Desactivar el Entorno Virtual

Cuando termines de trabajar:

```bash
deactivate
```

## Solución de Problemas

### Error: "python3: command not found"
- Asegúrate de tener Python 3 instalado
- En algunos sistemas, el comando puede ser `python` en lugar de `python3`

### Error: "pip: command not found"
- Instala pip: `python3 -m ensurepip --upgrade`

### Error al instalar dependencias
- Asegúrate de tener conexión a internet
- Intenta actualizar pip: `pip install --upgrade pip`
- Verifica que estás usando Python 3.8 o superior

### Error de conexión a bases de datos
- Verifica que las credenciales en `download_data.py` sean correctas
- Asegúrate de tener acceso a los servidores ClickHouse e InfluxDB

## Notas

- El entorno virtual (`venv`) está en `.gitignore` y no se sube al repositorio
- Cada desarrollador debe crear su propio entorno virtual
- Las dependencias se actualizan ejecutando: `pip install -r requirements.txt --upgrade`

