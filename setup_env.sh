#!/bin/bash
# Script para configurar el entorno virtual e instalar dependencias
# para el proyecto de descarga de datos fotovoltaicos

echo "=========================================="
echo "Configuración del Entorno Virtual"
echo "=========================================="
echo ""

# Verificar si Python 3 está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado"
    echo "Por favor instala Python 3.8 o superior"
    exit 1
fi

echo "✅ Python 3 encontrado: $(python3 --version)"
echo ""

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual 'venv'..."
    python3 -m venv venv
    echo "✅ Entorno virtual creado"
else
    echo "✅ El entorno virtual 'venv' ya existe"
fi

echo ""

# Activar entorno virtual
echo "🔌 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

echo ""

# Instalar dependencias
echo "📥 Instalando dependencias desde requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo ""
    echo "✅ Dependencias instaladas correctamente"
else
    echo "❌ Error: No se encontró el archivo requirements.txt"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Configuración completada"
echo "=========================================="
echo ""
echo "Para activar el entorno virtual en el futuro, ejecuta:"
echo "  source venv/bin/activate"
echo ""
echo "Para desactivar el entorno virtual, ejecuta:"
echo "  deactivate"
echo ""

