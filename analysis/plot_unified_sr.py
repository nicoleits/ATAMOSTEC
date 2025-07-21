import os
import pandas as pd
import matplotlib.pyplot as plt
from config import paths

# Definir rutas de los CSV unificados
unified_dir = os.path.join(paths.BASE_OUTPUT_CSV_DIR, 'unified_sr_csvs')
unified_daily_csv = os.path.join(unified_dir, 'unified_sr_daily.csv')
unified_weekly_csv = os.path.join(unified_dir, 'unified_sr_weekly.csv')

# Crear carpeta para los gráficos unificados
output_graph_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, 'unified_sr_graphs')
os.makedirs(output_graph_dir, exist_ok=True)

# Función para cargar y preparar datos
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return None
    
    # Cargar CSV con low_memory=False para evitar advertencias de tipos mixtos
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Columnas en {file_path}: {df.columns.tolist()}")
    
    # Crear una nueva columna 'date' unificada
    df['date'] = pd.NaT
    
    # Mapeo de columnas de fecha por sensor
    date_columns = {
        'dustiq': 'dustiq_Time_Local_Naive',
        'ref_cells': 'ref_cells__time',
        'iv600': 'iv600_fecha_dia' if 'iv600_fecha_dia' in df.columns else 'iv600_fecha'
    }
    
    # Convertir cada columna de fecha a datetime y unificar en 'date'
    for sensor, date_col in date_columns.items():
        if date_col in df.columns:
            # Convertir a datetime y asegurar que todas las fechas sean naive (sin zona horaria)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[date_col] = df[date_col].dt.tz_localize(None)
            
            # Usar la primera fecha válida encontrada
            mask = df['date'].isna()
            df.loc[mask, 'date'] = df.loc[mask, date_col]
    
    # Eliminar filas sin fecha
    df = df.dropna(subset=['date'])
    
    # Ordenar por fecha
    df = df.sort_values('date')
    
    return df

# Cargar datos unificados
df_daily = load_data(unified_daily_csv)
df_weekly = load_data(unified_weekly_csv)

# Función para graficar SR
def plot_sr(df, title, output_file):
    if df is None or df.empty:
        print(f"No hay datos para graficar: {title}")
        return
    
    # Filtrar columnas que contienen 'SR' o son valores de interés
    sr_columns = [col for col in df.columns if 'SR' in col or any(x in col for x in ['Pmax', 'Isc'])]
    
    if not sr_columns:
        print(f"No se encontraron columnas de SR para graficar en: {title}")
        return
    
    plt.figure(figsize=(15, 7))
    for col in sr_columns:
        # Convertir valores a numéricos, ignorando errores
        values = pd.to_numeric(df[col], errors='coerce')
        plt.plot(df['date'], values, label=col)
    
    plt.xlabel('Fecha', fontsize=16)
    plt.ylabel('Soiling Ratio [%]', fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

# Graficar SR diarios
if df_daily is not None:
    plot_sr(df_daily, 'Soiling Ratios Diarios Unificados', os.path.join(output_graph_dir, 'unified_sr_daily.png'))

# Graficar SR semanales
if df_weekly is not None:
    plot_sr(df_weekly, 'Soiling Ratios Semanales Unificados', os.path.join(output_graph_dir, 'unified_sr_weekly.png'))

print(f"Gráficos guardados en: {output_graph_dir}") 