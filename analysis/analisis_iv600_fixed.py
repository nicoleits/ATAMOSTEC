import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats
from datetime import datetime
import logging

# Agregar el directorio padre al path para importar módulos locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plot_utils import save_plot

def run_analysis():
    # Crear carpeta de salida para los gráficos
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'graficos_analisis_integrado_py', 'iv600')
    os.makedirs(output_dir, exist_ok=True)

    print("--- Iniciando Análisis de Soiling con datos IV600 desde CSV (VERSIÓN FILTRADA) ---")
    
    # 1. Cargar el archivo raw_iv600_data.csv (como en el notebook original)
    iv600_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datos', 'raw_iv600_data.csv')
    
    try:
        df_iv600_raw = pd.read_csv(
            iv600_csv_path,
            parse_dates=['timestamp'],
            index_col='timestamp'
        )
        print(f"Datos cargados exitosamente desde {iv600_csv_path}")
    except FileNotFoundError:
        print(f"ERROR: Archivo no encontrado en {iv600_csv_path}. Asegúrate de que el archivo exista.")
        df_iv600_raw = pd.DataFrame()
        
    if df_iv600_raw.empty:
        print("DataFrame df_iv600_raw está vacío. No se puede continuar con el análisis IV600.")
        return {}
    
    # 2. Procesar timezone (como en el notebook original)
    if df_iv600_raw.index.tz is not None:
        print(f"Timestamp original con timezone: {df_iv600_raw.index.tz}")
        df_iv600_raw.index = df_iv600_raw.index.tz_convert(None)
    
    # 3. Mapeo de columnas (como en el notebook original)
    column_mapping = {
        'module': 'Module',
        'pmp': 'Pmax',
        'isc': 'Isc',
        'voc': 'Voc',
        'imp': 'Imp',
        'vmp': 'Vmp'
    }
    
    df_iv600_processed = df_iv600_raw.rename(columns=column_mapping)
    
    print("\nDataFrame IV600 (df_iv600_processed) - Primeras filas:")
    print(df_iv600_processed.head())
    print(f"\nMódulos presentes: {df_iv600_processed['Module'].unique()}")
    
    # 4. Procesamiento Avanzado de Datos IV600 (como en el notebook original)
    modules_of_interest = ['1MD434', '1MD439', '1MD440']
    valid_modules = [mod for mod in modules_of_interest if mod in df_iv600_processed['Module'].unique()]
    
    if valid_modules:
        print(f"\nMódulos válidos encontrados: {valid_modules}")
        
        # Crear DataFrames de 5 minutos por módulo (PROCESAMIENTO CLAVE DEL NOTEBOOK)
        pmax_module_dfs = []
        isc_module_dfs = []
        valid_module_keys = []
        
        for mod_name in valid_modules:
            df_mod = df_iv600_processed[df_iv600_processed['Module'] == mod_name]
            if not df_mod.empty:
                # Resample a 5 minutos (PASO CRÍTICO)
                df_mod_resampled = df_mod[['Pmax', 'Isc']].resample('5min').mean()
                
                if not df_mod_resampled.empty:
                    pmax_module_dfs.append(df_mod_resampled[['Pmax']])
                    isc_module_dfs.append(df_mod_resampled[['Isc']])
                    valid_module_keys.append(mod_name)
        
        if pmax_module_dfs and isc_module_dfs:
            # Concatenar todos los módulos (MULTIINDEX COMO EN EL NOTEBOOK)
            df_pmax_5T = pd.concat(pmax_module_dfs, keys=valid_module_keys, axis=1)
            df_isc_5T = pd.concat(isc_module_dfs, keys=valid_module_keys, axis=1)
            
            print(f"\nDataFrame Pmax (5 min, multi-módulo) df_pmax_5T - Shape: {df_pmax_5T.shape}")
            print(f"DataFrame Isc (5 min, multi-módulo) df_isc_5T - Shape: {df_isc_5T.shape}")
            
            # 5. CORRECCIÓN DE GAPS TEMPORALES (PASO CRÍTICO FALTANTE EN EL SCRIPT ANTERIOR)
            print("\nAplicando corrección de gaps temporales...")
            df_pmax_corrected_5T = df_pmax_5T.copy()
            gaps_corrected_pmax = 0
            
            for col_tuple in df_pmax_corrected_5T.columns:
                for idx_timestamp in df_pmax_corrected_5T.index:
                    idx_future = idx_timestamp + pd.Timedelta(minutes=5)
                    if pd.isna(df_pmax_corrected_5T.loc[idx_timestamp, col_tuple]):
                        if (idx_future in df_pmax_corrected_5T.index and 
                            pd.notna(df_pmax_corrected_5T.loc[idx_future, col_tuple])):
                            df_pmax_corrected_5T.loc[idx_timestamp, col_tuple] = df_pmax_corrected_5T.loc[idx_future, col_tuple]
                            df_pmax_corrected_5T.loc[idx_future, col_tuple] = np.nan
                            gaps_corrected_pmax += 1

            df_isc_corrected_5T = df_isc_5T.copy()
            gaps_corrected_isc = 0
            
            for col_tuple in df_isc_corrected_5T.columns:
                for idx_timestamp in df_isc_corrected_5T.index:
                    idx_future = idx_timestamp + pd.Timedelta(minutes=5)
                    if pd.isna(df_isc_corrected_5T.loc[idx_timestamp, col_tuple]):
                        if (idx_future in df_isc_corrected_5T.index and 
                            pd.notna(df_isc_corrected_5T.loc[idx_future, col_tuple])):
                            df_isc_corrected_5T.loc[idx_timestamp, col_tuple] = df_isc_corrected_5T.loc[idx_future, col_tuple]
                            df_isc_corrected_5T.loc[idx_future, col_tuple] = np.nan
                            gaps_corrected_isc += 1
            
            print(f"Gaps corregidos Pmax: {gaps_corrected_pmax}")
            print(f"Gaps corregidos Isc: {gaps_corrected_isc}")
            
            # 6. Remuestreo a promedio diario (COMO EN EL NOTEBOOK)
            df_pmax_daily = df_pmax_corrected_5T.resample('1D').mean().dropna(how='all')
            df_isc_daily = df_isc_corrected_5T.resample('1D').mean().dropna(how='all')

            print(f"\nDataFrame Pmax diario shape: {df_pmax_daily.shape}")
            print(f"DataFrame Isc diario shape: {df_isc_daily.shape}")
            
            # 6.1. ✅ NUEVO: Resample diario con Q25
            df_pmax_daily_q25 = df_pmax_corrected_5T.resample('1D').quantile(0.25).dropna(how='all')
            df_isc_daily_q25 = df_isc_corrected_5T.resample('1D').quantile(0.25).dropna(how='all')
            
            print(f"DataFrame Pmax diario Q25 shape: {df_pmax_daily_q25.shape}")
            print(f"DataFrame Isc diario Q25 shape: {df_isc_daily_q25.shape}")
            
            # 6.1.5. ✅ NUEVO: ELIMINAR EL PRIMER DÍA DE DATOS
            print("\n--- Eliminando el primer día de datos ---")
            
            def remove_first_day(df):
                """
                Elimina el primer día de datos del DataFrame
                """
                if not df.empty and len(df) > 1:
                    first_date = df.index[0]
                    df_filtered = df.iloc[1:]  # Eliminar primera fila
                    print(f"  Eliminado primer día: {first_date.strftime('%Y-%m-%d')}")
                    print(f"  Shape antes: {len(df)}, después: {len(df_filtered)}")
                    return df_filtered
                else:
                    print("  No hay suficientes datos para eliminar el primer día")
                    return df
            
            # Aplicar eliminación del primer día a todos los DataFrames diarios
            df_pmax_daily_original_shape = df_pmax_daily.shape
            df_isc_daily_original_shape = df_isc_daily.shape
            df_pmax_daily_q25_original_shape = df_pmax_daily_q25.shape
            df_isc_daily_q25_original_shape = df_isc_daily_q25.shape
            
            df_pmax_daily = remove_first_day(df_pmax_daily)
            df_isc_daily = remove_first_day(df_isc_daily)
            df_pmax_daily_q25 = remove_first_day(df_pmax_daily_q25)
            df_isc_daily_q25 = remove_first_day(df_isc_daily_q25)
            
            print(f"📊 RESUMEN ELIMINACIÓN PRIMER DÍA:")
            print(f"  - Pmax diario: {df_pmax_daily_original_shape} → {df_pmax_daily.shape}")
            print(f"  - Isc diario: {df_isc_daily_original_shape} → {df_isc_daily.shape}")
            print(f"  - Pmax diario Q25: {df_pmax_daily_q25_original_shape} → {df_pmax_daily_q25.shape}")
            print(f"  - Isc diario Q25: {df_isc_daily_q25_original_shape} → {df_isc_daily_q25.shape}")
            
            # 6.2. ✅ NUEVO: FILTRO E INTERPOLACIÓN para valores bajos anómalos
            print("\n--- Aplicando filtro e interpolación para valores bajos ---")
            
            def apply_interpolation_filter(df_power, df_current, power_threshold=250, current_threshold=8.0):
                """
                Aplica filtro e interpolación para valores por debajo de umbrales mínimos
                
                Parameters:
                - df_power: DataFrame con datos de potencia
                - df_current: DataFrame con datos de corriente  
                - power_threshold: Umbral mínimo de potencia (W)
                - current_threshold: Umbral mínimo de corriente (A)
                
                Returns:
                - df_power_filtered, df_current_filtered: DataFrames con interpolación aplicada
                """
                df_power_filtered = df_power.copy()
                df_current_filtered = df_current.copy()
                
                interpolated_power_count = 0
                interpolated_current_count = 0
                
                # Filtro e interpolación para potencia
                for col in df_power_filtered.columns:
                    # Identificar valores por debajo del umbral
                    low_values_mask = df_power_filtered[col] < power_threshold
                    low_values_count = low_values_mask.sum()
                    
                    if low_values_count > 0:
                        print(f"  Potencia {col}: {low_values_count} valores < {power_threshold}W detectados")
                        interpolated_power_count += low_values_count
                        
                        # Reemplazar con NaN e interpolar
                        df_power_filtered.loc[low_values_mask, col] = np.nan
                        df_power_filtered[col] = df_power_filtered[col].interpolate(method='linear')
                
                # Filtro e interpolación para corriente
                for col in df_current_filtered.columns:
                    # Identificar valores por debajo del umbral
                    low_values_mask = df_current_filtered[col] < current_threshold
                    low_values_count = low_values_mask.sum()
                    
                    if low_values_count > 0:
                        print(f"  Corriente {col}: {low_values_count} valores < {current_threshold}A detectados")
                        interpolated_current_count += low_values_count
                        
                        # Reemplazar con NaN e interpolar
                        df_current_filtered.loc[low_values_mask, col] = np.nan
                        df_current_filtered[col] = df_current_filtered[col].interpolate(method='linear')
                
                return df_power_filtered, df_current_filtered, interpolated_power_count, interpolated_current_count
            
            # Aplicar filtro e interpolación a datos diarios promedio
            df_pmax_daily, df_isc_daily, interp_power_daily, interp_current_daily = apply_interpolation_filter(
                df_pmax_daily, df_isc_daily, power_threshold=250, current_threshold=8.0
            )
            
            # Aplicar filtro e interpolación a datos diarios Q25
            df_pmax_daily_q25, df_isc_daily_q25, interp_power_q25, interp_current_q25 = apply_interpolation_filter(
                df_pmax_daily_q25, df_isc_daily_q25, power_threshold=250, current_threshold=8.0
            )
            
            print(f"📊 RESUMEN INTERPOLACIÓN:")
            print(f"  - Potencia diaria promedio: {interp_power_daily} valores interpolados")
            print(f"  - Corriente diaria promedio: {interp_current_daily} valores interpolados")
            print(f"  - Potencia diaria Q25: {interp_power_q25} valores interpolados")
            print(f"  - Corriente diaria Q25: {interp_current_q25} valores interpolados")
            
            # 7. Resample semanal con cuantil 0.25 - USANDO DATOS FILTRADOS
            # CAMBIADO: Ahora usa los datos diarios Q25 filtrados en lugar de datos de 5min sin filtrar
            df_pmax_weekly = df_pmax_daily_q25.resample('W').mean().dropna(how='all')
            df_isc_weekly = df_isc_daily_q25.resample('W').mean().dropna(how='all')
            
            print(f"DataFrame Pmax semanal Q25 (desde datos filtrados) shape: {df_pmax_weekly.shape}")
            print(f"DataFrame Isc semanal Q25 (desde datos filtrados) shape: {df_isc_weekly.shape}")
            print("✅ Datos semanales ahora usan datos diarios filtrados como base")
            
            # 6.3. NUEVO: Gráficos de valores absolutos diarios de Potencia y Corriente
            print("\n--- Generando gráficos de valores absolutos diarios ---")
            
            # Definir paleta de colores para los gráficos
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            

            # Gráfico combinado de Potencia y Corriente (ejes separados)
            if not df_pmax_daily.empty and not df_isc_daily.empty:
                fig_combo, (ax_combo_pmax, ax_combo_isc) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
                
                # Subplot superior: Potencia
                for idx, col_tuple in enumerate(df_pmax_daily.columns):
                    if not df_pmax_daily[col_tuple].dropna().empty:
                        module_name = col_tuple[0]
                        color = colors[idx % len(colors)]
                        df_pmax_daily[col_tuple].plot(ax=ax_combo_pmax, style='-o', alpha=0.75, 
                                                    label=f'Pmax {module_name}', markersize=4, color=color)
                
                ax_combo_pmax.legend(fontsize=12)
                ax_combo_pmax.set_ylabel('Pmax Power [W]', fontsize=14)
                ax_combo_pmax.grid(True, alpha=0.3)
                ax_combo_pmax.set_title('Daily Power and Current - IV600', fontsize=18)
                ax_combo_pmax.tick_params(axis='both', labelsize=12)
                
                # Subplot inferior: Corriente
                for idx, col_tuple in enumerate(df_isc_daily.columns):
                    if not df_isc_daily[col_tuple].dropna().empty:
                        module_name = col_tuple[0]
                        color = colors[idx % len(colors)]
                        df_isc_daily[col_tuple].plot(ax=ax_combo_isc, style='-o', alpha=0.75, 
                                                   label=f'Isc {module_name}', markersize=4, color=color)
                
                ax_combo_isc.legend(fontsize=12)
                ax_combo_isc.set_ylabel('Isc Current [A]', fontsize=14)
                ax_combo_isc.set_xlabel('Date', fontsize=14)
                ax_combo_isc.grid(True, alpha=0.3)
                ax_combo_isc.tick_params(axis='both', labelsize=12)
                plt.xticks(rotation=30, ha='right', fontsize=12)
                
                plt.tight_layout()
                save_plot(plt.gcf(), 'potencia_corriente_diaria_combo_iv600', subdir='iv600')
                plt.close()
                print("✅ Gráfico combinado de potencia y corriente diaria IV600 generado")
        else:
            print("No hay datos válidos de módulos para continuar.")
            return {}
    else:
        print("No hay módulos válidos.")
        return {}

    # 8. Calcular Soiling Ratios (SR) - APLICANDO FILTRADO CONSISTENTE
    ref_mod_pmax_col = ('1MD439', 'Pmax')
    ref_mod_isc_col = ('1MD439', 'Isc')
    test_mod_434_pmax_col = ('1MD434', 'Pmax')
    test_mod_434_isc_col = ('1MD434', 'Isc')
    test_mod_440_pmax_col = ('1MD440', 'Pmax')
    test_mod_440_isc_col = ('1MD440', 'Isc')

    # === SOILING RATIOS DIARIOS ===
    sr_pmp_iv600 = pd.DataFrame(index=df_pmax_daily.index)
    sr_isc_iv600 = pd.DataFrame(index=df_isc_daily.index)

    # SR Pmp 434vs439 - CON FILTRADO
    if test_mod_434_pmax_col in df_pmax_daily.columns and ref_mod_pmax_col in df_pmax_daily.columns:
        sr_raw = 100 * df_pmax_daily[test_mod_434_pmax_col] / df_pmax_daily[ref_mod_pmax_col]
        sr_pmp_iv600['SR_Pmp_434vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]
        print(f"SR Pmp 434vs439: {len(sr_raw)} -> {len(sr_pmp_iv600['SR_Pmp_434vs439'].dropna())} después del filtrado")

    # SR Pmp 440vs439 - CON FILTRADO
    if test_mod_440_pmax_col in df_pmax_daily.columns and ref_mod_pmax_col in df_pmax_daily.columns:
        sr_raw = 100 * df_pmax_daily[test_mod_440_pmax_col] / df_pmax_daily[ref_mod_pmax_col]
        sr_pmp_iv600['SR_Pmp_440vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]
        print(f"SR Pmp 440vs439: {len(sr_raw)} -> {len(sr_pmp_iv600['SR_Pmp_440vs439'].dropna())} después del filtrado")

    # SR Isc 434vs439 - CON FILTRADO
    if test_mod_434_isc_col in df_isc_daily.columns and ref_mod_isc_col in df_isc_daily.columns:
        sr_raw = 100 * df_isc_daily[test_mod_434_isc_col] / df_isc_daily[ref_mod_isc_col]
        sr_isc_iv600['SR_Isc_434vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]
        print(f"SR Isc 434vs439: {len(sr_raw)} -> {len(sr_isc_iv600['SR_Isc_434vs439'].dropna())} después del filtrado")

    # SR Isc 440vs439 - CON FILTRADO
    if test_mod_440_isc_col in df_isc_daily.columns and ref_mod_isc_col in df_isc_daily.columns:
        sr_raw = 100 * df_isc_daily[test_mod_440_isc_col] / df_isc_daily[ref_mod_isc_col]
        sr_isc_iv600['SR_Isc_440vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]
        print(f"SR Isc 440vs439: {len(sr_raw)} -> {len(sr_isc_iv600['SR_Isc_440vs439'].dropna())} después del filtrado")

    # === SOILING RATIOS SEMANALES - CORREGIDO ===
    # Primero calculamos los SR diarios Q25 para ser consistentes
    sr_pmp_iv600_daily_q25_for_weekly = pd.DataFrame(index=df_pmax_daily_q25.index)
    sr_isc_iv600_daily_q25_for_weekly = pd.DataFrame(index=df_isc_daily_q25.index)

    # SR Pmp Q25 diarios - CON FILTRADO (para base semanal)
    if test_mod_434_pmax_col in df_pmax_daily_q25.columns and ref_mod_pmax_col in df_pmax_daily_q25.columns:
        sr_raw = 100 * df_pmax_daily_q25[test_mod_434_pmax_col] / df_pmax_daily_q25[ref_mod_pmax_col]
        sr_pmp_iv600_daily_q25_for_weekly['SR_Pmp_434vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]

    if test_mod_440_pmax_col in df_pmax_daily_q25.columns and ref_mod_pmax_col in df_pmax_daily_q25.columns:
        sr_raw = 100 * df_pmax_daily_q25[test_mod_440_pmax_col] / df_pmax_daily_q25[ref_mod_pmax_col]
        sr_pmp_iv600_daily_q25_for_weekly['SR_Pmp_440vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]

    # SR Isc Q25 diarios - CON FILTRADO (para base semanal)
    if test_mod_434_isc_col in df_isc_daily_q25.columns and ref_mod_isc_col in df_isc_daily_q25.columns:
        sr_raw = 100 * df_isc_daily_q25[test_mod_434_isc_col] / df_isc_daily_q25[ref_mod_isc_col]
        sr_isc_iv600_daily_q25_for_weekly['SR_Isc_434vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]

    if test_mod_440_isc_col in df_isc_daily_q25.columns and ref_mod_isc_col in df_isc_daily_q25.columns:
        sr_raw = 100 * df_isc_daily_q25[test_mod_440_isc_col] / df_isc_daily_q25[ref_mod_isc_col]
        sr_isc_iv600_daily_q25_for_weekly['SR_Isc_440vs439'] = sr_raw[(sr_raw >= 93) & (sr_raw <= 101)]

    # Ahora calculamos los promedios semanales A PARTIR de los SR diarios ya filtrados
    sr_pmp_iv600_weekly = sr_pmp_iv600_daily_q25_for_weekly.resample('W').mean().dropna(how='all')
    sr_isc_iv600_weekly = sr_isc_iv600_daily_q25_for_weekly.resample('W').mean().dropna(how='all')

    print(f"🔧 CORRECCIÓN: SR semanales ahora calculados desde SR diarios Q25 filtrados")
    print(f"   - SR Pmp semanal shape: {sr_pmp_iv600_weekly.shape}")
    print(f"   - SR Isc semanal shape: {sr_isc_iv600_weekly.shape}")

    print(f"\n=== RESUMEN SOILING RATIOS ===")
    print(f"SR Pmp diario: {sr_pmp_iv600.shape}")
    print(f"SR Isc diario: {sr_isc_iv600.shape}")
    print(f"SR Pmp semanal: {sr_pmp_iv600_weekly.shape}")
    print(f"SR Isc semanal: {sr_isc_iv600_weekly.shape}")

    # 9. GRÁFICOS - SIGUIENDO LA LÓGICA EXACTA DE LOS GRÁFICOS SIN PICOS
    
    # 9.1 Gráfico SR Pmp diario (FILTRADO)
    if not sr_pmp_iv600.empty:
        fig1, ax1 = plt.subplots(figsize=(15, 6))
        plot_legend1 = []
        
        for idx, col in enumerate(sr_pmp_iv600.columns):
            if not sr_pmp_iv600[col].dropna().empty:
                color = colors[idx % len(colors)]
                sr_pmp_iv600[col].plot(ax=ax1, style='--o', alpha=0.75, label=col, 
                                      markersize=4, color=color)
                plot_legend1.append(col)
        
        if plot_legend1:
            ax1.legend(plot_legend1, fontsize=14)
            ax1.set_ylabel('Pmp Soiling Ratio [%]', fontsize=16)
            ax1.set_xlabel('Date', fontsize=16)
            ax1.grid(True)
            ax1.set_title('Pmp Soiling Ratios - IV600 (Filtered 93-101%)', fontsize=20)
            ax1.set_ylim(90, 110)
            ax1.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=30, ha='right', fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            save_plot(plt.gcf(), 'sr_pmp_iv600_filtrado', subdir='iv600')
            plt.close()
            print("✅ Gráfico SR Pmp diario filtrado generado")

    # 9.2 Gráfico SR Isc diario (FILTRADO)
    if not sr_isc_iv600.empty:
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        plot_legend2 = []
        
        for idx, col in enumerate(sr_isc_iv600.columns):
            if not sr_isc_iv600[col].dropna().empty:
                color = colors[idx % len(colors)]
                sr_isc_iv600[col].plot(ax=ax2, style='--o', alpha=0.75, label=col, 
                                      markersize=4, color=color)
                plot_legend2.append(col)
        
        if plot_legend2:
            ax2.legend(plot_legend2, fontsize=14)
            ax2.set_ylabel('Isc Soiling Ratio [%]', fontsize=16)
            ax2.set_xlabel('Date', fontsize=16)
            ax2.grid(True)
            ax2.set_title('Isc Soiling Ratios - IV600 (Filtered 93-101%)', fontsize=20)
            ax2.set_ylim(90, 110)
            ax2.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=30, ha='right', fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            save_plot(plt.gcf(), 'sr_isc_iv600_filtrado', subdir='iv600')
            plt.close()
            print("✅ Gráfico SR Isc diario filtrado generado")

    # 9.3 Gráfico SR Pmp semanal Q25 (FILTRADO)
    if not sr_pmp_iv600_weekly.empty:
        fig3, ax3 = plt.subplots(figsize=(15, 6))
        plot_legend3 = []
        
        for idx, col in enumerate(sr_pmp_iv600_weekly.columns):
            if not sr_pmp_iv600_weekly[col].dropna().empty:
                color = colors[idx % len(colors)]
                sr_pmp_iv600_weekly[col].plot(ax=ax3, style='-o', alpha=0.8, label=col, 
                                            markersize=6, linewidth=2, color=color)
                plot_legend3.append(col)
        
        if plot_legend3:
            ax3.legend(plot_legend3, fontsize=14)
            ax3.set_ylabel('Pmp Soiling Ratio [%]', fontsize=16)
            ax3.set_xlabel('Date', fontsize=16)
            ax3.grid(True)
            ax3.set_title('Pmp Soiling Ratios - IV600 (Weekly, Filtered 93-101%)', fontsize=20)
            ax3.set_ylim(90, 110)
            ax3.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=30, ha='right', fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            save_plot(plt.gcf(), 'sr_pmp_iv600_semanal_q25_filtrado', subdir='iv600')
            plt.close()
            print("✅ Gráfico SR Pmp semanal Q25 filtrado generado")

    # 9.4 Gráfico SR Isc semanal Q25 (FILTRADO)
    if not sr_isc_iv600_weekly.empty:
        fig4, ax4 = plt.subplots(figsize=(15, 6))
        plot_legend4 = []
        
        for idx, col in enumerate(sr_isc_iv600_weekly.columns):
            if not sr_isc_iv600_weekly[col].dropna().empty:
                color = colors[idx % len(colors)]
                sr_isc_iv600_weekly[col].plot(ax=ax4, style='-o', alpha=0.8, label=col, 
                                            markersize=6, linewidth=2, color=color)
                plot_legend4.append(col)
        
        if plot_legend4:
            ax4.legend(plot_legend4, fontsize=14)
            ax4.set_ylabel('Isc Soiling Ratio [%]', fontsize=16)
            ax4.set_xlabel('Date', fontsize=16)
            ax4.grid(True)
            ax4.set_title('Isc Soiling Ratios - IV600 (Weekly Q25, Filtered 90-105%)', fontsize=20)
            ax4.set_ylim(90, 110)
            ax4.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=30, ha='right', fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            save_plot(plt.gcf(), 'sr_isc_iv600_semanal_q25_filtrado', subdir='iv600')
            plt.close()
            print("✅ Gráfico SR Isc semanal Q25 filtrado generado")

    # 9.5 Gráfico combinado Pmp vs Isc diario (FILTRADO)
    sr_pmp_434_col_name = 'SR_Pmp_434vs439'
    sr_isc_434_col_name = 'SR_Isc_434vs439'

    if (sr_pmp_434_col_name in sr_pmp_iv600.columns and not sr_pmp_iv600[sr_pmp_434_col_name].dropna().empty and
        sr_isc_434_col_name in sr_isc_iv600.columns and not sr_isc_iv600[sr_isc_434_col_name].dropna().empty):
        
        # Filtrar días que solo tienen datos de una serie pero no de la otra
        sr_pmp_clean = sr_pmp_iv600[sr_pmp_434_col_name].dropna()
        sr_isc_clean = sr_isc_iv600[sr_isc_434_col_name].dropna()
        
        # Encontrar fechas comunes (intersección)
        common_dates = sr_pmp_clean.index.intersection(sr_isc_clean.index)
        
        if len(common_dates) > 0:
            # Filtrar solo las fechas que tienen ambos datos
            sr_pmp_filtered = sr_pmp_clean.loc[common_dates]
            sr_isc_filtered = sr_isc_clean.loc[common_dates]
        
            fig5, ax5 = plt.subplots(figsize=(15, 6))
            sr_pmp_filtered.plot(ax=ax5, style='--o', alpha=0.75, 
                                 label=f'Pmp SR ({sr_pmp_434_col_name})', markersize=4)
            sr_isc_filtered.plot(ax=ax5, style='--o', alpha=0.75, 
                                 label=f'Isc SR ({sr_isc_434_col_name})', markersize=4)
        
            ax5.legend(fontsize=14)
            ax5.set_ylabel('Soiling Ratio [%]', fontsize=16)
            ax5.set_xlabel('Date', fontsize=16)
            ax5.grid(True)
            ax5.set_title('Pmp vs Isc SR Comparison (1MD434/1MD439) - IV600 (Filtered 93-101%)', fontsize=20)
            ax5.set_ylim(90, 110)
            ax5.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=30, ha='right', fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            save_plot(plt.gcf(), 'sr_comparison_pmp_vs_isc_iv600_filtrado', subdir='iv600')
            plt.close()
            print("✅ Gráfico comparación Pmp vs Isc diario filtrado generado")
        else:
            print("⚠️  No hay fechas comunes entre SR Pmp e Isc para el gráfico diario filtrado")

    # 9.6 Gráfico combinado Pmp vs Isc semanal Q25 (FILTRADO)
    if (not sr_pmp_iv600_weekly.empty and not sr_isc_iv600_weekly.empty and
        sr_pmp_434_col_name in sr_pmp_iv600_weekly.columns and not sr_pmp_iv600_weekly[sr_pmp_434_col_name].dropna().empty and
        sr_isc_434_col_name in sr_isc_iv600_weekly.columns and not sr_isc_iv600_weekly[sr_isc_434_col_name].dropna().empty):
        
        # Filtrar fechas comunes para datos semanales
        sr_pmp_weekly_clean = sr_pmp_iv600_weekly[sr_pmp_434_col_name].dropna()
        sr_isc_weekly_clean = sr_isc_iv600_weekly[sr_isc_434_col_name].dropna()
        
        common_dates_weekly = sr_pmp_weekly_clean.index.intersection(sr_isc_weekly_clean.index)
        
        if len(common_dates_weekly) > 0:
            sr_pmp_weekly_filtered = sr_pmp_weekly_clean.loc[common_dates_weekly]
            sr_isc_weekly_filtered = sr_isc_weekly_clean.loc[common_dates_weekly]
        
            fig6, ax6 = plt.subplots(figsize=(15, 6))
            sr_pmp_weekly_filtered.plot(ax=ax6, style='-o', alpha=0.8, 
                                        label=f'Pmp SR ({sr_pmp_434_col_name})', 
                                        markersize=6, linewidth=2)
            sr_isc_weekly_filtered.plot(ax=ax6, style='-o', alpha=0.8, 
                                        label=f'Isc SR ({sr_isc_434_col_name})', 
                                        markersize=6, linewidth=2)
        
            ax6.legend(fontsize=14)
            ax6.set_ylabel('Soiling Ratio [%]', fontsize=16)
            ax6.set_xlabel('Date', fontsize=16)
            ax6.grid(True)
            ax6.set_title('Pmp vs Isc SR Comparison (1MD434/1MD439) - IV600 (Weekly Q25, Filtered 93-101%)', fontsize=20)
            ax6.set_ylim(90, 110)
            ax6.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=30, ha='right', fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            save_plot(plt.gcf(), 'sr_comparison_pmp_vs_isc_iv600_semanal_q25_filtrado', subdir='iv600')
            plt.close()
            print("✅ Gráfico comparación Pmp vs Isc semanal Q25 filtrado generado")
        else:
            print("⚠️  No hay fechas comunes entre SR Pmp e Isc para el gráfico semanal Q25 filtrado")

    # 10. GRÁFICOS ADICIONALES CON LÍNEAS DE TENDENCIA, PENDIENTE Y R²
    
    # Función auxiliar para calcular y mostrar estadísticas de regresión
    def plot_with_trend_line(ax, y_data, label, color, marker_style, is_weekly=False):
        """Agregar línea de tendencia con estadísticas integradas en la leyenda"""
        # Filtrar datos válidos (sin NaN)
        valid_mask = ~pd.isna(y_data)
        if valid_mask.sum() < 2:  # Necesitamos al menos 2 puntos
            return None
            
        y_valid = y_data[valid_mask]
        
        # Convertir fechas a números para la regresión
        x_numeric = pd.to_numeric(y_valid.index)
        
        # Calcular regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_valid)
        
        # Crear línea de tendencia
        trend_line = slope * x_numeric + intercept
        
        # Calcular pendiente CORRECTA usando método directo
        total_days = (y_valid.index[-1] - y_valid.index[0]).days
        total_change = y_valid.values[-1] - y_valid.values[0]
        slope_per_day = total_change / total_days if total_days > 0 else 0
        
        # Si es semanal, convertir a %/semana
        if is_weekly:
            slope_per_period = slope_per_day * 7  # Convertir de %/día a %/semana
            period_unit = "semana"
        else:
            slope_per_period = slope_per_day
            period_unit = "día"
        
        # Crear etiqueta multilínea con estadísticas integradas
        multiline_label = f"{label}\nTrend: {slope_per_period:.4f} %/{period_unit} R² = {r_value**2:.3f}"
        
        # Plotear datos originales con etiqueta multilínea
        ax.plot(y_valid.index, y_valid, marker_style, alpha=0.7, label=multiline_label, 
                markersize=6, color=color)
        
        # Plotear línea de tendencia sin etiqueta
        ax.plot(y_valid.index, trend_line, '--', alpha=0.8, linewidth=2, 
                color=color)
        
        return {
            'slope': slope,
            'slope_per_period': slope_per_period,
            'period_unit': period_unit,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'total_days': total_days,
            'total_change': total_change
        }

    # 10.1 Gráfico comparación Pmp vs Isc diario FILTRADO con tendencias
    if (sr_pmp_434_col_name in sr_pmp_iv600.columns and not sr_pmp_iv600[sr_pmp_434_col_name].dropna().empty and
        sr_isc_434_col_name in sr_isc_iv600.columns and not sr_isc_iv600[sr_isc_434_col_name].dropna().empty):
        
        # Filtrar fechas comunes para gráfico con tendencias
        sr_pmp_trend_clean = sr_pmp_iv600[sr_pmp_434_col_name].dropna()
        sr_isc_trend_clean = sr_isc_iv600[sr_isc_434_col_name].dropna()
        
        common_dates_trend = sr_pmp_trend_clean.index.intersection(sr_isc_trend_clean.index)
        
        if len(common_dates_trend) > 0:
            sr_pmp_trend_filtered = sr_pmp_trend_clean.loc[common_dates_trend]
            sr_isc_trend_filtered = sr_isc_trend_clean.loc[common_dates_trend]
        
        fig7, ax7 = plt.subplots(figsize=(16, 8))
        
        # Plotear con líneas de tendencia
        stats_pmp = plot_with_trend_line(ax7, sr_pmp_trend_filtered, 
                                    'Pmp SR', '#1f77b4', '-o')
        stats_isc = plot_with_trend_line(ax7, sr_isc_trend_filtered, 
                                        'Isc SR', '#ff7f0e', '-o')
    
        # Configurar gráfico
        ax7.legend(fontsize=12, loc='upper right')
        ax7.set_ylabel('Soiling Ratio [%]', fontsize=16)
        ax7.set_xlabel('Date', fontsize=16)
        ax7.grid(True, alpha=0.3)
        ax7.set_title('SR Pmp & Isc - IV600', fontsize=18)
        ax7.set_ylim(90, 110)
        ax7.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        save_plot(plt.gcf(), 'sr_comparison_pmp_vs_isc_iv600_filtrado_with_trend', subdir='iv600')
        plt.close()
        print("✅ Gráfico comparación Pmp vs Isc diario filtrado con tendencias generado")
    else:
        print("⚠️  No hay fechas comunes entre SR Pmp e Isc para el gráfico diario filtrado con tendencias")

    # 10.2 Gráfico comparación Pmp vs Isc semanal Q25 FILTRADO con tendencias
    if (not sr_pmp_iv600_weekly.empty and not sr_isc_iv600_weekly.empty and
        sr_pmp_434_col_name in sr_pmp_iv600_weekly.columns and not sr_pmp_iv600_weekly[sr_pmp_434_col_name].dropna().empty and
        sr_isc_434_col_name in sr_isc_iv600_weekly.columns and not sr_isc_iv600_weekly[sr_isc_434_col_name].dropna().empty):
        
        # Filtrar fechas comunes para gráfico semanal con tendencias
        sr_pmp_weekly_trend_clean = sr_pmp_iv600_weekly[sr_pmp_434_col_name].dropna()
        sr_isc_weekly_trend_clean = sr_isc_iv600_weekly[sr_isc_434_col_name].dropna()
        
        common_dates_weekly_trend = sr_pmp_weekly_trend_clean.index.intersection(sr_isc_weekly_trend_clean.index)
        
        if len(common_dates_weekly_trend) > 0:
            sr_pmp_weekly_trend_filtered = sr_pmp_weekly_trend_clean.loc[common_dates_weekly_trend]
            sr_isc_weekly_trend_filtered = sr_isc_weekly_trend_clean.loc[common_dates_weekly_trend]
        
        fig8, ax8 = plt.subplots(figsize=(16, 8))
        
        # Plotear con líneas de tendencia
        stats_pmp_w = plot_with_trend_line(ax8, sr_pmp_weekly_trend_filtered, 
                                        'Pmp SR Semanal', '#1f77b4', '-o', is_weekly=True)
        stats_isc_w = plot_with_trend_line(ax8, sr_isc_weekly_trend_filtered, 
                                            'Isc SR Semanal', '#ff7f0e', '-o', is_weekly=True)
        
        # Configurar gráfico
        ax8.legend(fontsize=12, loc='upper right')
        ax8.set_ylabel('Soiling Ratio [%]', fontsize=16)
        ax8.set_xlabel('Date', fontsize=16)
        ax8.grid(True, alpha=0.3)
        ax8.set_title('SR Pmp & Isc - IV600 Q25', fontsize=18)
        ax8.set_ylim(90, 110)
        ax8.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        save_plot(plt.gcf(), 'sr_comparison_pmp_vs_isc_iv600_semanal_q25_filtrado_with_trend', subdir='iv600')
        plt.close()
        print("✅ Gráfico comparación Pmp vs Isc semanal Q25 filtrado con tendencias generado")
    else:
        print("⚠️  No hay fechas comunes entre SR Pmp e Isc para el gráfico semanal Q25 filtrado con tendencias")

    # 10.3 Gráfico comparación Pmp vs Isc semanal Q25 SIN FILTRAR con tendencias
    # Primero calculamos los SR sin filtrar
    sr_pmp_iv600_weekly_unfiltered = pd.DataFrame(index=df_pmax_weekly.index)
    sr_isc_iv600_weekly_unfiltered = pd.DataFrame(index=df_isc_weekly.index)

    # SR Pmp semanal 434vs439 - SIN FILTRADO
    if test_mod_434_pmax_col in df_pmax_weekly.columns and ref_mod_pmax_col in df_pmax_weekly.columns:
        sr_pmp_iv600_weekly_unfiltered['SR_Pmp_434vs439'] = 100 * df_pmax_weekly[test_mod_434_pmax_col] / df_pmax_weekly[ref_mod_pmax_col]

    # SR Isc semanal 434vs439 - SIN FILTRADO
    if test_mod_434_isc_col in df_isc_weekly.columns and ref_mod_isc_col in df_isc_weekly.columns:
        sr_isc_iv600_weekly_unfiltered['SR_Isc_434vs439'] = 100 * df_isc_weekly[test_mod_434_isc_col] / df_isc_weekly[ref_mod_isc_col]

    if (not sr_pmp_iv600_weekly_unfiltered.empty and not sr_isc_iv600_weekly_unfiltered.empty and
        sr_pmp_434_col_name in sr_pmp_iv600_weekly_unfiltered.columns and not sr_pmp_iv600_weekly_unfiltered[sr_pmp_434_col_name].dropna().empty and
        sr_isc_434_col_name in sr_isc_iv600_weekly_unfiltered.columns and not sr_isc_iv600_weekly_unfiltered[sr_isc_434_col_name].dropna().empty):
        
        # Filtrar fechas comunes para gráfico semanal sin filtrar con tendencias
        sr_pmp_weekly_unfilt_clean = sr_pmp_iv600_weekly_unfiltered[sr_pmp_434_col_name].dropna()
        sr_isc_weekly_unfilt_clean = sr_isc_iv600_weekly_unfiltered[sr_isc_434_col_name].dropna()
        
        common_dates_weekly_unfilt = sr_pmp_weekly_unfilt_clean.index.intersection(sr_isc_weekly_unfilt_clean.index)
        
        if len(common_dates_weekly_unfilt) > 0:
            sr_pmp_weekly_unfilt_filtered = sr_pmp_weekly_unfilt_clean.loc[common_dates_weekly_unfilt]
            sr_isc_weekly_unfilt_filtered = sr_isc_weekly_unfilt_clean.loc[common_dates_weekly_unfilt]
        
        fig9, ax9 = plt.subplots(figsize=(16, 8))
        
        # Plotear con líneas de tendencia
        stats_pmp_u = plot_with_trend_line(ax9, sr_pmp_weekly_unfilt_filtered, 
                                        'Pmp SR Semanal', '#1f77b4', '-o', is_weekly=True)
        stats_isc_u = plot_with_trend_line(ax9, sr_isc_weekly_unfilt_filtered, 
                                            'Isc SR Semanal', '#ff7f0e', '-o', is_weekly=True)
        
        # Configurar gráfico
        ax9.legend(fontsize=12, loc='upper right')
        ax9.set_ylabel('Soiling Ratio [%]', fontsize=16)
        ax9.set_xlabel('Date', fontsize=16)
        ax9.grid(True, alpha=0.3)
        ax9.set_title('SR Pmp & Isc - IV600 Q25', fontsize=18)
        ax9.set_ylim(90, 110)
        ax9.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        save_plot(plt.gcf(), 'sr_comparison_pmp_vs_isc_iv600_semanal_q25_with_trend', subdir='iv600')
        plt.close()
        print("✅ Gráfico comparación Pmp vs Isc semanal Q25 sin filtrar con tendencias generado")
    else:
        print("⚠️  No hay fechas comunes entre SR Pmp e Isc para el gráfico semanal Q25 sin filtrar con tendencias")

    # 10.4 Gráfico comparación Pmp vs Isc diario Q25 FILTRADO con tendencias
    # Usamos los SR diarios Q25 ya calculados
    sr_pmp_iv600_daily_q25 = sr_pmp_iv600_daily_q25_for_weekly
    sr_isc_iv600_daily_q25 = sr_isc_iv600_daily_q25_for_weekly

    if (sr_pmp_434_col_name in sr_pmp_iv600_daily_q25.columns and not sr_pmp_iv600_daily_q25[sr_pmp_434_col_name].dropna().empty and
        sr_isc_434_col_name in sr_isc_iv600_daily_q25.columns and not sr_isc_iv600_daily_q25[sr_isc_434_col_name].dropna().empty):
        
        # Filtrar fechas comunes para gráfico diario Q25 con tendencias
        sr_pmp_q25_clean = sr_pmp_iv600_daily_q25[sr_pmp_434_col_name].dropna()
        sr_isc_q25_clean = sr_isc_iv600_daily_q25[sr_isc_434_col_name].dropna()
        
        common_dates_q25 = sr_pmp_q25_clean.index.intersection(sr_isc_q25_clean.index)
        
        if len(common_dates_q25) > 0:
            sr_pmp_q25_filtered = sr_pmp_q25_clean.loc[common_dates_q25]
            sr_isc_q25_filtered = sr_isc_q25_clean.loc[common_dates_q25]
            
            fig_q25, ax_q25 = plt.subplots(figsize=(16, 8))
            
            # Plotear con líneas de tendencia
            stats_pmp_q25 = plot_with_trend_line(ax_q25, sr_pmp_q25_filtered, 
                                                'Pmp SR Daily Q25', '#1f77b4', '-o')
            stats_isc_q25 = plot_with_trend_line(ax_q25, sr_isc_q25_filtered, 
                                                'Isc SR Daily Q25', '#ff7f0e', '-o')
        
            # Configurar gráfico
            ax_q25.legend(fontsize=12, loc='upper right')
            ax_q25.set_ylabel('Soiling Ratio [%]', fontsize=16)
            ax_q25.set_xlabel('Date', fontsize=16)
            ax_q25.grid(True, alpha=0.3)
            ax_q25.set_title('SR Pmp & Isc - IV600 Q25', fontsize=18)
            ax_q25.set_ylim(90, 110)
            ax_q25.tick_params(axis='both', labelsize=14)
        
            plt.tight_layout()
            save_plot(plt.gcf(), 'sr_comparison_pmp_vs_isc_iv600_diario_q25_filtrado_with_trend', subdir='iv600')
            plt.close()
            print("✅ Gráfico comparación Pmp vs Isc diario Q25 filtrado con tendencias generado")
        else:
            print("⚠️  No hay fechas comunes entre SR Pmp e Isc para el gráfico diario Q25 filtrado con tendencias")

    # 10.5 Gráfico comparación Pmp vs Isc diario SIN FILTRAR con tendencias
    # Primero calculamos los SR diarios sin filtrar
    sr_pmp_iv600_unfiltered = pd.DataFrame(index=df_pmax_daily.index)
    sr_isc_iv600_unfiltered = pd.DataFrame(index=df_isc_daily.index)

    # SR Pmp 434vs439 - SIN FILTRADO
    if test_mod_434_pmax_col in df_pmax_daily.columns and ref_mod_pmax_col in df_pmax_daily.columns:
        sr_pmp_iv600_unfiltered['SR_Pmp_434vs439'] = 100 * df_pmax_daily[test_mod_434_pmax_col] / df_pmax_daily[ref_mod_pmax_col]

    # SR Isc 434vs439 - SIN FILTRADO
    if test_mod_434_isc_col in df_isc_daily.columns and ref_mod_isc_col in df_isc_daily.columns:
        sr_isc_iv600_unfiltered['SR_Isc_434vs439'] = 100 * df_isc_daily[test_mod_434_isc_col] / df_isc_daily[ref_mod_isc_col]

    if (sr_pmp_434_col_name in sr_pmp_iv600_unfiltered.columns and not sr_pmp_iv600_unfiltered[sr_pmp_434_col_name].dropna().empty and
        sr_isc_434_col_name in sr_isc_iv600_unfiltered.columns and not sr_isc_iv600_unfiltered[sr_isc_434_col_name].dropna().empty):
        
        # Filtrar fechas comunes para gráfico diario sin filtrar con tendencias
        sr_pmp_unfilt_clean = sr_pmp_iv600_unfiltered[sr_pmp_434_col_name].dropna()
        sr_isc_unfilt_clean = sr_isc_iv600_unfiltered[sr_isc_434_col_name].dropna()
        
        common_dates_unfilt = sr_pmp_unfilt_clean.index.intersection(sr_isc_unfilt_clean.index)
        
        if len(common_dates_unfilt) > 0:
            sr_pmp_unfilt_filtered = sr_pmp_unfilt_clean.loc[common_dates_unfilt]
            sr_isc_unfilt_filtered = sr_isc_unfilt_clean.loc[common_dates_unfilt]
            
            fig11, ax11 = plt.subplots(figsize=(16, 8))
        
        # Plotear con líneas de tendencia
            stats_pmp_d = plot_with_trend_line(ax11, sr_pmp_unfilt_filtered, 
                                          'Pmp SR', '#1f77b4', '-o')
            stats_isc_d = plot_with_trend_line(ax11, sr_isc_unfilt_filtered, 
                                              'Isc SR', '#ff7f0e', '-o')
        
        # Configurar gráfico
            ax11.legend(fontsize=12, loc='upper right')
            ax11.set_ylabel('Soiling Ratio [%]', fontsize=16)
            ax11.set_xlabel('Date', fontsize=16)
            ax11.grid(True, alpha=0.3)
            ax11.set_title('SR Pmp & Isc - IV600', fontsize=18)
            ax11.set_ylim(90, 110)
            ax11.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        save_plot(plt.gcf(), 'sr_comparison_pmp_vs_isc_iv600_with_trend', subdir='iv600')
        plt.close()
        print("✅ Gráfico comparación Pmp vs Isc diario sin filtrar con tendencias generado")
    else:
        print("⚠️  No hay fechas comunes entre SR Pmp e Isc para el gráfico diario sin filtrar con tendencias")

    # 11. EXPORTAR DATOS A CSV Y EXCEL
    print("\n--- Exportando datos IV600 ---")
    
    # Importar config para usar rutas estándar
    from config import paths
    
    # Crear carpeta para los archivos CSV (estructura estándar)
    csv_output_dir = os.path.join(paths.BASE_OUTPUT_CSV_DIR, 'iv600')
    os.makedirs(csv_output_dir, exist_ok=True)
    
    # Crear carpeta para los archivos Excel (legacy)
    excel_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datos', 'excel_exports')
    os.makedirs(excel_output_dir, exist_ok=True)
    
    # Nombre del archivo Excel
    excel_filename = os.path.join(excel_output_dir, 'soiling_ratios_iv600_completo.xlsx')
    
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Hoja 1: Soiling Ratios Diarios (promedio) - FILTRADOS
            if not sr_pmp_iv600.empty or not sr_isc_iv600.empty:
                sr_daily_combined = pd.concat([sr_pmp_iv600, sr_isc_iv600], axis=1)
                sr_daily_combined.to_excel(writer, sheet_name='SR_Diario_Promedio_Filtrado', index=True)
                print(f"✅ Hoja 'SR_Diario_Promedio_Filtrado' exportada: {sr_daily_combined.shape}")
            
            # Hoja 2: Soiling Ratios Diarios Q25 - FILTRADOS
            if 'sr_pmp_iv600_daily_q25' in locals() and 'sr_isc_iv600_daily_q25' in locals():
                if not sr_pmp_iv600_daily_q25.empty or not sr_isc_iv600_daily_q25.empty:
                    sr_daily_q25_combined = pd.concat([sr_pmp_iv600_daily_q25, sr_isc_iv600_daily_q25], axis=1)
                    sr_daily_q25_combined.to_excel(writer, sheet_name='SR_Diario_Q25_Filtrado', index=True)
                    print(f"✅ Hoja 'SR_Diario_Q25_Filtrado' exportada: {sr_daily_q25_combined.shape}")
            
            # Hoja 3: Soiling Ratios Semanales Q25 - FILTRADOS
            if not sr_pmp_iv600_weekly.empty or not sr_isc_iv600_weekly.empty:
                sr_weekly_combined = pd.concat([sr_pmp_iv600_weekly, sr_isc_iv600_weekly], axis=1)
                sr_weekly_combined.to_excel(writer, sheet_name='SR_Semanal_Q25_Filtrado', index=True)
                print(f"✅ Hoja 'SR_Semanal_Q25_Filtrado' exportada: {sr_weekly_combined.shape}")
            
            # Hoja 4: Soiling Ratios Diarios SIN FILTRAR
            if 'sr_pmp_iv600_unfiltered' in locals() and 'sr_isc_iv600_unfiltered' in locals():
                if not sr_pmp_iv600_unfiltered.empty or not sr_isc_iv600_unfiltered.empty:
                    sr_daily_unfiltered_combined = pd.concat([sr_pmp_iv600_unfiltered, sr_isc_iv600_unfiltered], axis=1)
                    sr_daily_unfiltered_combined.to_excel(writer, sheet_name='SR_Diario_Promedio_SinFiltrar', index=True)
                    print(f"✅ Hoja 'SR_Diario_Promedio_SinFiltrar' exportada: {sr_daily_unfiltered_combined.shape}")
            
            # Hoja 5: Soiling Ratios Semanales Q25 SIN FILTRAR
            if 'sr_pmp_iv600_weekly_unfiltered' in locals() and 'sr_isc_iv600_weekly_unfiltered' in locals():
                if not sr_pmp_iv600_weekly_unfiltered.empty or not sr_isc_iv600_weekly_unfiltered.empty:
                    sr_weekly_unfiltered_combined = pd.concat([sr_pmp_iv600_weekly_unfiltered, sr_isc_iv600_weekly_unfiltered], axis=1)
                    sr_weekly_unfiltered_combined.to_excel(writer, sheet_name='SR_Semanal_Q25_SinFiltrar', index=True)
                    print(f"✅ Hoja 'SR_Semanal_Q25_SinFiltrar' exportada: {sr_weekly_unfiltered_combined.shape}")
            
            # Hoja 6: Datos base Pmax diarios (promedio)
            if not df_pmax_daily.empty:
                df_pmax_daily.to_excel(writer, sheet_name='Datos_Pmax_Diario_Promedio', index=True)
                print(f"✅ Hoja 'Datos_Pmax_Diario_Promedio' exportada: {df_pmax_daily.shape}")
            
            # Hoja 7: Datos base Isc diarios (promedio)
            if not df_isc_daily.empty:
                df_isc_daily.to_excel(writer, sheet_name='Datos_Isc_Diario_Promedio', index=True)
                print(f"✅ Hoja 'Datos_Isc_Diario_Promedio' exportada: {df_isc_daily.shape}")
            
            # Hoja 8: Datos base Pmax diarios Q25
            if not df_pmax_daily_q25.empty:
                df_pmax_daily_q25.to_excel(writer, sheet_name='Datos_Pmax_Diario_Q25', index=True)
                print(f"✅ Hoja 'Datos_Pmax_Diario_Q25' exportada: {df_pmax_daily_q25.shape}")
            
            # Hoja 9: Datos base Isc diarios Q25
            if not df_isc_daily_q25.empty:
                df_isc_daily_q25.to_excel(writer, sheet_name='Datos_Isc_Diario_Q25', index=True)
                print(f"✅ Hoja 'Datos_Isc_Diario_Q25' exportada: {df_isc_daily_q25.shape}")
        
        print(f"\n📊 ARCHIVO EXCEL GUARDADO EN: {excel_filename}")
        print("🔍 Hojas incluidas:")
        print("   - SR_Diario_Promedio_Filtrado (Soiling Ratios diarios promedio filtrados)")
        print("   - SR_Diario_Q25_Filtrado (Soiling Ratios diarios Q25 filtrados)")
        print("   - SR_Semanal_Q25_Filtrado (Soiling Ratios semanales Q25 filtrados)")
        print("   - SR_Diario_Promedio_SinFiltrar (Soiling Ratios diarios promedio sin filtrar)")
        print("   - SR_Semanal_Q25_SinFiltrar (Soiling Ratios semanales Q25 sin filtrar)")
        print("   - Datos_Pmax_Diario_Promedio (Datos base Pmax diarios promedio)")
        print("   - Datos_Isc_Diario_Promedio (Datos base Isc diarios promedio)")
        print("   - Datos_Pmax_Diario_Q25 (Datos base Pmax diarios Q25)")
        print("   - Datos_Isc_Diario_Q25 (Datos base Isc diarios Q25)")
        
    except Exception as e:
        print(f"❌ Error al exportar a Excel: {e}")
        print("Asegúrate de tener openpyxl instalado: pip install openpyxl")

    # 12. EXPORTAR DATOS A CSV PARA GRÁFICO CONSOLIDADO
    print("\n--- Exportando datos IV600 a CSV para gráfico consolidado ---")
    
    try:
        # Exportar SOLO las columnas principales para gráfico consolidado
        pmp_col = 'SR_Pmp_434vs439'
        isc_col = 'SR_Isc_434vs439'
        dfs = []
        if pmp_col in sr_pmp_iv600_weekly.columns:
            dfs.append(sr_pmp_iv600_weekly[[pmp_col]])
        if isc_col in sr_isc_iv600_weekly.columns:
            dfs.append(sr_isc_iv600_weekly[[isc_col]])
        sr_weekly_combined = pd.concat(dfs, axis=1)
        
        # Renombrar columnas
        sr_weekly_combined.columns = ['SR_Pmax_IV600', 'SR_Isc_IV600']
        
        # Agregar columna de tiempo
        sr_weekly_combined.reset_index(inplace=True)
        sr_weekly_combined.rename(columns={sr_weekly_combined.columns[0]: 'timestamp'}, inplace=True)
        
        # Guardar CSV
        csv_filename = os.path.join(csv_output_dir, 'iv600_sr_semanal_q25.csv')
        sr_weekly_combined.to_csv(csv_filename, index=False)
        print(f"✅ CSV para gráfico consolidado guardado: {csv_filename}")
        print(f"   - Shape: {sr_weekly_combined.shape}")
        print(f"   - Columnas: {list(sr_weekly_combined.columns)}")
    except Exception as e:
        print(f"❌ Error al exportar CSV para gráfico consolidado: {e}")

    print("\n--- Visualización de Soiling Ratios IV600 FILTRADOS CON TENDENCIAS Finalizada ---")

    return {
        'sr_pmp_iv600': sr_pmp_iv600,
        'sr_isc_iv600': sr_isc_iv600,
        'sr_pmp_iv600_weekly': sr_pmp_iv600_weekly,
        'sr_isc_iv600_weekly': sr_isc_iv600_weekly,
        'excel_file': excel_filename if 'excel_filename' in locals() else None
    }

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando análisis IV600 Filtrado...")
    run_analysis() 