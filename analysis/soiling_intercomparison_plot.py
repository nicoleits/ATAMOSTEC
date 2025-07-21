import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Leer datos de DustIQ
df_dust_iq = pd.read_csv('datos/raw_dustiq_data.csv', parse_dates=['_time'])
df_dust_iq.set_index('_time', inplace=True)
df_dust_iq.index = df_dust_iq.index.tz_localize(None)  # Convertir a tz-naive
df_dust_iq = df_dust_iq.between_time('13:00', '18:00')
sr_dust_iq = df_dust_iq.loc['2024-07-23':, 'SR_C11_Avg']
sr_dust_iq = sr_dust_iq[sr_dust_iq > 70]  # Filtrar outliers

# Leer datos de Soiling Kit
df_soiling_kit = pd.read_csv('datos/soiling_kit_raw_data.csv', parse_dates=['_time'])
df_soiling_kit.set_index('_time', inplace=True)
df_soiling_kit.index = df_soiling_kit.index.tz_localize(None)  # Convertir a tz-naive

# Procesar datos del Soiling Kit
df_soiling_kit_1 = df_soiling_kit.loc['2024-07-23':'2024-10-23'].between_time('12:00', '18:00')
df_soiling_kit_2 = df_soiling_kit.loc['2024-10-23':'2025-12-31'].between_time('11:00','17:00')
df_soiling_kit = pd.concat([df_soiling_kit_1,df_soiling_kit_2]).sort_index()

# Calcular Soiling Ratio para el Soiling Kit
sr_soilingratio = 100 * df_soiling_kit['Isc(p)'].div(df_soiling_kit['Isc(e)'])
sr_soilingratio = sr_soilingratio[sr_soilingratio > 90].loc['2024-07-23':]

# Usar límite dinámico basado en los datos disponibles o final de 2025
# Usar el mínimo entre el último dato disponible y el final de 2025
max_available_date = max(sr_dust_iq.index.max() if not sr_dust_iq.empty else pd.Timestamp('2025-12-31'),
                        sr_soilingratio.index.max() if not sr_soilingratio.empty else pd.Timestamp('2025-12-31'))
end_date = min(max_available_date, pd.Timestamp('2025-12-31')).tz_localize(None)

sr_dust_iq = sr_dust_iq.loc[:end_date]
sr_soilingratio = sr_soilingratio.loc[:end_date]

# Obtener datos semanales
sr_dustiq_semanal = sr_dust_iq.resample('1W').quantile(0.25)
sr_soilingratio_semanal = sr_soilingratio.resample('1W').quantile(0.25)

# Aplicar límite a datos semanales
sr_dustiq_semanal = sr_dustiq_semanal.loc[:end_date]
sr_soilingratio_semanal = sr_soilingratio_semanal.loc[:end_date]

# Normalizar DustIQ semanal a 100% usando el valor máximo inicial
initial_period_dustiq = sr_dustiq_semanal.loc['2024-07-23':'2024-08-23']  # Primer mes de datos
max_initial_dustiq = initial_period_dustiq.max()
sr_dustiq_semanal = sr_dustiq_semanal * (100 / max_initial_dustiq)

# Normalizar Soiling Kit semanal a 100% usando el valor máximo inicial
initial_period = sr_soilingratio_semanal.loc['2024-07-23':'2024-08-23']  # Primer mes de datos
max_initial = initial_period.max()
sr_soilingratio_semanal = sr_soilingratio_semanal * (100 / max_initial)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(15, 7))

# Graficar
sr_dustiq_semanal.plot(ax=ax, style='--o', alpha=0.75, label='Dust IQ')
sr_soilingratio_semanal.plot(ax=ax, style='--o', alpha=0.75, label='Soiling Kit')

# Configurar el gráfico
ax.set_ylabel('Soiling Ratio [%]', fontsize=16)
ax.set_xlabel('Time [day]', fontsize=16)
ax.grid(True)
ax.set_title('Soiling Ratio - Intercomparison', fontsize=16)
ax.legend(fontsize=12)
ax.tick_params(axis='both', labelsize=14)
ax.set_ylim([90, 105])

# Ajustar el layout y guardar
plt.tight_layout()
plt.savefig("soiling_ratio_intercomparison_semanal.jpg", dpi=800)
plt.show() 