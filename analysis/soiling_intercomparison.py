import pandas as pd
import clickhouse_connect
import os
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from classes_codes import medio_dia_solar
from classes_codes import influxdb_class
from influxdb_client import InfluxDBClient, Point, Dialect
from influxdb_client.client.write_api import SYNCHRONOUS
import polars as pl
from utils.plot_utils import save_plot


All = slice(None)


#%%
# Configuration for InfluxDB client
url = "http://146.83.153.212:27017"
token = "piDbFR_bfRWO5Epu1IS96WbkNpSZZCYgwZZR29PcwUsxXwKdIyLMhVAhU4-5ohWeXIsX7Dp_X-WiPIDx0beafg=="
org = "atamostec"

# Ensure url is a string
if not isinstance(url, str):
    raise ValueError('"url" attribute is not str instance')

#%%
# Define date and time range
fecha_inicio = '01/07/2024'
fecha_final = '31/12/2025'
date_s = pd.to_datetime(fecha_inicio, dayfirst=True)
date_f = pd.to_datetime(fecha_final, dayfirst=True)


#%%Functions

medsolar = medio_dia_solar()
influx_class = influxdb_class(token, url, org)

#%%
# Query data

# bucket = "PSDA"
# tables = ["CACTUS_transmittance_losses"] #contiene ratios de fotoceldas 3,4,5
# attributes = ["FC3", "FC4", "FC5"]

# df_glasses = influx_class.query_influxdb(bucket,tables,attributes,date_s).droplevel(level=0,axis=1)

"Irradiancias Fotoceldas experimento con vidrios"
bucket = "meteo_psda"
tables = ["6852_Ftc"] #contiene irradiancias de fotoceldas 1,2,3,4,5
attributes = ["R_FC1_Avg","R_FC2_Avg","R_FC3_Avg", "R_FC4_Avg", "R_FC5_Avg"]

df_glasses = influx_class.query_influxdb(bucket,tables,attributes,date_s).droplevel(level=0,axis=1)
df_glasses = df_glasses.between_time('13:00', '18:00')

#%%
#PV Glasses data processing

df_glasses['Ref'] = df_glasses[['R_FC1_Avg','R_FC2_Avg']].mean(axis=1)
df_glasses_daily = df_glasses.between_time('13:00','18:00').resample('1d').sum().div(60000)

#%% Graficos RC PVGlasses

fig, ax = plt.subplots(figsize=(10, 5))
# df_glasses_daily.plot(ax=ax, style='--o', alpha=0.75)
df_glasses.plot(ax=ax, style='--o', alpha=0.75)


# ax.legend(['1RC411', '1RC410'])
# ax.set_ylim([94, 101])
ax.set_ylabel('Soiling Ratio [-]')
ax.set_title('Soiling Ratio - Intercomparison')
ax.grid()

plt.show()
# fig.savefig('Reference_cells.jpg',dpi=1000)

#%%
"Parametros Curvas IV PVStand"
bucket = "PSDA"
tables = ["PERC1_fixed_1MD43420160719", "PERC2_fixed_1MD43920160719"]
attributes = ["Imax", "Umax", 'Pmax']
df_pvstand = influx_class.query_influxdb(bucket, tables, attributes, date_s) #parametros curvas IV
df_pvstand.columns = df_pvstand.columns.set_levels(["MD434", "MD439"], level=0)
df_pvstand = df_pvstand.between_time('13:00', '18:00')
print(f"df_pvstand: {df_pvstand.head()}")

#%%
"Irradiancias Fotoceldas de estructura fija"
bucket = "PSDA"
tables = ["RefCellsFixed"]
attributes = ["1RC410(w.m-2)", "1RC411(w.m-2)", "1RC412(w.m-2)"]
df_rc_fix = influx_class.query_influxdb(bucket, tables, attributes, date_s).droplevel(level=0, axis=1)
df_rc_fix = df_rc_fix.between_time('13:00', '18:00')
print(f"df_rc_fix: {df_rc_fix.head()}")


tables = ["fixed_plant_atamo_1"]
df_rc_fix_dec = influx_class.query_influxdb(bucket, tables, attributes, date_s).droplevel(level=0, axis=1)
df_rc_fix_dec = df_rc_fix_dec.between_time('13:00', '18:00')
df_rc_fix = pd.concat([df_rc_fix, df_rc_fix_dec], axis=0)
print(f"df_rc_fix (after concat): {df_rc_fix.head()}")

#%%
"Temperatura PT100 Modulos MD434 y MD439"
bucket = "PSDA"
tables = ["TempModFixed"]
attributes = ["1TE416(C)", "1TE417(C)", "1TE418(C)","1TE419(C)"]
df_pt100_fix = influx_class.query_influxdb(bucket, tables, attributes, date_s).droplevel(level=0, axis=1)
df_pt100_fix = df_pt100_fix.between_time('13:00', '18:00')
print(f"df_rc_fix: {df_pt100_fix.head()}")

tables = ["fixed_plant_atamo_1"]
df_pt100_fix_dec = influx_class.query_influxdb(bucket, tables, attributes,  pd.to_datetime('05/12/2024', dayfirst=True)).droplevel(level=0, axis=1)
df_pt100_fix_dec = df_pt100_fix_dec.between_time('13:00', '18:00')
df_pt100_fix = pd.concat([df_pt100_fix, df_pt100_fix_dec], axis=0)

print(f"df_rc_fix: {df_pt100_fix.head()}")
#%%
"Soiling Ratio Dust IQ"
bucket = "PSDA"
tables = ["DustIQ"]
attributes = ["SR_C11_Avg", "SR_C12_Avg"]
df_dust_iq = influx_class.query_influxdb(bucket, tables, attributes, date_s).droplevel(level=0, axis=1)
df_dust_iq = df_dust_iq.between_time('13:00', '18:00')

print(f"df_dust_iq: {df_dust_iq.head()}")

#%% DustIQ plot
fig,ax = plt.subplots(figsize=(10,5))

df_dust_iq.SR_C11_Avg.loc['2024-07-23':].between_time('12:00', '13:00').resample('1W').mean().plot(ax=ax,style='--o',alpha=0.75)
df_dust_iq.SR_C11_Avg.loc['2024-07-23':].between_time('14:00', '15:00').resample('1W').mean().plot(ax=ax,style='--o',alpha=0.75)
df_dust_iq.SR_C11_Avg.loc['2024-07-23':].between_time('16:00', '17:00').resample('1W').mean().plot(ax=ax,style='--o',alpha=0.75)

ax.legend(['12 PM','2 PM','4 PM'])
ax.set_ylim([94,102])
ax.set_ylabel('Soiling Ratio [-]')
ax.set_title('Soiling Ratio - Dust IQ')
ax.grid()

#%%
"IV Curves Soiling Kit"
bucket = "PSDA"
tables = ["soilingkit"]
attributes = ["Isc(e)", "Isc(p)", "Te(C)", "Tp(C)"]
df_sk = influx_class.query_influxdb(bucket, tables, attributes, date_s).droplevel(level=0, axis=1)
df_soiling_kit = df_sk.copy()
#%%
df_soiling_kit_1 = df_soiling_kit.loc['2024-07-23':'2024-10-23'].between_time('12:00', '18:00')
df_soiling_kit_2 = df_soiling_kit.loc['2024-10-23':'2025-03-30'].between_time('11:00','17:00')
df_soiling_kit = pd.concat([df_soiling_kit_1,df_soiling_kit_2]).sort_index()

#%%
df_soiling_kit_1 = df_soiling_kit.loc['2024-07-23':'2024-10-27'].between_time('14:00', '16:00')
df_soiling_kit_2 = df_soiling_kit.loc['2024-10-27':'2025-03-30'].between_time('11:00','13:00')
df_soiling_kit = pd.concat([df_soiling_kit_1,df_soiling_kit_2]).sort_index()
print(f"df_soiling_kit: {df_soiling_kit.head()}")


#%%
# Data processing Isc correccion PVStand

df_pvstand.loc['2024-08-01':,('MD434','Icorr')] = df_pvstand.loc['2024-08-01':,('MD434','Imax')]*(1 + (0.0004*(25 - df_pt100_fix.loc['2024-08-01':,'1TE416(C)'])))
df_pvstand.loc['2024-08-01':,('MD439','Icorr')] = df_pvstand.loc['2024-08-01':,('MD439','Imax')]*(1 + (0.0004*(25 - df_pt100_fix.loc['2024-08-01':,'1TE418(C)'])))

# Data processing Isc correccion Soiling Kit

df_soiling_kit.loc['2024-07-23':,'Ie(c)'] = df_soiling_kit.loc['2024-07-23':,'Isc(e)']*(1 + (0.0004*(25 - df_soiling_kit.loc['2024-07-23':,'Te(C)'])))
df_soiling_kit.loc['2024-07-23':,'Ip(c)'] = df_soiling_kit.loc['2024-07-23':,'Isc(p)']*(1 + (0.0004*(25 - df_soiling_kit.loc['2024-07-23':,'Tp(C)'])))

# Data processing DustIQ

sr_dust_iq = df_dust_iq.loc['2024-07-23':, 'SR_C11_Avg']
sr_dust_iq = sr_dust_iq[sr_dust_iq > 70] #70 por outliers
print(f"sr_dust_iq: {sr_dust_iq.head()}\nmax: {sr_dust_iq.max()}\nmin: {sr_dust_iq.min()}")

# PVStand

sr_isc_pvstand = df_pvstand.loc['2024-08-01':, ('MD434', 'Imax')].div(df_pvstand.loc[:, ('MD439', 'Imax')])
sr_isc_pvstand = 100 * sr_isc_pvstand[(sr_isc_pvstand < 1) & (sr_isc_pvstand > 0.8)]
print(f"sr_isc_pvstand: {sr_isc_pvstand.head()}\nmax: {sr_isc_pvstand.max()}\nmin: {sr_isc_pvstand.min()}")

sr_isc_pvstand_c = df_pvstand.loc['2024-08-01':, ('MD434', 'Icorr')].div(df_pvstand.loc[:, ('MD439', 'Icorr')])
sr_isc_pvstand_c = 100 * sr_isc_pvstand_c[(sr_isc_pvstand_c < 1) & (sr_isc_pvstand_c > 0.8)]
print(f"sr_isc_pvstand_c: {sr_isc_pvstand_c.head()}\nmax: {sr_isc_pvstand_c.max()}\nmin: {sr_isc_pvstand_c.min()}")

sr_pmp_pvstand = df_pvstand.loc['2024-08-01':, ('MD434', 'Pmax')].div(df_pvstand.loc[:, ('MD439', 'Pmax')])
sr_pmp_pvstand = 100 * sr_pmp_pvstand[(sr_pmp_pvstand < 1) & (sr_pmp_pvstand > 0.8)]
print(f"sr_pmp_pvstand: {sr_pmp_pvstand.head()}\nmax: {sr_pmp_pvstand.max()}\nmin: {sr_pmp_pvstand.min()}")

sr_pmp_pvstand = sr_pmp_pvstand + 3

#%% RC Fija

sr_kwh_rc = df_rc_fix.between_time('13:00', '18:00').loc['2024-08-01':, ('1RC410(w.m-2)', '1RC411(w.m-2)')].div(df_rc_fix['1RC412(w.m-2)'].between_time('13:00', '18:00'), axis=0)
sr_kwh_rc = 100 * sr_kwh_rc[(sr_kwh_rc < 1.05) & (sr_kwh_rc > 0.8)]
sr_kwh_rc411 = sr_kwh_rc.loc['2024-08-01':,'1RC411(w.m-2)'] - 0.93
sr_kwh_rc411 = sr_kwh_rc411[sr_kwh_rc411 < 100]
print(f"sr_kwh_rc: {sr_kwh_rc.head()}\nmax: {sr_kwh_rc.max()}\nmin: {sr_kwh_rc.min()}")

fig, ax = plt.subplots(figsize=(10, 5))
sr_kwh_rc.loc['2024':, '1RC411(w.m-2)'].resample('1W').mean().sub(1.65).plot(ax=ax, style='--o', alpha=0.75)
sr_kwh_rc.loc['2024':, '1RC410(w.m-2)'].resample('1W').mean().sub(1.65).plot(ax=ax, style='--o', alpha=0.75)

ax.legend(['1RC411', '1RC410'])
ax.set_ylim([94, 101])
ax.set_ylabel('Soiling Ratio [-]')
ax.set_title('Soiling Ratio - Intercomparison')
ax.grid()

plt.show()
save_plot(fig, 'Reference_cells', subdir='intercomparison', dpi=1000)

#%% Soiling kit data processing
sr_soilingratio = 100 * df_soiling_kit['Isc(p)'].div(df_soiling_kit['Isc(e)'])
sr_soilingratio = sr_soilingratio[sr_soilingratio > 90].loc['2024-07-23':]
print(f"sr_soilingratio: {sr_soilingratio.head()}\nmax: {sr_soilingratio.max()}\nmin: {sr_soilingratio.min()}")

sr_soilingratio_c = 100 * df_soiling_kit['Ip(c)'].div(df_soiling_kit['Ie(c)'])
sr_soilingratio_c = sr_soilingratio_c[sr_soilingratio_c > 90].loc['2024-07-23':]
print(f"sr_soilingratio_C: {sr_soilingratio_c.head()}\nmax: {sr_soilingratio_c.max()}\nmin: {sr_soilingratio.min()}")

#Ajuste a 100%
# sr_soilingratio = sr_soilingratio - 1.127
# sr_soilingratio = sr_soilingratio[sr_soilingratio <= 100]

sr_dustiq_semanal = sr_dust_iq.loc['2024-07':'2025-03'].between_time('14:00','18:00').resample('1W').mean() + 0.5

sr_soilingratio_semanal = sr_soilingratio.loc['2024-07':'2025-03'].resample('1W').mean() - 1.6

#%% Soilingkit + Dustiq plots


fig,ax = plt.subplots(figsize=(10,5))

sr_dustiq_semanal.plot(ax=ax,style='--o',alpha=0.75)
sr_soilingratio_semanal.plot(ax=ax, style='--o', alpha=0.75)

ax.legend(['Dust IQ','SoilingRatio'])
ax.set_ylim([90,105])
ax.set_ylabel('Soiling Ratio [-]')
ax.set_title('Soiling Raito - Intercomparison')
ax.grid()

save_plot(fig, 'benchmark_SK_DustIQ', subdir='intercomparison', dpi=1000)
#%%
# Clickhouse client setup
host = "146.83.153.212"
# host = "172.24.61.95"
port = "30091"
user = "default"
pwrd = "Psda2020"
client_clickhouse = clickhouse_connect.get_client(host=host, port=port, username=user, password=pwrd)

#%%
# Query Clickhouse data IV600 Trazer
query_clickhouse = "SELECT * FROM ref_data.iv_curves_trazador_manual"
data_iv_curves = client_clickhouse.query(query_clickhouse)

curves_list = []
for curve in data_iv_curves.result_set:
    # Extraemos las listas de corrientes y voltajes.
    currents = curve[4]
    voltages = curve[3]
    # Calculamos la potencia para cada par corriente/voltaje.
    powers = [currents[i] * voltages[i] for i in range(len(currents))]
    # Extraemos la marca de tiempo, el módulo y otros parámetros:
    timestamp = curve[0]
    module = curve[2]
    pmp = max(powers)
    isc = max(currents)
    voc = max(voltages)
    imp = currents[np.argmax(powers)]
    vmp = voltages[np.argmax(powers)]
    # Se añade la fila procesada a la lista
    curves_list.append([timestamp, module, pmp, isc, voc, imp, vmp])

#%% IV600 angel

column_names = ["timestamp", "module", "pmp", "isc", "voc", "imp", "vmp"]
df_curves = pl.DataFrame(curves_list, schema=column_names)

# Se ordena el DataFrame por timestamp y se ajusta la zona horaria a "UTC".
df_curves = df_curves.sort("timestamp").with_columns(
    pl.col("timestamp").dt.replace_time_zone("UTC")
)

# Impresión del DataFrame original para verificación.
print("\n------------------- DataFrames extraidos -------------------")
print("\nDataFrame original:")
print(df_curves)

# =============================================================================
# Filtrado de Datos por Fecha y Por Módulo
# =============================================================================
# Se definen dos fechas de corte:
# - filter_date: fecha a partir de la cual se toman los datos.
# - filter_date_for_440: fecha específica para el módulo 1MD440.
filter_date = datetime(2024, 9, 25, tzinfo=timezone.utc)
filter_date_for_440 = datetime(2024, 11, 19, 0, 0, tzinfo=timezone.utc)

# Se crea un diccionario para almacenar DataFrames filtrados por módulo.
dfs_by_module_filtered = {}
for mod_name in df_curves["module"].unique():
    if mod_name != "Unknown Module":
        # Se filtran los datos para cada módulo a partir de filter_date y se ordenan por timestamp.
        df_module = df_curves.filter(
            (pl.col("module") == mod_name) & (pl.col("timestamp") >= filter_date)
        ).sort("timestamp")
        dfs_by_module_filtered[mod_name] = df_module

# Se extraen DataFrames específicos para cada módulo.
df_439 = dfs_by_module_filtered["1MD439"]
# Se filtra df_439 para obtener datos posteriores a filter_date_for_440 (para comparación con 1MD440)
df_439_for_df_440 = df_439.filter(pl.col("timestamp") >= filter_date_for_440).sort("timestamp")
df_440 = dfs_by_module_filtered["1MD440"]
df_434 = dfs_by_module_filtered["1MD434"]

# Impresión de los DataFrames filtrados para verificación.
print("\nDataFrame filtrado para el módulo 1MD439:")
print(df_439)
print("\nDataFrame filtrado para el módulo 1MD439 después de la fecha de corte para 1MD440:")
print(df_439_for_df_440)
print("\nDataFrame filtrado para el módulo 1MD434:")
print(df_434)
print("\nDataFrame filtrado para el módulo 1MD440:")
print(df_440)


# Agrupación para df_439
df_439_daily_counts = (
    df_439
    .group_by(pl.col("timestamp").dt.truncate("1d"))
    .agg([pl.len()])
    .rename({"timestamp": "date", "len": "measurements_per_day"})
)

# Agrupación para df_434
df_434_daily_counts = (
    df_434
    .group_by(pl.col("timestamp").dt.truncate("1d"))
    .agg([pl.len()])
    .rename({"timestamp": "date", "len": "measurements_per_day"})
)

# Agrupación para df_439_for_df_440
df_439_for_df_440_daily_counts = (
    df_439_for_df_440
    .group_by(pl.col("timestamp").dt.truncate("1d"))
    .agg([pl.len()])
    .rename({"timestamp": "date", "len": "measurements_per_day"})
)

# Agrupación para df_440
df_440_daily_counts = (
    df_440
    .group_by(pl.col("timestamp").dt.truncate("1d"))
    .agg([pl.len()])
    .rename({"timestamp": "date", "len": "measurements_per_day"})
)

# =============================================================================
# Comparación de Conteos Diarios entre Módulos 1MD434 y 1MD439
# =============================================================================
# Se une el DataFrame diario de 1MD439 con el de 1MD434 y se verifica que el número
# de mediciones diarias sea igual.
df_comparison = df_439_daily_counts.join(
    df_434_daily_counts, on="date", how="inner", suffix="_434"
)

df_comparison = df_comparison.with_columns(
    (pl.col("measurements_per_day") == pl.col("measurements_per_day_434")).alias("equal_counts")
)

differences_found = False
print("\nComparación entre 434 y 439:")
for row in df_comparison.iter_rows(named=True):
    if not row["equal_counts"]:
        differences_found = True
        print(f"El día {row['date']} NO tiene igual número de pruebas: {row['measurements_per_day']} vs {row['measurements_per_day_434']}")

if differences_found:
    print("Se sacarán las filas problemáticas de ambos DataFrames.")
else:
    print("Todos los días tienen igual número de pruebas.")

# =============================================================================
# Comparación de Conteos Diarios entre Módulos 1MD440 y 1MD439 (para fechas posteriores a filter_date_for_440)
# =============================================================================
df_comparison = df_439_for_df_440_daily_counts.join(
    df_440_daily_counts, on="date", how="inner", suffix="_440"
)   
df_comparison = df_comparison.with_columns(
    (pl.col("measurements_per_day") == pl.col("measurements_per_day_440")).alias("equal_counts")
)

differences_found = False
print("\nComparación entre 440 y 439_for_440:")
for row in df_comparison.iter_rows(named=True):
    if not row["equal_counts"]:
        differences_found = True
        print(f"El día {row['date']} NO tiene igual número de pruebas: {row['measurements_per_day']} vs {row['measurements_per_day_440']}")

if differences_found:
    print("Se sacarán las filas problemáticas de ambos DataFrames.")
    
    # Se extraen los días válidos donde el número de mediciones es igual.
    valid_dates = df_comparison.filter(pl.col("equal_counts")).select("date")
    valid_dates_list = valid_dates["date"].to_list()
    
    # Se filtran los DataFrames originales para eliminar los días problemáticos.
    df_439_for_df_440 = df_439_for_df_440.filter(
        pl.col("timestamp").dt.truncate("1d").is_in(valid_dates_list)
    )
    df_440 = df_440.filter(
        pl.col("timestamp").dt.truncate("1d").is_in(valid_dates_list)
    )
    
    print("\nDataFrame 1MD439 (post 1MD440 corte) limpio:")
    print(df_439_for_df_440)
    print("\nDataFrame 1MD440 limpio:")
    print(df_440)
else:
    print("Todos los días tienen igual número de pruebas.")

# =============================================================================
# Cálculo de Ratios entre Módulos
# =============================================================================
# Se calcula el ratio de la columna 'pmp' entre distintos módulos.
# En este ejemplo se calcula el ratio entre 1MD434 y 1MD439.
pmp_ratio = df_434["pmp"] / df_439["pmp"]
df_ratio_434_439 = pl.DataFrame({
    "timestamp_439": df_439["timestamp"],
    "timestamp_434": df_434["timestamp"],
    "pmp_439": df_439["pmp"],
    "pmp_434": df_434["pmp"],
    "pmp_ratio": pmp_ratio
})
print("\nDataFrame con el ratio de pmp entre 1MD434 y 1MD439:")

# Se guarda el resultado en un archivo de texto.
with open("pmp_ratio_434_439.txt", "w") as f:
    for i in df_ratio_434_439.iter_rows(named=True):
        f.write(f"{i['timestamp_439']}, {i['timestamp_434']}, {i['pmp_439']}, {i['pmp_434']}, {i['pmp_ratio']}\n")

# Se calcula el ratio entre 1MD440 y 1MD439 (para fechas posteriores a filter_date_for_440).
df_ratio_440_439 = pl.DataFrame({
    "timestamp_439": df_439_for_df_440["timestamp"],
    "timestamp_440": df_440["timestamp"],
    "pmp_439": df_439_for_df_440["pmp"],
    "pmp_440": df_440["pmp"],
    "pmp_ratio": df_440["pmp"] / df_439_for_df_440["pmp"]
})

print(df_ratio_434_439)
print(df_ratio_440_439)

#%% Graficos IV600


# =============================================================================
# Graficación de Ratios
# =============================================================================
# Se extraen los datos para graficar el ratio de pmp.
timestamps_434_439 = df_ratio_434_439["timestamp_439"].to_list()
ratio_434_439 = (df_ratio_434_439["pmp_ratio"]).to_list()

timestamps_440_439 = df_ratio_440_439["timestamp_439"].to_list()
ratio_440_439 = (df_ratio_440_439["pmp_ratio"]).to_list()

# Se crea una figura y se superponen ambas series de datos en un mismo gráfico.
plt.figure(figsize=(12, 6))
plt.plot(timestamps_434_439, ratio_434_439, marker="o", linestyle="-", label="1MD434 vs 1MD439")
# La siguiente línea está comentada, pero puede activarse para graficar también el ratio entre 1MD440 y 1MD439.
# plt.plot(timestamps_440_439, ratio_440_439, marker="s", linestyle="-", label="1MD440 vs 1MD439")
plt.title("Comparación de pmp Ratio")
plt.xlabel("Timestamp")
plt.ylabel("pmp Ratio")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.ylim(0.9, 1.1)
plt.show()



#%% IV600

df = pd.DataFrame(curves_list, columns=['StampTime', 'Module', 'Pmax', 'Imax', 'Umax', 'Imp', 'Ump']).set_index('StampTime')
df.index = df.index.tz_localize(None)
print(f"df (Clickhouse data): {df.head()}")

a = pd.concat([df[df['Module']=='1MD434'].drop(columns='Module'),],axis=1).resample('5T').mean()
b = pd.concat([df[df['Module']=='1MD439'].drop(columns='Module'),],axis=1).resample('5T').mean()
c = pd.concat([df[df['Module']=='1MD440'].drop(columns='Module'),],axis=1).resample('5T').mean()

df = pd.concat([a,b,c],keys=['1MD434','1MD439','1MD440'],axis=1).loc['2024-09-25':,(All,'Pmax')].dropna(how='all')
df_i = pd.concat([a,b,c],keys=['1MD434','1MD439','1MD440'],axis=1).loc['2024-09-25':,(All,'Imax')].dropna(how='all')

# Asegúrate de que el índice es datetime
df.index = pd.to_datetime(df.index)
df_i.index = pd.to_datetime(df_i.index)

# Crear una copia del DataFrame para no modificar el original directamente
df_corr = df.copy()
df_icorr = df_i.copy()

# Iterar sobre todas las columnas del DataFrame
for col in df.columns:
    for idx in df.index:
        # Crear el timestamp 5 minutos adelante
        idx_futuro = idx + pd.Timedelta(minutes=5)

        # Si hay NaN en el tiempo actual y un valor válido en el futuro, moverlo
        if pd.isna(df_corr.loc[idx, col]):
            if idx_futuro in df_corr.index and pd.notna(df_corr.loc[idx_futuro, col]):
                df_corr.loc[idx, col] = df_corr.loc[idx_futuro, col]
                df_corr.loc[idx_futuro, col] = np.nan  # Opcional: limpia el valor original

for col in df_i.columns:
    for idx in df_i.index:
        # Crear el timestamp 5 minutos adelante
        idx_futuro = idx + pd.Timedelta(minutes=5)

        # Si hay NaN en el tiempo actual y un valor válido en el futuro, moverlo
        if pd.isna(df_icorr.loc[idx, col]):
            if idx_futuro in df_icorr.index and pd.notna(df_icorr.loc[idx_futuro, col]):
                df_icorr.loc[idx, col] = df_icorr.loc[idx_futuro, col]
                df_icorr.loc[idx_futuro, col] = np.nan  # Opcional: limpia el valor original

# (Opcional) Eliminar filas que quedaron completamente vacías
df_corr = df_corr.resample('1D').mean().dropna(how='all')
df_icorr = df_icorr.resample('1D').mean().dropna(how='all')

sr_pmp_iv600 = pd.DataFrame()
sr_pmp_iv600['1MD434/439'] = 100*df_corr['1MD434','Pmax'].div(df_corr['1MD439','Pmax'])
sr_pmp_iv600['1MD440/439'] = 100*df_corr['1MD440','Pmax'].div(df_corr['1MD439','Pmax'])
sr_pmp_iv600['1MD434/439'] = sr_pmp_iv600['1MD434/439'][(sr_pmp_iv600['1MD434/439'] < 105) & (sr_pmp_iv600['1MD434/439'] > 90)]

df_icorr = df_icorr.resample('1D').mean().dropna(how='all')
sr_isc_iv600 = pd.DataFrame()
sr_isc_iv600['1MD434/439'] = 100*df_icorr['1MD434','Imax'].div(df_icorr['1MD439','Imax'])
sr_isc_iv600['1MD440/439'] = 100*df_icorr['1MD440','Imax'].div(df_icorr['1MD439','Imax'])
sr_isc_iv600['1MD434/439'] = sr_isc_iv600['1MD434/439'][(sr_isc_iv600['1MD434/439'] < 105) & (sr_isc_iv600['1MD434/439'] > 90)]


fig, ax = plt.subplots(figsize=(15, 5))
sr_pmp_iv600.plot(ax=ax,style='--o', alpha=0.75)
ax.legend(['SR 434','SR 440'])
ax.set_ylabel('Soiling Ratio [%]')
ax.set_xlabel('Time [day]')
ax.grid()
ax.set_title('Soiling Ratios IV600')
ax.set_ylim([80, 110])
save_plot(fig, 'SR_Pmp_IV600', subdir='intercomparison', dpi=1000)
plt.show()

fig, ax = plt.subplots(figsize=(15, 5))
sr_pmp_iv600['1MD434/439'].plot(ax=ax,style='--o', alpha=0.75)
sr_isc_iv600['1MD434/439'].plot(ax=ax,style='--o', alpha=0.75)
ax.legend(['Pmp SR 434','Isc SR 434'])
ax.set_ylabel('Soiling Ratio [%]')
ax.set_xlabel('Time [day]')
ax.grid()
ax.set_title('Soiling Ratios IV600')
ax.set_ylim([80, 110])
save_plot(fig, 'SR_434_Pmp_Isc_IV600', subdir='intercomparison', dpi=1000)
plt.show()

#%%


sr_pmp_iv600 = 100 * sr_pmp_iv600[(sr_pmp_iv600 < 1) & (sr_pmp_iv600 > 0.8)]





df_iv600 = pd.concat([df_corr[df_corr['Module'] == '1MD434'].drop(columns='Module'),
                      df_corr[df_corr['Module'] == '1MD439'].drop(columns='Module')], keys=['MD434', 'MD439'], axis=1).resample('1H').mean().dropna()
print(f"df_iv600: {df_iv600.head()}")

sr_isc_iv600 = df_iv600.loc['2024-08-01':, ('MD434', 'Imax')].div(df_iv600.loc[:, ('MD439', 'Imax')])
sr_isc_iv600 = 100 * sr_isc_iv600[(sr_isc_iv600 < 1) & (sr_isc_iv600 > 0.8)]
print(f"sr_isc_iv600: {sr_isc_iv600.head()}\nmax: {sr_isc_iv600.max()}\nmin: {sr_isc_iv600.min()}")

sr_pmp_iv600 = df_iv600.loc['2024-08-01':, ('MD434', 'Pmax')].div(df_iv600.loc[:, ('MD439', 'Pmax')])
sr_pmp_iv600 = 100 * sr_pmp_iv600[(sr_pmp_iv600 < 1) & (sr_pmp_iv600 > 0.8)]
print(f"sr_pmp_iv600: {sr_pmp_iv600.head()}\nmax: {sr_pmp_iv600.max()}\nmin: {sr_pmp_iv600.min()}")

#%%
# Plotting

# Plotting IV 600

fig, ax = plt.subplots(figsize=(15, 5))
sr_pmp_iv600.plot(ax=ax,style='--o', alpha=0.75)
ax.legend('SR Pmp IV600')
ax.set_ylabel('Soiling Ratio [%]')
ax.set_xlabel('Time [day]')
ax.grid()
ax.set_title('Soiling Ratios IV600')
# ax.set_ylim([96, 100])

plt.show()


fig, ax = plt.subplots(figsize=(15, 5))
sr_isc_iv600.resample('1d').mean().plot(ax=ax)
sr_pmp_iv600.resample('1d').mean().plot(ax=ax)
ax.legend(['SR Isc IV600','SR Pmp IV600'])
ax.set_ylabel('Soiling Ratio [%]')
ax.set_xlabel('Time [day]')
ax.grid()
ax.set_title('Soiling Ratios IV600')
# ax.set_ylim([96, 100])

plt.show()

#

fig, ax = plt.subplots(figsize=(15, 5))
sr_dust_iq.resample('1d').mean().plot(ax=ax)
sr_isc_pvstand.resample('1d').mean().plot(ax=ax)
sr_pmp_pvstand.resample('1d').mean().plot(ax=ax)
sr_kwh_rc411.resample('1d').mean().plot(ax=ax)
sr_soilingratio.resample('1d').mean().plot(ax=ax)
# sr_isc_iv600.resample('1d').mean().plot(ax=ax)
# sr_pmp_iv600.resample('1d').mean().plot(ax=ax)
ax.legend(['Dust IQ', 'Isc PVStand', 'Pmp PVStand',
           'RC Fixed 411','Soiling Kit'])
ax.set_ylabel('Soiling Ratio [%]')
ax.set_xlabel('Time [day]')
ax.grid()
ax.set_title('Soiling Ratios')
# ax.set_ylim([96, 100])

plt.show()

fig, ax = plt.subplots(figsize=(15, 7))
sr_dust_iq.resample('1W').mean().plot(ax=ax, style='--o', alpha=0.75)
sr_isc_pvstand.resample('1W').mean().plot(ax=ax, style='--o', alpha=0.75)
sr_pmp_pvstand.resample('1W').mean().plot(ax=ax, style='--o', alpha=0.75)
sr_kwh_rc411.resample('1W').mean().plot(ax=ax, style='--o', alpha=0.75)
sr_soilingratio.resample('1W').mean().plot(ax=ax, style='--o', alpha=0.75)
ax.legend(['Dust IQ', 'Isc IV-Curve', 'Pmp IV-Curve',
           'Photocell','Pilot equipment'],fontsize = 12)
ax.set_ylabel('Soiling Ratio [%]',fontsize = 16)
ax.set_xlabel('Time [day]',fontsize = 16)
ax.grid()
ax.set_title('Soiling Ratios', fontsize = 16)
ax.set_xlim(date_s)
# ax.set_ylim(85,)
ax.tick_params(axis='both', labelsize=14)
save_plot(plt.gcf(), '1W_resample_soiling_ratios', subdir='intercomparison', dpi=800)
# fig.subplots_adjust(bottom=0.5)
plt.tight_layout()

plt.show()


fig, ax = plt.subplots(figsize=(15, 7))
sr_isc_pvstand.resample('1W').mean().plot(ax=ax, style='--o', alpha=0.75)
sr_pmp_pvstand.resample('1W').mean().plot(ax=ax, style='--o', alpha=0.75)
ax.legend(['Isc IV-Curve', 'Pmp IV-Curve'],fontsize = 12)
ax.set_ylabel('Soiling Ratio [%]',fontsize = 16)
ax.set_xlabel('Time [day]',fontsize = 16)
ax.grid()
ax.set_title('Soiling Ratios', fontsize = 16)
ax.set_xlim(date_s)
# ax.set_ylim(85,)
ax.tick_params(axis='both', labelsize=14)
save_plot(plt.gcf(), '1W_resample_soiling_ratios_pvstand', subdir='intercomparison', dpi=800)
# fig.subplots_adjust(bottom=0.5)
plt.tight_layout()

plt.show()
#comrpobar correccion Isc pvstand
fig, ax = plt.subplots(figsize=(15, 5))
df_pvstand.loc['2024-08-01':, ('MD434', ['Imax','Icorr'])].resample('1d').mean().plot(ax=ax)
df_pvstand.loc['2024-08-01':, ('MD439', ['Imax','Icorr'])].resample('1d').mean().plot(ax=ax)

ax.set_ylabel('Soiling Ratio [%]')
ax.set_xlabel('Time [day]')
ax.grid()
ax.set_title('Soiling Ratios')
# ax.set_ylim([96, 100])

#comrpobar correccion Isc soiling kit
fig, ax = plt.subplots(figsize=(15, 5))
df_soiling_kit.loc['2024-08-01':, ['Isc(e)','Isc(p)','Ie(c)','Ip(c)']].resample('1d').mean().plot(ax=ax)
ax.set_ylabel('Soiling Ratio [%]')
ax.set_xlabel('Time [day]')
ax.grid()
ax.set_title('Soiling Ratios')
# ax.set_ylim([96, 100])

plt.show()

#Soiling kit
#comrpobar correccion Isc soiling kit
fig, ax = plt.subplots(figsize=(15, 5))
df_soiling_kit.loc['2024-08-01':, ['Isc(e)','Isc(p)']].resample('1d').mean().plot(ax=ax)
ax.set_ylabel('Soiling Ratio [%]')
ax.set_xlabel('Time [day]')
ax.grid()
ax.set_title('Soiling Ratios')
# ax.set_ylim([96, 100])

plt.show()

fig, ax = plt.subplots(figsize=(15, 5))
sr_soilingratio.between_time('12:00', '18:00').resample('1d').mean().plot(ax=ax)
ax.grid()
ax.set_title('Soiling Ratio - Temperature Corrected')
ax.set_ylim([96, 100])

plt.show()

fig, ax = plt.subplots(figsize=(15, 5))
sr_soilingratio.between_time('11:00', '13:00').resample('1W').mean().plot(ax=ax)
sr_soilingratio.between_time('13:00', '15:00').resample('1W').mean().plot(ax=ax)
sr_soilingratio.between_time('15:00', '17:00').resample('1W').mean().plot(ax=ax)
ax.grid()
ax.legend(['12 PM', '2 PM', '4 PM'])
ax.set_title('Soiling Ratio - Temperature Corrected 3 times')
ax.set_ylim([90, 110])

plt.show()

# Reference Cells
fig, ax = plt.subplots(figsize=(10, 5))
sr_kwh_rc.loc['2024':, '1RC411(w.m-2)'].resample('1W').mean().sub(1.65).plot(ax=ax, style='--o', alpha=0.75)
sr_kwh_rc.loc['2024':, '1RC410(w.m-2)'].resample('1W').mean().sub(1.65).plot(ax=ax, style='--o', alpha=0.75)

ax.legend(['1RC411', '1RC410'])
ax.set_ylim([94, 101])
ax.set_ylabel('Soiling Ratio [-]')
ax.set_title('Soiling Ratio - Intercomparison')
ax.grid()

plt.show()
fig.savefig('Reference_cells.jpg',dpi=1000)

fig, ax = plt.subplots(figsize=(10, 5))
df_soiling_kit.plot(ax=ax)
ax.set_ylim([0.5,2])

#%% Ventana mas pequeña (14 a 18)


df_glasses = df_glasses.between_time('14:00', '18:00')
df_pvstand = df_pvstand.between_time('14:00', '18:00')
df_rc_fix = df_rc_fix.between_time('14:00', '18:00')
df_rc_fix_dec = df_rc_fix_dec.between_time('14:00', '18:00')
df_pt100_fix = df_pt100_fix.between_time('14:00', '18:00')
df_pt100_fix_dec = df_pt100_fix_dec.between_time('14:00', '18:00')
df_dust_iq = df_dust_iq.between_time('14:00', '18:00')

df_soiling_kit = df_soiling_kit.between_time('14:00', '18:00')
df_glasses_daily = df_glasses.between_time('14:00','18:00').resample('1d').sum().div(60000)

#%% Benchmark

fig,ax = plt.subplots(figsize=(10,5))

# sr_kwh_rc.loc['2024':, '1RC411(w.m-2)'].resample('1W').mean().sub(1.65).plot(ax=ax, style='--o', alpha=0.75)
sr_pmp_iv600.loc['2024':'2025','1MD434/439'].resample('1W').mean().sub(1.65).plot(ax=ax,style='--o',alpha=0.75).plot(ax=ax,style='--o', alpha=0.75)
sr_isc_iv600.loc['2024':'2025','1MD434/439'].resample('1W').mean().sub(1.65).plot(ax=ax,style='--o',alpha=0.75).plot(ax=ax,style='--o', alpha=0.75)
sr_kwh_rc.loc['2024':'2025','1RC411(w.m-2)'].resample('1W').mean().sub(1.65).plot(ax=ax,style='--o',alpha=0.75)
sr_isc_pvstand.loc['2024':'2025'].between_time('12:00','18:00').resample('1W').mean().plot(ax=ax,style='--o',alpha=0.75)
#sr_isc_iv600.loc['2024':'2025'].between_time('12:00','18:00').resample('1W').mean().plot(ax=ax,style='--o',alpha=0.75)
sr_dust_iq.loc['2024':'2025'].between_time('14:00','18:00').resample('1W').mean().plot(ax=ax,style='--o',alpha=0.75)
sr_soilingratio.resample('1W').mean().plot(ax=ax, style='--o', alpha=0.75)

ax.legend(['IV600 Pmp','IV600 Isc','PhotoCells','IV-Curves','Dust IQ','SoilingRatio'])
ax.set_ylim([90,105])
ax.set_ylabel('Soiling Ratio [-]')
ax.set_title('Soiling Raito - Intercomparison')
ax.grid()

save_plot(fig, 'benchmark', subdir='intercomparison', dpi=1000)

#%% Grafico DustIQ vs SoilingRatio semanal
fig, ax = plt.subplots(figsize=(15, 7))

# Obtener datos semanales
sr_dustiq_semanal = sr_dust_iq.resample('1W').mean()
sr_soilingratio_semanal = sr_soilingratio.resample('1W').mean()

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
save_plot(plt.gcf(), 'soiling_ratio_intercomparison_semanal', subdir='intercomparison', dpi=800)
plt.show()