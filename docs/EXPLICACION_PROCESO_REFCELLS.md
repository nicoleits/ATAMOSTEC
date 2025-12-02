# Explicación del Proceso de Análisis de RefCells

## 📋 Resumen General

El análisis de RefCells calcula el **Soiling Ratio (SR)** y su **incertidumbre** usando el método GUM (Guide to the Expression of Uncertainty in Measurement). El proceso completo se ejecuta en varios módulos que trabajan en conjunto.

---

## 🚀 Punto de Entrada

### Comando de Ejecución
```bash
python3 -m analysis.ref_cells_analyzer
```

### Flujo de Ejecución

```
run_analysis() 
    ↓
analyze_ref_cells_data(raw_data_filepath)
    ↓
[Procesamiento de datos]
    ↓
run_uncertainty_propagation_analysis()
    ↓
[Generación de gráficos]
```

---

## 📦 Estructura de Módulos

### 1. **`ref_cells_analyzer.py`** (Módulo Principal)
   - **No usa clases**, solo funciones
   - **Función principal**: `analyze_ref_cells_data()`
   - **Responsabilidades**:
     - Carga y preprocesamiento de datos
     - Cálculo de SR
     - Filtrado y ajuste de datos
     - Generación de gráficos
     - Coordinación del análisis de incertidumbre

### 2. **`sr_uncertainty_propagation.py`** (Módulo de Incertidumbre)
   - **No usa clases**, solo funciones
   - **Funciones clave**:
     - `channel_u()`: Calcula incertidumbre por canal (fotocelda)
     - `propagate_sr_minute()`: Propaga incertidumbre minuto a minuto
     - `aggregate_with_uncertainty()`: Agrega datos con incertidumbre (diario/semanal/mensual)
     - `run_uncertainty_propagation_analysis()`: Función principal del módulo

---

## 🔄 Proceso Paso a Paso

### **PASO 1: Carga de Datos** (`analyze_ref_cells_data()`)
```python
# 1.1. Cargar CSV con datos brutos
df_ref_cells = pd.read_csv(raw_data_filepath, index_col='timestamp')

# 1.2. Convertir índice a datetime
df_ref_cells.index = pd.to_datetime(...)

# 1.3. Asegurar timezone UTC
df_ref_cells.index = df_ref_cells.index.tz_localize('UTC')
```

**Datos de entrada:**
- Archivo: `datos/refcells/refcells_data.csv`
- Columnas: `1RC411(w.m-2)` (celda sucia), `1RC412(w.m-2)` (celda limpia), etc.

---

### **PASO 2: Filtrado por Mediodía Solar** (Opcional)
```python
# Solo si se ejecuta run_analysis_solar_noon()
df_ref_cells = filter_by_solar_noon(df_ref_cells, hours_window=2.5)
```

**Función:** `filter_by_solar_noon()`
- Usa `UtilsMedioDiaSolar` para calcular mediodía solar real
- Filtra datos ±2.5 horas alrededor del mediodía solar

---

### **PASO 3: Cálculo de Soiling Ratio (SR)**
```python
# SR = 100 * S / C
# donde:
#   S = irradiancia celda sucia (1RC411)
#   C = irradiancia celda limpia (1RC412)

sr_df = (df_ref_cells[soiled_col] / df_ref_cells[clean_col]) * 100
```

**Filtros aplicados:**
- `C >= 200 W/m²` (umbral mínimo de irradiancia)
- `SR entre 0% y 200%` (valores razonables)

---

### **PASO 4: Análisis de Propagación de Incertidumbre**

#### 4.1. **Cálculo Minuto a Minuto** (`propagate_sr_minute()`)

```python
# Para cada minuto:
# 1. Calcular incertidumbre de cada canal
u_S = channel_u(S, u_add=2.5, u_scale=0.0125)  # Celda sucia
u_C = channel_u(C, u_add=2.5, u_scale=0.0125)  # Celda limpia

# 2. Calcular derivadas parciales
dSR_dS = 100 / C
dSR_dC = -100 * S / C²

# 3. Calcular varianza de SR (propagación de errores)
Var_SR = (dSR_dS)² * u_S² + (dSR_dC)² * u_C² + 2 * dSR_dS * dSR_dC * Cov(S,C)

# 4. Calcular incertidumbre expandida (k=2)
U_SR_k2 = k_expand * sqrt(Var_SR)
```

**Parámetros de incertidumbre:**
- `U_ADD_K2 = 5.0 W/m²` (incertidumbre aditiva, k=2)
- `U_SCALE_K2 = 0.025` (2.5% de escala, k=2)
- `rho = 0.0` (correlación entre S y C, asumida 0)

#### 4.2. **Agregación Temporal** (`aggregate_with_uncertainty()`)

```python
# Agregar SR a diferentes escalas temporales:
# - Diario (Q25): resample('D').quantile(0.25)
# - Semanal (Q25): resample('W-SUN').quantile(0.25)
# - Mensual (Q25): resample('M').quantile(0.25)

# Para cada agregado, calcular incertidumbre LOCAL:
# Promediar la incertidumbre minuto a minuto de ese período
U_agg = df_uncertainty['U_SR_k2_rel'].resample('D').mean()
```

**Archivos generados:**
- `sr_minute_with_uncertainty.csv` (datos minuto a minuto)
- `sr_daily_abs_with_U.csv` (datos diarios con incertidumbre)
- `sr_weekly_abs_with_U.csv` (datos semanales con incertidumbre)
- `sr_monthly_abs_with_U.csv` (datos mensuales con incertidumbre)

---

### **PASO 5: Filtrado y Ajuste de SR**

```python
# 5.1. Filtrar SR extremos
sr_filtered = sr_df[(sr_df >= sr_min) & (sr_df <= sr_max)]

# 5.2. Calcular Q25 (cuantil 25%) diario y semanal
df_daily_sr_q25 = sr_filtered.resample('D').quantile(0.25)
df_weekly_sr_q25 = sr_filtered.resample('W-SUN').quantile(0.25)

# 5.3. Ajustar a 100% (opcional, si REFCELLS_ADJUST_TO_100_FLAG = True)
# Ajusta el primer valor válido a 100%
serie_adjusted = serie + (100 - first_valid_value)
```

---

### **PASO 6: Generación de Gráficos**

#### 6.1. **Gráficos Combinados**
- `refcells_sr_combinado_semanal.png`: Todas las celdas, semanal
- `refcells_sr_combinado_diario.png`: Todas las celdas, diario

#### 6.2. **Gráficos Individuales por Celda**
- `refcell_1RC411wm2_sr_semanal_periodo_especifico.png`: Semanal con tendencia
- `refcell_1RC411wm2_sr_diario_q25_tendencia.png`: Diario con tendencia y barras de error
- `refcell_1RC411wm2_sr_3meses.png`: Primeros 3 meses
- `refcell_1RC411wm2_sr_semanal_3meses.png`: Primeros 3 meses semanal

**Funciones de generación:**
- `_generate_specific_cell_plot()`: Gráfico semanal con tendencia y barras de error
- `_generate_daily_q25_trend_plot()`: Gráfico diario con tendencia y barras de error
- `_generate_first_3_months_plot()`: Gráfico de primeros 3 meses
- `_generate_first_3_months_weekly_plot()`: Gráfico semanal primeros 3 meses

---

## 🔍 Detalles Técnicos Importantes

### **Cálculo de Tendencia**
```python
# Usa timestamps reales (no índices secuenciales)
x_days = (valid_dates - first_date).total_seconds() / 86400.0
slope_days, intercept, r_value, p_value, std_err = stats.linregress(x_days, y_valid)

# Convertir a %/semana para gráficos semanales
slope_weeks = slope_days * 7
```

### **Barras de Error**
```python
# Cargar datos de incertidumbre agregada
df_uncertainty = pd.read_csv('sr_daily_abs_with_U.csv')

# Para cada punto del gráfico:
u_rel = df_uncertainty.loc[date, 'U_rel_k2']  # Incertidumbre relativa (%)
yerr = u_rel * sr_value / 100.0  # Convertir a valor absoluto

# Graficar con errorbar
ax.errorbar(x, y, yerr=yerr, ...)
```

### **Límites del Eje Y**
```python
# Fijos: 50% a 110%
ax.set_ylim([50, 110])
```

---

## 📊 Flujo de Datos

```
Datos Brutos (CSV)
    ↓
Preprocesamiento (timezone, filtros)
    ↓
Cálculo SR = 100 * S / C
    ↓
┌─────────────────┬──────────────────┐
│                 │                  │
Análisis Normal   Análisis Mediodía Solar
│                 │                  │
│                 │                  │
↓                 ↓                  ↓
Propagación de Incertidumbre (GUM)
    ↓
┌─────────────────┬──────────────────┐
│                 │                  │
Minuto a Minuto   Agregación Temporal
│                 │                  │
│                 │                  │
↓                 ↓                  ↓
CSVs con          Gráficos con
Incertidumbre     Barras de Error
```

---

## 🎯 Diferencias entre Análisis Normal y Mediodía Solar

| Aspecto | Análisis Normal | Mediodía Solar |
|--------|----------------|----------------|
| **Datos** | Todos los datos (24h) | Solo ±2.5h alrededor mediodía |
| **Incertidumbre** | ~4.4-6.2% | ~3.6% (menor, condiciones más estables) |
| **Archivos** | `propagacion de errores/ref_cell/` | `propagacion de errores/ref_cell/mediodia_solar/` |
| **Gráficos** | `graficos_analisis_integrado_py/ref_cells/` | `graficos_analisis_integrado_py/ref_cells/mediodia_solar/` |

---

## 🔧 Configuración

### **Archivos de Configuración:**
- `config/paths.py`: Rutas de archivos
- `config/settings.py`: Parámetros (umbrales, flags, etc.)

### **Parámetros Clave:**
- `MIN_IRRADIANCE_THRESHOLD = 200 W/m²`
- `U_ADD_K2 = 5.0 W/m²`
- `U_SCALE_K2 = 0.025 (2.5%)`
- `REFCELLS_ADJUST_TO_100_FLAG`: Ajustar primer valor a 100%

---

## 📝 Resumen de Funciones Principales

### **`ref_cells_analyzer.py`**
1. `analyze_ref_cells_data()`: Función principal del análisis normal
2. `analyze_ref_cells_data_solar_noon()`: Función principal del análisis mediodía solar
3. `filter_by_solar_noon()`: Filtra datos por mediodía solar
4. `_generate_specific_cell_plot()`: Genera gráfico semanal individual
5. `_generate_daily_q25_trend_plot()`: Genera gráfico diario con tendencia
6. `_adjust_series_start_to_100()`: Ajusta serie para que empiece en 100%

### **`sr_uncertainty_propagation.py`**
1. `channel_u()`: Calcula incertidumbre por canal
2. `propagate_sr_minute()`: Propaga incertidumbre minuto a minuto
3. `aggregate_with_uncertainty()`: Agrega datos con incertidumbre
4. `run_uncertainty_propagation_analysis()`: Función principal del módulo

---

## ✅ No se Usan Clases

**Todo el código está basado en funciones**, no en clases. Esto hace el código más simple y directo:
- Fácil de entender
- Fácil de depurar
- Fácil de mantener
- No hay estado compartido entre funciones

---

## 🎓 Conceptos Clave

1. **GUM (Guide to the Expression of Uncertainty in Measurement)**: Método estándar para calcular incertidumbre
2. **Propagación de Errores**: Cómo se combinan las incertidumbres de las variables de entrada
3. **Derivadas Parciales**: Miden cómo cambia SR cuando cambian S o C
4. **Factor de Cobertura (k=2)**: Expande la incertidumbre a un nivel de confianza del 95%
5. **Incertidumbre Local vs Global**: La incertidumbre varía según las condiciones de cada período

---

¿Tienes alguna pregunta específica sobre alguna parte del proceso?

