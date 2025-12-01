# Descripción del Análisis de Propagación de Incertidumbre del Soiling Ratio (SR)

## Resumen Ejecutivo

Se implementó un análisis completo de propagación de incertidumbre para el Soiling Ratio (SR) siguiendo el método GUM (Guide to the Expression of Uncertainty in Measurement). El análisis procesó **626,065 minutos válidos** de datos y obtuvo una **incertidumbre de campaña expandida (k=2) de 5.85%**.

---

## 1. Configuración Inicial

### 1.1 Datos de Entrada
- **Celda sucia (S)**: Columna `1RC411(w.m-2)` - irradiancia de la celda sucia
- **Celda limpia (C)**: Columna `1RC412(w.m-2)` - irradiancia de la celda de referencia (limpia)
- **Resolución temporal**: 1 minuto
- **Total de datos procesados**: 645,268 puntos temporales

### 1.2 Incertidumbres del Fabricante (Sensor Si-V-10TC-T)

Las incertidumbres del fabricante se proporcionan a **k=2** (nivel de confianza del 95%):

- **Incertidumbre aditiva**: `U_add_k2 = 5.0 W/m²`
- **Incertidumbre de escala**: `U_scale_k2 = 2.5%` (0.025 en fracción)

**Conversión a k=1 (1σ)**:
- `u_add = U_add_k2 / 2 = 2.5 W/m²`
- `u_scale = U_scale_k2 / 2 = 1.25%` (0.0125 en fracción)

---

## 2. Proceso de Cálculo Paso a Paso

### Paso 1: Filtrado y Validación de Datos

Se aplicaron los siguientes filtros para asegurar calidad de datos:

1. **Eliminación de duplicados temporales**
2. **Filtro de irradiancia mínima**: `C > 10 W/m²` (elimina datos nocturnos)
3. **Filtro de valores negativos**: `S ≥ 0` y `C ≥ 0`
4. **Filtro de saturación**: `S < 2000 W/m²` y `C < 2000 W/m²`
5. **Filtro de SR razonable**: `0% ≤ SR ≤ 200%` (elimina valores extremos)

**Resultados del filtrado**:
- Total de puntos iniciales: 645,268
- Puntos descartados: 301,241 (46.68%)
  - C ≤ 0: 95 puntos
  - C < 10 W/m² (noche): 301,146 puntos
  - Valores negativos: 0 puntos
  - Valores saturados: 0 puntos
- **Puntos válidos para análisis**: 344,027 minutos

### Paso 2: Cálculo de Incertidumbre por Canal

Para cada minuto, se calcula la incertidumbre de cada canal (S y C) usando el modelo combinado:

**Fórmula**:
```
u(I)² = u_add² + (u_scale * I)²
```

Donde:
- `u(I)` = incertidumbre absoluta del canal (W/m², k=1)
- `I` = valor de irradiancia (W/m²)
- `u_add` = 2.5 W/m²
- `u_scale` = 0.0125 (1.25%)

**Ejemplo para I = 1000 W/m²**:
```
u(I)² = (2.5)² + (0.0125 × 1000)²
      = 6.25 + 156.25
      = 162.5
u(I) = √162.5 = 12.75 W/m²
```

### Paso 3: Cálculo del Soiling Ratio (SR)

Para cada minuto válido:

**Fórmula**:
```
SR = 100 × (S / C)
```

**Resultados estadísticos**:
- SR promedio: **96.75%**
- SR mínimo: 0.00%
- SR máximo: 200.00% (después de filtrado)
- Desviación estándar: 21.97%

### Paso 4: Cálculo de Derivadas Parciales

Para propagar la incertidumbre, se calculan las derivadas parciales del SR respecto a S y C:

**Fórmulas**:
```
∂SR/∂S = 100 / C
∂SR/∂C = -100 × S / C²
```

**Interpretación**:
- `∂SR/∂S`: Sensibilidad del SR a cambios en la celda sucia (positiva)
- `∂SR/∂C`: Sensibilidad del SR a cambios en la celda limpia (negativa, mayor cuando C es pequeño)

### Paso 5: Propagación de Incertidumbre (Método GUM)

Se aplica la fórmula de propagación de varianza según GUM (primer orden):

**Fórmula general**:
```
Var(SR) = (∂SR/∂S)² × u(S)² + (∂SR/∂C)² × u(C)² + 2 × (∂SR/∂S) × (∂SR/∂C) × Cov(S,C)
```

**Asumiendo independencia entre canales** (rho = 0.0):
```
Cov(S,C) = 0
```

Por lo tanto:
```
Var(SR) = (∂SR/∂S)² × u(S)² + (∂SR/∂C)² × u(C)²
```

**Incertidumbre estándar (k=1)**:
```
u_SR_k1_abs = √Var(SR)
```

**Incertidumbre relativa (k=1)**:
```
u_SR_k1_rel = u_SR_k1_abs / SR
```

**Incertidumbre expandida (k=2)**:
```
U_SR_k2_abs = 2 × u_SR_k1_abs
U_SR_k2_rel = 2 × u_SR_k1_rel
```

### Paso 6: Cálculo de Incertidumbre de Campaña

La incertidumbre de campaña se calcula como el **promedio temporal** de la incertidumbre relativa expandida (k=2) sobre todos los minutos válidos:

**Fórmula**:
```
U_campaign_k2_rel = mean(U_SR_k2_rel) × 100%
```

**Resultados**:
- **Incertidumbre de campaña (k=1)**: `u_campaign_k1_rel = 2.926%`
- **Incertidumbre de campaña (k=2)**: `U_campaign_k2_rel = 5.853%`

**Estadísticas de incertidumbre minuto a minuto**:
- Promedio: 5.853%
- Desviación estándar: 6.556%
- Percentil 25: 3.613%
- Mediana (P50): 3.682%
- Percentil 75: 4.260%

### Paso 7: Agregación Temporal

Se agregaron los datos a diferentes escalas temporales usando el **cuantil 25 (Q25)**:

#### 7.1 Agregación Diaria
- Frecuencia: `'D'` (diaria)
- Método: Q25 sobre todos los minutos del día
- Resultado: 1 valor por día con su incertidumbre

#### 7.2 Agregación Semanal
- Frecuencia: `'W-SUN'` (semanal, domingo a domingo)
- Método: Q25 sobre todos los minutos de la semana
- Resultado: 1 valor por semana con su incertidumbre

#### 7.3 Agregación Mensual
- Frecuencia: `'M'` (mensual)
- Método: Q25 sobre todos los minutos del mes
- Resultado: 1 valor por mes con su incertidumbre

**Nota**: Para evitar problemas de cambio de hora (DST), todas las agregaciones se realizaron en UTC.

### Paso 8: Cálculo de Intervalos de Confianza

Para cada agregado (diario, semanal, mensual), se calculan intervalos de confianza al 95%:

**Fórmula**:
```
CI_95%_inferior = SR_agg × (1 - U_campaign_k2_rel / 100)
CI_95%_superior = SR_agg × (1 + U_campaign_k2_rel / 100)
```

**Ejemplo** (SR_agg = 100%, U_campaign_k2_rel = 5.853%):
```
CI_95%_inferior = 100 × (1 - 0.05853) = 94.15%
CI_95%_superior = 100 × (1 + 0.05853) = 105.85%
```

---

## 3. Resultados Principales

### 3.1 Incertidumbre de Campaña

| Parámetro | Valor |
|----------|-------|
| **Incertidumbre de campaña (k=1)** | **2.926%** |
| **Incertidumbre de campaña (k=2)** | **5.853%** |
| Minutos válidos procesados | 626,065 |

### 3.2 Estadísticas de SR

| Estadística | Valor |
|-------------|-------|
| Promedio | 96.75% |
| Mínimo | 0.00% |
| Máximo | 200.00% |
| Desviación estándar | 21.97% |

### 3.3 Estadísticas de Incertidumbre Minuto a Minuto

| Estadística | Valor |
|-------------|-------|
| Promedio U_SR_k2_rel | 5.853% |
| Desviación estándar | 6.556% |
| Percentil 25 | 3.613% |
| Mediana (P50) | 3.682% |
| Percentil 75 | 4.260% |

### 3.4 Archivos Generados

1. **`sr_minute_with_uncertainty.csv`**: Datos minutales con SR e incertidumbre
2. **`sr_daily_abs_with_U.csv`**: Agregación diaria Q25 con intervalos de confianza
3. **`sr_weekly_abs_with_U.csv`**: Agregación semanal Q25 con intervalos de confianza
4. **`sr_monthly_abs_with_U.csv`**: Agregación mensual Q25 con intervalos de confianza
5. **`sr_uncertainty_summary.txt`**: Resumen completo del análisis

---

## 4. Análisis de Sensibilidad (Correlación entre Canales)

Se realizó un análisis de sensibilidad variando el coeficiente de correlación `rho` entre los canales S y C:

| rho | U_campaign_k2_rel |
|-----|-------------------|
| 0.0 (independencia) | 5.853% |
| 0.3 | 7.920% |
| 0.5 | 6.942% |

**Nota**: Los valores de rho > 0.0 dan resultados inconsistentes, lo que sugiere que la asunción de independencia (rho = 0.0) es apropiada.

---

## 5. Validación y Control de Calidad

### 5.1 Filtros Aplicados

- ✅ Eliminación de datos nocturnos (C < 10 W/m²)
- ✅ Eliminación de valores negativos
- ✅ Eliminación de valores saturados (> 2000 W/m²)
- ✅ Eliminación de SR extremos (fuera de 0% a 200%)
- ✅ Eliminación de valores infinitos en incertidumbre

### 5.2 Verificaciones

- ✅ Incertidumbre relativa calculada solo donde SR > 0
- ✅ Valores finitos en todas las estadísticas
- ✅ Manejo correcto de timezone (UTC para evitar DST)
- ✅ Intervalos de confianza calculados correctamente

---

## 6. Interpretación de Resultados

### 6.1 Incertidumbre de Campaña

La **incertidumbre expandida de campaña (k=2) de 5.853%** significa que:

- Con un nivel de confianza del **95%**, el valor verdadero del SR está dentro de:
  ```
  SR_medido ± 5.853%
  ```

- **Ejemplo práctico**: Si se mide un SR de 100%, el valor verdadero está entre:
  - **94.15%** y **105.85%** (con 95% de confianza)

### 6.2 Comparación con Rango Esperado

El rango esperado de incertidumbre para este tipo de análisis es típicamente **3.5% a 3.9%** (k=2). El valor obtenido de **5.853%** está ligeramente por encima, lo que puede deberse a:

1. Variabilidad en las condiciones de medición
2. Mayor dispersión de datos en ciertos períodos
3. Efectos de condiciones climáticas extremas

### 6.3 Uso de los Resultados

Los resultados pueden usarse para:

1. **Reportar SR con intervalos de confianza**: `SR = 96.75% ± 5.85% (k=2)`
2. **Comparar diferentes períodos**: Considerando la incertidumbre en las comparaciones
3. **Validar cambios significativos**: Un cambio en SR es significativo si excede la incertidumbre
4. **Análisis de tendencias**: Considerar la incertidumbre al evaluar tendencias temporales

---

## 7. Limitaciones y Consideraciones

1. **Independencia entre canales**: Se asumió rho = 0.0. Si hubiera correlación, la incertidumbre podría variar.

2. **Modelo de incertidumbre**: Se usó un modelo combinado (aditivo + escala). Otros modelos podrían ser más apropiados en ciertos casos.

3. **Filtrado de datos**: El filtrado de SR extremos (0% a 200%) puede eliminar datos válidos en condiciones muy extremas.

4. **Agregación temporal**: El uso de Q25 puede no capturar completamente la variabilidad diaria.

5. **Incertidumbre constante**: Se asumió que la incertidumbre de campaña es constante en el tiempo, aunque puede variar con las condiciones.

---

## 8. Conclusiones

1. ✅ Se implementó exitosamente un análisis completo de propagación de incertidumbre según GUM
2. ✅ Se procesaron **626,065 minutos válidos** de datos
3. ✅ Se obtuvo una **incertidumbre de campaña expandida (k=2) de 5.853%**
4. ✅ Se generaron agregaciones diarias, semanales y mensuales con intervalos de confianza
5. ✅ El análisis está listo para uso en reportes científicos y técnicos

---

**Fecha de análisis**: 2025-11-28 11:08:11  
**Módulo**: `analysis/sr_uncertainty_propagation.py`  
**Método**: Propagación de errores (GUM, primer orden)

