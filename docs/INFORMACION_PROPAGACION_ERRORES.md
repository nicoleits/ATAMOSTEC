# Información Necesaria para Propagación de Errores

Este documento lista toda la información necesaria para implementar la propagación de errores (GUM) en cada análisis del sistema.

---

## 1. SOILING KIT

### Cálculo Principal
- **SR = 100 × Isc(p) / Isc(e)**
- Con corrección de temperatura: `Isc_corr = Isc × (1 + α_isc × (T - T_ref))`

### Mediciones Utilizadas
- **Isc(e)**: Corriente de cortocircuito módulo expuesto (A)
- **Isc(p)**: Corriente de cortocircuito módulo protegido (A)
- **Te(C)**: Temperatura módulo expuesto (°C)
- **Tp(C)**: Temperatura módulo protegido (°C)

### Información Necesaria

#### Incertidumbres de Medición
- [ ] **u(Isc)**: Incertidumbre de medición de corriente de cortocircuito (A)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del amperímetro/multímetro
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_soiling_kit.py` líneas 39-44
    - `U_ISC_ADD_K2` (línea 39): Incertidumbre aditiva (A, k=2)
    - `U_ISC_SCALE_K2` (línea 40): Incertidumbre de escala (adimensional, k=2)

- [ ] **u(T)**: Incertidumbre de medición de temperatura (°C)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del sensor de temperatura
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_soiling_kit.py` líneas 48-52
    - `U_TEMP_ADD_K2` (línea 48): Incertidumbre aditiva (°C, k=2)

#### Coeficientes de Corrección
- [ ] **α_isc**: Coeficiente de temperatura de Isc (%/°C o 1/°C)
  - Valor actual: ¿?
  - Fuente: Datasheet del módulo o calibración
  - Incertidumbre de α_isc: ¿?
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_soiling_kit.py` línea 56
    - `U_ALPHA_ISC` (línea 56): Incertidumbre del coeficiente α_isc (1/°C, k=1)

- [ ] **T_ref**: Temperatura de referencia para corrección (°C)
  - Valor actual: ¿?
  - Incertidumbre: ¿?

#### Correlaciones
- [ ] **ρ(Isc(e), Isc(p))**: Correlación entre corrientes de módulos expuesto y protegido
  - Valor estimado: ¿?
  - Justificación: ¿Mismo instrumento de medición?

- [ ] **ρ(Te, Tp)**: Correlación entre temperaturas
  - Valor estimado: ¿?
  - Justificación: ¿Mismo sensor o ambiente similar?

#### Información del Equipo
- [ ] Modelo del amperímetro/multímetro usado
- [ ] Certificado de calibración del amperímetro
- [ ] Modelo del sensor de temperatura
- [ ] Certificado de calibración del sensor de temperatura
- [ ] Modelo del módulo fotovoltaico (para α_isc)

---

## 2. DUSTIQ

### Estado
✅ **YA IMPLEMENTADO** - Ver `analysis/sr_uncertainty_dustiq.py`

### Cálculo Principal
- **SR_C11_Avg**: Valor directo del sensor (ya es SR en %)

### Mediciones Utilizadas
- **SR_C11_Avg**: Soiling Ratio promedio del canal C11 (%)
- **SR_C12_Avg**: Soiling Ratio promedio del canal C12 (%)

### Información Necesaria

#### Incertidumbres del Sensor
- [x] **u(SR_DustIQ)**: Incertidumbre del sensor DustIQ (%)
  - Tipo: Aditiva y de escala
  - Valor k=2: **U_ADD_K2 = 0.1%**, **U_SCALE_K2 = 1%** ✅
  - Fuente: Especificaciones del fabricante (accuracy: ±0.1% of reading ±1%)
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_dustiq.py` líneas 37-44
    - `U_SR_ADD_K2` (línea 37): Incertidumbre aditiva (%, k=2) = 0.1%
    - `U_SR_SCALE_K2` (línea 38): Incertidumbre de escala (adimensional, k=2) = 0.01 (1%)

- [ ] **Rango de medición**: ¿?
- [ ] **Resolución**: ¿?

#### Especificaciones del Fabricante
- [ ] Modelo del sensor DustIQ
- [ ] Manual técnico con especificaciones de incertidumbre
- [ ] Certificado de calibración (si aplica)
- [ ] Condiciones de operación (temperatura, humedad, etc.)

#### Correlaciones
- [ ] **ρ(SR_C11, SR_C12)**: Correlación entre canales C11 y C12
  - Valor estimado: ¿?
  - Justificación: ¿Mismo sensor, diferentes canales?

---

## 3. PVSTAND

### Estado
✅ **IMPLEMENTADO** - Ver `analysis/sr_uncertainty_pvstand.py`  
⚠️ **Parcialmente completado**: Valores de IV tracer actualizados, faltan sensor de temperatura y coeficientes de módulo  
📋 **Ver guía de búsqueda**: `docs/GUIA_BUSQUEDA_INFORMACION_PVSTAND.md`

### Cálculo Principal
- **SR_Isc = 100 × Isc_soiled / Isc_reference**
- **SR_Pmax = 100 × Pmax_soiled / Pmax_reference**
- Con corrección de temperatura:
  - `Isc_corr = Isc × (1 + α_isc × (T - T_ref))`
  - `Pmax_corr = Pmax × (1 + β_pmax × (T - T_ref))`

### Mediciones Utilizadas
- **Isc_soiled**: Corriente de cortocircuito módulo sucio (A)
- **Isc_reference**: Corriente de cortocircuito módulo referencia (A)
- **Pmax_soiled**: Potencia máxima módulo sucio (W)
- **Pmax_reference**: Potencia máxima módulo referencia (W)
- **T_soiled**: Temperatura módulo sucio (°C)
- **T_reference**: Temperatura módulo referencia (°C)

### Información Necesaria

#### Incertidumbres de Medición
- [x] **u(Isc)**: Incertidumbre de medición de corriente (A)
  - Tipo: De escala
  - Valor k=2: **U_SCALE_K2 = 0.2%** ✅
  - Fuente: Especificaciones del fabricante del IV tracer (accuracy: 0.2% of reading)
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_pvstand.py` líneas 38-46
    - `U_ISC_ADD_K2` (línea 38): Incertidumbre aditiva (A, k=2) = 0.0 (sin componente aditiva)
    - `U_ISC_SCALE_K2` (línea 39): Incertidumbre de escala (adimensional, k=2) = 0.002 (0.2%) ✅

- [x] **u(Pmax)**: Incertidumbre de medición de potencia (W)
  - Tipo: De escala
  - Valor k=2: **U_SCALE_K2 = 0.4%** ✅
  - Fuente: Especificaciones del fabricante del IV tracer (accuracy: 0.4% of reading para MPP)
  - Nota: Puede depender de u(Isc) y u(Vmax)
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_pvstand.py` líneas 41-49
    - `U_PMAX_ADD_K2` (línea 41): Incertidumbre aditiva (W, k=2) = 0.0 (sin componente aditiva)
    - `U_PMAX_SCALE_K2` (línea 42): Incertidumbre de escala (adimensional, k=2) = 0.004 (0.4%) ✅

- [ ] **u(Vmax)**: Incertidumbre de medición de voltaje máximo (V)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del equipo IV tracer
  - **⚠️ NOTA**: Actualmente no se usa en el código (solo se usa Isc y Pmax)

- [ ] **u(T)**: Incertidumbre de medición de temperatura (°C)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del sensor de temperatura
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_pvstand.py` líneas 52-53
    - `U_TEMP_ADD_K2` (línea 52): Incertidumbre aditiva (°C, k=2)

#### Coeficientes de Corrección
- [ ] **α_isc**: Coeficiente de temperatura de Isc (%/°C o 1/°C)
  - Valor actual: ¿?
  - Fuente: Datasheet del módulo
  - Incertidumbre de α_isc: ¿?
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_pvstand.py` línea 56
    - `U_ALPHA_ISC` (línea 56): Incertidumbre del coeficiente α_isc (1/°C, k=1)

- [ ] **β_pmax**: Coeficiente de temperatura de Pmax (%/°C o 1/°C)
  - Valor actual: ¿?
  - Fuente: Datasheet del módulo
  - Incertidumbre de β_pmax: ¿?
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_pvstand.py` línea 57
    - `U_BETA_PMAX` (línea 57): Incertidumbre del coeficiente β_pmax (1/°C, k=1)

- [ ] **T_ref**: Temperatura de referencia para corrección (°C)
  - Valor actual: ¿?
  - Incertidumbre: ¿?

#### Correlaciones
- [ ] **ρ(Isc_soiled, Isc_reference)**: Correlación entre corrientes
  - Valor estimado: ¿?
  - Justificación: ¿Mismo instrumento de medición?

- [ ] **ρ(Pmax_soiled, Pmax_reference)**: Correlación entre potencias
  - Valor estimado: ¿?
  - Justificación: ¿Mismo instrumento de medición?

- [ ] **ρ(T_soiled, T_reference)**: Correlación entre temperaturas
  - Valor estimado: ¿?
  - Justificación: ¿Mismo sensor o ambiente similar?

- [ ] **ρ(Isc, Pmax)**: Correlación entre Isc y Pmax (mismo módulo)
  - Valor estimado: ¿?
  - Justificación: ¿Mediciones simultáneas del mismo módulo?

#### Información del Equipo
- [ ] Modelo del IV tracer usado
- [ ] Certificado de calibración del IV tracer
- [ ] Especificaciones técnicas del IV tracer (precisión, resolución)
- [ ] Modelo del sensor de temperatura
- [ ] Certificado de calibración del sensor de temperatura
- [ ] Modelo del módulo fotovoltaico (para α_isc y β_pmax)

---

## 4. PV GLASSES

### Estado
✅ **IMPLEMENTADO** - Ver `analysis/sr_uncertainty_pv_glasses.py`

### Cálculo Principal
- **SR = 100 × R_FCi_Avg / REF**
- Donde:
  - `REF = (R_FC1_Avg + R_FC2_Avg) / 2` (promedio de dos celdas de referencia limpias)
  - `R_FCi_Avg` son transmitancias de vidrios sucios (i = 3, 4, 5)
- **NO usa IV tracer**, usa las mismas fotoceldas que ref_cells (Si-V-10TC-T)
- Las masas se miden con una balanza, pero solo se usan para correlación, no en el cálculo del SR

### Mediciones Utilizadas
- **R_FC1_Avg**: Irradiancia celda de referencia 1 (W/m²)
- **R_FC2_Avg**: Irradiancia celda de referencia 2 (W/m²)
- **R_FC3_Avg**: Irradiancia celda sucia 3 (W/m²)
- **R_FC4_Avg**: Irradiancia celda sucia 4 (W/m²)
- **R_FC5_Avg**: Irradiancia celda sucia 5 (W/m²)
- **Masas**: Medidas con balanza (solo para correlación, no afecta incertidumbre de SR)

### Información Necesaria

#### Incertidumbres de Medición
- [x] **u(R_FCi)**: Incertidumbre de medición de irradiancia (W/m²)
  - Tipo: Aditiva y de escala
  - Valor k=2: **U_ADD_K2 = 5.0 W/m²**, **U_SCALE_K2 = 0.025 (2.5%)** ✅
  - Fuente: Especificaciones del fabricante (Si-V-10TC-T) - **MISMAS FOTOCELDAS QUE REF_CELLS**
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_pv_glasses.py` líneas 37-44
    - `U_ADD_K2` (línea 37): Incertidumbre aditiva (W/m², k=2) = 5.0 W/m² ✅
    - `U_SCALE_K2` (línea 38): Incertidumbre de escala (adimensional, k=2) = 0.025 (2.5%) ✅

#### Correlaciones
- [ ] **ρ(R_FC1, R_FC2)**: Correlación entre las dos celdas de referencia
  - Valor estimado: ¿?
  - Justificación: ¿Mismo tipo de celda, condiciones similares?
  
- [ ] **ρ(R_FCi, R_FCj)**: Correlación entre celdas sucias (i, j = 3, 4, 5)
  - Valor estimado: ¿?
  - Justificación: ¿Mismo tipo de celda, condiciones similares?

- [ ] **ρ(R_FCi, REF)**: Correlación entre celda sucia y referencia promedio
  - Valor estimado: ¿?
  - Justificación: ¿Mismo tipo de celda, condiciones similares?

#### Información del Equipo
- [x] Modelo de fotoceldas: **Si-V-10TC-T** (mismas que ref_cells) ✅
- [x] Especificaciones técnicas: Ya conocidas (ver sección 5 - Celdas de Referencia) ✅
- [ ] Modelo de balanza (solo para referencia, no afecta incertidumbre de SR)

---

## 5. CELDAS DE REFERENCIA (REF CELLS)

### Estado
✅ **YA IMPLEMENTADO** - Ver `analysis/sr_uncertainty_propagation.py`

### Cálculo Principal
- **SR = 100 × S / C**
- Donde S = irradiancia celda sucia (1RC411), C = irradiancia celda limpia (1RC412)

### Incertidumbres Usadas
- **u_add_k2 = 5.0 W/m²** (aditiva, k=2)
- **u_scale_k2 = 0.025** (2.5%, de escala, k=2)
- Fuente: Especificaciones del fabricante (Si-V-10TC-T)

---

## 6. ANÁLISIS IV600

### Estado
✅ **IMPLEMENTADO** - Ver `analysis/sr_uncertainty_iv600.py`

### Cálculo Principal
- **SR_Isc = 100 × Isc_sucio / Isc_referencia**
- **SR_Pmax = 100 × Pmax_sucio / Pmax_referencia**
- Módulos: 1MD434 (sucio) vs 1MD439 (referencia), 1MD440 (sucio) vs 1MD439 (referencia)

### Mediciones Utilizadas
- **Isc_sucio**: Corriente de cortocircuito módulo sucio (A)
- **Isc_referencia**: Corriente de cortocircuito módulo referencia (A)
- **Pmax_sucio**: Potencia máxima módulo sucio (W)
- **Pmax_referencia**: Potencia máxima módulo referencia (W)

### Información Necesaria

#### Incertidumbres de Medición (del Certificado de Calibración)
- [x] **u(Isc)**: Incertidumbre de medición de corriente (A)
  - Tipo: De escala
  - Valor k=2: **U_SCALE_K2 = 0.2%** ✅
  - Fuente: Certificado de calibración IV600 (accuracy: ±0.2%Isc)
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_iv600.py` líneas 40-43
    - `U_ISC_ADD_K2` (línea 40): Incertidumbre aditiva (A, k=2) = 0.0 A ✅
    - `U_ISC_SCALE_K2` (línea 41): Incertidumbre de escala (adimensional, k=2) = 0.002 (0.2%) ✅

- [x] **u(Pmax)**: Incertidumbre de medición de potencia (W)
  - Tipo: Aditiva y de escala
  - Valor k=2: **U_ADD_K2 = 6.0 W**, **U_SCALE_K2 = 1.0%** ✅
  - Fuente: Certificado de calibración IV600 (accuracy: ±1.0%lectura + 6 dgt)
  - Nota: 6 dgt = 6 dígitos × resolución (1 W para rango 50-9999 W)
  - **📍 Ubicación en código**: `analysis/sr_uncertainty_iv600.py` líneas 45-48
    - `U_PMAX_ADD_K2` (línea 45): Incertidumbre aditiva (W, k=2) = 6.0 W ✅
    - `U_PMAX_SCALE_K2` (línea 46): Incertidumbre de escala (adimensional, k=2) = 0.01 (1.0%) ✅

- [ ] **u(Voc)**: Incertidumbre de medición de voltaje (V)
  - Tipo: De escala
  - Valor k=2: **U_SCALE_K2 = 0.2%** (según certificado: ±0.2%Voc)
  - **⚠️ NOTA**: Actualmente no se usa en el código (solo se usa Isc y Pmax)

#### Condiciones del Certificado
- [x] Temperatura: **23°C ± 5°C** ✅
- [x] Humedad relativa: **<80%RH** ✅
- [x] Rango de corriente: **0.20 A a 40.00 A** ✅
- [x] Rango de potencia: **50 W a 9999 W** (y 10k-59.99k W) ✅
- [x] Voltaje mínimo: **VCC > 15V** (para corriente y voltaje), **VCC ≥ 30V** (para potencia) ✅

#### Correlaciones
- [ ] **ρ(Isc_soiled, Isc_reference)**: Correlación entre corrientes
  - Valor estimado: ¿?
  - Justificación: ¿Mismo instrumento de medición?

- [ ] **ρ(Pmax_soiled, Pmax_reference)**: Correlación entre potencias
  - Valor estimado: ¿?
  - Justificación: ¿Mismo instrumento de medición?

#### Información del Equipo
- [x] Modelo del IV tracer: **IV600 (IVCK)** ✅
- [x] Certificado de calibración: Valores incorporados ✅
- [x] Especificaciones técnicas: Según certificado ✅

---

## INFORMACIÓN GENERAL ADICIONAL

### Factores de Cobertura
- [ ] **k_expand**: Factor de cobertura para expandir incertidumbre (default: 2.0 para k=2)
  - ¿Usar k=2 para todos los análisis?
  - ¿Algún análisis requiere k diferente?

### Correlaciones Generales
- [ ] **Estrategia para estimar correlaciones**:
  - ¿Usar correlación empírica de datos?
  - ¿Asumir correlación = 0 (independencia)?
  - ¿Usar correlación = 1 (mismo instrumento)?

### Umbrales y Filtros
- [ ] ¿Mantener los mismos umbrales de filtrado que ya existen?
- [ ] ¿Agregar filtros basados en incertidumbre?

### Formato de Salida
- [ ] ¿Mismo formato que ref_cells (CSV con incertidumbre, gráficos)?
- [ ] ¿Guardar en carpetas separadas por análisis dentro de "propagacion de errores"?

---

## PRIORIDADES SUGERIDAS

1. **Alta Prioridad**:
   - Soiling Kit (análisis fundamental)
   - PVStand (análisis principal de banco de pruebas)

2. **Media Prioridad**:
   - DustIQ (sensor directo, más simple)
   - PV Glasses (similar a PVStand)

3. **Baja Prioridad**:
   - IV600 (requiere revisión del análisis específico)

---

## NOTAS

- Para cada análisis, se necesita al menos:
  1. **Incertidumbres del fabricante** (manual técnico, certificado de calibración)
  2. **Coeficientes de corrección** y sus incertidumbres
  3. **Información sobre correlaciones** entre mediciones

- Si no se tiene información exacta, se pueden usar valores estimados razonables basados en:
  - Especificaciones típicas de equipos similares
  - Estándares de la industria
  - Análisis de sensibilidad para determinar qué incertidumbres son más críticas

---

## 📍 GUÍA RÁPIDA: Dónde Reemplazar Valores en el Código

### Soiling Kit
**Archivo**: `ATAMOSTEC/analysis/sr_uncertainty_soiling_kit.py`

| Variable | Línea | Valor Actual (Estimado) | Qué Reemplazar |
|----------|-------|------------------------|----------------|
| `U_ISC_ADD_K2` | 39 | `0.01` A | Incertidumbre aditiva de corriente (k=2) del amperímetro |
| `U_ISC_SCALE_K2` | 40 | `0.01` (1%) | Incertidumbre de escala de corriente (k=2) del amperímetro |
| `U_TEMP_ADD_K2` | 48 | `1.0` °C | Incertidumbre aditiva de temperatura (k=2) del sensor |
| `U_ALPHA_ISC` | 56 | `0.0001` 1/°C | Incertidumbre del coeficiente α_isc (k=1) |

### PVStand
**Archivo**: `ATAMOSTEC/analysis/sr_uncertainty_pvstand.py`

| Variable | Línea | Valor Actual | Estado |
|----------|-------|--------------|--------|
| `U_ISC_ADD_K2` | 38 | `0.0` A | ✅ Sin componente aditiva |
| `U_ISC_SCALE_K2` | 39 | `0.002` (0.2%) | ✅ Valor del fabricante (accuracy: 0.2%) |
| `U_PMAX_ADD_K2` | 41 | `0.0` W | ✅ Sin componente aditiva |
| `U_PMAX_SCALE_K2` | 42 | `0.004` (0.4%) | ✅ Valor del fabricante (accuracy: 0.4% MPP) |
| `U_TEMP_ADD_K2` | 52 | `1.0` °C | ⚠️ Valor estimado (necesita certificado) |
| `U_ALPHA_ISC` | 56 | `0.0001` 1/°C | ⚠️ Valor estimado (necesita datasheet) |
| `U_BETA_PMAX` | 57 | `0.0001` 1/°C | ⚠️ Valor estimado (necesita datasheet) |

### DustIQ
**Archivo**: `ATAMOSTEC/analysis/sr_uncertainty_dustiq.py`

✅ **Ya tiene valores reales del fabricante** (accuracy: ±0.1% of reading ±1%):
- `U_SR_ADD_K2` (línea 37): `0.1` % ✅
- `U_SR_SCALE_K2` (línea 38): `0.01` (1%) ✅

### PV Glasses
**Archivo**: `ATAMOSTEC/analysis/sr_uncertainty_pv_glasses.py`

✅ **Ya tiene valores reales del fabricante** (Si-V-10TC-T, mismas fotoceldas que ref_cells):
- `U_ADD_K2` (línea 37): `5.0` W/m² ✅
- `U_SCALE_K2` (línea 38): `0.025` (2.5%) ✅

### IV600
**Archivo**: `ATAMOSTEC/analysis/sr_uncertainty_iv600.py`

✅ **Ya tiene valores reales del certificado de calibración** (IV600):
- `U_ISC_ADD_K2` (línea 40): `0.0` A ✅ (sin componente aditiva)
- `U_ISC_SCALE_K2` (línea 41): `0.002` (0.2%) ✅
- `U_PMAX_ADD_K2` (línea 45): `6.0` W ✅ (6 dígitos × 1 W)
- `U_PMAX_SCALE_K2` (línea 46): `0.01` (1.0%) ✅

### Celdas de Referencia
**Archivo**: `ATAMOSTEC/analysis/sr_uncertainty_propagation.py`

✅ **Ya tiene valores reales del fabricante** (Si-V-10TC-T):
- `U_ADD_K2` (línea 37): `5.0` W/m² ✅
- `U_SCALE_K2` (línea 38): `0.025` (2.5%) ✅

---

## 📋 Formato de los Valores en Certificados

Los certificados típicamente reportan incertidumbres expandidas a **k=2** (95% confianza).

**Ejemplo de certificado:**
```
Incertidumbre expandida (k=2):
- Corriente: U = 0.02 A (aditiva) + 0.5% (de escala)
- Potencia: U = 2.0 W (aditiva) + 1.0% (de escala)
- Temperatura: U = 0.5 °C (aditiva)
```

**Cómo ingresar:**
- **Aditiva (k=2)**: Si certificado dice `U = 0.02 A (k=2)` → `U_ISC_ADD_K2 = 0.02`
- **Escala (k=2)**: Si certificado dice `U = 0.5% (k=2)` → `U_ISC_SCALE_K2 = 0.005` (0.5% = 0.005 en fracción)

**⚠️ IMPORTANTE**: El código automáticamente convierte a k=1 dividiendo por 2.0, no necesitas hacerlo manualmente.


