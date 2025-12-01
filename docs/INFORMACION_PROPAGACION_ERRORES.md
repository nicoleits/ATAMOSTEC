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

- [ ] **u(T)**: Incertidumbre de medición de temperatura (°C)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del sensor de temperatura

#### Coeficientes de Corrección
- [ ] **α_isc**: Coeficiente de temperatura de Isc (%/°C o 1/°C)
  - Valor actual: ¿?
  - Fuente: Datasheet del módulo o calibración
  - Incertidumbre de α_isc: ¿?

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

### Cálculo Principal
- **SR_C11_Avg**: Valor directo del sensor (ya es SR en %)

### Mediciones Utilizadas
- **SR_C11_Avg**: Soiling Ratio promedio del canal C11 (%)
- **SR_C12_Avg**: Soiling Ratio promedio del canal C12 (%)

### Información Necesaria

#### Incertidumbres del Sensor
- [ ] **u(SR_DustIQ)**: Incertidumbre del sensor DustIQ (%)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Manual técnico o certificado de calibración del DustIQ

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
- [ ] **u(Isc)**: Incertidumbre de medición de corriente (A)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del equipo IV tracer

- [ ] **u(Pmax)**: Incertidumbre de medición de potencia (W)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del equipo IV tracer
  - Nota: Puede depender de u(Isc) y u(Vmax)

- [ ] **u(Vmax)**: Incertidumbre de medición de voltaje máximo (V)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del equipo IV tracer

- [ ] **u(T)**: Incertidumbre de medición de temperatura (°C)
  - Tipo: Aditiva y/o de escala
  - Valor k=2: ¿?
  - Fuente: Especificaciones del sensor de temperatura

#### Coeficientes de Corrección
- [ ] **α_isc**: Coeficiente de temperatura de Isc (%/°C o 1/°C)
  - Valor actual: ¿?
  - Fuente: Datasheet del módulo
  - Incertidumbre de α_isc: ¿?

- [ ] **β_pmax**: Coeficiente de temperatura de Pmax (%/°C o 1/°C)
  - Valor actual: ¿?
  - Fuente: Datasheet del módulo
  - Incertidumbre de β_pmax: ¿?

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

### Cálculo Principal
- Similar a PVStand: **SR = 100 × Isc_soiled / Isc_reference** o **SR = 100 × Pmax_soiled / Pmax_reference**
- Con corrección de temperatura

### Mediciones Utilizadas
- Similar a PVStand: Isc, Pmax, temperaturas

### Información Necesaria
- [ ] **Misma información que PVStand** (ver sección 3)
- [ ] **Diferencias específicas de PV Glasses**:
  - [ ] Tipo de vidrio usado
  - [ ] Transmitancia del vidrio (si afecta la incertidumbre)
  - [ ] Cualquier corrección adicional específica de vidrios

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

### Cálculo Principal
- Análisis de curvas IV
- Posiblemente cálculo de SR similar a PVStand

### Información Necesaria
- [ ] **Revisar análisis específico** para determinar cálculos exactos
- [ ] **u(I)**: Incertidumbre de corriente del IV600
- [ ] **u(V)**: Incertidumbre de voltaje del IV600
- [ ] **u(P)**: Incertidumbre de potencia (derivada de I y V)
- [ ] Modelo del equipo IV600
- [ ] Certificado de calibración

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


