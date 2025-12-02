# Guía de Búsqueda de Información para PVStand

Esta guía te ayuda a identificar exactamente qué información buscar y dónde encontrarla para completar la propagación de errores de PVStand.

---

## 📋 Checklist de Información a Buscar

### 1. EQUIPO IV TRACER (Medición de Isc y Pmax)

#### Información Básica
- [ ] **Modelo del IV tracer**: ¿Qué marca y modelo se usa?
- [ ] **Número de serie**: Para identificar el certificado correcto
- [ ] **Ubicación del equipo**: ¿Dónde está instalado?

#### Certificado de Calibración del IV Tracer
Buscar en:
- Archivos físicos de certificados de calibración
- Base de datos de calibraciones del laboratorio
- Documentación del proyecto

**Qué buscar en el certificado:**
- [ ] **Incertidumbre de corriente (Isc)** a k=2:
  - Componente aditiva: ¿? A (k=2)
  - Componente de escala: ¿? % (k=2)
  - Ejemplo: "U = 0.02 A (aditiva) + 0.5% (de escala) a k=2"

- [ ] **Incertidumbre de potencia (Pmax)** a k=2:
  - Componente aditiva: ¿? W (k=2)
  - Componente de escala: ¿? % (k=2)
  - Ejemplo: "U = 2.0 W (aditiva) + 1.0% (de escala) a k=2"

- [ ] **Fecha de calibración**: ¿Cuándo fue calibrado?
- [ ] **Vigencia del certificado**: ¿Está vigente?

#### Manual Técnico del IV Tracer
Si no hay certificado, buscar en:
- Manual del usuario
- Especificaciones técnicas
- Datasheet del fabricante

**Qué buscar:**
- [ ] Precisión de medición de corriente
- [ ] Precisión de medición de potencia
- [ ] Resolución
- [ ] Rango de medición

---

### 2. SENSOR DE TEMPERATURA

#### Información Básica
- [ ] **Modelo del sensor**: ¿Qué sensor se usa? (ej: PT100, termopar, etc.)
- [ ] **Número de serie**: Si aplica
- [ ] **Ubicación**: ¿Dónde está instalado? (módulo sucio vs referencia)

#### Certificado de Calibración del Sensor de Temperatura
Buscar en:
- Archivos físicos de certificados
- Base de datos de calibraciones

**Qué buscar en el certificado:**
- [ ] **Incertidumbre de temperatura** a k=2:
  - Componente aditiva: ¿? °C (k=2)
  - Componente de escala (si aplica): ¿? % (k=2)
  - Ejemplo: "U = 0.5 °C (aditiva) a k=2"

- [ ] **Rango de calibración**: ¿En qué rango de temperatura fue calibrado?
- [ ] **Fecha de calibración**: ¿Cuándo fue calibrado?

#### Manual Técnico del Sensor
Si no hay certificado:
- [ ] Precisión del sensor
- [ ] Resolución
- [ ] Rango de operación

**Nota**: Según el código, se usan sensores `1TE416(C)` (módulo sucio) y `1TE418(C)` (módulo referencia). Verificar si son el mismo modelo o diferentes.

---

### 3. MÓDULO FOTOVOLTAICO

#### Información Básica
- [ ] **Modelo del módulo**: ¿Qué módulo se usa? (ej: PERC, monocristalino, etc.)
- [ ] **Fabricante**: ¿Quién lo fabricó?
- [ ] **Módulos usados**: 
  - Módulo sucio: `perc1fixed` (según settings)
  - Módulo referencia: `perc2fixed` (según settings)

#### Datasheet del Módulo
Buscar en:
- Archivos del proyecto
- Sitio web del fabricante
- Documentación técnica

**Qué buscar en el datasheet:**
- [ ] **Coeficiente de temperatura de Isc (α_isc)**:
  - Valor: ¿? %/°C o 1/°C
  - Valor actual en código: `-0.0004` o `0.0004` (según settings)
  - Incertidumbre de α_isc: ¿? (típicamente 10-20% del valor)

- [ ] **Coeficiente de temperatura de Pmax (β_pmax)**:
  - Valor: ¿? %/°C o 1/°C
  - Valor actual en código: `+0.0037` o `-0.0037` (según settings)
  - Incertidumbre de β_pmax: ¿? (típicamente 10-20% del valor)

- [ ] **Temperatura de referencia (T_ref)**:
  - Valor: Típicamente 25°C (STC)
  - Incertidumbre: ¿? (generalmente despreciable)

---

## 📍 Dónde Buscar la Información

### 1. Archivos Físicos
- [ ] Carpeta de certificados de calibración
- [ ] Archivo de documentación del proyecto
- [ ] Manuales de equipos guardados

### 2. Base de Datos / Sistema de Gestión
- [ ] Sistema de gestión de calibraciones del laboratorio
- [ ] Base de datos de equipos
- [ ] Sistema de trazabilidad

### 3. Documentación del Proyecto
- [ ] Informes técnicos
- [ ] Documentación de instalación
- [ ] Especificaciones de compra

### 4. Fabricantes
- [ ] Sitio web del fabricante del IV tracer
- [ ] Sitio web del fabricante del sensor de temperatura
- [ ] Sitio web del fabricante del módulo fotovoltaico

### 5. Contactos
- [ ] Responsable de calibraciones del laboratorio
- [ ] Técnico que instaló los equipos
- [ ] Proveedor de los equipos

---

## 🔍 Preguntas Clave para Identificar el Equipo

### Para el IV Tracer:
1. ¿Qué marca/modelo de IV tracer se usa en PVStand?
2. ¿Hay un certificado de calibración reciente?
3. ¿Dónde está guardada la documentación del equipo?

### Para el Sensor de Temperatura:
1. ¿Qué tipo de sensor se usa? (PT100, termopar, etc.)
2. ¿Los sensores `1TE416(C)` y `1TE418(C)` son del mismo modelo?
3. ¿Hay certificados de calibración para estos sensores?

### Para el Módulo:
1. ¿Qué modelo de módulo fotovoltaico se usa?
2. ¿Tienes el datasheet del módulo?
3. ¿Los módulos sucio y referencia son del mismo modelo?

---

## 📝 Formato de la Información que Necesitas

Una vez que encuentres la información, necesitarás estos valores específicos:

### Del Certificado de Calibración del IV Tracer:

**Ejemplo de formato típico:**
```
Incertidumbre expandida (k=2, 95% confianza):

Corriente (Isc):
- Aditiva: U = 0.015 A (k=2)
- De escala: U = 0.3% of reading (k=2)

Potencia (Pmax):
- Aditiva: U = 1.5 W (k=2)
- De escala: U = 0.5% of reading (k=2)
```

**Valores a extraer:**
- `U_ISC_ADD_K2 = 0.015` A
- `U_ISC_SCALE_K2 = 0.003` (0.3% = 0.003 en fracción)
- `U_PMAX_ADD_K2 = 1.5` W
- `U_PMAX_SCALE_K2 = 0.005` (0.5% = 0.005 en fracción)

### Del Certificado de Calibración del Sensor de Temperatura:

**Ejemplo:**
```
Incertidumbre expandida (k=2):
- Temperatura: U = 0.5 °C (aditiva, k=2)
```

**Valor a extraer:**
- `U_TEMP_ADD_K2 = 0.5` °C

### Del Datasheet del Módulo:

**Ejemplo:**
```
Temperature Coefficients:
- α_isc = 0.0004 /°C (0.04% /°C)
- β_pmax = -0.0037 /°C (-0.37% /°C)

Incertidumbre típica: ±10% del valor del coeficiente
```

**Valores a extraer:**
- `U_ALPHA_ISC = 0.00004` 1/°C (10% de 0.0004)
- `U_BETA_PMAX = 0.00037` 1/°C (10% de 0.0037)

---

## ⚠️ Notas Importantes

1. **Si no encuentras certificados de calibración:**
   - Usa las especificaciones del fabricante del manual técnico
   - Los valores típicos de incertidumbre para IV tracers son:
     - Corriente: 0.01-0.05 A (aditiva) + 0.5-1% (escala)
     - Potencia: 1-5 W (aditiva) + 0.5-1% (escala)

2. **Si no encuentras incertidumbre de coeficientes de temperatura:**
   - Usa 10-20% del valor del coeficiente como estimación razonable
   - Ejemplo: Si α_isc = 0.0004, entonces u(α) ≈ 0.00004-0.00008

3. **Vigencia de certificados:**
   - Verifica que los certificados estén vigentes
   - Si están vencidos, busca certificados más recientes o usa valores conservadores

4. **Múltiples sensores:**
   - Si hay sensores diferentes para módulo sucio y referencia, necesitarás certificados para ambos
   - Si son del mismo modelo, puedes usar el mismo certificado

---

## 📞 Contactos Útiles

Si necesitas ayuda para encontrar la información:

1. **Responsable de calibraciones**: ¿Quién gestiona las calibraciones?
2. **Técnico del proyecto**: ¿Quién instaló/configuró los equipos?
3. **Proveedor de equipos**: ¿Quién vendió/suministró los equipos?

---

## ✅ Una Vez que Tengas la Información

Después de encontrar los valores, actualiza el archivo:
**`ATAMOSTEC/analysis/sr_uncertainty_pvstand.py`**

Reemplaza los valores en las líneas indicadas en la tabla de la sección "📍 GUÍA RÁPIDA" del documento `INFORMACION_PROPAGACION_ERRORES.md`.

---

**Última actualización**: 2025-01-XX  
**Archivo relacionado**: `docs/INFORMACION_PROPAGACION_ERRORES.md`

