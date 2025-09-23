# Análisis PV Glasses con Cuantil 25 (Q25)

## 📊 Descripción

El análisis PV Glasses Q25 es una implementación alternativa y más robusta del análisis tradicional de PV Glasses. Utiliza el **cuantil 25 (Q25)** en lugar de promedios, lo que lo hace significativamente más resistente a outliers y datos anómalos.

## 🎯 Problema Resuelto

### Problema Original
- El análisis tradicional de PV Glasses perdía datos del **2025-08-07** debido a que eran considerados outliers por el filtro IQR
- Los valores de R_FC5_Avg (636.8 - 691.1) estaban por debajo del límite inferior IQR (691.5)
- Esto resultaba en pérdida de información valiosa para el análisis

### Solución Q25
- **No aplica filtro IQR** por defecto (más permisivo)
- Usa **cuantil 25** en lugar de promedios (más robusto)
- **Conserva datos anómalos** que pueden representar condiciones reales (días muy sucios)
- Proporciona una **visión más conservadora** del rendimiento

## 🚀 Cómo Usar

### Opción 1: Script Independiente
```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar análisis Q25
python run_pv_glasses_q25.py

# Ver ayuda
python run_pv_glasses_q25.py --help
```

### Opción 2: Menú Principal
```bash
python main.py
# Seleccionar opción 14: "PV Glasses Q25 (Cuantil 25)"
```

## 📁 Archivos de Entrada

- `datos/raw_pv_glasses_data.csv` - Datos raw de PV Glasses
- `datos/20241114 Calendario toma de muestras soiling.xlsx` - Calendario de muestras

## 📄 Archivos de Salida

### CSV Procesados
- `datos_procesados_analisis_integrado_py/pv_glasses_q25/datos_q25_diarios.csv`
- `datos_procesados_analisis_integrado_py/pv_glasses_q25/seleccion_irradiancia_q25.csv`
- `datos_procesados_analisis_integrado_py/pv_glasses_q25/soiling_ratios_q25.csv`

### Gráficos
- `graficos_analisis_integrado_py/pv_glasses_q25/SR_Q25_Periodo_*_MasasCorregidas.png`
- `graficos_analisis_integrado_py/pv_glasses_q25/SR_Q25_por_Periodo_Barras.png`

## 🔄 Comparación: Tradicional vs Q25

| Aspecto | Análisis Tradicional | Análisis Q25 |
|---------|---------------------|--------------|
| **Estadística** | Promedio | Cuantil 25 |
| **Filtro IQR** | Sí (restrictivo) | No (permisivo) |
| **Outliers** | Elimina datos anómalos | Conserva datos anómalos |
| **Robustez** | Sensible a outliers | Resistente a outliers |
| **Interpretación** | Rendimiento promedio | Rendimiento conservador |
| **Datos 2025-08-07** | ❌ Perdidos | ✅ Conservados |

## 📊 Interpretación de Resultados

### Cuantil 25 (Q25)
- Representa el valor por debajo del cual está el **25% de los datos**
- Es más **conservador** que el promedio
- Útil para análisis de **peor caso** (worst-case scenario)
- Menos afectado por **valores extremos altos**

### Casos de Uso
- **Garantías de rendimiento**: Q25 proporciona estimaciones conservadoras
- **Análisis de riesgo**: Identificar el rendimiento en condiciones adversas
- **Datos con outliers**: Cuando hay muchos valores anómalos
- **Validación cruzada**: Comparar con análisis tradicional

## ⚙️ Configuración

### Parámetros Principales
```python
usar_mediodia_solar_real = True        # Filtro de mediodía solar
intervalo_minutos_mediodia = 60        # Ventana ±60 minutos
filtrar_outliers_iqr = False          # No usar IQR (recomendado)
umbral_irradiancia_ref = 300          # Filtro REF >= 300 W/m²
```

### Personalización
El script puede modificarse fácilmente para:
- Cambiar el cuantil (ej: Q10, Q50, Q75)
- Ajustar filtros temporales
- Modificar umbrales de irradiancia
- Agregar nuevas métricas estadísticas

## 📈 Ventajas del Análisis Q25

1. **Robustez Estadística**
   - Menos sensible a outliers
   - Estimaciones más estables
   - Mejor para datos con alta variabilidad

2. **Conservación de Datos**
   - No pierde información valiosa
   - Incluye condiciones extremas
   - Mayor representatividad temporal

3. **Análisis Complementario**
   - Se puede usar junto al análisis tradicional
   - Proporciona diferentes perspectivas
   - Validación cruzada de resultados

4. **Aplicabilidad Práctica**
   - Útil para garantías de performance
   - Análisis de riesgo operacional
   - Planificación conservadora

## 🔧 Mantenimiento

### Logs
- Los logs se guardan en `pv_glasses_q25.log`
- Nivel de detalle: INFO
- Incluye timestamps y trazabilidad de errores

### Actualización
- El código está modularizado para fácil mantenimiento
- Separación clara entre procesamiento y visualización
- Documentación inline extensiva

## 🤝 Contribución

Para mejorar el análisis Q25:
1. Documentar casos de uso específicos
2. Agregar nuevas métricas estadísticas robustas
3. Implementar visualizaciones comparativas
4. Optimizar rendimiento para datasets grandes

## 📞 Soporte

Para problemas o preguntas sobre el análisis Q25:
- Revisar logs en `pv_glasses_q25.log`
- Verificar archivos de entrada requeridos
- Comparar resultados con análisis tradicional
- Documentar diferencias significativas encontradas
