# GUÍA DE INICIO RÁPIDO

Ejecuta el proyecto completo en 5 pasos simples.

---

## Paso 1: Instalar Dependencias (1 minuto)

```bash
pip install -r requirements.txt
```

Verifica la instalación:
```bash
python -c "import numpy, torch, gymnasium; print('✓ Dependencias OK')"
```

---

## Paso 2: Ejecutar Demostración Rápida (10 minutos)

Para una prueba rápida del proyecto:

```bash
python demo.py
```

Esto ejecutará:
- ✓ Entorno epidemiológico básico
- ✓ Entrenamiento acelerado (5 iteraciones)
- ✓ Evaluación de política
- ✓ Análisis y gráficos

---

## Paso 3: Entrenar Modelo Completo (5 minutos en GPU)

Para entrenar el modelo con 50 iteraciones (100,000+ pasos):

```bash
python main.py --mode train --iterations 50 --save_path model.pt
```

Parámetros opcionales:
- `--iterations N`: Número de iteraciones (default: 50)
- `--save_path PATH`: Donde guardar el modelo (default: /mnt/user-data/outputs/epidemic_policy.pt)

**Salida esperada:**
```
============================================================
ENTRENAMIENTO DE POLÍTICA PPO EPIDEMIOLÓGICA
============================================================
[Iteración 1/50] Recolectando experiencias...
[Iteración 1/50] Actualizando política...
  Recompensa promedio: 2.4563
  Longitud episodio: 58
  Pérdida Actor: 0.012345
  Pérdida Crítico: 0.000234

... (más iteraciones)
```

---

## Paso 4: Probar Modelo Entrenado (2 minutos)

Una vez entrenado, evalúa el modelo:

```bash
python main.py --mode test --checkpoint model.pt --episodes_test 10
```

Muestra:
- Recompensa promedio en 10 episodios
- Dinámicas epidemiológicas
- Desempeño de la política

---

## Paso 5: Análisis Completo (5 minutos)

Genera gráficos, análisis y reportes:

```bash
python analysis.py --model_path model.pt --episodes 5 --output_dir results/
```

Genera archivos:
- `epidemic_curves.png` - Curvas SEIR
- `control_policies.png` - Políticas aprendidas
- `training_progress.png` - Evolución del entrenamiento
- `epidemic_analysis_report.txt` - Reporte estadístico

---

## Comandos Frecuentes

### Solo Entrenar
```bash
python main.py --mode train --iterations 100
```

### Solo Probar
```bash
python main.py --mode test --checkpoint model.pt --episodes_test 5
```

### Análisis sin Entrenar
```bash
python analysis.py --model_path model.pt
```

### Demo Completa
```bash
python demo.py
```

---

## Solución de Problemas

### ❌ "ModuleNotFoundError: No module named 'torch'"

**Solución:**
```bash
pip install torch
```

Para GPU NVIDIA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ❌ "CUDA out of memory"

**Solución:**
```bash
# Reducir tamaño de lote en el código
trainer = PPOTrainer(env=env, batch_size=32)

# O usar CPU
python main.py --mode train --iterations 50  # Automáticamente detecta CPU
```

### ❌ Entrenamiento muy lento en CPU

**Esperado:** CPU toma 3-5 minutos por iteración
**Solución:** Usar GPU si disponible, o reducir iteraciones

### ❌ Gráficos no se guardan

**Verificación:**
```bash
ls -la /mnt/user-data/outputs/
```

Si directorio no existe:
```bash
mkdir -p /mnt/user-data/outputs
```

---

## Archivos Generados

Después de entrenar y analizar:

```
/mnt/user-data/outputs/
├── epidemic_policy.pt              # Modelo entrenado
├── epidemic_policy_results.npy     # Datos de prueba
├── epidemic_curves.png             # Gráfico
├── control_policies.png            # Gráfico
├── training_progress.png           # Gráfico
└── epidemic_analysis_report.txt    # Reporte
```

---

## Configuración Recomendada

### Para Desarrollo/Testing
```bash
python main.py --mode train --iterations 5 --save_path test_model.pt
# Toma: 2-3 minutos
# Resultado: Modelo básico funcional
```

### Para Producción Académica
```bash
python main.py --mode train --iterations 50 --save_path epidemic_policy.pt
# Toma: 20-30 minutos en GPU
# Resultado: Modelo convergente
```

### Para Investigación Avanzada
```bash
python main.py --mode train --iterations 200 --save_path advanced_policy.pt
# Toma: 1-2 horas en GPU
# Resultado: Política altamente optimizada
```

---

## Uso en Jupyter Notebook

```python
from epidemic_env import EpidemicEnvironment
from ppo_trainer import PPOTrainer
from analysis import EpidemicAnalyzer
import torch

# Crear entorno
env = EpidemicEnvironment()

# Crear entrenador
trainer = PPOTrainer(env=env, device='cuda')

# Entrenar
trainer.train(num_iterations=10)

# Evaluar
rewards, data = trainer.test_policy(num_episodes=3)

# Analizar
analyzer = EpidemicAnalyzer(env, trainer)
trajectories, actions = analyzer.simulate_with_policy(num_episodes=3)
analyzer.plot_epidemic_curves(trajectories)
analyzer.generate_report(trajectories, actions)
```

---

## Siguientes Pasos

1. **Leer documentación completa**:
   ```bash
   cat README.md
   ```

2. **Entender formulación matemática**:
   ```bash
   cat MATHEMATICAL_FORMULATION.md
   ```

3. **Experimentar con parámetros**:
   - Editar `epidemic_env.py` para cambiar dinámicas
   - Editar `ppo_trainer.py` para hiperparámetros
   - Editar `main.py` para configuración de entrenamiento

4. **Análisis avanzado**:
   - Extender `analysis.py` con gráficos personalizados
   - Guardar y comparar múltiples políticas
   - Validar con poblaciones escaladas (N > 100)

---

**¡Listo para empezar!**

¿Preguntas? Consulta el README.md o MATHEMATICAL_FORMULATION.md
