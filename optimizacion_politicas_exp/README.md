# *Experimento 2:* Optimización de Políticas de Intervención No Farmacéutica (NPI) contra el Andes Orthohantavirus (ANDV)
## Reinforcement Learning ccon PPO

---

## 📋 Descripción del Proyecto

Este proyecto implementa un **Modelo Basado en Agentes (ABM) vectorizado** que simula la propagación epidemiológica del **Andes Orthohantavirus (ANDV)**, el principal agente causal del Síndrome Cardiopulmonar por Hantavirus (SCPH) en América del Sur y el único hantavirus con capacidad documentada de transmisión interhumana.

La optimización de políticas de **intervención no farmacéutica** se realiza mediante **Proximal Policy Optimization (PPO)**, un algoritmo de aprendizaje por refuerzo que converge a políticas de control epidemiológico efectivas frente a brotes con alta letalidad y dinámicas mixtas (zoonóticas e interhumanas).

### Características Clave

✓ **Modelo Epidemiológico Sofisticado**

* Dinámicas SEIR: Susceptibles → Latentes → Sintomáticos → Recuperados/Fallecidos
* Transmisión zoonótica (focos rurales / exposición al colilargo *Oligoryzomys longicaudatus*) e interhumana (contacto estrecho)
* Período de incubación estocástico: 9-33 días (característico de ANDV)
* Retraso en detección de casos: 1-3 días (confirmación RT-PCR / serología)
* Tasa de letalidad: 35% (rango histórico documentado del SCPH: 25-40%)

✓ **Arquitectura Vectorizada con NumPy/SciPy**

* Sin bucles iterativos en Python (compilados en C)
* Cálculo de distancias matricial: O(N) con `scipy.spatial.distance.cdist`
* Tiempo por paso simulado: < 0.1 ms (N=100)
* Entrenamiento completo: 2-5 minutos en GPU RTX 3060

✓ **Espacios Continuos de Acción**

* Mascarillas N95 / Respiradores de alta eficiencia (eficacia 60% frente a aerosoles)
* Confinamiento residencial y restricción de movilidad rural
* Control de aforos en zonas públicas y centros comunitarios
* Desratización y control de reservorios ambientales (con recuperación ecológica)
* Aislamiento estricto de casos sintomáticos / sospechosos

✓ **Función de Recompensa Multiobjetivo**

* α=0.20: Impacto económico
* β=0.65: Salud pública (priorizado por alta letalidad del ANDV)
* γ=0.15: Sostenibilidad social
* Penalización de -10.0 por cada fallecimiento

✓ **Algoritmo PPO Robusto**

* Arquitectura Actor-Crítico independiente
* Clipping de gradiente: ε=0.2
* Ventaja Generalizada (GAE): λ=0.95, γ=0.99
* Bonificación de entropía para exploración

---

## 🏗️ Arquitectura del Código

```
.
├── epidemic_env.py          # Entorno Gymnasium vectorizado para ANDV
├── ppo_trainer.py           # Algoritmo PPO + redes neuronales
├── main.py                  # Script principal (entrenamiento/inferencia)
├── analysis.py              # Análisis y visualización
├── requirements.txt         # Dependencias Python
└── README.md                # Esta documentación

```

### Archivos Principales

#### 1. `epidemic_env.py` - Simulación Epidemiológica

**Clase `EpidemicEnvironment(gym.Env)**`

Implementa el entorno Gymnasium con:

* **Geometría Espacial**: Plano continuo 2D [0,10] × [0,10]
* 100 agentes humanos distribuidos en ~33 casas
* 5 focos zoonóticos rurales (hábitat del ratón colilargo)
* 5 zonas seguras urbanas


* **Dinámicas de Transmisión**:
```
P(Infección) = 1 - ∏(1 - λ_zoo) * ∏(1 - λ_hum)

λ_zoo = β_zoo * hazard * (1 - 0.6*a_masks) * I(distancia < r_zoo)
λ_hum = β_hum * (1 - 0.6*a_masks) * I(distancia < r_hum)

```


* **Métodos Clave**:
* `step(action)`: Ejecuta un paso temporal (día)
* `_update_mobility()`: Movimiento pendular vectorizado
* `_calculate_infections()`: Cálculo de contagios zoonóticos e interhumanos
* `_update_health_transitions()`: Progresión epidemiológica del ANDV
* `_get_observation()`: Estado agregado normalizado
* `_calculate_reward()`: Función multiobjetivo



#### 2. `ppo_trainer.py` - Aprendizaje por Refuerzo

**Clase `ActorCriticNetwork(nn.Module)**`

Red neuronal con arquitectura separada:

```
Input [5] → Capas compartidas → [64] → [64]
                                  ↓
                        ┌─────────┴────────┐
                        ↓                  ↓
                    Actor μ,σ          Critic V
                   Sigmoid [5]        Linear [1]

```

* **Actor**: Genera μ y σ de distribución Gaussiana diagonal
* **Crítico**: Estima valor V(s_t) para GAE

**Clase `PPOTrainer**`

Implementación de PPO con:

* **Colección de Experiencias**: Rollouts de 2048 pasos
* **Cálculo de Advantages**: GAE con λ=0.95
* **Actualización de Política**:
```
L^CLIP(θ) = min(r_t * Â_t, clip(r_t, 1±ε) * Â_t)
L^VF(φ) = (V(s_t) - V_target)²
L_total = L^CLIP + 0.5*L^VF + 0.01*Entropy

```



#### 3. `main.py` - Interfaz Principal

Proporciona dos modos:

**Modo Entrenamiento**:

```bash
python main.py --mode train --iterations 50

```

* 50 iteraciones × 2048 pasos = 102,400 pasos de entorno
* Guarda modelo en `/mnt/user-data/outputs/epidemic_policy.pt`
* Tiempo estimado: 2-5 minutos (GPU RTX 3060)

**Modo Prueba**:

```bash
python main.py --mode test --checkpoint model.pt --episodes_test 5

```

* Evalúa política entrenada sin exploración
* Muestra recompensas y dinámicas de contención

#### 4. `analysis.py` - Análisis de Resultados

**Clase `EpidemicAnalyzer**`

Genera análisis completos con visualizaciones:

```python
analyzer = EpidemicAnalyzer(env, trainer)

# Simular con política entrenada
trajectories, actions = analyzer.simulate_with_policy(num_episodes=3)

# Gráficos
analyzer.plot_epidemic_curves(trajectories)      # Curvas SEIR para ANDV
analyzer.plot_control_policies(actions)          # Políticas de control
analyzer.plot_training_progress(trainer)         # Progreso del entrenamiento
analyzer.generate_report(trajectories, actions)  # Reporte estadístico

```

---

## 🚀 Cómo Usar

### Instalación de Dependencias

```bash
pip install -r requirements.txt

```

Dependencias principales:

* **NumPy** (1.24+): Operaciones vectorizadas
* **SciPy** (1.11+): Cálculo de distancias
* **Gymnasium** (0.29+): Interfaz RL
* **PyTorch** (2.0+): Redes neuronales
* **Matplotlib** (3.7+): Visualización

### Entrenamiento Básico

```python
from epidemic_env import EpidemicEnvironment
from ppo_trainer import PPOTrainer

# Crear entorno
env = EpidemicEnvironment(
    population_size=500,
    simulation_days=150,
    num_houses=165,
    num_zoonotic_foci=25,
    num_safe_zones=25
)

# Crear entrenador
trainer = PPOTrainer(
    env=env,
    learning_rate=3e-4,
    gamma=0.99,
    lambda_gae=0.95,
    clip_ratio=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01,
    batch_size=64,
    epochs_per_update=10,
    device='cuda'  # o 'cpu'
)

# Entrenar
trainer.train(num_iterations=50)

# Guardar
trainer.save_policy('epidemic_policy.pt')

# Probar
rewards, data = trainer.test_policy(num_episodes=5)

```

### Análisis y Visualización

```python
from analysis import EpidemicAnalyzer

analyzer = EpidemicAnalyzer(env, trainer)

# Simular con política
trajectories, actions = analyzer.simulate_with_policy(num_episodes=3)

# Generar gráficos y reporte
analyzer.plot_epidemic_curves(trajectories, save_path='curves.png')
analyzer.plot_control_policies(actions, save_path='policies.png')
analyzer.plot_training_progress(trainer, save_path='progress.png')
analyzer.generate_report(trajectories, actions, save_dir='./results')

```

---

## 📊 Parámetros Epidemiológicos (Basados en ANDV)

Todos los valores reflejan la literatura clínica y epidemiológica documentada para el Andes Orthohantavirus:

| Parámetro | Valor | Justificación Clínica / Epidemiológica |
| --- | --- | --- |
| **Tamaño poblacional (N)** | 500 | Micropoblación rural/periurbana expuesta |
| **Duración simulación (T)** | 150 días | Permite capturar múltiples ciclos de incubación |
| **Período de latencia** | 9-33 días | Período de incubación prolongado del ANDV (mediana 18-22 días) |
| **Período infeccioso** | 5-10 días | Fase cardiopulmonar agudizada y eliminación viral |
| **Retraso detección** | 1-3 días | Demora de laboratorio para confirmación ELISA/RT-PCR |
| **CFR (tasa letalidad)** | 35% | Consistente con la letalidad histórica del SCPH por ANDV (25-40%) |
| **β_zoonótico** | 65% | Exposición a aerosoles de deyecciones de *O. longicaudatus* |
| **β_interhumano** | 5% | Probabilidad por contacto estrecho o intrafamiliar |
| **Eficacia mascarilla** | 60% | Protección con respiradores N95 frente a aerosoles |
| **Radio zoonótico** | 0.15 | Distancia de riesgo en focos de roedores/galpones |
| **Radio humano** | 0.12 | Proximidad física en espacio cerrado |

---

## 🎯 Dinámicas Aprendidas

El algoritmo PPO converge a políticas optimizadas para mitigar brotes de ANDV:

### 1. **Mitigación Preventiva y Anticipación**

* Intervenciones **tempranas** en la fase subclínica para compensar el largo período de incubación (18-22 días).
* Evita la trampa de la "histéresis epidemiológica", donde las medidas tardías no evitan el colapso debido a contagios silenciosos ya producidos.

### 2. **Focalización de Recursos**

* Intervención estricta en focos rurales/zoonóticos combinada con desratización periódica.
* Restricción focalizada de movilidad y aislamiento temprano de contactos estrechos intrafamiliares para cortar la transmisión interhumana.

### 3. **Equilibrio Multiobjetivo**

* Priorización severa de la salud pública debido a la penalización por alta letalidad (CFR 35%).
* Minimización del impacto socioeconómico al evitar confinamientos totales prolongados una vez contenido el foco.

---

## 🔧 Hiperparámetros Ajustables

### Parámetros del Entorno

```python
EpidemicEnvironment(
    population_size=500,         # Tamaño de la población
    simulation_days=150,         # Horizonte temporal del experimento
    num_houses=165,              # Cantidad total de casas
    num_zoonotic_foci=25,        # Puntos de interés con probabilidad de contagio zoonótica 
    num_safe_zones=25            # Puntos de interés SIN probabilidad de contagio zoonótica 
)

```

### Parámetros PPO

```python
PPOTrainer(
    learning_rate=3e-4,         # Tasa de aprendizaje (α)
    gamma=0.99,                 # Factor de descuento
    lambda_gae=0.95,            # Parámetro GAE (λ)
    clip_ratio=0.2,             # Rango de clipping (ε)
    value_coeff=0.5,            # Coeficiente de pérdida de valor
    entropy_coeff=0.01,         # Coeficiente de entropía para exploración
    batch_size=64,              # Tamaño de mini-lote
    epochs_per_update=10        # Épocas de optimización por ciclo
)

```

---

## 📝 Estructura de Salida

El entrenamiento genera:

```
/mnt/user-data/outputs/
├── epidemic_policy.pt              # Pesos de la red entrenada
├── epidemic_policy_results.npy     # Datos de prueba (rewards, trayectorias)
├── epidemic_curves.png             # Curvas SEIR para Andes Orthohantavirus
├── control_policies.png            # Visualización de las políticas aplicadas
├── training_progress.png           # Evolución de las métricas de PPO
└── epidemic_analysis_report.txt    # Reporte estadístico detallado

```
---

## 🎓 Referencias Científicas

1. **Andes Orthohantavirus**: Padula et al. *Transmission of Hantavirus Pulmonary Syndrome in Argentina*. N Engl J Med.
2. **Modelos Epidemiológicos**: Kermack-McKendrick SEIR para patógenos zoonóticos.
3. **Aprendizaje por Refuerzo**: Schulman et al. (2017), *Proximal Policy Optimization Algorithms*.
4. **Simulación Basada en Agentes**: Métodos de computación matricial vectorizada NumPy/SciPy.

---

## 📄 Licencia y Uso Académico

Este proyecto es de **uso estrictamente académico y educativo**.

* ✓ Uso permitido en cursos de Machine Learning, Aprendizaje por Refuerzo y Epidemiología Computacional.
* ✓ Modificación libre para investigación académica.
* ✗ Uso comercial no autorizado.
* ✗ No constituye una herramienta de diagnóstico médico ni sustituye las directrices de los organismos oficiales de salud pública.
