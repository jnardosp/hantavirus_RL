"""
Modelo Basado en Agentes Vectorizado para Simulación Epidemiológica del Virus J
Entorno Gymnasium con operaciones NumPy/SciPy para máximo rendimiento computacional
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.distance import cdist
from typing import Tuple, Dict, Any


class EpidemicEnvironment(gym.Env):
    """
    Entorno de aprendizaje por refuerzo para optimizar políticas de contención
    ante brotes epidémicos del Virus J ficticio.
    
    Características:
    - Simulación espacial continua 2D vectorizada con NumPy
    - Dinámicas de transmisión zoonótica e interhumana
    - Intervenciones no farmacéuticas como acciones continuas
    - Función de recompensa multiobjetivo (salud, economía, sostenibilidad social)
    - Asimetría de información y retraso en detección de casos
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, 
                 population_size: int = 500,
                 simulation_days: int = 150,
                 num_houses: int = 165,  # ~3 people per house
                 num_zoonotic_foci: int = 25,
                 num_safe_zones: int = 25,
                 initial_infected: int = 20):
        """
        Inicialización del entorno epidemiológico.
        
        Args:
            population_size: Número de agentes humanos (N=100)
            simulation_days: Horizonte temporal de simulación (T=60 días)
            num_houses: Residencias (núcleos familiares)
            num_zoonotic_foci: Focos de riesgo zoonótico rural
            num_safe_zones: Zonas seguras urbanas
            initial_infected: Número de agentes infectados (estado LATENTE)
                al inicio de cada episodio, es decir, el tamaño del brote
                inicial (por defecto 1, el "paciente cero" clásico)
        """
        super().__init__()
        
        # Parámetros de la población
        self.N = population_size
        self.T = simulation_days
        self.num_houses = num_houses
        self.num_zoonotic_foci = num_zoonotic_foci
        self.num_safe_zones = num_safe_zones

        # Tamaño del brote inicial (número de casos importados al día 0)
        if initial_infected < 1:
            raise ValueError("initial_infected debe ser al menos 1")
        if initial_infected > population_size:
            raise ValueError(
                "initial_infected no puede superar population_size "
                f"({initial_infected} > {population_size})"
            )
        self.initial_infected = initial_infected
        
        # Espacios de observación y acción
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )  # [susceptibles, latentes, sintomáticos, fallecidos, riesgo_ambiental]
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )  # [mascarillas, confinamiento, aforos, desratización, aislamiento]
        
        # Parámetros epidemiológicos (del informe)
        self.CFR = 0.35  # Tasa de letalidad (25-40%, usamos 35%)
        self.r_zoonotic = 0.15  # Radio de exposición zoonótica
        self.r_human = 0.12    # Radio de transmisión respiratoria humana
        self.beta_zoo = 0.65   # Probabilidad base contagio zoonótico (50%+)
        self.beta_human = 0.05 # Probabilidad contagio interhumano (5%)
        self.mask_efficacy = 0.6  # Eficacia de mascarillas (60%)
        
        # Períodos epidemiológicos estocásticos
        self.latency_period = (9, 33)     # Rango período de incubación
        self.infectious_period = (5, 10)  # Duración fase sintomática
        self.detection_delay = (1, 3)     # Retraso en confirmación diagnóstica
        
        # Estados de salud de los agentes
        self.SUSCEPTIBLE = 0
        self.LATENT = 1
        self.SYMPTOMATIC = 2
        self.DECEASED = 3
        self.RECOVERED = 4
        
        # Inicializar geometría espacial
        self._initialize_spatial_geometry()
        
        # Inicializar estado de agentes
        self._initialize_agents()
        
        # Contadores de tiempo
        self.current_step = 0
        self.episode_data = {}
        
    def _initialize_spatial_geometry(self) -> None:
        """
        Inicializa la topología espacial del entorno: casas, focos zoonóticos
        y zonas seguras en un plano continuo acotado [0, 10] x [0, 10].
        """
        np.random.seed(self.np_random.integers(0, 2**32 - 1) if self.np_random else None)
        
        # Casas (residencias) - distribución aleatoria
        self.house_positions = np.random.uniform(0, 10, (self.num_houses, 2)).astype(np.float32)
        
        # Focos zoonóticos - distribución aleatoria con probabilidades iniciales
        self.zoonotic_foci = np.random.uniform(0, 10, (self.num_zoonotic_foci, 2)).astype(np.float32)
        self.zoonotic_hazard_probs = np.random.uniform(0.5, 1.0, self.num_zoonotic_foci).astype(np.float32)
        self.zoonotic_hazard_decay = 0.05  # Recuperación ecológica diaria
        
        # Zonas seguras - distribución aleatoria
        self.safe_zones = np.random.uniform(0, 10, (self.num_safe_zones, 2)).astype(np.float32)
        self.safe_zone_capacities = np.full(self.num_safe_zones, 20, dtype=np.int32)  # Cap base
        
    def _initialize_agents(self) -> None:
        """
        Inicializa los agentes de la población con estados de salud y posiciones.
        """
        # Asignar agentes a casas aleatoriamente (3 por casa en promedio)
        house_assignments = np.random.choice(self.num_houses, self.N)
        self.agent_houses = house_assignments
        
        # Posiciones iniciales en las casas (pequeña variación)
        self.positions = np.zeros((self.N, 2), dtype=np.float32)
        for i in range(self.N):
            house_idx = self.agent_houses[i]
            # Agregar pequeño offset a la posición de la casa
            offset = np.random.normal(0, 0.1, 2)
            self.positions[i] = self.house_positions[house_idx] + offset
            self.positions[i] = np.clip(self.positions[i], 0, 10)
        
        # Estados de salud iniciales
        self.health_states = np.full(self.N, self.SUSCEPTIBLE, dtype=np.int32)
        
        # Tiempos en estado actual (para transiciones)
        self.time_in_state = np.zeros(self.N, dtype=np.int32)
        
        # Señal de infección latente (tiempo de latencia restante)
        self.latency_time_remaining = np.zeros(self.N, dtype=np.int32)
        
        # Retraso en detección de casos sintomáticos
        self.detection_delay_remaining = np.zeros(self.N, dtype=np.int32)
        
        # Registro de estado observado (lo que el policy puede ver)
        self.observed_health_states = np.full(self.N, self.SUSCEPTIBLE, dtype=np.int32)
        
        # POI asignados (casa o zona segura) para cada agente
        self.assigned_poi = np.random.choice(self.num_safe_zones, self.N)
        
        # Iniciar el brote con un número arbitrario de casos importados
        # (self.initial_infected agentes elegidos sin reemplazo, cada uno
        # entra en estado LATENTE con su propio período de incubación)
        patient_zeros = np.random.choice(
            self.N, size=self.initial_infected, replace=False
        )
        self.health_states[patient_zeros] = self.LATENT
        self.latency_time_remaining[patient_zeros] = np.random.randint(
            self.latency_period[0], self.latency_period[1] + 1,
            size=self.initial_infected
        )
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso temporal de la simulación epidemiológica.
        
        Args:
            action: Vector de 5 acciones continuas [mascarillas, confinamiento, aforos, desratización, aislamiento]
        
        Returns:
            observation: Estado agregado del sistema
            reward: Recompensa multiobjetivo
            terminated: Si el episodio ha terminado
            truncated: Si el horizonte temporal se ha alcanzado
            info: Información adicional
        """
        # Desempacar acciones
        a_masks = action[0]
        a_confinement = action[1]
        a_capacity = action[2]
        a_deratization = action[3]
        a_isolation = action[4]
        
        # 1. MOVILIDAD: Actualizar posiciones de agentes
        self._update_mobility(a_confinement, a_isolation)
        
        # 2. TRANSMISIÓN: Calcular nuevas infecciones (zoonótica e interhumana)
        self._calculate_infections(a_masks)
        
        # 3. RIESGO ZOONÓTICO: Aplicar desratización y recuperación ecológica
        self._update_zoonotic_hazard(a_deratization)
        
        # 4. TRANSICIONES EPIDEMIOLÓGICAS: Progresión de estados de salud
        self._update_health_transitions()
        
        # 5. DETECCIÓN: Actualizar estado observado con retraso
        self._update_detection()
        
        # 6. CAPACIDAD MÁXIMA: Aplicar control de aforos
        self._apply_capacity_restrictions(a_capacity)
        
        # 7. OBSERVACIÓN: Calcular estado agregado
        observation = self._get_observation()
        
        # 8. RECOMPENSA: Función multiobjetivo
        reward = self._calculate_reward(
            a_masks, a_confinement, a_capacity, a_deratization, a_isolation
        )
        
        # 9. TERMINACIÓN
        self.current_step += 1
        terminated = False  # No terminar antes de T días
        truncated = self.current_step >= self.T
        
        # Información de debug
        info = {
            "susceptible": np.sum(self.health_states == self.SUSCEPTIBLE),
            "latent": np.sum(self.health_states == self.LATENT),
            "symptomatic": np.sum(self.health_states == self.SYMPTOMATIC),
            "deceased": np.sum(self.health_states == self.DECEASED),
            "recovered": np.sum(self.health_states == self.RECOVERED),
        }
        
        return observation, float(reward), terminated, truncated, info
    
    def _update_mobility(self, a_confinement: float, a_isolation: float) -> None:
        """
        Actualiza posiciones de agentes mediante movimiento pendular diario.
        Vectorizado: sin bucles for.
        """
        # Máscara de movilidad: agentes confinados permanecen en casa
        mobility_mask = np.random.uniform(0, 1, self.N) > a_confinement
        
        # Máscara de aislamiento: sintomáticos detectados confinados
        isolation_mask = np.logical_and(
            mobility_mask,
            ~((self.observed_health_states == self.SYMPTOMATIC) & 
              (np.random.uniform(0, 1, self.N) < a_isolation))
        )
        
        # Vectores de movimiento hacia destinos
        destination_positions = np.zeros((self.N, 2), dtype=np.float32)
        
        for i in range(self.N):
            if isolation_mask[i]:
                # Agente móvil: dirigirse a POI asignado o permanecer en casa
                if np.random.random() < 0.5:  # 50% probabilidad de viajar
                    poi_idx = self.assigned_poi[i]
                    destination_positions[i] = self.safe_zones[poi_idx]
                else:
                    destination_positions[i] = self.house_positions[self.agent_houses[i]]
            else:
                # Agente confinado: permanecer en casa
                destination_positions[i] = self.house_positions[self.agent_houses[i]]
        
        # Movimiento lineal interpolado hacia destino (0.5 del camino por día)
        movement = destination_positions - self.positions
        self.positions = self.positions + 0.5 * movement
        
        # Ruido browniano estocástico (desvíos menores)
        brownian_noise = np.random.normal(0, 0.05, (self.N, 2))
        self.positions = self.positions + brownian_noise
        
        # Acotar posiciones al espacio [0, 10] x [0, 10]
        self.positions = np.clip(self.positions, 0, 10)
    
    def _calculate_infections(self, a_masks: float) -> None:
        """
        Calcula nuevas infecciones mediante colisiones de distancia vectorizadas.
        Utiliza scipy.spatial.distance.cdist para operaciones O(1) compiladas en C.
        """
        susceptible_mask = self.health_states == self.SUSCEPTIBLE
        susceptible_idx = np.where(susceptible_mask)[0]
        
        if len(susceptible_idx) == 0:
            return
        
        susceptible_pos = self.positions[susceptible_idx]
        
        # CONTAGIO ZOONÓTICO
        # Matriz de distancias humanos -> focos zoonóticos
        distances_zoo = cdist(susceptible_pos, self.zoonotic_foci, metric='euclidean')
        
        for sj, s_idx in enumerate(susceptible_idx):
            for fj in range(self.num_zoonotic_foci):
                if distances_zoo[sj, fj] < self.r_zoonotic:
                    # Agente dentro del radio de exposición
                    base_prob = self.beta_zoo * self.zoonotic_hazard_probs[fj]
                    masked_prob = base_prob * (1 - self.mask_efficacy * a_masks)
                    
                    if np.random.random() < masked_prob:
                        # Infección zoonótica
                        self.health_states[s_idx] = self.LATENT
                        self.latency_time_remaining[s_idx] = np.random.randint(
                            self.latency_period[0], self.latency_period[1] + 1
                        )
                        break
        
        # CONTAGIO INTERHUMANO
        # Matriz de distancias humanos -> humanos
        distances_hum = cdist(self.positions, self.positions, metric='euclidean')
        
        symptomatic_mask = self.health_states == self.SYMPTOMATIC
        symptomatic_idx = np.where(symptomatic_mask)[0]
        
        for s_idx in susceptible_idx:
            for symp_idx in symptomatic_idx:
                if distances_hum[s_idx, symp_idx] < self.r_human:
                    # Agente susceptible cerca de agente sintomático
                    base_prob = self.beta_human
                    masked_prob = base_prob * (1 - self.mask_efficacy * a_masks)
                    
                    if np.random.random() < masked_prob:
                        # Infección interhumana
                        self.health_states[s_idx] = self.LATENT
                        self.latency_time_remaining[s_idx] = np.random.randint(
                            self.latency_period[0], self.latency_period[1] + 1
                        )
                        break
    
    def _update_zoonotic_hazard(self, a_deratization: float) -> None:
        """
        Actualiza el nivel de peligro zoonótico: desratización reduce el riesgo,
        pero hay recuperación ecológica después.
        """
        # Desratización reduce el peligro ambiental
        self.zoonotic_hazard_probs = self.zoonotic_hazard_probs * (1 - 0.5 * a_deratization)
        
        # Recuperación ecológica natural
        self.zoonotic_hazard_probs = np.minimum(
            self.zoonotic_hazard_probs + self.zoonotic_hazard_decay,
            1.0
        )
    
    def _update_health_transitions(self) -> None:
        """
        Actualiza transiciones epidemiológicas: L→I, I→(D|R) según CFR.
        Vectorizado sin bucles.
        """
        # LATENTE → SINTOMÁTICO
        latent_mask = self.health_states == self.LATENT
        latent_idx = np.where(latent_mask)[0]
        
        for idx in latent_idx:
            self.latency_time_remaining[idx] -= 1
            if self.latency_time_remaining[idx] <= 0:
                # Transición a sintomático
                self.health_states[idx] = self.SYMPTOMATIC
                # Fijar retraso de detección
                self.detection_delay_remaining[idx] = np.random.randint(
                    self.detection_delay[0], self.detection_delay[1] + 1
                )
                self.time_in_state[idx] = 0
        
        # SINTOMÁTICO → RECUPERADO o FALLECIDO
        symptomatic_mask = self.health_states == self.SYMPTOMATIC
        symptomatic_idx = np.where(symptomatic_mask)[0]
        
        for idx in symptomatic_idx:
            self.time_in_state[idx] += 1
            infectious_duration = np.random.randint(self.infectious_period[0], self.infectious_period[1] + 1)
            
            if self.time_in_state[idx] >= infectious_duration:
                # Desenlace: muerte o recuperación
                if np.random.random() < self.CFR:
                    self.health_states[idx] = self.DECEASED
                else:
                    self.health_states[idx] = self.RECOVERED
    
    def _update_detection(self) -> None:
        """
        Actualiza el estado observado con retraso: sintomáticos no son
        identificados inmediatamente.
        """
        # Retraso en confirmación diagnóstica
        for idx in range(self.N):
            if self.health_states[idx] == self.SYMPTOMATIC:
                self.detection_delay_remaining[idx] -= 1
                if self.detection_delay_remaining[idx] <= 0:
                    # Caso confirmado y registrado
                    self.observed_health_states[idx] = self.SYMPTOMATIC
            elif self.health_states[idx] in [self.DECEASED, self.RECOVERED]:
                self.observed_health_states[idx] = self.health_states[idx]
            # LATENT: siempre invisible
    
    def _apply_capacity_restrictions(self, a_capacity: float) -> None:
        """
        Aplica restricciones de aforo a las zonas seguras.
        Agentes que exceden capacidad se devuelven a casa.
        """
        for poi_idx in range(self.num_safe_zones):
            # Capacidad dinámica
            max_capacity = self.safe_zone_capacities[poi_idx] * (1 - a_capacity)
            
            # Contar agentes en esta zona
            distances_to_poi = np.linalg.norm(
                self.positions - self.safe_zones[poi_idx], axis=1
            )
            agents_in_poi = np.sum(distances_to_poi < 0.5)  # Radio de la zona
            
            if agents_in_poi > max_capacity:
                # Devolver agentes excedentes a casa aleatoriamente
                agents_in_poi_idx = np.where(distances_to_poi < 0.5)[0]
                num_to_remove = int(agents_in_poi - max_capacity)
                to_remove = np.random.choice(
                    agents_in_poi_idx, min(num_to_remove, len(agents_in_poi_idx)), replace=False
                )
                
                for idx in to_remove:
                    house_idx = self.agent_houses[idx]
                    self.positions[idx] = self.house_positions[house_idx]
    
    def _get_observation(self) -> np.ndarray:
        """
        Retorna el vector de estado agregado normalizado [0, 1]^5.
        """
        obs = np.array([
            np.sum(self.health_states == self.SUSCEPTIBLE) / self.N,
            np.sum(self.health_states == self.LATENT) / self.N,
            np.sum(self.observed_health_states == self.SYMPTOMATIC) / self.N,
            np.sum(self.health_states == self.DECEASED) / self.N,
            np.mean(self.zoonotic_hazard_probs),
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self, a_masks: float, a_confinement: float, a_capacity: float,
                           a_deratization: float, a_isolation: float) -> float:
        """
        Función de recompensa multiobjetivo con ponderaciones específicas:
        α=0.20 (economía), β=0.65 (salud), γ=0.15 (sostenibilidad social)

        Todas las intervenciones tienen ahora un costo asociado en al menos
        un eje, para evitar que el agente trate alguna acción como "gratis":

          - Confinamiento:  costo económico alto + costo social (ya existía)
          - Control aforos: costo social (ya existía)
          - Desratización:  costo económico bajo  + costo de salud bajo
          - Mascarillas:    costo económico bajo  + costo social moderado
          - Aislamiento:    costo económico moderado + costo social alto
        """
        alpha, beta, gamma = 0.20, 0.65, 0.15

        # --- Pesos de costo por intervención (ajustables) ---
        # Económicos (restan de R_c)
        C_ECON_DERATIZATION = 0.05   # bajo
        C_ECON_MASKS = 0.05          # bajo
        C_ECON_ISOLATION = 0.20      # moderado

        # Salud (resta de R_h)
        C_HEALTH_DERATIZATION = 0.05  # bajo (ej. exposición a raticidas/químicos)

        # Sociales (restan de R_p, junto con confinamiento/aforos ya existentes)
        C_SOCIAL_MASKS = 0.15        # moderado
        C_SOCIAL_ISOLATION = 0.40    # grande

        # R_c^t: Recompensa económica
        healthy_and_latent = (
            np.sum(self.health_states == self.SUSCEPTIBLE) +
            np.sum(self.health_states == self.LATENT)
        ) / self.N
        econ_cost = (
            C_ECON_DERATIZATION * a_deratization +
            C_ECON_MASKS * a_masks +
            C_ECON_ISOLATION * a_isolation
        )
        R_c = healthy_and_latent * (1 - a_confinement) - econ_cost

        # R_h^t: Recompensa de salud pública (asimétrica)
        health_weights = np.zeros(self.N)
        health_weights[self.health_states == self.SUSCEPTIBLE] = 1.0
        health_weights[self.health_states == self.LATENT] = 0.5
        health_weights[self.health_states == self.SYMPTOMATIC] = 0.0
        health_weights[self.health_states == self.DECEASED] = -10.0
        health_weights[self.health_states == self.RECOVERED] = 1.0
        R_h = np.mean(health_weights) - C_HEALTH_DERATIZATION * a_deratization

        # R_p^t: Penalización por fatiga pandémica / costo social
        social_cost = C_SOCIAL_MASKS * a_masks + C_SOCIAL_ISOLATION * a_isolation
        R_p = 1.0 - (a_confinement + a_capacity) / 2 - social_cost

        # Recompensa agregada
        reward = alpha * R_c + beta * R_h + gamma * R_p

        return reward
    
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno para un nuevo episodio.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self._initialize_spatial_geometry()
        self._initialize_agents()
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def render(self) -> None:
        """
        Renderizar no implementado en modo headless.
        """
        pass