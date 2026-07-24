"""
Implementación de Proximal Policy Optimization (PPO) con arquitectura Actor-Crítico
para optimización de políticas de contención epidemiológica del Hantavirus Andino.

Características:
- Redes neuronales parametrizadas independientemente para Actor y Crítico
- Clipping de región de confianza con ε=0.2
- Ventaja Generalizada (GAE) con λ=0.95 y γ=0.99
- Coeficientes de pérdida: c_1=0.5 (valor), c_2=0.01 (entropía)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ActorCriticNetwork(nn.Module):
    """
    Red neuronal Actor-Crítico con arquitectura separada pero compartiendo
    capas de características iniciales.
    """
    
    def __init__(self, input_size=5, action_size=5, hidden_size=64):
        """
        Args:
            input_size: Dimensión del espacio de observación (5)
            action_size: Dimensión del espacio de acciones (5)
            hidden_size: Número de unidades en capas ocultas
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Capas compartidas de características
        self.shared_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # RED DEL ACTOR: π_θ(a_t | s_t)
        # Genera parámetros μ y σ de distribución gaussiana diagonal
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Sigmoid()  # Confinar acciones a [0, 1]
        )
        
        self.actor_log_std = nn.Parameter(
            torch.zeros(action_size)
        )
        
        # RED DEL CRÍTICO: V_φ(s_t)
        # Estima valor acumulado esperado
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state):
        """
        Forward pass computando media, desv. estándar y valor.
        """
        features = self.shared_fc(state)
        
        # Actor: media y desv. estándar
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        # Crítico: valor
        value = self.critic(features)
        
        return mean, std, value
    
    def get_action_and_value(self, state):
        """
        Muestrea acción de política y obtiene estimación de valor.
        """
        mean, std, value = self.forward(state)
        
        # Distribución normal gaussiana diagonal
        dist = Normal(mean, std)
        action = dist.sample()
        
        # Clamp acciones a [0, 1]
        action = torch.clamp(action, 0, 1)
        
        # Log probabilidad de la acción (para ratio de importancia)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def get_value(self, state):
        """
        Obtiene estimación de valor sin muestrear acción.
        """
        features = self.shared_fc(state)
        value = self.critic(features)
        return value
    
    def get_action_log_prob(self, state, action):
        """
        Calcula log probabilidad de acción dada bajo política actual.
        """
        mean, std, _ = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class PPOTrainer:
    """
    Entrenador de Proximal Policy Optimization para el agente de control
    epidemiológico.
    """
    
    def __init__(self,
                 env,
                 learning_rate=3e-4,
                 gamma=0.99,
                 lambda_gae=0.95,
                 clip_ratio=0.2,
                 value_coeff=0.5,
                 entropy_coeff=0.01,
                 batch_size=64,
                 epochs_per_update=10,
                 device='cpu'):
        """
        Args:
            env: Entorno Gymnasium
            learning_rate: Tasa de aprendizaje Adam (α = 3e-4)
            gamma: Factor de descuento (γ = 0.99)
            lambda_gae: Parámetro GAE (λ = 0.95)
            clip_ratio: Parámetro de clipping (ε = 0.2)
            value_coeff: Coeficiente pérdida del valor (c_1 = 0.5)
            entropy_coeff: Coeficiente bonificación entropía (c_2 = 0.01)
            batch_size: Tamaño de lote (64)
            epochs_per_update: Épocas PPO por actualización (10)
            device: 'cpu' o 'cuda'
        """
        self.env = env
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        self.device = torch.device(device)
        
        # Red Actor-Crítico
        self.network = ActorCriticNetwork(
            input_size=5, action_size=5, hidden_size=64
        ).to(self.device)
        
        # Optimizador Adam
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Almacenamiento de transiciones
        self.storage = {
            'states': deque(maxlen=2048),
            'actions': deque(maxlen=2048),
            'rewards': deque(maxlen=2048),
            'values': deque(maxlen=2048),
            'log_probs': deque(maxlen=2048),
            'dones': deque(maxlen=2048),
        }
        
        # Estadísticas de entrenamiento
        self.train_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
        }
    
    def collect_rollout(self, num_steps=2048):
        """
        Recolecta trayectorias del entorno mediante rolleouts.
        Horizonte de rollout: T=2048 pasos.
        
        Args:
            num_steps: Número de pasos de ambiente a recolectar
        """
        state, _ = self.env.reset()
        state = torch.FloatTensor(state).to(self.device)
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            with torch.no_grad():
                action, log_prob, value = self.network.get_action_and_value(state.unsqueeze(0))
            
            # Ejecutar acción en entorno
            action_np = action.squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Almacenar experiencia
            self.storage['states'].append(state.cpu().numpy())
            self.storage['actions'].append(action.squeeze(0).cpu().numpy())
            self.storage['rewards'].append(reward)
            self.storage['values'].append(value.squeeze(0).item())
            self.storage['log_probs'].append(log_prob.squeeze(0).item())
            self.storage['dones'].append(terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            
            # Transición
            state = torch.FloatTensor(next_state).to(self.device)
            
            if terminated or truncated:
                self.train_stats['episode_rewards'].append(episode_reward)
                self.train_stats['episode_lengths'].append(episode_length)
                
                state, _ = self.env.reset()
                state = torch.FloatTensor(state).to(self.device)
                episode_reward = 0
                episode_length = 0
    
    def compute_advantages(self):
        """
        Calcula ventajas generalizadas (GAE) con λ=0.95 y γ=0.99.
        Formula: Â_t = Σ (γλ)^l δ_{t+l}^V
        donde δ_t^V = R_t + γV(s_{t+1}) - V(s_t)
        """
        states = torch.FloatTensor(np.array(self.storage['states'])).to(self.device)
        rewards = torch.FloatTensor(np.array(self.storage['rewards'])).to(self.device)
        values = torch.FloatTensor(np.array(self.storage['values'])).to(self.device)
        dones = torch.FloatTensor(np.array(self.storage['dones'])).float().to(self.device)
        
        # Obtener últimos valores
        with torch.no_grad():
            next_value = self.network.get_value(states[-1:])
        
        # Calcular advantages y returns
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value.squeeze().item()
            else:
                next_val = values[t + 1].item()
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values
        
        # Normalizar advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self):
        """
        Actualiza la política y la red de valor mediante PPO.
        Realiza 10 épocas de optimización sobre el mini-batch.
        """
        advantages, returns = self.compute_advantages()
        
        states = torch.FloatTensor(np.array(self.storage['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.storage['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.storage['log_probs'])).to(self.device)
        
        num_data = len(states)
        indices = np.arange(num_data)
        
        actor_losses = []
        critic_losses = []
        
        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices)
            
            for i in range(0, num_data, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                log_probs, entropy = self.network.get_action_log_prob(
                    batch_states, batch_actions
                )
                values = self.network.get_value(batch_states).squeeze(-1)
                
                # Ratio de importancia
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PÉRDIDA DEL ACTOR (PPO clipped)
                # L^CLIP(θ) = min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # PÉRDIDA DEL CRÍTICO (MSE)
                # L^VF(φ) = (V(s_t) - V_targ)^2
                critic_loss = (values - batch_returns).pow(2).mean()
                
                # BONIFICACIÓN DE ENTROPÍA
                # Promueve exploración
                entropy_bonus = -self.entropy_coeff * entropy.mean()
                
                # Pérdida total
                loss = actor_loss + self.value_coeff * critic_loss + entropy_bonus
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
        
        # Almacenar estadísticas
        self.train_stats['actor_losses'].append(np.mean(actor_losses))
        self.train_stats['critic_losses'].append(np.mean(critic_losses))
        
        # Limpiar storage
        for key in self.storage:
            self.storage[key].clear()
    
    def train(self, num_iterations=100):
        """
        Loop principal de entrenamiento.
        
        Args:
            num_iterations: Número de iteraciones (cada una con rollout de 2048 pasos)
        """
        print("=" * 60)
        print("ENTRENAMIENTO DE POLÍTICA PPO EPIDEMIOLÓGICA")
        print("=" * 60)
        print(f"Configuración:")
        print(f"  - Horizonte de rollout: 2048 pasos")
        print(f"  - Tamaño de lote: 64")
        print(f"  - Épocas PPO por iteración: 10")
        print(f"  - Factor de descuento γ: 0.99")
        print(f"  - Parámetro GAE λ: 0.95")
        print(f"  - Parámetro clip ε: 0.2")
        print("=" * 60 + "\n")
        
        for iteration in range(num_iterations):
            print(f"[Iteración {iteration + 1}/{num_iterations}] Recolectando experiencias...")
            self.collect_rollout(num_steps=2048)
            
            print(f"[Iteración {iteration + 1}/{num_iterations}] Actualizando política...")
            self.update_policy()
            
            # Estadísticas
            if len(self.train_stats['episode_rewards']) > 0:
                avg_reward = np.mean(self.train_stats['episode_rewards'][-10:])
                avg_length = np.mean(self.train_stats['episode_lengths'][-10:])
                actor_loss = self.train_stats['actor_losses'][-1]
                critic_loss = self.train_stats['critic_losses'][-1]
                
                print(f"  Recompensa promedio: {avg_reward:.4f}")
                print(f"  Longitud episodio: {avg_length:.0f}")
                print(f"  Pérdida Actor: {actor_loss:.6f}")
                print(f"  Pérdida Crítico: {critic_loss:.6f}\n")
    
    def save_policy(self, filepath):
        """Guarda la política entrenada."""
        torch.save(self.network.state_dict(), filepath)
        print(f"Política guardada en: {filepath}")
    
    def load_policy(self, filepath):
        """Carga una política entrenada."""
        self.network.load_state_dict(torch.load(filepath))
        print(f"Política cargada desde: {filepath}")
    
    # Agregamos max_steps=150 como variable editable en los argumentos
    def test_policy(self, num_episodes=5, max_steps=150):
        """
        Prueba la política entrenada sin exploración.
        """
        print("\n" + "=" * 60)
        print(f"PRUEBA DE POLÍTICA ENTRENADA ({max_steps} días)")
        print("=" * 60 + "\n")
        
        episode_rewards = []
        episode_data = []
        
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            
            episode_reward = 0
            episode_log = {
                'susceptible': [],
                'latent': [],
                'symptomatic': [],
                'deceased': [],
                'actions': []
            }
            
            done = False
            step = 0
            
            # Reemplazamos el 60 por la variable max_steps
            while not done and step < max_steps:
                with torch.no_grad():
                    mean, std, value = self.network.forward(state.unsqueeze(0))
                
                # Usar media (acción determinista) para prueba
                action = mean.squeeze(0).cpu().numpy()
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_log['susceptible'].append(info['susceptible'])
                episode_log['latent'].append(info['latent'])
                episode_log['symptomatic'].append(info['symptomatic'])
                episode_log['deceased'].append(info['deceased'])
                episode_log['actions'].append(action)
                
                episode_reward += reward
                state = torch.FloatTensor(next_state).to(self.device)
                
                done = terminated or truncated
                step += 1
            
            episode_rewards.append(episode_reward)
            episode_data.append(episode_log)
            
            print(f"Episodio {ep + 1}: Recompensa = {episode_reward:.4f}, Pasos = {step}")
        
        print(f"\nRecompensa promedio (prueba): {np.mean(episode_rewards):.4f}")
        
        return episode_rewards, episode_data
