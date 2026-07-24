"""
Análisis y visualización de dinámicas epidemiológicas simuladas bajo políticas PPO.

Genera:
- Gráficos de trayectorias de compartimentos SEIR
- Análisis de políticas de control aprendidas
- Comparativas con/sin intervención
- Estadísticas epidemiológicas agregadas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple
import torch
from epidemic_env import EpidemicEnvironment
from ppo_trainer import PPOTrainer
import os


class EpidemicAnalyzer:
    """
    Analizador de dinámicas epidemiológicas simuladas.
    Genera visualizaciones e informes de resultados.
    """
    
    def __init__(self, env: EpidemicEnvironment, trainer: PPOTrainer):
        self.env = env
        self.trainer = trainer
        self.device = trainer.device
        
    def simulate_with_policy(self, num_episodes: int = 3, max_steps: int = 150) -> Tuple[List, List]:
        """
        Simula episodios usando la política aprendida y recolecta datos
        de dinámicas epidemiológicas.
        
        Returns:
            Tupla con datos agregados de trayectorias
        """
        all_trajectories = []
        all_actions = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).to(self.device)
            
            trajectory = {
                'susceptible': [],
                'latent': [],
                'symptomatic': [],
                'deceased': [],
                'recovered': [],
                'time': []
            }
            
            actions_log = {
                'mascarillas': [],
                'confinamiento': [],
                'aforos': [],
                'desratización': [],
                'aislamiento': []
            }
            
            done = False
            step = 0
            
            while not done and step < max_steps:
                with torch.no_grad():
                    mean, std, value = self.trainer.network.forward(state.unsqueeze(0))
                
                # Usar media determinista
                action = mean.squeeze(0).cpu().numpy()
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Registrar trayectoria
                trajectory['susceptible'].append(info['susceptible'])
                trajectory['latent'].append(info['latent'])
                trajectory['symptomatic'].append(info['symptomatic'])
                trajectory['deceased'].append(info['deceased'])
                trajectory['recovered'].append(info['recovered'])
                trajectory['time'].append(step)
                
                # Registrar acciones
                actions_log['mascarillas'].append(action[0])
                actions_log['confinamiento'].append(action[1])
                actions_log['aforos'].append(action[2])
                actions_log['desratización'].append(action[3])
                actions_log['aislamiento'].append(action[4])
                
                state = torch.FloatTensor(next_state).to(self.device)
                done = terminated or truncated
                step += 1
            
            all_trajectories.append(trajectory)
            all_actions.append(actions_log)
        
        return all_trajectories, all_actions
    
    def plot_epidemic_curves(self, trajectories: List, save_path: str = None, max_steps: int = 150) -> None:
        """
        Grafica las curvas epidemiológicas SEIR agregadas.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Dinámicas Epidemiológicas', 
                     fontsize=14, fontweight='bold')
        
        # Agregar trayectorias
        num_episodes = len(trajectories)
        
        for traj in trajectories:
            time = np.array(traj['time'])
            
            # Susceptibles
            axes[0, 0].plot(time, traj['susceptible'], label='S (Susceptibles)', 
                          linewidth=2, alpha=0.7)
            
            # Latentes
            axes[0, 1].plot(time, traj['latent'], label='L (Latentes/Incubación)', 
                          linewidth=2, alpha=0.7)
            
            # Sintomáticos
            axes[0, 2].plot(time, traj['symptomatic'], label='I (Sintomáticos)', 
                          linewidth=2, alpha=0.7)
            
            # Recuperados
            axes[1, 0].plot(time, traj['recovered'], label='R (Recuperados)', 
                          linewidth=2, alpha=0.7)
            
            # Fallecidos
            axes[1, 1].plot(time, traj['deceased'], label='D (Fallecidos)', 
                          linewidth=2, alpha=0.7)
        
        # Configurar ejes
        titles = ['Susceptibles', 'Latentes', 'Sintomáticos', 'Recuperados', 'Fallecidos']
        for ax, title in zip(axes.flat, titles):
            ax.set_xlabel('Tiempo (días)', fontsize=11)
            ax.set_ylabel('Número de agentes', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_steps)
        
        # Eliminar el 6to subplot vacío de la cuadrícula 2x3
        fig.delaxes(axes.flat[5])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de curvas epidemiológicas guardado: {save_path}")
        else:
            plt.show()
    
    def plot_control_policies(self, actions: List, save_path: str = None) -> None:
        """
        Grafica las políticas de intervención aprendidas.
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle('Políticas de Intervención No Farmacéutica Aprendidas', 
                     fontsize=14, fontweight='bold')
        
        # Títulos visuales de los gráficos
        action_names = [
            'Mascarillas N95',
            'Confinamiento',
            'Control de Aforos',
            'Desratización',
            'Aislamiento'
        ]
        
        # Las llaves exactas con las que se guardaron los datos en el diccionario
        action_keys = [
            'mascarillas', 
            'confinamiento', 
            'aforos', 
            'desratización', 
            'aislamiento'
        ]
        
        colors = ['blue', 'green', 'orange', 'purple', 'red']
        
        # Iterar sobre cada tipo de acción para graficarla
        for action_idx, (action_name, key, color) in enumerate(zip(action_names, action_keys, colors)):
            ax = axes.flat[action_idx]
            
            # Graficar la curva de cada episodio para esta acción
            for ep_idx, ep_actions in enumerate(actions):
                action_values = ep_actions[key]
                time = np.arange(len(action_values))
                ax.plot(time, action_values, label=f'Episodio {ep_idx + 1}', 
                       linewidth=2, alpha=0.7, color=color)
            
            # Configurar visualmente el subplot
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Tiempo (días)', fontsize=10)
            ax.set_ylabel('Intensidad [0, 1]', fontsize=10)
            ax.set_title(action_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if actions:
                ax.legend(fontsize=9)
        
        # Eliminar el 6to subplot que queda vacío en la cuadrícula 2x3
        fig.delaxes(axes.flat[5])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de políticas de control guardado: {save_path}")
        else:
            plt.show()
    
    def plot_training_progress(self, trainer: PPOTrainer, save_path: str = None) -> None:
        """
        Grafica el progreso del entrenamiento.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Progreso del Entrenamiento - PPO Epidemiológico', 
                     fontsize=14, fontweight='bold')
        
        # Recompensas por episodio
        if len(trainer.train_stats['episode_rewards']) > 0:
            axes[0, 0].plot(trainer.train_stats['episode_rewards'], linewidth=1.5, color='blue')
            axes[0, 0].set_xlabel('Número de episodio', fontsize=11)
            axes[0, 0].set_ylabel('Recompensa acumulada', fontsize=11)
            axes[0, 0].set_title('Recompensa por Episodio', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Longitud de episodios
        if len(trainer.train_stats['episode_lengths']) > 0:
            axes[0, 1].plot(trainer.train_stats['episode_lengths'], linewidth=1.5, color='green')
            axes[0, 1].set_xlabel('Número de episodio', fontsize=11)
            axes[0, 1].set_ylabel('Longitud del episodio (pasos)', fontsize=11)
            axes[0, 1].set_title('Longitud del Episodio', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Pérdida del Actor
        if len(trainer.train_stats['actor_losses']) > 0:
            axes[1, 0].plot(trainer.train_stats['actor_losses'], linewidth=1.5, color='red')
            axes[1, 0].set_xlabel('Número de iteración', fontsize=11)
            axes[1, 0].set_ylabel('Pérdida del Actor', fontsize=11)
            axes[1, 0].set_title('Pérdida Actor (PPO-Clip)', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Pérdida del Crítico
        if len(trainer.train_stats['critic_losses']) > 0:
            axes[1, 1].plot(trainer.train_stats['critic_losses'], linewidth=1.5, color='purple')
            axes[1, 1].set_xlabel('Número de iteración', fontsize=11)
            axes[1, 1].set_ylabel('Pérdida del Crítico (MSE)', fontsize=11)
            axes[1, 1].set_title('Pérdida Crítico (Valor)', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de progreso de entrenamiento guardado: {save_path}")
        else:
            plt.show()
    
    def generate_report(self, trajectories: List, actions: List, save_dir: str = None) -> str:
        """
        Genera un reporte textual con estadísticas epidemiológicas.
        """
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE ANÁLISIS EPIDEMIOLÓGICO - SIMULACIÓN DEL HANTAVIRUS")
        report.append("=" * 80 + "\n")
        
        report.append("ESTADÍSTICAS AGREGADAS DE LA SIMULACIÓN\n")
        
        num_episodes = len(trajectories)
        final_susceptible = []
        final_latent = []
        final_symptomatic = []
        final_deceased = []
        final_recovered = []
        max_symptomatic = []
        total_cases = []
        
        for traj in trajectories:
            final_susceptible.append(traj['susceptible'][-1])
            final_latent.append(traj['latent'][-1])
            final_symptomatic.append(traj['symptomatic'][-1])
            final_deceased.append(traj['deceased'][-1])
            final_recovered.append(traj['recovered'][-1])
            max_symptomatic.append(max(traj['symptomatic']))
            total_cases.append(sum(1 for x in traj['deceased'] if x > 0))
        
        report.append(f"Número de episodios simulados: {num_episodes}")
        report.append(f"Duración de cada simulación: 60 días")
        report.append(f"Tamaño poblacional: 100 agentes\n")
        
        report.append("ESTADO FINAL (Promedio):")
        report.append(f"  - Susceptibles: {np.mean(final_susceptible):.2f} (±{np.std(final_susceptible):.2f})")
        report.append(f"  - Latentes: {np.mean(final_latent):.2f} (±{np.std(final_latent):.2f})")
        report.append(f"  - Sintomáticos: {np.mean(final_symptomatic):.2f} (±{np.std(final_symptomatic):.2f})")
        report.append(f"  - Fallecidos: {np.mean(final_deceased):.2f} (±{np.std(final_deceased):.2f})")
        report.append(f"  - Recuperados: {np.mean(final_recovered):.2f} (±{np.std(final_recovered):.2f})\n")
        
        report.append("PICOS EPIDEMIOLÓGICOS:")
        report.append(f"  - Máximo de casos sintomáticos: {np.mean(max_symptomatic):.2f} (±{np.std(max_symptomatic):.2f})")
        report.append(f"  - Tasa de ataque (% infectados): {100 * np.mean([1 - s/self.env.N for s in final_susceptible]):.2f}%")
        report.append(f"  - Tasa de letalidad bruta: {100 * np.mean([d/100 for d in final_deceased]):.2f}%\n")
        
        # Políticas promedio
        report.append("POLÍTICAS DE INTERVENCIÓN PROMEDIO:\n")
        
        avg_masks = np.mean([np.mean(act['mascarillas']) for act in actions])
        avg_confinement = np.mean([np.mean(act['confinamiento']) for act in actions])
        avg_capacity = np.mean([np.mean(act['aforos']) for act in actions])
        avg_deratization = np.mean([np.mean(act['desratización']) for act in actions])
        avg_isolation = np.mean([np.mean(act['aislamiento']) for act in actions])
        
        report.append(f"  - Intensidad promedio mascarillas: {avg_masks:.3f} (0=nula, 1=máxima)")
        report.append(f"  - Intensidad promedio confinamiento: {avg_confinement:.3f}")
        report.append(f"  - Intensidad promedio control aforos: {avg_capacity:.3f}")
        report.append(f"  - Intensidad promedio desratización: {avg_deratization:.3f}")
        report.append(f"  - Intensidad promedio aislamiento: {avg_isolation:.3f}\n")
        
        report.append("=" * 80)
        report.append("FIN DEL REPORTE")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, 'epidemic_analysis_report.txt')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\nReporte guardado en: {filepath}")
        
        print("\n" + report_text)
        
        return report_text


def main():
    """
    Script principal para análisis y visualización de resultados.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Análisis y visualización de simulaciones epidemiológicas"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/mnt/user-data/outputs/epidemic_policy.pt',
        help='Ruta al modelo entrenado'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Número de episodios a simular para análisis'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/user-data/outputs',
        help='Directorio para guardar gráficos y reportes'
    )
    parser.add_argument(
        '--initial_infected',
        type=int,
        default=20,
        help='Número de personas infectadas (brote inicial) al día 0'
    )
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Crear entorno
    env = EpidemicEnvironment(
        population_size=500,
        simulation_days=150,
        num_houses=165,
        num_zoonotic_foci=25,
        num_safe_zones=25,
        initial_infected=args.initial_infected
    )
    
    # Crear entrenador
    trainer = PPOTrainer(env=env, device=device)
    
    # Cargar modelo y estadísticas
    if os.path.exists(args.model_path):
        trainer.load_policy(args.model_path)
        print(f"Modelo cargado desde: {args.model_path}")
        
        # Buscar el archivo de resultados (.npy) asociado al modelo
        results_path = args.model_path.replace('.pt', '_results.npy')
        if os.path.exists(results_path):
            print("Cargando historial de entrenamiento...")
            stats_data = np.load(results_path, allow_pickle=True).item()
            
            # Poblar el trainer con los datos históricos para poder graficarlos
            if 'train_rewards' in stats_data:
                trainer.train_stats['episode_rewards'] = stats_data['train_rewards']
            if 'train_lengths' in stats_data:
                trainer.train_stats['episode_lengths'] = stats_data['train_lengths']
            if 'actor_losses' in stats_data:
                trainer.train_stats['actor_losses'] = stats_data['actor_losses']
            if 'critic_losses' in stats_data:
                trainer.train_stats['critic_losses'] = stats_data['critic_losses']
        else:
            print(f"Advertencia: No se encontró {results_path}. El gráfico de progreso quedará en blanco.")
            
    else:
        print(f"Advertencia: No se encontró modelo en {args.model_path}")
        print("Usando red inicializada aleatoriamente")
    
    # Crear analizador
    analyzer = EpidemicAnalyzer(env, trainer)
    
    # Ejecutar simulaciones
    print("\nEjecutando simulaciones para análisis...")
    trajectories, actions = analyzer.simulate_with_policy(num_episodes=args.episodes)
    
    # Generar visualizaciones
    print("Generando visualizaciones...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyzer.plot_epidemic_curves(
        trajectories,
        save_path=os.path.join(args.output_dir, 'epidemic_curves.png')
    )
    
    analyzer.plot_control_policies(
        actions,
        save_path=os.path.join(args.output_dir, 'control_policies.png')
    )
    
    analyzer.plot_training_progress(
        trainer,
        save_path=os.path.join(args.output_dir, 'training_progress.png')
    )
    
    # Generar reporte
    analyzer.generate_report(
        trajectories, actions,
        save_dir=args.output_dir
    )
    
    print("\n✓ Análisis completado. Archivos guardados en:", args.output_dir)


if __name__ == '__main__':
    main()
