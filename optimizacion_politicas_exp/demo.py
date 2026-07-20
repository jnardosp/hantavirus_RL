"""
SCRIPT DE DEMOSTRACIÓN RÁPIDA
Ejecuta una prueba completa del proyecto en ~10 minutos
"""

import numpy as np
import torch
from epidemic_env import EpidemicEnvironment
from ppo_trainer import PPOTrainer
from analysis import EpidemicAnalyzer
import os


def demo_basic_environment():
    """
    Demostración 1: Funcionalidad básica del entorno
    """
    print("\n" + "="*70)
    print("DEMO 1: ENTORNO EPIDEMIOLÓGICO BÁSICO")
    print("="*70)
    
    env = EpidemicEnvironment(
        population_size=100,
        simulation_days=60,
        num_houses=33,
        num_zoonotic_foci=5,
        num_safe_zones=5
    )
    
    print("Entorno creado exitosamente")
    print(f"  - Espacio de observación: {env.observation_space}")
    print(f"  - Espacio de acciones: {env.action_space}")
    
    # Simular 5 pasos con política nula
    print("\nEjecutando 5 pasos de simulación (sin intervención)...")
    obs, _ = env.reset()
    
    for step in range(5):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Sin intervención
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nPaso {step + 1}:")
        print(f"  Susceptibles: {info['susceptible']}")
        print(f"  Latentes: {info['latent']}")
        print(f"  Sintomáticos: {info['symptomatic']}")
        print(f"  Fallecidos: {info['deceased']}")
        print(f"  Recompensa: {reward:.4f}")
    
    print("\n✓ Demo 1 completada")
    return env


def demo_ppo_training():
    """
    Demostración 2: Entrenamiento PPO acelerado (5 iteraciones)
    """
    print("\n" + "="*70)
    print("DEMO 2: ENTRENAMIENTO PPO ACELERADO")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device.upper()}")
    
    env = EpidemicEnvironment(
        population_size=100,
        simulation_days=60,
        num_houses=33,
        num_zoonotic_foci=5,
        num_safe_zones=5
    )
    
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
        device=device
    )
    
    print("\nEntrenando modelo durante 5 iteraciones (10,240 pasos)...")
    print("(Esto toma ~30 segundos en GPU, ~3 minutos en CPU)\n")
    
    trainer.train(num_iterations=5)
    
    print("\n✓ Demo 2 completada")
    return trainer


def demo_policy_evaluation():
    """
    Demostración 3: Evaluación de política entrenada
    """
    print("\n" + "="*70)
    print("DEMO 3: EVALUACIÓN DE POLÍTICA ENTRENADA")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = EpidemicEnvironment(
        population_size=100,
        simulation_days=60,
        num_houses=33,
        num_zoonotic_foci=5,
        num_safe_zones=5
    )
    
    trainer = PPOTrainer(env=env, device=device)
    
    # Cargar modelo si existe, sino entrenar brevemente
    model_path = '/mnt/user-data/outputs/epidemic_policy.pt'
    if os.path.exists(model_path):
        trainer.load_policy(model_path)
        print(f"Modelo cargado desde: {model_path}")
    else:
        print("Modelo no encontrado. Ejecutando entrenamiento rápido...")
        trainer.train(num_iterations=3)
    
    print("\nEvaluando política en 3 episodios de prueba...")
    rewards, data = trainer.test_policy(num_episodes=3)
    
    print(f"\nResultados de prueba:")
    print(f"  - Recompensa promedio: {np.mean(rewards):.4f}")
    print(f"  - Desviación estándar: {np.std(rewards):.4f}")
    print(f"  - Máximo: {np.max(rewards):.4f}")
    print(f"  - Mínimo: {np.min(rewards):.4f}")
    
    print("\n✓ Demo 3 completada")
    return trainer


def demo_analysis():
    """
    Demostración 4: Análisis y generación de reportes
    """
    print("\n" + "="*70)
    print("DEMO 4: ANÁLISIS Y VISUALIZACIÓN")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = EpidemicEnvironment(
        population_size=100,
        simulation_days=60,
        num_houses=33,
        num_zoonotic_foci=5,
        num_safe_zones=5
    )
    
    trainer = PPOTrainer(env=env, device=device)
    
    # Cargar o entrenar modelo
    model_path = '/mnt/user-data/outputs/epidemic_policy.pt'
    if os.path.exists(model_path):
        trainer.load_policy(model_path)
    else:
        print("Entrenando modelo rápido (3 iteraciones)...")
        trainer.train(num_iterations=3)
    
    analyzer = EpidemicAnalyzer(env, trainer)
    
    print("\nSimulando 2 episodios para análisis...")
    trajectories, actions = analyzer.simulate_with_policy(num_episodes=2)
    
    print("Generando gráficos...")
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    
    try:
        analyzer.plot_epidemic_curves(
            trajectories,
            save_path='/mnt/user-data/outputs/demo_epidemic_curves.png'
        )
        print("  ✓ Gráfico de curvas epidemiológicas")
    except Exception as e:
        print(f"  ✗ Error en gráfico de curvas: {e}")
    
    try:
        analyzer.plot_control_policies(
            actions,
            save_path='/mnt/user-data/outputs/demo_control_policies.png'
        )
        print("  ✓ Gráfico de políticas de control")
    except Exception as e:
        print(f"  ✗ Error en gráfico de políticas: {e}")
    
    try:
        analyzer.plot_training_progress(
            trainer,
            save_path='/mnt/user-data/outputs/demo_training_progress.png'
        )
        print("  ✓ Gráfico de progreso de entrenamiento")
    except Exception as e:
        print(f"  ✗ Error en gráfico de progreso: {e}")
    
    print("\nGenerando reporte...")
    analyzer.generate_report(
        trajectories, actions,
        save_dir='/mnt/user-data/outputs'
    )
    
    print("\n✓ Demo 4 completada")
    print(f"\nArchivos guardados en: /mnt/user-data/outputs/")


def main():
    """
    Ejecutar todas las demostraciones
    """
    print("\n" + "="*70)
    print("DEMOSTRACIÓN COMPLETA DEL PROYECTO")
    print("Optimización de Políticas Epidemiológicas con PPO")
    print("="*70)
    
    try:
        # Demo 1: Entorno básico
        env = demo_basic_environment()
        
        # Demo 2: Entrenamiento PPO
        trainer = demo_ppo_training()
        
        # Demo 3: Evaluación
        demo_policy_evaluation()
        
        # Demo 4: Análisis
        demo_analysis()
        
        print("\n" + "="*70)
        print("✓ TODAS LAS DEMOSTRACIONES COMPLETADAS EXITOSAMENTE")
        print("="*70)
        
        print("\nPróximos pasos:")
        print("1. Para entrenamiento completo:")
        print("   python main.py --mode train --iterations 100")
        print("\n2. Para análisis completo:")
        print("   python analysis.py --model_path epidemic_policy.pt --episodes 5")
        print("\n3. Para consultar documentación:")
        print("   cat README.md")
        
    except Exception as e:
        print(f"\n✗ Error durante la demostración: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
