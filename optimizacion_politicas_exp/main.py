"""
Script principal para entrenamiento e inferencia de la política PPO
de optimización de intervenciones no farmacéuticas contra el Hantavirus Andino.

Uso:
    python main.py --mode train --iterations 50
    python main.py --mode test --checkpoint model.pt
"""

import argparse
import numpy as np
import torch
import sys
import os

# Importar módulos del proyecto
from epidemic_env import EpidemicEnvironment
from ppo_trainer import PPOTrainer


def setup_environment():
    """
    Configura el entorno de ejecución y parámetros globales.
    """
    # Configurar semilla aleatoria para reproducibilidad
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Detectar dispositivo (CPU o GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo detectado: {device.upper()}")
    
    return device


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento e inferencia de política PPO epidemiológica"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test'],
        default='train',
        help='Modo de ejecución: entrenamiento o prueba'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Número de iteraciones de entrenamiento (cada iteración = 2048 pasos)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Ruta al checkpoint de modelo entrenado para prueba'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='model.pt',
        help='Ruta donde guardar el modelo entrenado'
    )
    parser.add_argument(
        '--episodes_test',
        type=int,
        default=5,
        help='Número de episodios a ejecutar en modo prueba'
    )
    parser.add_argument(
        '--initial_infected',
        type=int,
        default=None,
        help='Número de personas infectadas (brote inicial) al día 0 (fijo, ignorado si se usan --initial_infected_min/max)'
    )
    parser.add_argument(
        '--initial_infected_min',
        type=int,
        default=5,
        help='Límite inferior del rango aleatorio de brote inicial (requiere --initial_infected_max)'
    )
    parser.add_argument(
        '--initial_infected_max',
        type=int,
        default=80,
        help='Límite superior del rango aleatorio de brote inicial (requiere --initial_infected_min)'
    )
    
    args = parser.parse_args()
    
    # Configurar entorno
    device = setup_environment()
    
    # Crear instancia del entorno epidemiológico
    print("\nInicializando entorno epidemiológico...")
    if args.initial_infected_min is not None and args.initial_infected_max is not None:
        env = EpidemicEnvironment(
            population_size=500,
            simulation_days=150,
            num_houses=165,
            num_zoonotic_foci=25,
            num_safe_zones=25,
            initial_infected_range=(args.initial_infected_min, args.initial_infected_max)
        )
        print(f"✓ Entorno inicializado correctamente (brote inicial aleatorio: "
              f"{args.initial_infected_min}-{args.initial_infected_max} casos por episodio)")
    else:
        env = EpidemicEnvironment(
            population_size=500,
            simulation_days=150,
            num_houses=165,
            num_zoonotic_foci=25,
            num_safe_zones=25,
            initial_infected=args.initial_infected
        )
        print(f"✓ Entorno inicializado correctamente (brote inicial fijo: {args.initial_infected} casos)")
    
    # Crear entrenador PPO
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
    
    if args.mode == 'train':
        print("\n" + "="*70)
        print("MODO: ENTRENAMIENTO DE POLÍTICA PPO")
        print("="*70)
        print(f"Parámetros de entrenamiento:")
        print(f"  - Brote inicial: {args.initial_infected} casos")
        print(f"  - Iteraciones: {args.iterations}")
        print(f"  - Pasos por iteración: 2048")
        print(f"  - Pasos totales aprox.: {args.iterations * 2048}")
        print(f"  - Tasa aprendizaje: 3e-4")
        print(f"  - Factor descuento γ: 0.99")
        print(f"  - Parámetro GAE λ: 0.95")
        print(f"  - Parámetro clipping ε: 0.2")
        print(f"  - Dispositivo: {device.upper()}")
        print("="*70)
        
        # Ejecutar entrenamiento
        trainer.train(num_iterations=args.iterations)
        
        # Guardar modelo
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Solo intenta crear la carpeta si hay una ruta especificada
            os.makedirs(save_dir, exist_ok=True)
        trainer.save_policy(args.save_path)
        
        # Prueba rápida del modelo entrenado
        print("\nEjecutando pruebas del modelo entrenado...")
        episode_rewards, episode_data = trainer.test_policy(num_episodes=args.episodes_test)
        
        # Guardar datos de prueba
        
        results_file = args.save_path.replace('.pt', '_results.npy')
        np.save(results_file, {
            'episode_rewards': episode_rewards,
            'episode_data': episode_data,
            'train_rewards': trainer.train_stats['episode_rewards'],
            'train_lengths': trainer.train_stats['episode_lengths'],
            # ¡Estas son las dos líneas nuevas que debes agregar!
            'actor_losses': trainer.train_stats['actor_losses'],
            'critic_losses': trainer.train_stats['critic_losses']
        }, allow_pickle=True)
        print(f"Resultados guardados en: {results_file}")

    elif args.mode == 'test':
        if args.checkpoint is None:
            print("Error: Debe especificar --checkpoint para modo prueba")
            sys.exit(1)
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Archivo {args.checkpoint} no encontrado")
            sys.exit(1)
        
        print("\n" + "="*70)
        print("MODO: PRUEBA DE POLÍTICA")
        print("="*70)
        print(f"Cargando modelo desde: {args.checkpoint}")
        
        trainer.load_policy(args.checkpoint)
        print("✓ Modelo cargado correctamente")
        
        # Ejecutar pruebas
        episode_rewards, episode_data = trainer.test_policy(
            num_episodes=args.episodes_test
        )
        
        print(f"\nRecompensa promedio: {np.mean(episode_rewards):.4f}")
        print(f"Desviación estándar: {np.std(episode_rewards):.4f}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
