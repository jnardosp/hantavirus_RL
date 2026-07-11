from torch import nn

# we created a conversion from day estimations to simulation-step values
from rate_conversion import estimate_steps_per_day, day_to_step_prob

grid_size = 50

steps_per_day = estimate_steps_per_day(grid_size)

# Environment Parameters
env_config = {
    "simulation_time": 512,        
    "grid_size": grid_size,               
    "n_humans": 40,                
    "n_infected": 10,              
    "beta": day_to_step_prob(0.3, steps_per_day),   # Transmission rate during contact median reproductive number (R) was 2.12                
    "initial_agent_adherence": 0.05,  
    "distance_decay": 0.15,    
    "lethality": day_to_step_prob(0.32, steps_per_day),    # case fatality rate (CFR) of 32.35% (other reports between 20-50%)
    "immunity_loss_prob": 0.25,    # no evidence suggesting short-term reinfection.
    "recovery_rate": day_to_step_prob(0.06, steps_per_day),    # acute phase and resolution clears within 14 to 20 days post-prodrome
    "adherence_penalty_factor": 1, 
    "adherence_effectiveness": 0.2,         
    "movement_type": "continuous_random",  
    "movement_scale": 1,          
    "visibility_radius": -1,               
    "reinfection_count": 0,        
    "safe_distance": 3,    # aerosolized and droplet boundaries dictate this strict spatial threshold                    
    "init_agent_distance": 5,      
    "max_distance_for_beta_calculation": 3,    # match safe_distance
    "reward_type": "potential_field", 
    "reward_ablation": "full",              
    "render_mode": None           
}

# PPO Hyperparameters
ppo_config = {
    "policy_type": "MultiInputPolicy",
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[256, 256, 256, 256],  
            vf=[256, 256, 256, 256]  
        ),
        activation_fn=nn.ReLU,  
        ortho_init=True,        
    ),
    "batch_size": 2048,          
    "n_steps": 1024,             
    "n_epochs": 5,                
    "learning_rate": 3e-4,        
    "gamma": 0.96,             
    "gae_lambda": 0.95,         
    "target_kl": 0.04,           
    "clip_range": 0.2,            
    "ent_coef": 0.02,             
    "normalize_advantage": True,   
    "total_timesteps": 500000,
    "n_envs": 4                   
}

# Logging and Saving and evals 
save_config = {
    "base_log_path": "logs",
    "save_freq": 250_000,          
    "save_replay_buffer": True,
    "verbose": 1,
    "eval_freq": 250_000,          
} 