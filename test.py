from ContagionRL.environment import SIRSDEnvironment
import gymnasium as gym
import ContagionRL.registerSIRSD  # Register the environment

# Test basic environment creation
env = SIRSDEnvironment()
print('Environment created successfully!')
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)