import time

import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)
from task1-testing-rl import RoboboEnv

# Create the environment
robobo = SimulationRobobo()  # or HardwareRobobo()
env = RoboboEnv(robobo)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create the RL model
model = DDPG("MlpPolicy", env, action_noise, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ddpg_robobo"+f"{time.time()}")

# Load the model
model = DDPG.load("ddpg_robobo")