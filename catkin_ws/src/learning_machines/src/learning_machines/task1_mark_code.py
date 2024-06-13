import time

import gym
from gym import spaces
from stable_baselines3 import DQN
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


class RoboboEnv(gym.Env):
    def __init__(self, robobo: IRobobo, collision_threshold=400):
        super(RoboboEnv, self).__init__()
        self.robobo = robobo

        # parameters
        self.collision_threshold = collision_threshold

        # Define action and observation space
        # Actions: 0 (forward), 1 (backward), 2 (turn 45 left), 3 (turn 45 right)
        self.action_space = spaces.Discrete(4)

        # Observation: First 2 are the motor speeds, the other 8 are IR sensor readings
        # CHECK THE ORDER OF THE SENSORS!
        # Define the low and high arrays: motor speed range is [-1, 1]; IR is [0, 100]
        low = np.array([0] * 8, dtype=np.float32)
        high = np.array([self.collision_threshold] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        # Reset the state of the environment to an initial state
        # Implement any reset logic if required
        sensor_data = self.robobo.read_irs()
        return np.array(sensor_data)

    def translate_action(self, action):
        # move backward
        if action == 0:
            return -1, -1
        # move forward
        elif action == 1:
            return 1, 1
        # turn 45 degrees left
        elif action == 2:
            return 0.5, -0.5
        # turn 45 degrees right
        elif action == 3:
            return -0.5, 0.5

    def step(self, action):
        # Execute one time step within the environment

        left_motor, right_motor = self.translate_action(action)

        # execute the action
        self.robobo.move(100 * left_motor, 100 * right_motor, 200)

        # TODO save this to the observation space, clip IR values to 0, 100
        # current implementation regards anything above 100 as identical to 100- something to think about
        # sensor_data = np.clip(self.robobo.read_irs(), 0, self.collision_threshold)
        sensor_data = np.array(self.robobo.read_irs(), np.float32)

        # TODO: different filters for each sensor (i.e. center sensor high value is different to LL sensor)
        sensor_data[sensor_data > self.collision_threshold] = self.collision_threshold
        sensor_data[sensor_data < 10] = 0

        print(sensor_data)

        # Reward logic based on sensor data
        # reward low sensor data and fast movement
        # need to find a better way for both motor and sensor data computation
        forward_bonus = 2 if action == 1 else 0
        reward = ((abs(left_motor + right_motor) * (1 - (np.max(sensor_data) / self.collision_threshold)))
                  - 20 * sum(sensor_data[sensor_data == self.collision_threshold])
                  + forward_bonus)

        # TODO define termination condition- implement a timer for the simulator maybe
        done = False

        return np.array(sensor_data), reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        # removed this because it shouldnt be the env stopping the task
        pass


def run_task1(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    try:
        env = RoboboEnv(rob)

        n_actions = env.action_space.n
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Create the RL model
        model = DQN("MlpPolicy", env, verbose=1)

        # Train the model
        model.learn(total_timesteps=10000)

        # Save the model
        model.save(str(FIGRURES_DIR / "ddpg_robobo"+f"{time.time()}"))

        # Load the model
        model = DQN.load("ddpg_robobo")

    # except Exception as e:
    #    print(e)

    finally:
        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()
