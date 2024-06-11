import gym
from gym import spaces
import numpy as np
from robobo_interface import IRobobo, SimulationRobobo
class RoboboEnv(gym.Env):
    def __init__(self, robobo: IRobobo):
        super(RoboboEnv, self).__init__()
        self.robobo = robobo

        # Define action and observation space
        # Actions: 0 (stop), 1 (forward), 2 (backward), 3 (turn 45 left), 4 (turn 45 right)
        self.action_space = spaces.Discrete(5)

        # Observation: First 2 are the motor speeds, the other 8 are IR sensor readings
        # CHECK THE ORDER OF THE SENSORS!
        # Define the low and high arrays: motor speed range is [-1, 1]; IR is [0, 100]
        low = np.array([-1, -1] + [0] * 8, dtype=np.float32)
        high = np.array([1, 1] + [100] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.play_simulation()
    def reset(self):
        # Reset the state of the environment to an initial state
        # Implement any reset logic if required
        sensor_data = self.robobo.read_irs()
        return np.array(sensor_data)

    def step(self, action):
        # Execute one time step within the environment

        left_motor = 0
        right_motor = 0
        # TODO maybe do this check in a method for better readability?
        # stop
        if action == 0:
            left_motor = 0
            right_motor = 0
        # move forward
        elif action == 1:
            left_motor = 1
            right_motor = 1
        # move backward
        elif action == -1:
            left_motor = -1
            right_motor = -1
        # turn 45 degrees left
        elif action == 2:
            left_motor = 0.5
            right_motor = -0.5
        # turn 45 degrees right
        elif action == 3:
            left_motor = -0.5
            right_motor = 0.5

        # execute the action
        self.robobo.move(100 * left_motor, 100 * right_motor, 200)

        # TODO save this to the observation space, clip IR values to 0, 100
        # current implementation regards anything above 100 as identical to 100- something to think about
        sensor_data = np.clip(self.robobo.read_irs(), 0, 100)

        # Reward logic based on sensor data
        # reward low sensor data and fast movement
        # need to find a better way for both motor and sensor data computation
        reward = abs(left_motor+right_motor) * (1 - (np.avg(sensor_data)/100))

        # collision punishment, I'm assuming 100 is collision but this is MOST LIKELY not true.
        if any(data>99 for data in sensor_data):
            reward = -10

        # TODO define termination condition- implement a timer for the simulator maybe
        done = False

        return np.concatenate((np.array([left_motor, right_motor]), np.array(sensor_data))), reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.stop_simulation()