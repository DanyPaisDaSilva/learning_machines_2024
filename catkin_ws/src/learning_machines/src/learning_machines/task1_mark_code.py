import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from matplotlib import pyplot as plt
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
from time import sleep


class RoboboEnv(gym.Env):
    def __init__(self, robobo: IRobobo, collision_threshold=1000):
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

        self.track_reward = []
        self.track_sensors = []

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

        return 0, 0

    def filter_hardware(self, sensor_data):
        max_counts = 0
        # high pass filter
        # backL
        if sensor_data[0] > 800:
            sensor_data[0] = 800
            max_counts += 1
        # backR
        if sensor_data[1] > 300:
            sensor_data[1] = 300
            max_counts += 1
        # frontL
        if sensor_data[2] > 800:
            sensor_data[2] = 800
            max_counts += 1
        # frontR
        if sensor_data[3] > 800:
            sensor_data[3] = 800
            max_counts += 1
        # frontC
        if sensor_data[4] > 400:
            sensor_data[4] = 400
            max_counts += 1
        # frontRR
        if sensor_data[5] > 500:
            sensor_data[5] = 500
            max_counts += 1
        # backC
        if sensor_data[6] > 800:
            sensor_data[6] = 800
            max_counts += 1
        # frontLL
        if sensor_data[7] > 500:
            sensor_data[6] = 500
            max_counts += 1

        # low pass filter
        # backL
        if sensor_data[0] < 15:
            sensor_data[0] = 0
        # backR
        if sensor_data[1] < 15:
            sensor_data[1] = 0
        # frontL
        if sensor_data[2] < 65:
            sensor_data[2] = 0
        # frontR
        if sensor_data[3] < 30:
            sensor_data[3] = 0
        # frontC
        if sensor_data[4] < 20:
            sensor_data[4] = 0
        # frontRR
        if sensor_data[5] < 20:
            sensor_data[5] = 0
        # backC
        if sensor_data[6] < 30:
            sensor_data[6] = 0
        # frontLL
        if sensor_data[7] < 15:
            sensor_data[6] = 0

        return  max_counts

    def filter_simulation(self, sensor_data):
        max_counts = 0
        # high pass filter
        # backL
        if sensor_data[0] > 800:
            sensor_data[0] = 800
            max_counts += 1
        # backR
        if sensor_data[1] > 300:
            sensor_data[1] = 300
            max_counts += 1
        # frontL
        if sensor_data[2] > 800:
            sensor_data[2] = 800
            max_counts += 1
        # frontR
        if sensor_data[3] > 800:
            sensor_data[3] = 800
            max_counts += 1
        # frontC
        if sensor_data[4] > 400:
            sensor_data[4] = 400
            max_counts += 1
        # frontRR
        if sensor_data[5] > 500:
            sensor_data[5] = 500
            max_counts += 1
        # backC
        if sensor_data[6] > 800:
            sensor_data[6] = 800
            max_counts += 1
        # frontLL
        if sensor_data[7] > 500:
            sensor_data[6] = 500
            max_counts += 1

        # low pass filter
        # backL
        if sensor_data[0] < 15:
            sensor_data[0] = 0
        # backR
        if sensor_data[1] < 15:
            sensor_data[1] = 0
        # frontL
        if sensor_data[2] < 65:
            sensor_data[2] = 0
        # frontR
        if sensor_data[3] < 30:
            sensor_data[3] = 0
        # frontC
        if sensor_data[4] < 20:
            sensor_data[4] = 0
        # frontRR
        if sensor_data[5] < 20:
            sensor_data[5] = 0
        # backC
        if sensor_data[6] < 30:
            sensor_data[6] = 0
        # frontLL
        if sensor_data[7] < 15:
            sensor_data[6] = 0

        return  max_counts

    def step(self, action):

        # Execute one time step within the environment
        left_motor, right_motor = self.translate_action(action)

        # execute the action
        blockid = self.robobo.move(int(100 * left_motor), int(100 * right_motor), 200)
        if blockid in self.robobo._used_pids: self.robobo._used_pids.remove(blockid)
        sleep(0.2)

        # TODO save this to the observation space, clip IR values to 0, 100
        # current implementation regards anything above 100 as identical to 100- something to think about
        # sensor_data = np.clip(self.robobo.read_irs(), 0, self.collision_threshold)
        sensor_data = np.array(self.robobo.read_irs(), np.float32)

        # TODO: different filters for each sensor (i.e. center sensor high value is different to LL sensor)
        if isinstance(self.robobo, SimulationRobobo):
            collision_counts = self.filter_simulation(sensor_data)
        else:
            collision_counts = self.filter_hardware(sensor_data)

        # Reward logic based on sensor data
        # reward low sensor data and fast movement
        # need to find a better way for both motor and sensor data computation

        forward_bonus = 5 if action == 1 else 1

        reward = ((abs(left_motor + right_motor) * forward_bonus * (
                1 - (np.max(sensor_data) / self.collision_threshold)))
                  - 10 * collision_counts
                  )

        print(f"ACTION {action},\nSENSOR DATA: {sensor_data},\n REWARD: {reward}")

        # TODO define termination condition- implement a timer for the simulator maybe
        done = False

        # plot stuff
        self.track_reward.append(reward)
        self.track_sensors.append(sensor_data)

        return np.array(sensor_data), reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.stop_simulation()


def run_task1(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    env = RoboboEnv(rob)

    n_actions = env.action_space.n
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    config_default = {
        "batch_size": 8,
        "buffer_size": 10000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.5,
        "gamma": 0.5,
        "gradient_steps": 4,
        "learning_rate": 0.001,
        "learning_starts": 150,
        "target_update_interval": 8,
        "train_freq": 4,
    }

    # Create the RL model
    model = DQN("MlpPolicy", env, verbose=1, **config_default)

    # Train the model
    model.learn(total_timesteps=200)

    # Save the model
    model.save(str(FIGRURES_DIR / "ddpg_robobo"+f"{time.time()}"))

    # Load the model
    model = DQN.load("ddpg_robobo")
    env.close()

    # plot the plots
    plot_sensor_data(env.track_sensors)
    plot_reward(env.track_reward)

    # except Exception as e:
    #    print(e)
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def plot_sensor_data(sensor_data_list):
    # Transpose the data to separate each sensor's readings
    data_transposed = list(zip(*sensor_data_list))

    # Sensor names
    sensors = ["BackL", "BackR", "FrontL", "FrontR", "FrontC", "FrontRR", "BackC", "FrontLL"]

    time_points = list(range(1, len(sensor_data_list) + 1))

    # Plotting the data
    plt.figure(figsize=(16, 8))
    for i, sensor in enumerate(sensors):
        plt.plot(time_points, data_transposed[i], label=sensor)

    plt.xlabel('Timestep')
    plt.ylabel('Sensor Readings')
    plt.title('Sensor Data Over Timesteps')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Show plot
    plt.savefig(str(FIGRURES_DIR / "sensor_data.png"))
    plt.show()


def plot_reward(reward_data_list):
    time_points = list(range(1, len(reward_data_list) + 1))

    # Plotting the data
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, reward_data_list)

    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title('Reward Data Over Timesteps')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Show plot
    plt.savefig(str(FIGRURES_DIR / "reward_data.png"))
    plt.show()
