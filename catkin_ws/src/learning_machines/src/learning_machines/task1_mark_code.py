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


class RoboboEnv(gym.Env):
    def __init__(self, robobo: IRobobo, collision_threshold=1000):
        super(RoboboEnv, self).__init__()
        self.robobo = robobo

        # parameters
        self.collision_threshold = collision_threshold

        # Define action and observation space
        # Actions: 0 (forward), 1 (backward), 2 (turn 45 left), 3 (turn 45 right)
        self.action_space = spaces.Discrete(5)

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

    def translate_action_bad(self, action):
        # move forward
        if action == 0:
            return 1, 1
        # turn 45 degrees left
        elif action == 1:
            return 0.5, -0.5
        # turn 45 degrees right
        elif action == 2:
            return -0.5, 0.5
        return 1, 1

    def translate_action_test(self, action):
        # move forward
        if action == 0:
            return -1, -1
        elif action == 0:
            return 1, 1
        # turn 45 degrees left
        elif action == 1:
            return 0.5, 0
        # turn 45 degrees right
        elif action == 2:
            return 0, 0.5
        elif action == 3:
            return -0.5, 0
        # turn 45 degrees right
        elif action == 4:
            return 0, -0.5
        return 0, 0

    def step(self, action):

        # Execute one time step within the environment
        left_motor, right_motor = self.translate_action_test(action)

        # execute the action
        blockid = self.robobo.move(100 * left_motor, 100 * right_motor, 200)
        self.robobo._used_pids.remove(blockid)

        # TODO save this to the observation space, clip IR values to 0, 100
        # current implementation regards anything above 100 as identical to 100- something to think about
        # sensor_data = np.clip(self.robobo.read_irs(), 0, self.collision_threshold)
        sensor_data = np.array(self.robobo.read_irs(), np.float32)

        # TODO: different filters for each sensor (i.e. center sensor high value is different to LL sensor)
        sensor_data[sensor_data > self.collision_threshold] = self.collision_threshold
        sensor_data[sensor_data < 10] = 0

        # Reward logic based on sensor data
        # reward low sensor data and fast movement
        # need to find a better way for both motor and sensor data computation
        forward_bonus = 2 if action == 1 else 0
        reward = ((abs(left_motor + right_motor) * (1 - (np.max(sensor_data) / self.collision_threshold)))
                  - 2 * len(sensor_data[sensor_data == self.collision_threshold])
                  + forward_bonus)

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
        # removed this because it shouldnt be the env stopping the task
        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.stop_simulation()


def run_task1(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

        env = RoboboEnv(rob)

        n_actions = env.action_space.n
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        """
        Config for when we need it later
        config = {
            "batch_size": 32,
            "buffer_size": 10000,
            "exploration_final_eps": 0.02,
            "exploration_fraction": 0.1,
            "gamma": 0.99,
            "gradient_steps": 4,
            "learning_rate": 1e-4,
            "learning_starts": 10000,
            "target_update_interval": 1000,
            "train_freq": 4,
        }
        """

        # Create the RL model
        model = DQN("MlpPolicy", env, verbose=1)

        # Train the model
        model.learn(total_timesteps=2000)

        # Save the model
        # model.save(str(FIGRURES_DIR / "ddpg_robobo"+f"{time.time()}"))

        # Load the model
        # model = DQN.load("ddpg_robobo")
        env.close()
        print("is done?")

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
    plt.figure(figsize=(16, 12))
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
