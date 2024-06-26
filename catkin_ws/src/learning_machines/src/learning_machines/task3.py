import gym
from gym import spaces
from stable_baselines3 import DQN
import numpy as np
from data_files import FIGURES_DIR, MODELS_DIR
import cv2
from matplotlib import pyplot as plt
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)
from datetime import datetime
from time import time

# RUN CONFIG PARAMETERS

load_model = False
load_and_train = False  # load_model has to be False
model_path = str(MODELS_DIR / "dqn_robobo_2024-06-21_10-56-04.zip")

print_output = False  # mostly for reward and action output
save_model = True


##################
# CV2 operations #
##################

def apply_mask(img, state="RED"):
    if state == "RED":
        # apply red mask
        return cv2.inRange(img, (45, 70, 70), (85, 255, 255))
    else:
        # apply green mask
        return cv2.inRange(img, (45, 70, 70), (85, 255, 255))


def set_resolution(img, resolution=32):
    # sim photo format is (512, 512)
    # irl photo format is (1536, 2048)
    return cv2.resize(img, (resolution, resolution))


def add_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype('uint8')
    noisy_image = cv2.add(img, gaussian_noise)
    return noisy_image


def process_image(img):
    return binarify_image(set_resolution(apply_morphology(apply_mask(img))))


def binarify_image(img):
    _, binary_image = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return (binary_image / 255).astype(np.uint8)


def process_image_w_noise(img):
    return binarify_image(apply_morphology(set_resolution(apply_morphology(apply_mask(add_noise(img))))))


def apply_morphology(image):
    # Step 2: Closing operation to fill small holes
    closing_kernel_size = 5  # Kernel size for closing
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, closing_kernel)

    opening_kernel_size = 3  # Small kernel size for opening
    opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, opening_kernel)

    # Perform the 'opening' operation, which is equivalent to erosion followed by dilation.
    return opened_image


####################
# static functions #
####################


# Dany reward function

def split_img_scores(img):
    # assuming img is processed and sides are equal length
    side_length = img.shape[0]
    split_length = side_length // 3

    # count pixels in
    split_L_score = np.sum(img[:, 0:split_length + 1])
    split_C_score = np.sum(img[:, split_length + 2:split_length * 2])
    split_R_score = np.sum(img[:, split_length * 2:side_length])

    # max count, with offset (because we assume that objects are rarely at top/bot
    max_C = (side_length - 10) * (split_length - 1)
    max_S = (side_length - 10) * split_length

    # max value +- 1.18
    return split_L_score / max_S, split_C_score / max_C, split_R_score / max_S


def calc_reward(img, action, timesteps=0):
    reward = 0

    split_L_score, split_C_score, split_R_score = split_img_scores(img)

    if split_L_score < split_C_score and split_R_score < split_C_score:
        # good if a lot of green in center
        reward = split_C_score + 2
        # reward going forward when a lot of green in center
        if action == 0:
            reward += 1
    elif split_R_score < split_L_score:
        reward = split_L_score + 0.5
        # reward turning in the right direction
        if action == 1:
            reward += 0.5
    elif split_L_score < split_R_score:
        reward = split_R_score + 0.5
        # reward turning in the right direction
        if action == 2:
            reward += 0.5
    # if they are equal (i.e. (0, 0, 0) )
    else:
        # reward turning if nothing can be seen
        if action == 1 or action == 2:
            reward = 0.1

    # punish going backwards only if camera detects something
    if action == 3 and not (split_L_score + split_C_score + split_R_score == 0):
        reward -= 1

    return reward


def get_reward(img, action=0, state="RED"):
    if state == "RED":
        return red_reward(img, action)
    else:
        return green_reward(img)


def red_reward(img, action):
    return calc_reward(img, action)


def green_reward(img):
    pass


# translates action to left and right movement
def translate_action(action):
    # move forward
    if action == 0:
        return 1, 1
    # turn 45 degrees left
    elif action == 1:
        return 0.5, -0.5
    # turn 45 degrees right
    elif action == 2:
        return -0.5, 0.5
    # if nothing works return stay still
    return 0, 0


class RoboboEnv(gym.Env):
    def __init__(self, robobo: IRobobo):
        super(RoboboEnv, self).__init__()
        self.robobo = robobo

        # Define action and observation space
        self.action_space = spaces.Discrete(3)

        # load example image
        setup_img = set_resolution(process_image(self.get_image()))

        # set low/high values using that example image
        low = np.zeros(setup_img.shape)
        high = np.ones(setup_img.shape)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)

        self.center_multiplier = 5
        # timesteps taken, 900 = 3 mins (with moving = 0.2)
        self.track_reward = []
        self.state = "RED"  # either "RED" or "GREEN"

    def get_image(self):
        img = self.robobo.get_image_front()
        # encode to hsv
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img_hsv

    def reset(self):
        # Reset the state of the environment to an initial state
        if isinstance(self.robobo, SimulationRobobo):
            # reset environment
            if not self.robobo.is_stopped():
                self.robobo.stop_simulation()
            self.robobo.play_simulation()

            # reset phone tilt & wheels
            self.robobo.set_phone_tilt(95, 50)
            self.robobo.reset_wheels()

            # TODO: check default position in arena_approach.ttt for
            #  position.x, position.y, position.z as well as for
            #  orientation.yaw, orientation.pitch, orientation.roll
            # self.robobo.set_position((0, 0, 0), (0, 0, 0))
            # also randomize food pos
        return set_resolution(process_image(self.get_image()))

    # Execute one time step within the environment
    def step(self, action):

        # translate action
        left_motor, right_motor = translate_action(action)

        # execute the action
        blockid = self.robobo.move(int(100 * left_motor), int(100 * right_motor), 200)
        if blockid in self.robobo._used_pids: self.robobo._used_pids.remove(blockid)

        # preprocess image
        image_masked = process_image(self.get_image())

        reward = get_reward(image_masked, action, self.state)

        self.track_reward.append(reward)

        if print_output:
            print(f"ACTION {action} with REWARD: {reward}")

        done = False
        # if all food collected
        if self.robobo.base_detects_food():
            done = True
            print("Food collected!")
        # if robot stuck / too much time passed (480s = 8 min) --> restart
        if isinstance(self.robobo, SimulationRobobo) and self.robobo.get_sim_time() > 480:
            done = True
            print("Ran out of time :(")

        return image_masked, reward, done, {}

    def close(self):
        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.stop_simulation()


def run_task2(rob: IRobobo):
    env = RoboboEnv(rob)

    config_default = {
        "batch_size": 64,
        "buffer_size": 10000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.25,
        "gamma": 0.95,
        "gradient_steps": 4,
        "learning_rate": 0.001,
        "learning_starts": 150,
        "target_update_interval": 8,
        "train_freq": 4,
    }

    if load_model:
        # Load the model
        model = DQN.load(model_path)
        model.set_env(env)

        # test the trained model
        obs = env.reset()
        print("START RUNNING LOADED MODEL")
        for _ in range(100000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
        print("FINISHED RUNNING LOADED MODEL")

    else:
        if load_and_train:
            print("LOADED MODEL")
            model = DQN.load(model_path)
            model.set_env(env)
        else:
            # Create the RL model
            model = DQN("MlpPolicy", env, verbose=1, **config_default)
        try:
            # Train the model
            print("TRAINING MODEL")
            start_time = time()
            model.learn(total_timesteps=100000)
            end_time = time()
            print(f"TRAINING MODEL FINISHED WITH RUNTIME: {end_time - start_time:.2f}s")
        except Exception as e:
            print(e)
        finally:
            # Save the model
            if save_model:
                print("SAVING MODEL")
                save_path = str(MODELS_DIR / f"dqn_robobo_t3_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
                model.save(save_path)
                print(f'MODEL SAVED UNDER {save_path}.zip')

    # close env
    env.close()