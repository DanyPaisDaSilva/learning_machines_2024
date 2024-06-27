import gym
from gym import spaces
from stable_baselines3 import DQN
import numpy as np
import cv2
from data_files import FIGURES_DIR, MODELS_DIR
from matplotlib import pyplot as plt
import traceback
from datetime import datetime
from time import time
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


##################
# CV2 operations #
##################

def apply_mask(img, state="RED"):
    if state == "RED":
        # apply red mask
        return apply_red_mask(img)
    else:
        # apply green mask
        return cv2.inRange(img, (45, 70, 70), (85, 255, 255))


def apply_red_mask(img):
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(img, lower_red, upper_red)
    mask2 = cv2.inRange(img, lower_red2, upper_red2)

    return mask1 + mask2


def set_resolution(img, resolution=16):
    # sim photo format is (512, 512)
    # irl photo format is (1536, 2048)
    return cv2.resize(img, (resolution, resolution))


def max_pooling(img, kernel_size=(16, 16)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def binarify_image(img):
    _, binary_image = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return (binary_image / 255).astype(np.uint8)


def apply_morphology(image):
    closing_kernel_size = 5
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, closing_kernel)

    opening_kernel_size = 3
    opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, opening_kernel)

    return opened_image


def process_image(img):
    return binarify_image(set_resolution(max_pooling(apply_mask(img))))
    # return binarify_image(set_resolution(max_pooling((apply_morphology(apply_mask(img))))))


# RUN CONFIG PARAMETERS

load_model = False
load_and_train = False  # load_model has to be False
model_path = str(MODELS_DIR / "dqn_robobo_2024-06-21_10-56-04.zip")

print_output = True  # mostly for reward and action output
save_model = True


####################
# static functions #
####################


# Dany reward function

def split_img_scores(img):
    # assuming img is processed and sides are equal length
    side_length = img.shape[0]
    split_length = side_length // 3
    remainder = img.shape[0] % 3

    # count pixels in
    split_L_score = np.sum(img[:, 0:split_length])
    split_C_score = np.sum(img[:, split_length:split_length * 2 + remainder])
    split_R_score = np.sum(img[:, split_length * 2 + remainder:side_length])

    # max count, with offset (because we assume that objects are rarely at top/bot
    max_value = side_length * side_length

    # a bit more lenient on center, because it's more important
    return split_L_score / max_value, split_C_score / (max_value - 5), split_R_score / max_value


def get_reward(img, action, food_base_distance=0):
    reward = 0
    red_c_center = False

    split_L_score, split_C_score, split_R_score = split_img_scores(img)

    if split_L_score < split_C_score and split_R_score < split_C_score:
        # good if a lot of green in center
        reward = split_C_score + 3
        # reward going forward when a lot of green in center
        if action == 0:
            reward += 2
        red_c_center = True
    elif split_R_score < split_L_score:
        reward = split_L_score + 0.5
        # reward turning in the right direction
        if action == 1:
            reward += 1
    elif split_L_score < split_R_score:
        reward = split_R_score + 0.5
        # reward turning in the right direction
        if action == 2:
            reward += 1
    # if they are equal (i.e. (0, 0, 0) )
    else:
        # reward turning if nothing can be seen
        if action == 1 or action == 2:
            reward = 0.1

    reward += food_base_distance

    return reward, red_c_center


# translates action to left and right movement
def translate_action(action):
    # move forward
    if action == 0:
        return 1, 1
    # turn 45 degrees left
    elif action == 1:
        return 0.5, 0
    # turn 45 degrees right
    elif action == 2:
        return 0, 0.5
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

        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)
        self.observation_space = spaces.Dict(
            {
                "mask": spaces.Box(low=low, high=high, dtype=np.uint8),
                "red_or_green": spaces.Discrete(2)
            }
        )

        self.center_multiplier = 5
        self.track_reward = []
        self.state = "RED"  # either "RED" or "GREEN"
        self.red_c_history = [0] * 10  # gives history of last 0.2*size (0.2*10 = 2s)

    def red_hist_insert(self, red_c_state):
        self.red_c_history.pop()
        if red_c_state:
            self.red_c_history.insert(0, 1)
        else:
            self.red_c_history.insert(0, 0)

    def check_grabbing(self):
        # robobo.read_iris()[4] = FrontC sensor data
        front_c_sensor_data = self.robobo.read_iris()[4]

        if isinstance(self.robobo, SimulationRobobo):
            treshold = 20  # min 34
        else:
            treshold = 50  # min 66

        if front_c_sensor_data > treshold:
            # calc avg of red_c_history
            if sum(self.red_c_history) / len(self.red_c_history) > 0:
                if self.state == "RED":
                    if print_output: print("Changed to GREEN state")
                    self.state = "GREEN"
                # if false, keep GREEN state
            else:
                if print_output: print("Changed to RED state")
                self.state = "RED"
        else:
            if self.state == "GREEN":
                if print_output: print("Changed to RED state")
                self.state = "RED"
            # if false, keep RED state

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
            self.robobo.set_phone_tilt(105, 50)
            self.robobo.reset_wheels()

        self.state = "RED"
        self.red_c_history = [0]
        return {"mask": binarify_image(process_image(self.get_image())), "red_or_green": 0}

    # Execute one time step within the environment
    def step(self, action):

        # translate action
        left_motor, right_motor = translate_action(action)

        # execute the action
        blockid = self.robobo.move(int(100 * left_motor), int(100 * right_motor), 200)
        if blockid in self.robobo._used_pids: self.robobo._used_pids.remove(blockid)

        # preprocess image
        image_masked = process_image(self.get_image())

        red_to_green_distance = 0
        if isinstance(self.robobo, SimulationRobobo):
            if self.state == "RED":
                red_to_green_distance = -1
            else:
                red_to_green_distance = np.linalg.norm(np.array(self.robobo.base_position())-np.array(self.robobo.get_position()))
            red_to_green_distance = self.robobo._base_food_distance()
            print(f"red_to_green_distance: {red_to_green_distance}")

        reward, red_c_state = get_reward(image_masked, action, red_to_green_distance)

        # update red history
        self.red_hist_insert(red_c_state)
        # check grab
        self.check_grabbing()

        # cam test
        # if reward > 0:
        #    cv2.imwrite(str(FIGURES_DIR / f"test_img_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpeg"),
        #                image_masked * 255)

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

        return {"mask": image_masked, "red_or_green": 1 if self.state == "RED" else 0}, reward, done, {}

    def close(self):
        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.stop_simulation()


def run_task3(rob: IRobobo):
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
            model = DQN("MultiInputPolicy", env, verbose=1, **config_default)
        try:
            # Train the model
            print("TRAINING MODEL")
            start_time = time()
            model.learn(total_timesteps=2000)
            end_time = time()
            print(f"TRAINING MODEL FINISHED WITH RUNTIME: {end_time - start_time:.2f}s")
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # Save the model
            if save_model:
                print("SAVING MODEL")
                save_path = str(MODELS_DIR / f"dqn_robobo_t3_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
                model.save(save_path)
                print(f'MODEL SAVED UNDER {save_path}.zip')

    # close env
    env.close()


# big image testing
"""
if reward > 0:
    size = 64
    state = "RED"
    image1 = self.robobo.get_image_front()
    image2 = set_resolution(binarify_image(apply_mask(self.get_image(), state)) * 255, size)
    image3 = set_resolution(max_pooling(binarify_image(apply_mask(self.get_image(), state))) * 255, size)
    # Concatenate the images horizontally
    combined_image = cv2.hconcat([image2, image3])
    cv2.imwrite(str(FIGURES_DIR / f"test_img_2{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpeg"),
                image1)
    cv2.imwrite(str(FIGURES_DIR / f"test_img_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpeg"),
                combined_image)
"""
