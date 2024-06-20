import gym
from gym import spaces
from stable_baselines3 import DQN
import numpy as np
from data_files import FIGURES_DIR, MODELS_DIR
import cv2
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

# load a model file?
load_model = False
model_path = str(MODELS_DIR / "dqn_robobo_2024-06-18_14-07-29.zip")
print_output = True


##################
# CV2 operations #
##################

def apply_mask(img):
    # TODO: test if this is good for both sim and irl
    return cv2.inRange(img, (45, 70, 70), (85, 255, 255))


def set_resolution(img, resolution=64):
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

# mark reward function
def calculate_weighted_area_score(img, coefficient):
    height, width = img.shape
    center_width = width // 2

    # Define region of interest
    left_bound = int(center_width - 0.15 * width)
    right_bound = int(center_width + 0.15 * width)

    # Extract ROI
    roi = img[:, left_bound:right_bound]

    # Count white pixels in ROI
    white_pixels_on_center = cv2.countNonZero(roi)

    # Calculate effective area in the ROI
    effective_area_in_roi = white_pixels_on_center * coefficient

    # Calculate the remaining white pixels in the img
    white_pixels_off_center = cv2.countNonZero(img) - white_pixels_on_center

    # Step 4: Calculate the combined effective area
    combined_effective_area = white_pixels_off_center + effective_area_in_roi

    # Calculate the percentage of the effective area
    total_pixel_count = img.size  # Equivalent to height * width
    weighted_area_score = (combined_effective_area / total_pixel_count)

    return weighted_area_score


# Dany reward function

def split_img_scores(img):
    # assuming img is processed and sides are equal length
    side_length = img.shape[0]
    split_length = side_length // 3

    split_L_score = np.sum(img[:, 0:split_length+1])
    split_C_score = np.sum(img[:, split_length+2:split_length * 2])
    split_R_score = np.sum(img[:, split_length * 2:side_length])

    max_C = (side_length - 10) * (split_length-1)
    max_S = (side_length - 10) * split_length

    return split_L_score/max_S, split_C_score/max_C, split_R_score/max_S


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
    # move backward
    elif action == 3:
        return -1, -1
    return 0, 0


class RoboboEnv(gym.Env):
    def __init__(self, robobo: IRobobo):
        super(RoboboEnv, self).__init__()
        self.robobo = robobo

        # Define action and observation space
        self.action_space = spaces.Discrete(4)

        # load example image
        setup_img = set_resolution(process_image(self.get_image()))

        # set low/high values using that example image
        low = np.zeros(setup_img.shape)
        high = np.ones(setup_img.shape)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)

        self.center_multiplier = 5
        # TODO: change irl phone position/tilt so that it looks forward

    def get_image(self):
        img = self.robobo.get_image_front()
        # encode to hsv
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img_hsv

    def reset(self):
        # Reset the state of the environment to an initial state
        if isinstance(self.robobo, SimulationRobobo):
            if not self.robobo.is_stopped():
                self.robobo.stop_simulation()
            self.robobo.play_simulation()

            print(f"tilt :{self.robobo.read_phone_tilt()}")
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

        # calculates the weighted area of "food" on camera to determine reward.
        # low: 0, high: 22 for coefficient=5 (don't ask about the numbers)
        weighted_area_score = calculate_weighted_area_score(image_masked, self.center_multiplier)

        # test camera
        #if weighted_area_score > 0:
        #    cv2.imwrite(str(FIGURES_DIR / f"test_img_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpeg"),
        #                image_masked * 255)

        # reward logic based on camera data:
        #   1) how centered is green
        #   2) how much green do we see
        # reward speed
        # reward moving forward if we see a lot of green
        # turning if no box

        # reward function: change in area from previous state.
        reward = weighted_area_score

        if print_output: print(f"ACTION {action} with REWARD: {reward}")

        done = False
        if isinstance(self.robobo, SimulationRobobo) and self.robobo.nr_food_collected() >= 7:
            done = True
            print("Collected all the food!")


        return image_masked, reward, done, {}

    def close(self):
        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.stop_simulation()


def run_task2(rob: IRobobo):
    env = RoboboEnv(rob)

    config_default = {
        "batch_size": 8,
        "buffer_size": 10000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.5,
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
    else:
        # Create the RL model
        model = DQN("MlpPolicy", env, verbose=1, **config_default)

    # Train the model
    start_time = time()
    model.learn(total_timesteps=2000)
    end_time = time()
    print(f"{end_time - start_time:.2f}")

    # Save the model
    save_path = str(MODELS_DIR / f"dqn_robobo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    model.save(save_path)
    print(f'model saved under {save_path}.zip')

    # close env
    env.close()

    """
    # load the model
    loaded_model = PPO.load("ppo_cartpole")

    # test the trained model
    obs = env.reset()
    for _ in range(1000):
        action, _states = loaded_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    """
