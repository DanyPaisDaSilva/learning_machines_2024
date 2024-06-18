"""
NOTES FOR TASK 2
# Randomize the object generation (through lua)
# Randomize the starting point OR 1000 steps in one arena, then 1000 in another arena, etc etc

# Reality vs Simulation
Add noise to the simulated camera to help with reality adjustment
Randomize the camera orientation slightly

# Reward function:
The center of the object should be as close as possible.
-> what if there are two objects? steer to the closest one (largest area)

Multiplying the different objectives: robot pays equal attention to both
Adding the different objectives: robot can do one or the other independently

## What do we reward?
- Finding new greens
- The DIFFERENCE in green area, vs one frame and the next



# Image processing:
Use OpenCV
Normalize image dimensions!
Filter for green, use HSV
Try to find the center of the object


TASK FLOWCHART
- Search for objects
- Move to the closest object (the closest being the largest sized mask)
-- Left area vs right area: You want to move towards the largest one...
- Collide with it! (move forward until it disappears)
- If you do not see green, reward the robot for turning (UNTIL green is seen, then punish staying in place)
"""
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from matplotlib import pyplot as plt
from data_files import FIGRURES_DIR
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

# CV2 operations

def apply_mask(img):
    # TODO: test if this is good for both sim and irl
    return cv2.inRange(img, (45, 70, 70), (85, 255, 255))


def set_resolution(img):
    # TODO: check irl resolution of camera
    # sim photo format is (512, 512)
    return cv2.resize(img, (128, 128))


def add_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype('uint8')
    noisy_image = cv2.add(img, gaussian_noise)
    return noisy_image


# other ideas:
# blurring/smoothing
# contour/edge detection

def process_image(img):
    return set_resolution(apply_mask(img))

def process_image_w_noise(img):
    return set_resolution(apply_mask(add_noise(img)))

def process_image_full(img):
    # apply morphology imported from image processing test- order or morphology and resolution???
    return set_resolution(apply_morphology(apply_mask(img)))

def calculate_weighted_area_score(mask, coefficient):
    height, width = mask.shape
    center_width = width // 2

    # Define region of interest
    left_bound = int(center_width - 0.15 * width)
    right_bound = int(center_width + 0.15 * width)

    # Extract ROI
    roi = mask[:, left_bound:right_bound]

    # Count white pixels in ROI
    white_pixels_on_center = cv2.countNonZero(roi)

    # Calculate effective area in the ROI
    effective_area_in_roi = white_pixels_on_center * coefficient

    # Calculate the remaining white pixels in the mask
    white_pixels_off_center = cv2.countNonZero(mask) - white_pixels_on_center

    # Step 4: Calculate the combined effective area
    combined_effective_area = white_pixels_off_center + effective_area_in_roi

    # Calculate the percentage of the effective area
    total_pixel_count = mask.size  # Equivalent to height * width
    weighted_area_score = (combined_effective_area / total_pixel_count) * 100 * 10

    return weighted_area_score

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

class RoboboEnv(gym.Env):
    def __init__(self, robobo: IRobobo):
        super(RoboboEnv, self).__init__()
        self.robobo = robobo

        # Define action and observation space
        self.action_space = spaces.Discrete(4)

        # load example image
        setup_img = process_image(self.get_image())

        # down sample image to 32x32 so that we aren't absolutely destroyed by high dimensionality
        low = np.zeros((32, 32))
        high = np.ones((32, 32))
        self.observation_space = spaces.Box(low=low, high=high, dtype=int)

        self.center_multiplier = 5
        self.old_reward = 0
        # TODO: change phone position/tilt so that it looks forward

    def get_image(self):
        img = self.robobo.get_image_front()
        # encode to hsv
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img_hsv
    def reset(self):
        # Reset the state of the environment to an initial state
        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.reset_wheels()
            # TODO: check default position in arena_approach.ttt for
            #  position.x, position.y, position.z as well as for
            #  orientation.yaw, orientation.pitch, orientation.roll
            #self.robobo.set_position((0,0,0), (0,0,0))
            # also randomize food pos

    # TODO I don't want backward as an option for this task.
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

    def step(self, action):
        # Execute one time step within the environment
        left_motor, right_motor = self.translate_action(action)

        # execute the action
        blockid = self.robobo.move(int(100 * left_motor), int(100 * right_motor), 200)
        if blockid in self.robobo._used_pids: self.robobo._used_pids.remove(blockid)

        # gets the down sampled (128x128), masked version of the image.
        image_masked = process_image_full(self.get_image())

        # calculates the weighted area of "food" on camera to determine reward.
        # low: 0, high: 22 for coefficient=5 (don't ask about the numbers)
        weighted_area_score = calculate_weighted_area_score(image_masked, self.center_multiplier)

        # reward logic based on camera data:
        #   1) how centered is green
        #   2) how much green do we see
        # reward speed
        # reward moving forward if we see a lot of green
        # turning if no box

        # reward function: change in area from previous state.
        # TODO add a BIG reward if the robot collides with a food object
        reward = weighted_area_score - self.old_reward
        self.old_reward = reward

        print(f"ACTION {action} with REWARD: {reward}")

        # TODO define termination condition- implement a timer for the simulator maybe
        done = False

        return image_masked, reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        if isinstance(self.robobo, SimulationRobobo):
            self.robobo.stop_simulation()


def run_task2(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    env = RoboboEnv(rob)

    n_actions = env.action_space.n

    # TODO: change this?
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

    # Create the RL model
    model = DQN("MlpPolicy", env, verbose=1, **config_default)

    # Train the model
    model.learn(total_timesteps=200)

    # Save the model
    # model.save(str(FIGRURES_DIR / f"ddpg_robobo_{time()}"))

    # Load the model
    # model = DQN.load("ddpg_robobo")
    env.close()

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
