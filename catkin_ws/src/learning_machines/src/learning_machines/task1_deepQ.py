import numpy as np
import random
import tensorflow as tf
from collections import deque
import gym  # Using OpenAI Gym for environment




# actions
# move r,l,f,f_short,f_long,b(?)

# states
# sensory data

# model (including init)
# step function: read_s --> action --> read_s
# reward function (reward low sensory data) + (speed) + (punish collision)

# Optimizer does the learning
# replay buffer add later
