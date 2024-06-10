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


def run_task1(rob: IRobobo, duration=30000):
    '''
    does everything needed to demo task0
    :param rob: robobo object
    :param duration: in miliseconds
    :return: void
    '''

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    direction = 1  # 1 = go in front, -1 = go backwards, 0 = stop
    # loop over 0.2 sec (can be reduced)
    while duration > 0:
        if isinstance(rob, SimulationRobobo):
            rob.reset_wheels()
        # read sensory data
        sensor_da   ta = rob.read_irs()
        # preprocess it

        #
        duration -= 200

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

