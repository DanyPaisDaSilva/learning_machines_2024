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


def do_task0(rob: IRobobo, duration=60000):
    '''
    does everything needed to demo task0
    :param rob: robobo object
    :param duration: in miliseconds
    :return: void
    '''

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    direction = 1  # 1 = go in front, -1 = go backwards, 0 = stop
    # loop over 0.5 sec (can be reduced)
    while duration > 0:
        sensor_data = rob.read_irs()
        new_dir = react_to_sensors(sensor_data)
        if new_dir is not None:
            direction = new_dir

        rob.move(100 * direction, 100 * direction, 500)
        duration -= 500

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def react_to_sensors(sensor_data):
    """
    return new direction value depending on the sensor data
    :param sensor_data: iris data
    :return: new direction
    """
    # sensor_data = [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
    # "enumerate" the data
    backL = sensor_data[0]
    backR = sensor_data[1]
    frontL = sensor_data[2]
    frontR = sensor_data[3]
    frontC = sensor_data[4]
    frontRR = sensor_data[5]
    backC = sensor_data[6]
    frontLL = sensor_data[7]

    if frontL > 80 or frontR > 80 or frontC > 100:  # or frontRR > 100 or frontLL > 100
        return -1
    if backL > 80 or backR > 80 or backC > 100:
        return 1
    return None
