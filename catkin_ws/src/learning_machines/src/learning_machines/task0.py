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

import pandas as pd
import matplotlib.pyplot as plt
import time

def do_task0(rob: IRobobo, duration=10000):
    '''
    does everything needed to demo task0
    :param rob: robobo object
    :param duration: in miliseconds
    :return: void
    '''

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    direction = 1  # 1 = go in front, -1 = go backwards, 0 = stop

    # Lists to store data
    timestamps = []
    sensor_data_list = []
    direction_list = []

    # loop over 0.5 sec (can be reduced)
    start_time = time.time()
    while duration > 0:
        rob.reset_wheels()
        rob.move(100 * direction, 100 * direction, 500)
        sensor_data = rob.read_irs()
        new_dir = react_to_sensors(sensor_data)
        if new_dir is not None:
            print("Change direction")
            direction = new_dir
        rob.sleep(0.5)
        duration -= 1000

        # Collect data
        timestamps.append(time.time() - start_time)
        sensor_data_list.append(sensor_data)
        direction_list.append(direction)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    # Plot results
    plot_data(timestamps, sensor_data_list, direction_list)

def plot_data(timestamps, sensor_data_list, direction_list):
    # Convert collected data to DataFrame
    data = {
        'time': timestamps,
        'sensor_data': sensor_data_list,
        'direction': direction_list
    }
    df = pd.DataFrame(data)

    # Plotting sensor data and direction against time
    fig, ax1 = plt.subplots()

    # Plotting sensor data (assuming we plot the sum of all sensor readings)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Sensor Data', color='tab:blue')
    ax1.plot(df['time'], df['sensor_data'].apply(sum), color='tab:blue', label='Sensor Data')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Creating a second y-axis for direction
    ax2 = ax1.twinx()
    ax2.set_ylabel('Direction', color='tab:red')
    ax2.plot(df['time'], df['direction'], color='tab:red', label='Direction')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and legend
    plt.title('Sensor Data and Direction Over Time')
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

    # Show plot
    plt.show()

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

    if frontL > 100 or frontR > 100 or frontC > 100: # or frontRR > 100 or frontLL > 100
        return -1
    if backL > 100 or backR > 100 or backC > 100:
        return 1
    return None
