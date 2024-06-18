#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, do_task0, run_task1, run_task2


if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # Testing
    # run_all_actions(rob)

    # TASK 0
    # print("Start")
    # do_task0(rob)
    # print("Finish")

    # TASK 1
    print("Start")
    run_task2(rob)
    print("Finish")
