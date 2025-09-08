#from utils.data_saving import DataHandler
import xlsxwriter
import numpy as np
import csv



# my_data = DataHandler(f'{my_save_path}/data_handler_test')

# agent_location = np.array([1, 2])
# target_location = np.array([3, 4])
# elapsed_steps = 5

# my_data.save_data_state(elapsed_steps, agent_location, target_location)

# my_data.save_data_action(1, 2, 3, 4, 5, 0, 0)

# my_data.close()


save_path_experiment = '/home/m4/Documents'
with open(f'{save_path_experiment}/experiment_data.csv', 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["elapsed_steps", "state", "agent_location_x", "agent_location_y", "target_location_x", "target_location_y", "piezo", "vpp", "frequency", "reward", "cumulative_reward", "terminated", "truncated"])


    for i in range(220):
        # write random value 
        with open(f'{save_path_experiment}/experiment_data.csv', 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([i, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, True, True])
