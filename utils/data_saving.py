import xlsxwriter
import os
import time
import cv2
import csv
import numpy as np


class DataHandler_Excel():
    
    def __init__(self, save_path):
            current_time = time.strftime("%Y%m%d-%H%M%S")
            experiment_path = os.path.join(save_path, f"experiment/{current_time}")
            os.makedirs(experiment_path, exist_ok=True)

            self.save_path = experiment_path
            self.workbook = xlsxwriter.Workbook(f'{self.save_path}/experiment_data.xlsx')
            self.worksheet = self.workbook.add_worksheet('Experiment data')
            self.worksheet.write('A1', 'Step')
            self.worksheet.write('B1', 'Piezo')
            self.worksheet.write('C1', 'Vpp')
            self.worksheet.write('D1', 'Frequency')
            self.worksheet.write('E1', 'Agent location x')
            self.worksheet.write('F1', 'Agent location y')
            self.worksheet.write('G1', 'Target location x')
            self.worksheet.write('H1', 'Target location y')
            self.worksheet.write('I1', 'Reward')
            self.worksheet.write('J1', 'Cumulative reward')
            self.worksheet.write('K1', 'Terminated')
            self.worksheet.write('L1', 'Truncated')     
            self.rowIndex = 2
            

    # TODO: Ensure that this is in sync with the spreadsheet
    def save_image(self, image):
        cv2.imwrite(f'{self.save_path}/image_' + str(self.rowIndex)+'.png', image)

    def save_initial_image(self, image):
        cv2.imwrite(f'{self.save_path}/image_1.png', image)
    
    def save_data_state(self,  elapsed_steps, agent_location, target_location):
        self.worksheet.write(f'A{self.rowIndex}', elapsed_steps)
        self.worksheet.write(f'E{self.rowIndex}', agent_location[0])
        self.worksheet.write(f'F{self.rowIndex}', agent_location[1])
        self.worksheet.write(f'G{self.rowIndex}', target_location[0])
        self.worksheet.write(f'H{self.rowIndex}', target_location[1])

    def save_data_action(self, piezo, vpp, freq, reward, cumulative_reward, terminated, truncated):
        self.worksheet.write(f'B{self.rowIndex}', piezo)
        self.worksheet.write(f'C{self.rowIndex}', vpp)
        self.worksheet.write(f'D{self.rowIndex}', freq)
        self.worksheet.write(f'I{self.rowIndex}', reward)
        self.worksheet.write(f'J{self.rowIndex}', cumulative_reward)
        self.worksheet.write(f'K{self.rowIndex}', terminated)
        self.worksheet.write(f'L{self.rowIndex}', truncated)
         
        self.rowIndex += 1

    def close(self):
        self.workbook.close()


if __name__ == "__main__":
    save_path_experiment = '/home/m4/Documents'
    with open(f'{save_path_experiment}/experiment_data.csv', 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["elapsed_steps", "state", "agent_location_x", "agent_location_y", "target_location_x", "target_location_y", "piezo", "vpp", "frequency", "reward", "cumulative_reward", "terminated", "truncated"])

    

    for i in range(220):
        # write random value 
        writer.writerow([i, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, False, False])