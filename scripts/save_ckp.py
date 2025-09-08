import os
import sys
import glob
from datetime import datetime
import time

# SAVE_DIRECTORY = "/home/m4/logdir/16_actions_racetrack_real_vpp_from_area_02"
SAVE_DIRECTORY = "/home/mahmoud/logdir/mixed_10_envs_no_flow_by_the_wall"
BACKUP_DIRECTORY = SAVE_DIRECTORY+"/checkpoints_backup"
CHECKPOINT_FILE = "checkpoint.ckpt"
TIMEDELTA = 180*60 # 360 minutes  

def main():
    init_time = time.time()
    while True:
        curr_time = time.time()
        curr_ckp = SAVE_DIRECTORY+"/"+CHECKPOINT_FILE
        if os.path.exists(curr_ckp):
            curr_ckp_time = os.path.getctime(curr_ckp)
            
            if not os.path.exists(BACKUP_DIRECTORY):
                os.makedirs(BACKUP_DIRECTORY)
            
            saved_ckp = glob.glob(BACKUP_DIRECTORY+"/**/*.ckpt")
            if len(saved_ckp) == 0:
                print('Backing up checkpoint file')
                dir_name = datetime.fromtimestamp(curr_ckp_time).strftime("%d-%m-%Y_%H-%M-%S")
                os.makedirs(f"{BACKUP_DIRECTORY}/Backup_{dir_name}")
                os.system(f"cp {curr_ckp} {BACKUP_DIRECTORY}/Backup_{dir_name}")
                print(f"Backup of {curr_ckp} created successfully")
            else:
                latest_ckp = max(saved_ckp, key=os.path.getctime)
                latest_ckp_time = os.path.getctime(latest_ckp)

                if curr_time - latest_ckp_time > TIMEDELTA and curr_ckp_time > latest_ckp_time:
                    print('Backing up checkpoint file')
                    # dir_name = time.strftime("%d-%m-%Y_%H-%M-%S")
                    dir_name = datetime.fromtimestamp(curr_ckp_time).strftime("%d-%m-%Y_%H-%M-%S")
                    os.makedirs(f"{BACKUP_DIRECTORY}/Backup_{dir_name}")
                    os.system(f"cp {curr_ckp} {BACKUP_DIRECTORY}/Backup_{dir_name}")
                    print(f"Backup of {curr_ckp} created successfully")
                else:
                    print(f'Last backup was {(curr_time - latest_ckp_time)/60} minutes ago, at {datetime.fromtimestamp(latest_ckp_time).strftime("%S-%M-%H %d-%m-%Y ")}')
                    print(f'Last checkpoint was {(curr_time - curr_ckp_time)/60} minutes ago, at {datetime.fromtimestamp(curr_ckp_time).strftime("%S-%M-%H %d-%m-%Y ")}')
                    print(f'No need to backup checkpoint file\n')
        time.sleep(15)

if __name__ == "__main__":
    main()