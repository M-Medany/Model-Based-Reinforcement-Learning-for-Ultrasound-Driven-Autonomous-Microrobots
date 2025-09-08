import os
import cv2
import glob
from tqdm import tqdm
from multiprocessing import Pool

# # parent_path = '/media/m4/Mahmoud_T7_Touch/Dreamer_Real_v2/32_actions_racetrack_real_mlp_keys_02/experiment_piezo_fixed_20240204-193332_run_0/original_data/'
# parent_path = '/media/m4/Mahmoud_T7_Touch/Dreamer_Real_v2/32_actions_racetrack_real_mlp_keys_02/experiment_piezo_fixed_20240205-100418_run_0/original_data/'
# parent_path2 = '/media/m4/Mahmoud_T7_Touch/Dreamer_Real_v2/32_actions_racetrack_real_mlp_keys_02/experiment_piezo_fixed_20240205-140720_run_0/original_data/'
# # parent_path3 = '/media/m4/Mahmoud_T7_Touch/Dreamer_Real_v2/32_actions_racetrack_real_mlp_keys_02/experiment_piezo_fixed_20240205-170016_run_0/original_data/'

# paths = os.listdir(parent_path)

# for path in paths:
#     if "jpeg" not in path:
#         path = parent_path + path
#         orig = cv2.imread(path)
#         cv2.imwrite(f'{path[:-3]}jpeg', orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
#         os.remove(path)
# /media/m4/Mahmoud_T7_Touch/Dreamer_Real_v2/**/original_data/*.png
experiment_paths = glob.glob("/media/m4/Mahmoud_T7_Touch/Dreamer_Real_v2/**/original_data/*.png")

# experiment_paths = glob.glob("/media/m4/Mahmoud_BACK_UP/Backup_DL_hard_drive/Dreamer_v2_backup_before_friday/**/original_data/*.png")

# for path in tqdm(experiment_paths, total=len(experiment_paths)):
#     orig = cv2.imread(path)
#     cv2.imwrite(f'{path[:-3]}jpeg', orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
#     os.remove(path)

def converting(path):
    orig = cv2.imread(path)
    cv2.imwrite(f'{path[:-3]}jpeg', orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
    os.remove(path)
    
if __name__ == "__main__":
    with Pool(processes=16) as pool:
        pool.map(converting, experiment_paths)