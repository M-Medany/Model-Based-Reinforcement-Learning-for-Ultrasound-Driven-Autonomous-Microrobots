import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# # Load data into a list
# data_list = [
#     np.load('/home/m4/logdir/run18/replay/20231025T190204F870668-6SGPrNMrBLExrij3opjEWi-0000000000000000000000-200.npz'),
#     np.load('/home/m4/logdir/run18/replay/20231025T190204F870668-6SGPrNMrBLExrij3opjEWi-70t9Xs2a2y1k7K3Twd7PNR-1024.npz'),
#     np.load('/home/m4/logdir/run18/replay/20231025T190705F689200-70t9Xs2a2y1k7K3Twd7PNR-3jXWm8ruKpdQurklMENRZi-1024.npz'),
#     np.load('/home/m4/logdir/run18/replay/20231025T191013F573798-3jXWm8ruKpdQurklMENRZi-53Rq1oQ3eb2A8eplvAVkBx-1024.npz'),
#     np.load('/home/m4/logdir/run18/replay/20231025T191321F771452-53Rq1oQ3eb2A8eplvAVkBx-6BI9nVlpktHxuCuuzd8IhP-1024.npz'),
#     np.load('/home/m4/logdir/run18/replay/20231024T175738F185850-1YvcgqKLLEZZhCRpkeWfzH-06yEHBjtFjYVXXOtxTxUJC-1024.npz'),
#     np.load('/home/m4/logdir/run18/replay/20231024T174859F104522-055hK5CfmWDYg2nf7uV9XE-53Gc9HYtTMEMY6MN6HAkyF-1024.npz'),
#     np.load('/home/m4/logdir/run18/replay/20231024T174859F104522-055hK5CfmWDYg2nf7uV9XE-53Gc9HYtTMEMY6MN6HAkyF-1024.npz')
# ]

def load_npz_files(directory):
    data_list = []
    
    # List all files in the directory
    files = os.listdir(directory)
    
    for file in files:
        file_path = os.path.join(directory, file)
        loaded_data = np.load(file_path)
        data_list.append(loaded_data)
    
    return data_list

# Directory path where .npz files are located
directory_path = '/home/m4/logdir/run18/replay'

# Call the function to load the .npz files
data_list = load_npz_files(directory_path)

exclude_file = '20231025T190705F689200-70t9Xs2a2y1k7K3Twd7PNR-3jXWm8ruKpdQurklMENRZi-1024.npz'

filtered_list = data_list





# Initialize an empty list to store distances for each data file
distances_list = []

# Initialize an empty list to store rewards for each data file
rewards_list = []

# Iterate over each data file and calculate distances and rewards
for data in data_list:
    agent_position = data["agent_position"]
    target_position = data['target_position']
    reward_data = data['reward']

    distances = []
    rewards = []

    # Calculate distances and append them to the list
    for agent_pos, target_pos in zip(agent_position[3:], target_position[3:]):
        distance = np.linalg.norm(agent_pos - target_pos)
        distances.append(distance)

    # Append rewards to the list
    cleaned_rewards = reward_data[3:]
    rewards = list(cleaned_rewards)

    distances_list.append(distances)
    rewards_list.append(rewards)

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

# Plot individual distances
for i, distances in enumerate(distances_list):
    ax1.plot(distances, label=f'Data {i+1}')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Distance')
ax1.set_title('Distance between Agent and Target')
ax1.legend()

# Plot mean distances
mean_distances = [np.mean(distances) for distances in distances_list]
ax2.plot(mean_distances, marker='o')
ax2.set_xlabel('Data Set')
ax2.set_ylabel('Mean Distance')
ax2.set_title('Mean Distance between Agent and Target')
ax2.set_xticks(range(len(mean_distances)))
ax2.set_xticklabels([f'Data {i+1}' for i in range(len(mean_distances))])

# Plot mean rewards
mean_rewards = [np.mean(rewards) for rewards in rewards_list]
ax3.plot(mean_rewards, marker='o', color='orange')
ax3.set_xlabel('Data Set')
ax3.set_ylabel('Mean Reward')
ax3.set_title('Mean Reward')
ax3.set_xticks(range(len(mean_rewards)))
ax3.set_xticklabels([f'Data {i+1}' for i in range(len(mean_rewards))])

plt.tight_layout()
plt.show()