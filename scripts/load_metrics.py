import json
import matplotlib.pyplot as plt

file_path = '/home/m4/logdir/run18/metrics.jsonl'

model_loss_mean_values = []  # This list will store the "train/model_loss_mean" values
steps = []

with open(file_path, 'r') as file:
    for line in file:
        json_data = json.loads(line)
        if 'train/model_loss_mean' in json_data:
            model_loss_mean_value = json_data['train/model_loss_mean']
            model_loss_mean_values.append(model_loss_mean_value)
            step = json_data['step']
            steps.append(step)
            

# Create an incremental index as the x-axis values
model_loss_mean_values = model_loss_mean_values[3:]
steps = steps[3:]
#time_values = list(range(4, len(model_loss_mean_values) + 4))

plt.plot(steps, model_loss_mean_values, marker='o', linestyle='-', color='b')
plt.title('Train Model Loss Mean Over Time')
plt.xlabel('Steps')
plt.ylabel('Model Loss Mean Value')
plt.grid(True)
plt.savefig('../results/run_18.png')
plt.show()
