import matplotlib.pyplot as plt 
from pick import pick
import os 
import json

models_dirs = []

for model in os.listdir('.'):
    if os.path.isdir(model) and not model.startswith('.'):
        models_dirs.append(model)
assert(len(models_dirs) >= 1)

model_dir, _ = pick(models_dirs, "Which model you want to visualize?",)

history_files =  []

for file in os.listdir(model_dir):
    if file.endswith('.json'):
        history_files.append(file)

# If no history files this means you need to train the model
assert(len(history_files) >= 1)

history_file, _ = pick(history_files, "Which history you want to visualize?")

history_file = os.path.join(model_dir, history_file)

history = json.loads(open(history_file, 'r').read())

plt.plot(history['loss'], label='Loss')
if history.get('val_loss'):
    plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()