import numpy as np 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt 

datasets = tfds.list_builders()
ds = tfds.load('cats_vs_dogs', download=True)
ds = tfds.as_numpy(ds)
print(ds)
_, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.flatten()

i = 0
for example in ds['train']:
    axes[i].imshow(example['image'], cmap='gray_r')
    axes[i].set_title(example['label'])
    axes[i].axis('off')
    i+=1

    if i == 9:
        break

plt.savefig('sample.png')

