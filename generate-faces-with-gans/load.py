import numpy as np 
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt 

datasets = tfds.list_builders()
ds = tfds.load('celeb_a')
print(ds)

_, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.flatten()
for i in range(len(axes)):
    axes[i].imshow(ds['image'][i])
    axes[i].axis('off')

plt.savefig('sample.png')