import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
import json
import os

print('Load Data...')
(X_train, y_train), (X_test, y_test) = load_data()
print('Data', X_train.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0

print('Build the model..')
# example of a 3-block vgg style architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy, 
    optimizer='adam', 
    metrics=['accuracy']
)

print(model.summary())

print('Start Traning..')

epochs = None 
batch_size = None
verbose = None
while not epochs:
    epochs_input = input('Num. of Epochs\n')
    if epochs_input:
        epochs = int(epochs_input)

while not batch_size:
    batch_input = input('Num. of batches\n')
    if batch_input:
        batch_size = int(batch_input)

while not verbose:
    verbose_input = input('Verbose\n')
    if verbose_input:
        verbose = int(verbose_input)

history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=epochs, batch_size=batch_size, 
    verbose=verbose
).history

# write the data to json file 
LOGS = './logs'
if not os.path.isdir(LOGS):
    os.mkdir(LOGS)

log_file = os.path.join(LOGS, f"history.json")
with open(log_file, 'w') as f:
    f.write(json.dumps(history))
# Save the model
model.save('model.h5')