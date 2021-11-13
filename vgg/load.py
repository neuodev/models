import numpy as np 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = VGG16()
print(model.summary())

image = load_img('mug.jpg', target_size=(224, 224))
image = img_to_array(image)

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the VGG model
image = preprocess_input(image)

# Making predictions
yhat = model.predict(image)
print('Yhat: ', yhat)

# Labels 
label = decode_predictions(yhat)
print('decode_predictions: ', label)
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))