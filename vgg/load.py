import numpy as np 
import tensorflow as tf 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from urllib.request import urlopen
import cv2
import matplotlib.pyplot as plt 

model = VGG16()
print(model.summary())

image_url = None
while not image_url:
    image_input = input('Enter Image Url: \n')
    if image_input:
        image_url = image_input

def read_image_from_url(url):
    img_array = None
    with urlopen(url) as request:
        img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return  cv2.resize(img, (224, 224))

image = read_image_from_url(image_input)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# # prepare the image for the VGG model
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
