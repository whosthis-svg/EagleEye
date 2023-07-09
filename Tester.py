import os
import numpy as np
from keras.utils import  load_img,img_to_array
from keras.models import load_model
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from PIL import Image

# DATASET_PATH = "C:\\Users\\61493\\Documents\\Dev\\Datasets\\Animals\\animals\\animals"

# # Load the model
# model = load_model('animal_classifier_model.h5')  # replace with the path to your saved model

# # Get the class names
# class_names = os.listdir(DATASET_PATH)

# # Load an image file to test, resizing it to the original model image size
# img_path = ''  # replace with the path to the image file you want to test
# img = load_img(img_path, target_size=(224, 224))  # replace 224, 224 with the image size your model was trained on

# # Convert the image data to a numpy array suitable for keras
# img_array = img_to_array(img)
# img_batch = np.expand_dims(img_array, axis=0)

# # Normalize the image data to the scale used in training
# img_batch /= 255.0

# # Run the image through the model and get the index of the highest score
# predictions = model.predict(img_batch)
# class_index = np.argmax(predictions)

# # Get the name of the class with the highest score
# class_name = class_names[class_index]
# confidence = np.max(predictions)

# print(f"Prediction: {class_name}")
# print(f"Confidence: {confidence}")


# Load the model
model = load_model('unetplusplus_model.h5')  # replace with the path to your saved model

# Get the class names


# Load an image file to test, resizing it to the original model image size
img_path = 'HD3.jpg'  # replace with the path to the image file you want to test
img = load_img(img_path, target_size=(128, 128))  # replace 224, 224 with the image size your model was trained on

# Convert the image data to a numpy array suitable for keras
img_array = img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# Normalize the image data to the scale used in training
img_batch /= 255.0

# Run the image through the model and get the index of the highest score
predictions = model.predict(img_batch)
class_index = np.argmax(predictions)

# Get the name of the class with the highest score

confidence = np.max(predictions)

print(f"Prediction: {class_index}")
print(f"Confidence: {confidence}")


# color_map = [
#     (0, 0, 0),         # background
#     (128, 0, 0),       # aeroplane
#     (0, 128, 0),       # bicycle
#     (128, 128, 0),     # bird
#     (0, 0, 128),       # boat
#     (128, 0, 128),     # bottle
#     (0, 128, 128),     # bus
#     (128, 128, 128),   # car
#     (64, 0, 0),        # cat
#     (192, 0, 0),       # chair
#     (64, 128, 0),      # cow
#     (192, 128, 0),     # dining table
#     (64, 0, 128),      # dog
#     (192, 0, 128),     # horse
#     (64, 128, 128),    # motorbike
#     (192, 128, 128),   # person
#     (0, 64, 0),        # potted plant
#     (128, 64, 0),      # sheep
#     (0, 192, 0),       # sofa
#     (128, 192, 0),     # train
#     (0, 64, 128),      # tv/monitor
# ]
