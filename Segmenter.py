import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
from keras.utils import to_categorical

# Define the paths to the dataset and image set files
dataset_dir = 'C:/Users/61493/Documents/Dev/Datasets/VOCdevkit/VOC2012'
train_file_path = 'C:/Users/61493/Documents/Dev/Datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
val_file_path = 'C:/Users/61493/Documents/Dev/Datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
image_dir = os.path.join(dataset_dir, 'JPEGImages')
mask_dir = os.path.join(dataset_dir, 'SegmentationClass')
save_model_path = 'model.ckpt'

color_map = [
    (0, 0, 0),         # background
    (128, 0, 0),       # aeroplane
    (0, 128, 0),       # bicycle
    (128, 128, 0),     # bird
    (0, 0, 128),       # boat
    (128, 0, 128),     # bottle
    (0, 128, 128),     # bus
    (128, 128, 128),   # car
    (64, 0, 0),        # cat
    (192, 0, 0),       # chair
    (64, 128, 0),      # cow
    (192, 128, 0),     # dining table
    (64, 0, 128),      # dog
    (192, 0, 128),     # horse
    (64, 128, 128),    # motorbike
    (192, 128, 128),   # person
    (0, 64, 0),        # potted plant
    (128, 64, 0),      # sheep
    (0, 192, 0),       # sofa
    (128, 192, 0),     # train
    (0, 64, 128),      # tv/monitor
]
# Load and preprocess the image and mask
def load_image(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    return image

def color_to_label(pixel, color_map):
    for i, color in enumerate(color_map):
        if np.array_equal(pixel, color):
            return i
    return 0

def load_mask(mask_path, color_map):
    mask = load_img(mask_path, target_size=(128,128))
    mask = img_to_array(mask)
    mask_label = np.zeros((128, 128), dtype=np.int)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask_label[i, j] = color_to_label(mask[i, j, :], color_map)
    return mask_label

# Load and preprocess the dataset
def load_dataset(image_set_file):
    images = []
    masks = []
    with open(image_set_file, 'r') as file:
        for line in file:
            filename = line.strip()
            image_path = os.path.join(image_dir, filename + '.jpg')
            mask_path = os.path.join(mask_dir, filename + '.png')
            image = load_image(image_path)           
            mask = load_mask(mask_path,color_map)          
            images.append(image)
            masks.append(mask)
    images = np.array(images)
    masks = np.array(masks)
    return images, masks

import tensorflow as tf

# Build UNet++ model
def unet_plusplus(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bridge
    conv_bridge = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)

    # Decoder
    upconv4 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv_bridge)
    concat4 = tf.keras.layers.concatenate([conv4, upconv4], axis=-1)
    upconv3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(concat4)
    concat3 = tf.keras.layers.concatenate([conv3, upconv3], axis=-1)
    upconv2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(concat3)
    concat2 = tf.keras.layers.concatenate([conv2, upconv2], axis=-1)
    upconv1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(concat2)

    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(upconv1)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# Specify input shape and number of classes
input_shape = (128, 128, 3)
num_classes = 21

# Create UNet++ model
model = unet_plusplus(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load training and validation datasets
train_images, train_masks = load_dataset(train_file_path)
val_images, val_masks = load_dataset(val_file_path)
train_masks = np.expand_dims(train_masks, axis=-1)
val_masks = np.expand_dims(val_masks, axis=-1)

resized_train_masks = np.zeros((train_masks.shape[0], input_shape[0], input_shape[1], num_classes), dtype=np.float32)
resized_val_masks = np.zeros((val_masks.shape[0], input_shape[0], input_shape[1], num_classes), dtype=np.float32)

for i in range(train_masks.shape[0]):
    train_mask = train_masks[i, :, :, 0]
    resized_train_mask = resize(train_mask, (input_shape[0], input_shape[1]), order=0, anti_aliasing=False)
    resized_train_mask = resized_train_mask.astype(int)
    resized_train_mask = to_categorical(resized_train_mask, num_classes=num_classes)
    resized_train_masks[i, :, :, :] = resized_train_mask

for i in range(val_masks.shape[0]):
    val_mask = val_masks[i, :, :, 0]
    resized_val_mask = resize(val_mask, (input_shape[0], input_shape[1]), order=0, anti_aliasing=False)
    resized_val_mask = resized_val_mask.astype(int)
    resized_val_mask = to_categorical(resized_val_mask, num_classes=num_classes)
    resized_val_masks[i, :, :, :] = resized_val_mask

# Train the model and save it
model.fit(train_images, resized_train_masks, validation_data=(val_images, resized_val_masks), epochs=10, batch_size=8)

# Save the trained model
model.save('Seg_model.h5')