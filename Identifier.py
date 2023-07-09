import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

# define some constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 21
LR = 1e-3

# define the path to your dataset
DATASET_PATH = "C:\\Users\\61493\\Documents\\Dev\\Datasets\\Animals\\animals\\animals"


# create an instance of ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2, # use 20% of the data for validation
)

# prepare iterators
train_it = datagen.flow_from_directory(
    DATASET_PATH,
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
)

val_it = datagen.flow_from_directory(
    DATASET_PATH,
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
)
# get class labels for the training dataset
train_labels = train_it.classes

# compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels),y= train_labels)
class_weights = dict(enumerate(class_weights))

# define the model
baseModel =ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(91, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become the actual model to train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# compile the model
opt = Adam(learning_rate=LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)  # stop training when the validation loss does not improve for 3 consecutive epochs
lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)  # reduce learning rate by a factor of 0.2 if the validation loss does not improve for 3 consecutive epochs

# train the model
model.fit(train_it, steps_per_epoch=train_it.samples//BATCH_SIZE, validation_data=val_it, validation_steps=val_it.samples//BATCH_SIZE, epochs=EPOCHS,callbacks=[early_stop, lr_decay],class_weight=class_weights)

# unfreeze the baseModel layers
for layer in baseModel.layers:
    layer.trainable = True

# recompile the model with a lower learning rate
opt_finetune = Adam(learning_rate=LR/10)
model.compile(loss="categorical_crossentropy", optimizer=opt_finetune, metrics=["accuracy"])

# fine-tune the model
model.fit(train_it, steps_per_epoch=train_it.samples//BATCH_SIZE, validation_data=val_it, validation_steps=val_it.samples//BATCH_SIZE, epochs=7, callbacks=[early_stop, lr_decay],class_weight=class_weights)

# save the model
model.save("animal_classifier_model.h5")
