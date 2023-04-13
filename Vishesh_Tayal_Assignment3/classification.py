import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
import glob
from tensorflow import keras
import matplotlib.image as mpimg
from keras import backend as K
from skimage import color
from skimage.measure import label, regionprops, regionprops_table
import shutil

img_width, img_height = 150, 150

temp_labels_1 = pd.read_csv("PatientData/Patient_1_Labels.csv")
condition = temp_labels_1["Label"] > 1
temp_labels_1.loc[condition, "Label"] = [1]

temp_labels_2 = pd.read_csv("PatientData/Patient_2_Labels.csv")
condition = temp_labels_2["Label"] > 1
temp_labels_2.loc[condition, "Label"] = [1]

temp_labels_3 = pd.read_csv("PatientData/Patient_3_Labels.csv")
condition = temp_labels_3["Label"] > 1
temp_labels_3.loc[condition, "Label"] = [1]

temp_labels_4 = pd.read_csv("PatientData/Patient_4_Labels.csv")
condition = temp_labels_4["Label"] > 1
temp_labels_4.loc[condition, "Label"] = [1]

temp_labels_5 = pd.read_csv("PatientData/Patient_5_Labels.csv")
condition = temp_labels_5["Label"] > 1
temp_labels_5.loc[condition, "Label"] = [1]

os.makedirs("Train/Resting")
os.makedirs("Train/Noise")
os.makedirs("Validation/Resting")
os.makedirs("Validation/Noise")

path1 = "Train/Noise"
path2 = "Train/Resting"
path3 = "Validation/Noise"
path4 = "Validation/Resting"

for file in glob.glob('PatientData/Patient_1/*_thresh.png', recursive=True):
    name = file.split("/")[2].split("_")[1]
    label_value = temp_labels_1.loc[temp_labels_1["IC"] == int(name), "Label"]
    img = cv2.imread(file)
    # img = slicing_contouring(file)
    if int(label_value) >= 1:
        cv2.imwrite(os.path.join(path2, name + "_1.png"), img)
    else:
        cv2.imwrite(os.path.join(path1, name + "_1.png"), img)

for file in glob.glob('PatientData/Patient_2/*_thresh.png', recursive=True):
    name = file.split("/")[2].split("_")[1]
    label_value = temp_labels_2.loc[temp_labels_2["IC"] == int(name), "Label"]
    img = cv2.imread(file)
    # img = slicing_contouring(file)
    if int(label_value) >= 1:
        cv2.imwrite(os.path.join(path2, name + "_2.png"), img)
    else:
        cv2.imwrite(os.path.join(path1, name + "_2.png"), img)

for file in glob.glob('PatientData/Patient_3/*_thresh.png', recursive=True):
    name = file.split("/")[2].split("_")[1]
    label_value = temp_labels_3.loc[temp_labels_3["IC"] == int(name), "Label"]
    img = cv2.imread(file)
    # img = slicing_contouring(file)
    if int(label_value) >= 1:
        cv2.imwrite(os.path.join(path2, name + "_3.png"), img)
    else:
        cv2.imwrite(os.path.join(path1, name + "_3.png"), img)

for file in glob.glob('PatientData/Patient_4/*_thresh.png', recursive=True):
    name = file.split("/")[2].split("_")[1]
    label_value = temp_labels_4.loc[temp_labels_4["IC"] == int(name), "Label"]
    img = cv2.imread(file)
    # img = slicing_contouring(file)
    if int(label_value) >= 1:
        cv2.imwrite(os.path.join(path2, name + "_4.png"), img)
    else:
        cv2.imwrite(os.path.join(path1, name + "_4.png"), img)

for file in glob.glob('PatientData/Patient_5/*_thresh.png', recursive=True):
    name = file.split("/")[2].split("_")[1]
    label_value = temp_labels_5.loc[temp_labels_5["IC"] == int(name), "Label"]
    img = cv2.imread(file)
    # img = slicing_contouring(file)
    if int(label_value) >= 1:
        cv2.imwrite(os.path.join(path4, name + "_5.png"), img)
    else:
        cv2.imwrite(os.path.join(path3, name + "_5.png"), img)

train_dir = "Train"
val_dir = "Validation"
train_samples = 0
validation_samples = 0

for root_dir, cur_dir, files in os.walk(r"Train"):
    train_samples += len(files)

for root_dir, cur_dir, files in os.walk(r"Validation"):
    validation_samples += len(files)

epochs = 25
batch_size = 32

print(train_samples, validation_samples)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# initializing the cnn
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name="precision")])

train_datagen = ImageDataGenerator(rescale=1 / 255.0)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=9,
    epochs=epochs,
    validation_data=validation_generator,
)

model.save('model_saved.h5')
