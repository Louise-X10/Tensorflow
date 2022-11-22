import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import datetime


# print("TensorFlow version:", tf.__version__)

# Define hyper parameters
n_epoch = 100
batch_size = 100

data_dir = './images'
img_height = 600
img_width = 514
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Begin CNN model!")
model2 = models.Sequential([
    layers.Rescaling(1./255), 
    layers.Conv2D(32, (3, 3), activation='relu', name="conv1"),
    layers.MaxPooling2D((2, 2), name="pool1"),
    layers.Conv2D(64, (3, 3), activation='relu', name="conv2"),
    layers.MaxPooling2D((2, 2), name="pool2"),
    layers.Conv2D(64, (3, 3), activation='relu', name="conv3"),
    layers.MaxPooling2D((2, 2), name="pool3"),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])


model2.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

model2.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
'''
model2.summary()

log_dir = "logs/cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model2.fit(train_imgs, train_lbs, epochs=n_epoch, batch_size = batch_size,
          validation_data=(test_imgs, test_lbs),
          callbacks=[tensorboard_callback],
          verbose = 2)

test_loss2, test_acc2 = model2.evaluate(features_test, labels_test, verbose=2)

'''