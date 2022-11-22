import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import datetime


# print("TensorFlow version:", tf.__version__)

# Define hyper parameters
n_epoch = 1000
batch_size = 100

# Data prepocessing

# load normalized color histograms with labels
features = pd.read_csv("features.txt", header=None)
features = features.iloc[:, 0:-1] # remove filename column
labels = features.iloc[:,0]
features = features.iloc[:,1:]

features_train, features_test, labels_train, labels_test  = train_test_split(features, labels, test_size = 0.2)


# NN model
print("Begin NN model!")
model = tf.keras.Sequential([
    layers.Dense(1000, activation='sigmoid', input_shape=(768,), name="hidden_layer1"),
    layers.Dense(1000, activation='sigmoid', name="hidden_layer2"),
    layers.Dense(1, name="final")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

# tensorboard
log_dir = "logs/nn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(features_train, labels_train, epochs=n_epoch, batch_size = batch_size,
          validation_data=(features_test, labels_test),
          callbacks=[tensorboard_callback],
          verbose = 2)

test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)