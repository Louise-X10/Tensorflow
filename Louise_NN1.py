#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:58:02 2022

@author: liuyilouise.xu
"""

from tensorflow.keras import datasets

import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# normalize data
train_images, test_images = train_images / 255.0, test_images / 255.0

# NN model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# NN model with softmax on output
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

np.argmax(predictions[0])