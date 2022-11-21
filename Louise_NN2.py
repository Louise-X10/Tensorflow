#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:37:53 2022

@author: liuyilouise.xu
"""

from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]

# Python optimisation variables
epochs = 10
batch_size = 100

# normalize the input images by dividing by 255.0
x_train = x_train / 255.0
x_test = x_test / 255.0

# convert x_test to tensor to pass through model 
# (train data will be converted to tensors on the fly)
x_test = tf.Variable(x_test)

# declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random.normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random.normal([10]), name='b2')

# set up feedforward loop
def nn_model(x_input, W1, b1, W2, b2):
    # flatten the input image from 28 x 28 to 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, W2), b2)
    return logits

def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                           logits=logits))
    return cross_entropy

# set up optimizer
optimizer = tf.keras.optimizers.Adam()


for epoch in range(epochs):
    batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
    # create tensors
    batch_x = tf.Variable(batch_x)
    batch_y = tf.Variable(batch_y)
    # create a one hot vector
    batch_y = tf.one_hot(batch_y, 10)
    
    # set up context for computing gradient
    with tf.GradientTape() as tape:
        logits = nn_model(batch_x, W1, b1, W2, b2)
        loss = loss_fn(logits, batch_y)
    
    # compute gradient
    gradients = tape.gradient(loss, [W1, b1, W2, b2])
    
    # perform backward propogation
    optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
    
    test_logits = nn_model(x_test, W1, b1, W2, b2)
    max_idxs = tf.argmax(test_logits, axis=1)
    test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
    
    print(f"Epoch: {epoch + 1}, loss={loss:.3f}, test set      accuracy={test_acc*100:.3f}%")

print("\nTraining complete!")