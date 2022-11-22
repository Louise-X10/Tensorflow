#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:13:40 2022

@author: liuyilouise.xu
"""

import matplotlib.pyplot as plt

train_acc = [0.5306, 0.8433, 0.9258]
val_acc = [0.7648, 0.9192, 0.9477]

train_loss = [0246.8056, 0.1587, 0.0846]
val_loss= [0.2116, 0.1030, 0.0627]

epochs = [1, 2, 3]


plt.title("Accuracy plot")
plt.plot(epochs, val_acc, label="validate")
plt.plot(epochs, train_acc, label="train")
plt.legend(loc='upper center')
plt.show()

plt.title("Error loss plot")
plt.plot(epochs, train_loss, label="validate")
plt.plot(epochs, val_loss, label="train")
plt.legend(loc='upper center')