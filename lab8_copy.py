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
'''
# part 4
unknown_features=[]
directory = './not_known'
for filename in os.listdir(directory):
    img_filename = os.path.join(directory, filename)
    hist = encode_img(img_filename)
    hist = np.true_divide(hist-np.min(hist), np.max(hist)-np.min(hist)) # minmax normalize histogram
    hist = np.append(hist, "#" + img_filename)
    unknown_features.append(list(hist))
unknown_features = np.asarray(unknown_features)


W = np.random.uniform(-1, 1, 769)
W, accuracies, train_errors = perceptron_train(features_train, W, 50, 0.001, n_epoch)

def perceptron_test(features, W):
    rows = features.shape[0]

    names = features[:, -1]
    X = np.delete(features, -1,1)# remove name column
    X0 = np.ones(shape=(rows,1))
    X = np.append(X0, X, axis=1) # add X0 column
    X = X.astype(np.float) # cast as float

    activation = np.dot(X, W)
    prediction = np.sign(activation)
    compare = np.stack((prediction, names), axis=1) # return prediction and name col
    return compare


compare = perceptron_test(unknown_features, W)
np.savetxt("unknown_results.txt", compare, delimiter=",", fmt='%s')
'''
