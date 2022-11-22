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


# CNN model
print("Begin CNN model!")
model2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(768, 0, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2))
    ])
model.summary()

# add NN model
model2.add(layers.Flatten())
model2.add(layers.Dense(64, activation='relu'))
model2.add(layers.Dense(1))

model2.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])


log_dir = "logs/cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model2.fit(features_train, labels_train, epochs=n_epoch, batch_size = batch_size,
          validation_data=(features_test, labels_test),
          callbacks=[tensorboard_callback],
          verbose = 2)

test_loss2, test_acc2 = model2.evaluate(features_test, labels_test, verbose=2)

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

