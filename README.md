
# Files
- `Louise_NN1.py`, `Louise_NN2.py`, `Louise_CNN1.py`: completed online tutorial files
- `lab8_NN.py`, `lab8_CNN.py`: aurora image classification files
- `features.txt`: color histogram, required for input of `lab8_NN.py`

Inputs to `lab8_CNN.py` are the raw images, uploaded to Google Drive. Results of CNN that are named "reshape" are trained on color histograms instead of raw images so should be neglected. Only `CNN-epoch3.png` is trained on the raw images, although no logs are saved and only 3 epochs were run due to time constraint. For all visualizations, orange is training and blue is validation.

# Tensorflow

To run in tensorflow environment, run the following code: 

```
python3 -m venv . # create virtual environment in current directory
source bin/activate
pip install --upgrade pip
pip install tensorflow # set up tensorflow
python lab8.py # run code
```

# Models

NN model uses 2 fully-connected hidden layers of 1000 neurons, connected to 1 output neuron. Mean squared error was used as the loss function so that the output value is compared with the target value.

CNN model uses 3 layers of convolution and padding, followed by 2 layers of fully-connected neurons, reducing the flattened output of CNN to 64 neurons and then to 1 output neuron. The loss function is also mean squared error. 
