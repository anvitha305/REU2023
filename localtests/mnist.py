import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
import keras

# modules for plotting in this notebook
import numpy as np
import matplotlib.pyplot as plt
import random

# load MNIST data for the training and test images and labels
# [need them to evaluate the model]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# import the MNIST model to evaluate it on a local machine
model = keras.models.load_model("mnist.keras")

# get loss and accuracy statistics
loss, acc = new_model.evaluate(x_test, y_test, verbose=2)

# predict values to build a confusion matrix
y_pred = model.predict(x_test)

# predictions and ground-truth to one-hot vectors
y_pred_classes = np.argmax(Y_pred,axis = 1)
y_true = np.argmax(y_test,axis = 1)

# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes) 

# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g')
