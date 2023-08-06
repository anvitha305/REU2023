import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
import keras
import seaborn as sns
# modules for plotting
import numpy as np
import matplotlib.pyplot as plt
import random
from pylab import savefig


# import the MNIST model to evaluate it on a local machine
model = keras.models.load_model("mnist.keras")

# get loss and accuracy statistics
loss, acc = model.evaluate(x_test, y_test, verbose=2)
with open("mnist_stats.txt", "w+") as f:
    f.write("Loss: " + str(loss)+"\n" + "Accuracy: "+str(acc))
# predict values to build a confusion matrix
y_pred = model.predict(x_test)

# predictions and ground-truth to one-hot vectors
y_pred_classes = np.argmax(y_pred,axis = 1)
y_true = np.argmax(y_test,axis = 1)



# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes) 

# plot confusion matrix
plt.figure(figsize=(10, 8))
mn = sns.heatmap(confusion_mtx, annot=True, fmt='g')
plt.xlabel("True Numbers")
plt.ylabel("Predicted Numbers")
plt.title("MNIST Dataset Number Prediction Confusion Matrix")
plt.savefig('mnist_local.png', dpi=400)

