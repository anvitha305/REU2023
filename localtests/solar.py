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
# getting dataset from Roboflow
from roboflow import Roboflow
import os
#print("hiiii :3")
# use api key to get dataset
if not os.path.exists("solar-panel-infrared-images-5/"):
    f = open("secret.txt", "r")
    key = f.readline()[:-1]
    rf = Roboflow(str(key))
    project = rf.workspace("solarpanelimages").project("solar-panel-infrared-images")
    dataset = project.version(5).download("yolov5")
test_set_loc = "solar-panel-infrared-images-5/test/images/"
random_test_image = random.choice(os.listdir(test_set_loc))
print(random_test_image)
os.system("python3 detect.py --weights solar_model --img 416 --conf 0.1 --source " + test_set_loc)

