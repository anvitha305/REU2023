import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
import keras
# modules for plotting
import random
# getting dataset from Roboflow
from roboflow import Roboflow
import os
#print("hiiii :3")
# use api key to get dataset
if not os.path.exists("animals-yolov5-1/"):
    f = open("secret.txt", "r")
    key = f.readline()[:-1]
    rf = Roboflow(str(key))
    project = rf.workspace("yolo-2xmbu").project("animals-yolov5-gslnk")
    dataset = project.version(1).download("yolov5")
if not os.path.exists("yolov5/"):
    os.system("git clone https://github.com/ultralytics/yolov5")
    os.system("pip3 install -qr yolov5/requirements.txt")
    os.system("cd ..")
test_set_loc = "animals-yolov5-1/test/images/"
random_test_image = random.choice(os.listdir(test_set_loc))
os.system("python3 yolov5/detect.py --weights animals.pt --img 416 --conf 0.1 --source " + test_set_loc+random_test_image)
os.system("python3 yolov5/val.py --weights animals.pt --img 416 --conf 0.1 --data animals-yolov5-1/data.yaml")
