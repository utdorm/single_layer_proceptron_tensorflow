from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# from tqdm import tqdm

# IMG_SIZE = 24
IMG_SIZE = 224
# IMG_SIZE = 28

DATADIR = './test/'

CATEGORIES = ['add', 'div', 'mlt', 'sbt']

testing_data = []

def create_testing_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                testing_data.append([new_array, class_num])  # add this to our testing_data
            
            except Exception as e:
               print("general exception", e, os.path.join(path,img))

create_testing_data()

import random
random.shuffle(testing_data)

X = []
labels = []

for features,label in testing_data:
    X.append(features)
    labels.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
labels = np.array(labels)

import pickle

pickle_out = open("X_tesing.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_testing.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

