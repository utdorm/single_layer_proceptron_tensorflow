import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os
from imutils import paths
import cv2
import pickle

# IMG_SIZE = 50
IMG_SIZE = 224
# IMG_SIZE = 28

DATADIR = 'dataset'
imagePaths = sorted(list(paths.list_images(DATADIR)))

training_data = []

# initialize the data and label_classes
data = []
label_classes = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    image_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    data.append(image_array)

    l = imagePath.split(os.path.sep)[-2].split("_")
    label_classes.append(l)

data = np.array(data, dtype="float") / 255.0
label_classes = np.array(label_classes)

# print("[INFO] class label_classes:")
mlb = MultiLabelBinarizer()
label_classes = mlb.fit_transform(label_classes)

# Check if the data is correctly labelled 
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i, label))
print ("\n[INFO]: Class in order: \n", mlb.classes_)

import random
random.shuffle(data)

X_training_samples = []

for features in data:
    X_training_samples.append(features) 

X_training_samples = np.array(X_training_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("2X_training.pickle","wb")
pickle.dump(X_training_samples, pickle_out)
pickle_out.close()

pickle_out = open("2y_training.pickle","wb")
pickle.dump(label_classes, pickle_out)
pickle_out.close()

pickle_out = open("2mlb.pickle","wb")
pickle.dump(mlb, pickle_out)
pickle_out.close()
