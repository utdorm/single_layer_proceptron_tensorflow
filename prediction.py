import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
import pickle


MODEL_NAME = "32x64x16-4-out_nodes_3-layers_1562861842"
TEST_SAMPLE = './examples/3.png'


def main():

    testing_arr = getTestingData(TEST_SAMPLE)

    preview = cv2.imread(TEST_SAMPLE)
    output = imutils.resize(preview, width=500)

    pickle_in = open("mlb.pickle","rb")
    mlb = pickle.load(pickle_in)

    model = tf.keras.models.load_model(MODEL_NAME)

    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    print("\nClassify image: ")
    proba = model.predict(testing_arr)[0]
    print(proba)


    classes_ = np.argsort(proba)[::-1][:4]
    for (i, j) in enumerate(classes_):
            # build the label and draw the label on the image
            label = "{}: {}".format(mlb.classes_[j], proba[j])
            cv2.putText(output, label, (10, (i * 30) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
        # print("{}: \t{:.1f}".format(label, p * 100))
        print("{}: \t{:.1f}".format(label, p))


   
    # # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    
def getTestingData(filepath):
    # IMG_SIZE = 224
    # IMG_SIZE = 50 
    IMG_SIZE = 28
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.



if __name__ == "__main__": 
    main()
