import cv2
import tensorflow as tf

def main():

    test_sample = './test/sub.png'

    CATEGORIES = ['add', 'div', 'mlt', 'sbt']

    model = tf.keras.models.load_model('64-nodes_2-layers-1560590292.model')

    prediction = model.predict([getTestingData(test_sample)])

    print("------------------------\n", prediction)

    # print("------------------------\n",CATEGORIES[int(prediction[0][0])])


def getTestingData(filepath):
    IMG_SIZE = 28  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.







if __name__ == "__main__": 

    main()