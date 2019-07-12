import keras
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense, Flatten
from keras.callbacks import TensorBoard
import pickle
import time
import numpy as np


""" 
    Variable to check:
        -Epoch/iteration = 50
        -Data 28x28px
"""

""" uncomment this if the GPU ran out of memory """
# tf.Session(config=tf.ConfigProto(allow_growth=True))


getMdoelName = ""

dense_layers = [2]
layer_sizes = [16, 28, 32, 64]


def main ():

    pickle_in = open("X_training.pickle","rb")
    x_training = pickle.load(pickle_in)

    pickle_in = open("y_training.pickle","rb")
    y_training = pickle.load(pickle_in)

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for layer_size_2 in layer_sizes:
                for layer_size_3 in layer_sizes:

                    model = Sequential()

                    model.add(Dense(layer_size, activation='relu', input_shape=x_training.shape[1:]))

                    model.add(Flatten())

                    model.add(Dense(layer_size_2, activation='relu'))

                    model.add(Dense(layer_size_3, activation='relu'))

                    model.add(Dense(2, activation='sigmoid'))

                    getModelName = "{}x{}x{}-nodes_{}-layers-{}".format(layer_size, layer_size_2, layer_size_3, dense_layer, int(time.time()))
                    initTensorboard = TensorBoard(log_dir="binary_model_logs_2layer/{}".format(getModelName))

                    model.compile(
                                    loss='binary_crossentropy',
                                    optimizer='adam',
                                    metrics=['accuracy']
                                )

                    model.fit(
                                x_training, y_training, 
                                batch_size=16, 
                                epochs=50, 
                                validation_split=0.2,
                                callbacks = [initTensorboard]
                            )
                    
                    model.summary()
        
    # model.save("{}.model".format(getModelName))


if __name__ == "__main__":

    main()