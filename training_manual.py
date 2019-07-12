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


def main ():

    pickle_in = open("X_training.pickle","rb")
    x_training = pickle.load(pickle_in)

    pickle_in = open("y_training.pickle","rb")
    y_training = pickle.load(pickle_in)
    
    model = Sequential()

    # first layer
    model.add(Dense(32, activation='relu', input_shape=x_training.shape[1:]))
    model.add(Flatten())

    # second layer
    model.add(Dense(64, activation='relu',))
    
    model.add(Dense(16, activation='relu',))

    model.add(Dense(4, activation='sigmoid'))

    getModelName = "32x64x16-3-out_nodes_3-layers_{}".format(int(time.time()))
    initTensorboard = TensorBoard(log_dir="temp_logs/{}".format(getModelName))

    grad_descent = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name="GradientDescent")

    model.compile(
                    loss='mean_squard_error',
                    optimizer=grad_descent,
                    metrics=['accuracy']
                )

    model.fit(
                x_training, y_training, 
                batch_size=8, 
                epochs=500, 
                validation_split=0.2, 
                callbacks = [initTensorboard]
            )
    
    model.summary()
    model.save("32x64x16-3-out_nodes_3-layers_{}".format(int(time.time())))

    
if __name__ == "__main__":

    main()