import keras
from keras.models import Sequential 
from keras.utils import plot_model, to_categorical
from keras.layers import Dense, Flatten
from keras.callbacks import TensorBoard
import pickle
import time
import numpy as np

# dense_layers = [1, 2, 3]
# layer_sizes = [1, 32, 64, 128]
# 64-nodes_3-layers-1559982529 - Best result

dense_layers = [1, 2, 3]
layer_sizes = [32, 64, 128, 256]

""" Accuracy Best """


""" Low Loss """



def main():

    pickle_in = open("X_training.pickle","rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y_training.pickle","rb")
    y = pickle.load(pickle_in)

    X_train = X/255.0
    y_train = to_categorical(y, num_classes=None)

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            getModelName = "{}-nodes_{}-layers-{}".format(layer_size, dense_layer, int(time.time()))
            
            model = Sequential()

            model.add(Dense(layer_size, activation='relu', input_shape=X.shape[1:]))

            model.add(Flatten())

            for _ in dense_layers:
                model.add(Dense(layer_size, activation='relu'))

            model.add(Dense(4, activation='sigmoid'))

            initTensorboard = TensorBoard(log_dir="multiclass_logs/{}".format(getModelName))

            model.compile(
                            loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'],
                            callbacks = [initTensorboard]
                        )

            model.fit(
                        X_train, y_train, 
                        batch_size=32, 
                        epochs=10, 
                        validation_split=0.1
                    )
            

if __name__ == "__main__": 

    main()