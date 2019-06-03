import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import TensorBoard
import pickle
import time

getModelName = 'Plus-notPlus-1layer-1500node.{}'.format(int(time.time()))
initTensorboard = TensorBoard(log_dir= './logs/{}'.format(getModelName)) 

nodePerLayer = 1
# nodePerLayer = 100
# nodePerLayer = 1500


# ep = 1
# ep = 10
# ep = 100
# ep = 1500


def main():

    pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y.pickle","rb")
    y = pickle.load(pickle_in)

    X = X/255.0

    model = Sequential()

    model.add(Dense(nodePerLayer, activation='relu', input_shape = X.shape[1:]))

    model.add(Flatten())

    # model.add(Dense(64))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                    optimizer='adam', 
                    metrics=['accuracy'])

    model.fit(X, y, epochs=10, 
                batch_size=32, 
                validation_split=0.1,
                callbacks=[initTensorboard])


if __name__ == "__main__": 

    main()


