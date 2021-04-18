from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, InputLayer, GlobalAveragePooling2D
from keras import Input


def create_model():
    model = Sequential()
    model.add(Input(shape=(1365//4, 2048//4, 3)))
    model.add(Conv2D(32, 3))
    model.add(Conv2D(32, 3))
    model.add(MaxPooling2D())
    model.add(Activation("relu"))

    model.add(Conv2D(64, 3))
    model.add(Conv2D(64, 3))
    model.add(MaxPooling2D())
    model.add(Activation("relu"))

    model.add(Conv2D(128, 3))
    model.add(Conv2D(128, 3))
    model.add(MaxPooling2D())
    model.add(Activation("relu"))

    model.add(Conv2D(512, 3))
    model.add(Conv2D(512, 3))
    model.add(MaxPooling2D())
    model.add(Activation("relu"))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation="softmax"))

    model.summary()

    return model



