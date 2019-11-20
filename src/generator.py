from keras.layers import Dense, Dropout, Input, ReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam


def generator():
    generator = Sequential()
    generator.add(Dense(units=512, input_dim=100))
    generator.add(ReLU())

    generator.add(Dense(units=512))
    generator.add(ReLU())

    generator.add(Dense(units=1024))
    generator.add(ReLU())

    generator.add(Dense(units=784, activation="tanh"))

    generator.compile(loss="binary_crossentropy",
                      optimizer=Adam(lr=0.0001, beta_1=0.5))
    return generator
