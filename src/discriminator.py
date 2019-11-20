from keras.layers import Dense, Dropout, Input, ReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam


def discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units=1024, input_dim=784))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(units=512))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(units=256))
    discriminator.add(ReLU())

    discriminator.add(Dense(units=1, activation="sigmoid"))

    discriminator.compile(loss="binary_crossentropy",
                          optimizer=Adam(lr=0.0001, beta_1=0.5))
    return discriminator
