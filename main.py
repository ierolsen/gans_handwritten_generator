from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from src.discriminator import discriminator
from src.generator import generator
from src.gans_model import gan

discriminator = discriminator()
generator = generator()
gan = gan(discriminator,generator)

#gan.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32)-127.5)/127.5


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])



epochs = 150
batch_size = 256

for e in range(epochs):
    for _ in range(batch_size):
        noise = np.random.normal(0, 1, [batch_size, 100])

        generated_images = generator.predict(noise)

        image_batch = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]

        x = np.concatenate([image_batch, generated_images])

        y_dis = np.zeros(batch_size * 2)
        y_dis[:batch_size] = 1

        discriminator.trainable = True
        discriminator.train_on_batch(x, y_dis)

        noise = np.random.normal(0, 1, [batch_size, 100])

        y_gen = np.ones(batch_size)

        discriminator.trainable = False

        gan.train_on_batch(noise, y_gen)
    print("epochs: ", e)


generator.save_weights('mnist_gans_model.h5')


noise= np.random.normal(loc=0, scale=1, size=[100, 100])
generated_images = generator.predict(noise)
generated_images = generated_images.reshape(100,28,28)
plt.imshow(generated_images[66], interpolation='nearest')
plt.axis('off')
plt.show()
plt.imsave("generated_image.png",generated_images[66])
