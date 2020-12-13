import tensorflow as tf
from tensorflow.keras import layers



class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(filters=64 , kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))
        self.model.add(layers.Conv2D(filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))
        self.model.add(layers.Conv2D(filters=512, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))
        self.model.add(layers.Flatten())

    def call(self, x):
        return self.model(x)


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(2048)
        self.model.add(layers.Reshape(target_shape=(4, 4, 128), input_shape=(None, 1024))),
        self.model.add(layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))
        self.model.add(layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

    def call(self, x):
        return self.model(x)

