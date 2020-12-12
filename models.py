### models.py ###

import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, ReLU, LeakyReLU, BatchNormalization, Conv2DTranspose, Dropout
from keras.layers.convolutional import UpSampling2D, Conv2D
from settings import *



class ResBlock(tf.keras.layers.Layer):
    def __init__(self, resample='up', filt=4):
        """
        :param resample: None, 'down', or 'up
        """
        super(ResBlock, self).__init__()
        self.resample = resample
        self.filt = filt

    def __call__(self, x):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64 * self.filt, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

        x = Conv2D(64 * self.filt, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = ReLU()(x)

        x = Conv2D(64 * self.filt, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = ReLU()(x)

        return x


class Generator(tf.keras.Model):

    def __init__(self):
        """
        Implementation of the Generator.
        """
        super(Generator, self).__init__(name='generator')

        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=(NOISE_SHAPE,), batch_size=BATCH_SIZE))
        self.model.add(Dense(units=8 * IMG_SIZE * IMG_SIZE, activation='relu'))
        self.model.add(Reshape((IMG_SIZE // 8, IMG_SIZE // 8, -1)))
        self.model.add(ResBlock(resample='up', filt=4))
        self.model.add(ResBlock(resample='up', filt=2))
        self.model.add(ResBlock(resample='up', filt=1))
        self.model.add(Conv2D(3, kernel_size=3, strides=1, padding="same", activation="tanh", use_bias=False))

    def call(self, x):
        x = self.model(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        """
        Implementation of the Discriminator.
        """
        super(Discriminator, self).__init__(name='discriminator')

        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

        self.model.add(Conv2D(64, kernel_size=4, strides=(2, 2), padding="same", use_bias=False))
        self.model.add(LeakyReLU())
        self.model.add(Conv2D(128, kernel_size=4, strides=(2, 2), padding="same", use_bias=False))
        self.model.add(LeakyReLU())
        self.model.add(Conv2D(256, kernel_size=4, strides=(2, 2), padding="same", use_bias=False))
        self.model.add(Conv2D(256, kernel_size=4, strides=(2, 2), padding="same", use_bias=False))
        self.model.add(LeakyReLU())
        self.model.add(Conv2D(512, kernel_size=4, strides=(2, 2), padding="same", use_bias=False))
        self.model.add(LeakyReLU())

        self.model.add(Conv2D(1, kernel_size=3, strides=1, padding="same", use_bias=False))

        self.model.add(Flatten())
        self.model.add(Dense(units=1, activation=None))

    def call(self, x):
        x = self.model(x)
        return x


