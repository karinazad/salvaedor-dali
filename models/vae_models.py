import tensorflow as tf
from tensorflow.keras import layers
from settings import *


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim=LATENT_DIM):
        super(Encoder, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))
        self.model.add(layers.Conv2D(filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))
        self.model.add(layers.Conv2D(filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))
        self.model.add(layers.Conv2D(filters=512, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(16, activation='relu'))

        self.dense_z = layers.Dense(latent_dim, name="z_mean")
        self.log_var_z = layers.Dense(latent_dim, name="z_log_var")

    def call(self, x):
        x = self.model(x)

        z_mean = self.dense_z(x)
        z_log_var = self.log_var_z(x)
        z = Sampling()([z_mean, z_log_var])

        return [z_mean, z_log_var, z]


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim = LATENT_DIM):
        super(Decoder, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(latent_dim,)))
        self.model.add(layers.Dense(DIM * DIM * 64, activation="relu"))
        self.model.add(layers.Reshape((DIM, DIM, 64)))
        self.model.add(layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
        self.model.add(layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
        self.model.add(layers.Conv2DTranspose(CHANNELS, 3, activation="sigmoid", padding="same"))

    def call(self, x):
        x = self.model(x)
        return x
