import tensorflow as tf
from tensorflow.keras import layers
from settings import *


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), batch_size=BATCH_SIZE)
        self.conv1 = layers.Conv2D(filters=64 , kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        self.conv2 = layers.Conv2D(filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        self.conv3 = layers.Conv2D(filters=512, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(16, activation = 'relu')

        self.dense_z = layers.Dense(LATENT_DIM, name="z_mean")
        self.log_var_z =  layers.Dense(LATENT_DIM, name="z_log_var")

    def sample(self, x):
        z_mean, z_log_var = x
        batch_size = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
        sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return sample

    def call(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)

        z_mean = self.dense_z(x)
        z_log_var = self.log_var_z(x)
        z = self.sample([z_mean, z_log_var])

        return [z_mean, z_log_var, z]


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.input = tf.keras.Input(shape=(LATENT_DIM,))
        self.dense = layers.Dense(7 * 7 * 64, activation="relu")
        self.reshape = layers.Reshape((7, 7, 64))
        self.conv1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.conv3 = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")


    def call(self, x):
        x = self.input(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

