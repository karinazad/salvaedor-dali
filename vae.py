import numpy as np
import tensorflow as tf
from vae_models import Encoder, Decoder
from settings import *


class VAE(tf.keras.Model):
    def __init__(self, encoder_path = None, decoder_path = None):
        super(VAE, self).__init__()
        self.latent_dim = LATENT_DIM

        if encoder_path:
            self.encoder = tf.keras.models.load_model(encoder_path)
        else:
            self.encoder = Encoder()

        if decoder_path:
            self.decoder = tf.keras.models.load_model(decoder_path)
        else:
            self.decoder = Decoder()

    def loss_fn(self, real, reconstr):
        loss = tf.keras.losses.binary_crossentropy(real, reconstr)
        loss = tf.reduce_mean(loss)
        loss *= IMG_SIZE * IMG_SIZE

        return loss

    def kl_loss(self, z_mean, z_log_var):
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return kl_loss

    def train_step(self, real):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(real)
            reconstr = self.decoder(z)
            reconstr_error = self.loss_fn(real, reconstr)
            kl_loss = self.kl_loss(z_mean, z_log_var)
            loss = reconstr_error + kl_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return { "loss": loss, "reconstruction_loss": reconstr_error, "kl_loss": kl_loss}
