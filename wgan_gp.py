import tensorflow as tf
import matplotlib.pyplot as plt
from gan_models import Generator, Discriminator
from settings import *
from utils.plot import plot_images

class WGAN_GP:
    def __init__(self, generator=None, discriminator=None, g_optimizer=None, d_optimizer=None,
                 gradient_penalty=100):

        if generator is None:
            self.G = Generator()
        else:
            self.G = generator

        if discriminator is None:
            self.D = Discriminator()
        else:
            self.D = discriminator

        if g_optimizer is None:
            self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        else:
            self.G_optimizer = g_optimizer

        if d_optimizer is None:
            self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        else:
            self.D_optimizer = d_optimizer

        self.gp_weight = gradient_penalty
        self.batch_size = BATCH_SIZE
        self.seed = tf.random.normal([16, NOISE_SHAPE])
        self.history = {"G_loss": [], "D_loss": [], "gradient_penalty": []}

    def generator_loss(self, fake_score):
        return -tf.math.reduce_mean(fake_score)

    def discriminator_loss(self, fake_score, real_score):
        return tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

    def gradient_penalty(self, real_samples, fake_samples):
        alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        real_samples = tf.cast(real_samples, tf.float32)
        diff = fake_samples - real_samples

        interpolated = real_samples + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.D(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    # @tf.function
    def G_train_step(self):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            fake_score = self.D(fake_samples, training=True)
            G_loss = self.generator_loss(fake_score)

        G_gradients = tape.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients((zip(G_gradients, self.G.trainable_variables)))

        return G_loss

    # @tf.function
    def D_train_step(self, real_samples):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            real_score = self.D(real_samples, training=True)
            fake_score = self.D(fake_samples, training=True)

            D_loss = self.discriminator_loss(real_score, fake_score)
            GP = self.gradient_penalty(real_samples, fake_samples) * self.gp_weight
            D_loss = D_loss + GP

        D_gradients = tape.gradient(D_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients((zip(D_gradients, self.D.trainable_variables)))

        return D_loss, GP

    def generate_samples(self, number=None):
        if number is None:
            number = self.batch_size
        z = tf.random.normal([number, NOISE_SHAPE])
        generated = self.G(z)

        return generated

    def create_dataset(self, inputs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(inputs.shape[0], seed=0).batch(self.batch_size, drop_remainder=True)
        return dataset

    def plot_images(self):
        z = tf.random.normal([16, 128])
        predictions = self.G(z, training=False)
        fig = plt.figure(figsize=(8, 8))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i])

        # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def train(self, inputs, epochs, step_log=10, show_images=True):

        print('Start training....')
        plot_images(self.G, seed=self.seed)

        for epoch in range(epochs):
            dataset = self.create_dataset(inputs)
            print(f"Epoch {epoch}/{epochs}:")

            step = 0
            for sample_batch in dataset:
                G_loss = self.G_train_step()
                D_loss, GP = self.D_train_step(sample_batch)

                if type(step_log) in (int, float):
                    if step % step_log == 0:
                        self.history["G_loss"].append(G_loss.numpy())
                        self.history["D_loss"].append(D_loss.numpy())
                        self.history['gradient_penalty'].append(GP.numpy())

                        print(f'\t Step {step}/{len(dataset) // self.batch_size}: \t Generator: {G_loss.numpy()}'
                              f' \t Discriminator: {D_loss.numpy()}')

                        if show_images:
                            plot_images(self.G, seed=self.seed)

                step += 1

            if step_log == 'epoch':
                self.history["G_loss"].append(G_loss.numpy())
                self.history["D_loss"].append(D_loss.numpy())
                self.history['gradient_penalty'].append(GP.numpy())

                print(f'\t Step {step}/{len(dataset) // self.batch_size} \t Generator: {G_loss.numpy()}'
                      f' \t Discriminator: {D_loss.numpy()}')
                if show_images:
                    plot_images(self.G, seed=self.seed)
