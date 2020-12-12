import tensorflow as tf
from keras.layers import  Dense, Reshape, Flatten, ReLU, LeakyReLU, BatchNormalization, Conv2DTranspose, Dropout
from keras.layers.convolutional import  Conv2D
from settings import *
from utils.plot import plot_images


class GAN_MNIST:
    def __init__(self):
        self.G = Generator_MNIST()
        self.D = Discriminator_MNIST()

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.G_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.D_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.batch_size = BATCH_SIZE
        self.history = {"G_loss": [], "D_loss": []}
        self.seed = tf.random.normal([16, NOISE_SHAPE])

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    def G_train_step(self):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            fake_score = self.D(fake_samples, training=True)
            G_loss = self.generator_loss(fake_score)

        G_gradients = tape.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients((zip(G_gradients, self.G.trainable_variables)))

        return G_loss

    def D_train_step(self, real_samples):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            real_score = self.D(real_samples, training=True)
            fake_score = self.D(fake_samples, training=True)

            D_loss = self.discriminator_loss(real_score, fake_score)

        D_gradients = tape.gradient(D_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients((zip(D_gradients, self.D.trainable_variables)))

        return D_loss


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


    def train(self, inputs, epochs, step_log=50, show_images = True):

        for epoch in range(epochs):
            dataset = self.create_dataset(inputs)
            print(f"Epoch {epoch}/{epochs}:")

            step = 0
            for sample_batch in dataset:
                G_loss = self.G_train_step()
                D_loss, GP = self.D_train_step(sample_batch)

                if step % step_log == 0:
                    self.history["G_loss"].append(G_loss.numpy())
                    self.history["D_loss"].append(D_loss.numpy())

                    print(f'\t Step {step}/{len(dataset) // self.batch_size} \t Generator: {G_loss.numpy()}'
                          f' \t Discriminator: {D_loss.numpy()}')
                    if show_images:
                        plot_images(self.G, seed = self.seed)

                step += 1




class Generator_MNIST(tf.keras.Model):
    def __init__(self):
        """
        Implementation of the Generator.
        """
        super(Generator_MNIST, self).__init__(name='generator')
        self.model = tf.keras.Sequential()
        self.model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_SHAPE,)))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        self.model.add(Reshape((7, 7, 256)))
        assert self.model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        self.model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 7, 7, 128)
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        self.model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 14, 14, 64)
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        self.model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, 28, 28, 1)

    def call(self, x):
        x = self.model(x)
        return x


class Discriminator_MNIST(tf.keras.Model):
    def __init__(self):
        """
        Implementation of the Generator.
        """
        super(Discriminator_MNIST, self).__init__(name='discriminator')

        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                              input_shape=[28, 28, 1]))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())
        self.model.add(Dense(1))

    def call(self, x):
        x = self.model(x)
        return x