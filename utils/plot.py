import tensorflow as tf
import matplotlib.pyplot as plt
from settings import *

def plot_images(generator, seed=None):
    if seed is None:
        seed = tf.random.normal([16, NOISE_SHAPE])

    predictions = generator(seed, training=False)
    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')

    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
