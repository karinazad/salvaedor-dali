import numpy as np
import os
import matplotlib.pyplot as plt
from settings import *


def plot_real_images(images, save_name=None):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()

    if save_name:
        plt.savefig(os.path.join(IMAGE_PATH, 'examples_real', save_name))


def plot_generated_images(decoder, samples=5, scale=10, save_name=None):
    n = samples * samples
    scale = scale

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    fig, axes = plt.subplots(samples, samples, figsize=(samples * 3, samples * 3))
    axes = axes.ravel()

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            image = x_decoded[0]
            axes[i].imshow(image)
            axes[i].axis('off')
    plt.show()

    if save_name:
        plt.savefig(os.path.join(IMAGE_PATH, 'examples_generated', save_name))

def plot_latent(decoder):
    # Adapted from Keras
    n = 30
    digit_size = IMG_SIZE
    scale = 1
    figsize = 20
    figure = np.zeros((digit_size * n, digit_size * n, 3))

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, 3)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
                :] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.show()


def plot_label_clusters(encoder, data, labels=None):
    # Adapted from Keras
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plot_history_vae(model):
    total_loss = model.history.history['loss']
    rec_loss = model.history.history['reconstruction_loss']
    kl_loss = model.history.history['kl_loss']
    x = np.arange(len(total_loss))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.ravel()

    axes[0].plot(x, total_loss, label='Total loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(x, rec_loss, label='Reconstruction loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(x, kl_loss, label='KL loss')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.show()