import numpy as np
import os
import matplotlib.pyplot as plt
from settings import *


def plot_real_images(images, save_path=False):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()

    if save_path:
        plt.savefig(save_path)


def plot_generated_images(decoder, samples=4, scale=1, save_path=None):
    n = samples * samples
    digit_size = IMG_SIZE

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    fig, axes = plt.subplots(samples, samples, figsize=(samples * 3, samples * 3))
    axes = axes.ravel()

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            if LATENT_DIM > 2:
                z_sample = np.random.normal(size=(LATENT_DIM,)).reshape(1, -1)
            x_decoded = decoder.predict(z_sample)
            image = x_decoded[0]
            axes[i].imshow(image)
            axes[i].axis('off')

    plt.show()

    if save_path:
        plt.savefig(save_path)


def plot_decoded(encoder, decoder, selected, title=None, save_path = None):
    cols = len(selected)
    fig, axes = plt.subplots(2, cols, figsize=(2 * cols, 4))
    axes = axes.ravel()
    encoded,_,_ = encoder(selected)
    decoded = decoder(encoded)

    for i in range(cols):
        axes[i].imshow(selected[i])
        axes[i + cols].imshow(decoded[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i+cols].set_xticks([])
        axes[i+cols].set_yticks([])

        if i == 0:
            axes[i].set_ylabel('Real Images')
            axes[i + cols].set_ylabel('Reconstruction')

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_latent(decoder, samples=10, latent_dim = LATENT_DIM, scale=1, save_path = None):
    digit_size = IMG_SIZE
    figsize = 12
    n = samples
    figure = np.zeros((digit_size * n, digit_size * n, 3))

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            if LATENT_DIM > 2:
                z_sample = np.random.normal(size=(latent_dim,)).reshape(1, -1) * 2 * scale - scale
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, 3)
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size,
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

    if save_path:
        plt.savefig(save_path)


def plot_label_clusters(encoder, data, labels=None, save_path = None):
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

    if save_path:
        plt.savefig(save_path)

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