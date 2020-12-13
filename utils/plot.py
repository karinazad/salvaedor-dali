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
    digit_size = IMG_SIZE

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