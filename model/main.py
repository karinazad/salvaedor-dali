import numpy as np
from vae import VAE
from settings import *


def generate_image(style):
    vae = VAE(encoder_path=ENCODER_PATH, decoder_path=DECODER_PATH)

    artists = np.load('./data/labels/artist.npy')
    genres = np.load('./data/labels/genres.npy')
    images = np.load('./data/processed/images_proc.npy')


if __name__ == '__main__':
    style = input()
    generate_image(style)
