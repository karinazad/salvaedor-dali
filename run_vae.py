import os
import sys
import numpy as np
import keras

from utils.image_processing import preprocess_images
from utils.plot import plot_decoded, plot_real_images
from models.vae import VAE
from settings import *


def run_vae(inputs, artist_or_genre, if_exists_load=True):
    saved_weights = os.listdir(os.path.join(ROOT_DIR, 'weights'))

    # if the artist or genre is in the save models, load the weights
    if artist_or_genre in saved_weights and if_exists_load:
        print('Found existing model...')
        encoder_path = os.path.join(ROOT_DIR, 'weights', artist_or_genre, 'encoder')
        decoder_path = os.path.join(ROOT_DIR, 'weights', artist_or_genre, 'decoder')

        vae = VAE(encoder_path=encoder_path, decoder_path=decoder_path)
        vae.compile(optimizer=keras.optimizers.Adam())
        print('... and loaded the weights')

    # or train a new model
    else:
        print('No existing model found. Starting to train...')
        vae = VAE()
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(inputs, epochs=1000, verbose=0, batch_size=32)
        print('... finished training')

    return vae


def save_generated_images(vae, inputs, artist = None, genre = None):
    if artist:
        SAVE_PATH = os.path.join(ROOT_DIR, 'generated', artist)
    elif genre:
        SAVE_PATH = os.path.join(ROOT_DIR, 'generated', genre)
    else:
        SAVE_PATH = os.path.join(ROOT_DIR, 'generated', 'unspecified')

    created = False
    suffix, i = '', 0

    while not created:
        if i > 10:
            raise Warning('Too many folders!')
        try:
            os.makedirs(SAVE_PATH + suffix)
            created = True
            SAVE_PATH = SAVE_PATH + suffix
        except:
            i += 1
            suffix = f'({i})'

    for i in range(len(inputs) // 8):
        if i == 0:
            plot_real_images(inputs[:16], save_path=os.path.join(SAVE_PATH, f'{artist}_real'))
        plot_decoded(vae.encoder, vae.decoder, inputs[6 * i:6 * (i + 1)],
                         save_path=os.path.join(SAVE_PATH, f'{artist}_{i}'))