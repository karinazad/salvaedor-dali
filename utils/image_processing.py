from settings import *
import numpy as np
import os

def preprocess_images(images, shuffle=False):
    images = (images / 255).astype(float)

    if shuffle:
        order = np.random.permutation(len(images))
        images = images[order]
    else:
        order = None

    return images, order

def pick_genre(genre, shuffle=True, images=None, genres=None):
    if images is None:
        images = np.load(os.path.join(IMAGE_PATH, '/processed/images.proc.npy'))
    if genres is None:
        genres = np.load(os.path.join(IMAGE_PATH, 'labels/genres.npy'))

    assert genre in genres

    genre_images = images[np.where(genres == genre)[0]]

    if shuffle:
        order = np.random.permutation(len(genre_images))
        genre_images = genre_images[order]

    return genre_images


def pick_artist(artist, shuffle=True, images=None, artists=None):
    if images is None:
        images = np.load(os.path.join(IMAGE_PATH, '/processed/images.proc.npy'))
    if artists is None:
        artists = np.load(os.path.join(IMAGE_PATH, 'labels/artists.npy'))

    assert artist in artists

    artist_images = images[np.where(artist == artists)[0]]

    if shuffle:
        order = np.random.permutation(len(artist_images))
        artist_images = artist_images[order]

    return artist_images
