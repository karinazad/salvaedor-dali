
import pandas as pd
from PIL import Image
from IPython.display import clear_output
import os
import sys
import numpy as np


from settings import *


def load_images(n_images=None, path=IMAGE_PATH, resize_shape=(64, 64), bw=False):
    """
    Returns an array of pixels for each file in the directory that it in the RGB format.
    :param path: (str) path to where the training images are stored
    :param n_images: (int) number of images to process, if None, all images in the directory are processed
    :param resize_shape: (tuple) shape of the final images
    :param bw: if True, return only one dimension (one channel)
    :return:
    images: (ndarray) array of pixels of shape (N, resize_shape[0], resize_shape[1])

    """
    images = []
    path = os.path.join(path, 'resized')
    files = os.listdir(path)
    if n_images and n_images < len(files):
        files = files[:n_images]

    for i, file in enumerate(files):
        img = Image.open(os.path.join(path, file)).resize(resize_shape)
        img = np.array(img)
        if img.ndim == 3:
            if bw:
                images.append(img[:, :, 0])
            else:
                images.append(img)

        print('Loading images: ', end=' ')
        sys.stdout.write(f"{i + 1}/{len(files)}")
        sys.stdout.flush()
        clear_output(wait=True)

    return np.array(images)


from PIL import Image
from IPython.display import clear_output
import os
import sys
import numpy as np


# from settings import *

def load_images(n_images=None, path=IMAGE_PATH, resize_shape=(64, 64), bw=False):
    """
    Returns an array of pixels for each file in the directory that it in the RGB format.
    :param path: (str) path to where the training images are stored
    :param n_images: (int) number of images to process, if None, all images in the directory are processed
    :param resize_shape: (tuple) shape of the final images
    :param bw: if True, return only one dimension (one channel)
    :return:
    images: (ndarray) array of pixels of shape (N, resize_shape[0], resize_shape[1])

    """
    images = []
    path = os.path.join(path, 'resized')
    files = os.listdir(path)
    if n_images and n_images < len(files):
        files = files[:n_images]

    for i, file in enumerate(files):
        img = Image.open(os.path.join(path, file)).resize(resize_shape)
        img = np.array(img)
        if img.ndim == 3:
            if bw:
                images.append(img[:, :, 0])
            else:
                images.append(img)

        print('Loading images: ', end=' ')
        sys.stdout.write(f"{i + 1}/{len(files)}")
        sys.stdout.flush()
        clear_output(wait=True)

    return np.array(images)


def preprocess_images(images, shuffle=False):
    images = (images / 255).astype(float)

    if shuffle:
        order = np.random.permutation(len(images))
        images = images[order]
    else:
        order = None

    return images, order


def create_labels_artist(path=None, genres=False):
    if path is None:
        path = os.path.join(IMAGE_PATH, 'resized')

    files = os.listdir(path)
    artists = pd.read_csv(os.path.join(IMAGE_PATH, 'artists.csv'))
    artists_names = artists['name'].str.replace(' ', '_').values

    tags = []
    for i, file in enumerate(files):
        try:
            artist_index = int(np.where([artist in file for artist in artists_names])[0])
            tag = artists_names[artist_index]
            if genres:
                tag = artists['genre'][artist_index]
                if len(tag.split(',')) > 1:
                    tag = tag.split(',')[0]
        except:
            if file.split('_')[0] == 'Albrecht':
                tag = 'Albrecht_Durer'
            else:
                tag = ''

        tags.append(tag)

    return np.array(tags)


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
