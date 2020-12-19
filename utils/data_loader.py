from PIL import Image
from IPython.display import clear_output
import os
import sys
import numpy as np
import pandas as pd
from settings import *


def load_images(n_images=None, root_path=None, resize_shape=(64, 64), bw=False):
    """
    Returns an array of pixels for each file in the directory that it in the RGB format.
    :param path: (str) path to where the training images are stored
    :param n_images: (int) number of images to process, if None, all images in the directory are processed
    :param resize_shape: (tuple) shape of the final images
    :param bw: if True, return only one dimension (one channel)
    :return:
    images: (ndarray) array of pixels of shape (N, resize_shape[0], resize_shape[1])

    """

    if root_path is None:
        try:
            root_path = ROOT_DIR
        except:
            root_path = os.path.dirname(os.path.abspath(__file__))

    images_path = os.path.join(root_path, 'resized')
    info = pd.read_csv(os.path.join(ROOT_DIR, 'data/labels/artists.csv'))

    files = os.listdir(images_path)

    if n_images and n_images < len(files):
        files = files[:n_images]

    data = {'images': [], 'artist': [], 'genre': []}

    for i, filename in enumerate(files):
        img = Image.open(os.path.join(images_path, filename)).resize(resize_shape)
        img = np.array(img)

        if img.ndim == 3:
            if bw:
                data['images'].append(img[:, :, 0])
            else:
                data['images'].append(img)

            artist_, genre_ = find_artist_genre(filename, info)
            data['artist'].append(artist_)
            data['genre'].append(genre_)

        print('Loading images: ', end=' ')
        sys.stdout.write(f"{i + 1}/{len(files)}")
        sys.stdout.flush()
        clear_output(wait=True)

    images = np.array(data['images'])
    artist = np.array(data['artist'])
    genre = np.array(data['genre'])

    return images, artist, genre


def find_artist_genre(filename, df):
    artists_names = df['name'].str.replace(' ', '_').values

    try:
        artist_index = int(np.where([artist in filename for artist in artists_names])[0])
        artist = artists_names[artist_index]
        genre = df['genre'][artist_index]

    except:
        if filename.split('_')[0] == 'Albrecht':
            artist = 'Albrecht_Durer'
            genre = 'Northern Renaissance'
        else:
            artist = ''
            genre = ''

    return artist, genre
