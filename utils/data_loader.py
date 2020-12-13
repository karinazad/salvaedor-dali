from PIL import Image
from IPython.display import clear_output
import os
import sys
import numpy as np
from settings import *

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


def preprocess_images(images):
    images = images.reshape((images.shape[0], IMG_SIZE, IMG_SIZE, CHANNELS)) / 255.
    return images.astype(float)
