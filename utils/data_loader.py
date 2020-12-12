from PIL import Image
import os
import numpy as np
from settings import *


def load_images(path=IMAGE_PATH, resize_shape=(64, 64)):
    """
    Returns an array of pixels for each file in the directory that it in the RGB format.
    :param path: path to where the training images are stored
    :param resize_shape: shape of the final images
    :return:
    images: (ndarray) array of pixels of shape (N, resize_shape[0], resize_shape[1])

    """
    images = []
    for file in os.listdir(path):
        img = Image.open(os.path.join(path, file)).resize(resize_shape)
        img = np.array(img)
        if img.ndim == 3:
            images.append(img)

    return np.array(images)