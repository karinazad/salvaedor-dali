from PIL import Image
from IPython.display import clear_output
import os
import sys
import numpy as np
from settings import *



def load_images(path=IMAGE_PATH, n_images=None, resize_shape=(64, 64)):
    """
    Returns an array of pixels for each file in the directory that it in the RGB format.
    :param path: (str) path to where the training images are stored
    :param n_images: (int) number of images to process, if None, all images in the directory are processed
    :param resize_shape: (tuple) shape of the final images
    :return:
    images: (ndarray) array of pixels of shape (N, resize_shape[0], resize_shape[1])

    """
    images = []
    files = os.listdir(path)
    if n_images:
        files = files[:n_images]

    for i, file in enumerate(files):
        img = Image.open(os.path.join(path, file)).resize(resize_shape)
        img = np.array(img)
        if img.ndim == 3:
            images.append(img)

        print('Loading images: ', end=' ')
        print(f'{i}/{len(files)}', end='\r')
        sys.stdout.write(f"{i}/{len(files)}")
        sys.stdout.flush()
        # for IPython notebooks
        clear_output(wait=True)

    return np.array(images)
