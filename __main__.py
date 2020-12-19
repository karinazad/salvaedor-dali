import numpy as np
import os
from utils.data_loader import *
from models.vae import VAE

if __name__ == '__main__':
    full_image_path = os.path.join(IMAGE_PATH, 'processed/images_proc.npy')
    images = np.load(full_image_path)
    print(len(images))
    print('loaded images')

    print('loading model:')
    vae =VAE()
    print('done')

