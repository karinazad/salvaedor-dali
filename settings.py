import os

try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    ROOT_DIR = ''

try:
    IMAGE_PATH = os.path.join(ROOT_DIR, 'data')
except:
    IMAGE_PATH = ''


NOISE_SHAPE = 128
BATCH_SIZE = 128
IMG_SIZE = 64

DIM = None
if IMG_SIZE == 28:
    DIM = 7
elif IMG_SIZE == 64:
    DIM = 16
else:
    raise Warning('Provide custom parameter for dimensions.')

CHANNELS = 3
LATENT_DIM = 5
