# Path to training dataset

IMAGE_PATH = ""

NOISE_SHAPE = 128
BATCH_SIZE = 32

IMG_SIZE = 64

if IMG_SIZE == 28:
    DIM = 7
elif IMG_SIZE == 64:
    DIM = 16
else:
    raise Warning('Provide custom parameter for dimensions.')

CHANNELS = 3
LATENT_DIM = 2
