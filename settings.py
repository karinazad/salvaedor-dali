# Path to training dataset
IMAGE_PATH = "/content/drive/MyDrive/Colab Notebooks/MicheGANgelo/images/"

ENCODER_PATH = "/content/drive/MyDrive/Colab Notebooks/MicheGANgelo/models/encoder_lat5_arch3"
DECODER_PATH = "/content/drive/MyDrive/Colab Notebooks/MicheGANgelo/models/decoder_lat5_arch3"

NOISE_SHAPE = 128
BATCH_SIZE = 32

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
