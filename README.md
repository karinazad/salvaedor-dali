# Sal**vae**dor Dali
Reproduce classical paintings using Variational Autoencoders (VAEs). 

![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/examples_real/surrealism.png)

## Variational Autoencoders
VAE is an autoencoder whose encodings distribution is regularised during the training in order to ensure that its latent space has good properties allowing us to generate some new data.

Variational Autoencoders (VAEs) learn to encode regularized distributions during training and restore or genereate new data during inference. Regularization ensures thtat the latent space has good properties. The model is able to reconstruct images (see Reconstruction by VAE) by passing into the encoder and the decoder but generate new data as well. This is done by passing random or uniform samples from latent space through the decoder (see Latent Space Exploration).

## Dataset
The dataset was obtained from Kaggle (https://www.kaggle.com/ikarus777/best-artworks-of-all-time). Images were resized to 64x64.
There are over 8500 images with the most common genres being Impressionism (1647), Post-Impressionism (1048), Symbolims (666), and Surrealism (435).
And there are over 50 artists, such as Vincent van Gogh, Edgar Degas, Albrecht Durer, Pablo Picasso, and Salvador Dali.

## Reconstruction by VAE
Examples of reconstructed paintings.

#### Frida Kahlo
![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/Frida_Kahlo/Frida_Kahlo_0.png)
![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/Frida_Kahlo/Frida_Kahlo_1.png)


#### Andy Warhol
![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/Andy_Warhol(1)/Andy_Warhol_1.png)
![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/Andy_Warhol(1)/Andy_Warhol_3.png)

#### Vincent van Gogh
![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/Vincent_van_Gogh(1)/Vincent_van_Gogh_1.png)
![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/Vincent_van_Gogh(1)/Vincent_van_Gogh_2.png)


#### Salvador Dali
![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/Salvador_Dali/Salvador_Dali_0.png)
![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/Salvador_Dali/Salvador_Dali_1.png)


## Latent space exploration
