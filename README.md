# Sal**vae**dor Dali
Reproduce classical paintings using Variational Autoencoders (VAEs). 

![alt text](https://raw.githubusercontent.com/karinazad/salvaedor-dali/main/generated/examples_real/surrealism.png)

## Install and run
Installation
```
git clone https://github.com/karinazad/salvaedor-dali.git
cd salvaedor-dali
pip install -r requirements.txt
```

Run 
```
python run_vae.py
```

The following prompt will appear:
```
Please type in the name of a painter (e.g.: Salvador Dali, Vincent van Gogh etc.)

```
Type in the name of a painter. So far, the model supports: 

Albrecht Durer, Alfred Sisley, Amedeo Modigliani, Andrei Rublev, Andy Warhol, Camille Pissarro, Caravaggio, Claude Monet, Diego Rivera, Diego Velazquez, Edgar Degas, Edouard Manet, Edvard Munch, El Greco, Eugene Delacroix, Francisco Goya, Frida Kahlo, Georges Seurat, Giotto di Bondone, Gustav Klimt, Gustave Courbet, Henri Matisse, Henri Rousseau, Henri de Toulouse-Lautrec, Hieronymus Bosch, Jackson Pollock, Jan van Eyck, Joan Miro, Kazimir Malevich, Leonardo da Vinci, Marc Chagall, Michelangelo, Mikhail Vrubel, Pablo Picasso, Paul Cezanne, Paul Gauguin, Paul Klee, Peter Paul Rubens, Pierre-Auguste Renoir, Piet Mondrian, Pieter Bruegel, Raphael, Rembrandt, Rene Magritte, Salvador Dali, Sandro Botticelli, Titian, Vasiliy Kandinskiy, Vincent van Gogh, William Turner.

To skip training and use pre-trained network, select one of the following:
Andy Warhol, Frida Kahlo, Vincent van Gogh, Paul Cezzane, Salvador Dali.


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
