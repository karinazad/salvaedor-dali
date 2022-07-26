# Sal**vae**dor Dali
Reproduce classical paintings using Conditional Variational Autoencoder (CVAE). 

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

Vincent Van Gogh, Andy Warhol, Claude Monet, Salvador Dali, Paul Gauguin, Pablo Picasso, and Fridа Kahlo.

<!-- Albrecht Durer, Alfred Sisley, Amedeo Modigliani, Andrei Rublev, Andy Warhol, Camille Pissarro, Caravaggio, Claude Monet, Diego Rivera, Diego Velazquez, Edgar Degas, Edouard Manet, Edvard Munch, El Greco, Eugene Delacroix, Francisco Goya, Frida Kahlo, Georges Seurat, Giotto di Bondone, Gustav Klimt, Gustave Courbet, Henri Matisse, Henri Rousseau, Henri de Toulouse-Lautrec, Hieronymus Bosch, Jackson Pollock, Jan van Eyck, Joan Miro, Kazimir Malevich, Leonardo da Vinci, Marc Chagall, Michelangelo, Mikhail Vrubel, Pablo Picasso, Paul Cezanne, Paul Gauguin, Paul Klee, Peter Paul Rubens, Pierre-Auguste Renoir, Piet Mondrian, Pieter Bruegel, Raphael, Rembrandt, Rene Magritte, Salvador Dali, Sandro Botticelli, Titian, Vasiliy Kandinskiy, Vincent van Gogh, William Turner.

To skip training and use pre-trained network, select one of the following:

Andy Warhol, Frida Kahlo, Vincent van Gogh, Paul Cezzane, Salvador Dali. -->


## Conditional Variational Autoencoder
Variational Autoencoders (VAEs) learn a regularized distributions of the training data during training. Regularization ensures thtat the latent space has good properties (i.e., most commonly, is a normal distribution). The model is able to reconstruct images (see Reconstruction by VAE) by passing into the encoder and the decoder but generate new data as well. This is done by sampling from the latent space and passing it to the generator.

Conditional VAE is an extension of VAE which enables us to specify the type of samples that should be generated. In this case, it corresponds to the name of the author. 

## Dataset
The dataset was obtained from Kaggle (https://www.kaggle.com/ikarus777/best-artworks-of-all-time). Images were resized to 64x64.
There are over 8500 images with the most common genres being Impressionism (1647), Post-Impressionism (1048), Symbolims (666), and Surrealism (435).
And there are over 50 artists, such as Vincent van Gogh, Edgar Degas, Albrecht Durer, Pablo Picasso, and Salvador Dali.



