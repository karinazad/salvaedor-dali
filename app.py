from flask import Flask, render_template, request, redirect, url_for
from flask import send_file
import numpy as np
import os
import sys

from run_vae import run_vae, save_generated_images
from utils.image_processing import preprocess_images
from settings import *

app = Flask(__name__)
app.static_folder = 'static'

artists = [{
    'artist': 'Vincent van Gogh',
    'genre': 'Surrealism'},
    {'artist': 'Salvador Dali',
     'genre': 'Surrealism'}]


IMAGES_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html', data=None)


@app.route('/about')
def about():
    return render_template('about.html', posts=artists, title='About')


@app.route('/error')
def error():
    return render_template('error.html')


@app.route('/', methods=['POST'])
def show_result():
    for key, value in request.form.items():

        if value[0] == 'G':
            type = 'genre'
            picked = value[2:]
        else:
            type = 'artist'
            picked = value

    image_src = "general_2.png"

    images = np.load(os.path.join(ROOT_DIR, 'data/processed/64x64/images.npy'))
    artists, genres = np.load(os.path.join(ROOT_DIR, 'data/processed/64x64/labels.npy'))

    images, order = preprocess_images(images, shuffle=True)
    artists, genres = artists[order], genres[order]

    print('Please type in the name of a painter (e.g.: Salvador Dali, Vincent van Gogh etc.)')
    artist_or_genre = input()
    artist_or_genre = artist_or_genre.replace(' ', '_')

    if artist_or_genre:
        if artist_or_genre in np.unique(artists):
            artist = artist_or_genre
            genre = None
            print('Your selected artist is ' + artist)
            inputs = images[artists == artist]

        elif artist_or_genre in np.unique(genres):
            genre = artist_or_genre
            artist = None
            print('Your selected genre is ' + genre)
            inputs = images[genres == genre]

        else:
            print('Sorry, this option is not available yet.')
            sys.exit()

    print('Number of images: ' + len(inputs))

    vae = run_vae(images[:50], artist_or_genre)

    if artist:
        save_generated_images(vae, inputs, artist = artist)
    elif genre:
        save_generated_images(vae, inputs, genre=genre)
    else:
        save_generated_images(vae, inputs)


    return render_template('result.html', data=None, artist=picked, type=type, image=image_src)


if __name__ == '__main__':
    app.run(debug=True)
