from flask import Flask, render_template, request, redirect, url_for
from flask import send_file
import os


app = Flask(__name__)
app.static_folder = 'static'

artists = [{
    'artist': 'Vincent van Gogh',
    'genre': 'Surrealism'
},
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
    return render_template('about.html', posts=posts, title='About')


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

    return render_template('result.html', data=None, artist=picked, type=type, image=image_src)


if __name__ == '__main__':
    app.run(debug=True)
