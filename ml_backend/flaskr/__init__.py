import os
import numpy as np
import base64
import io
from PIL import Image
from flask import Flask, request
from flask_cors import CORS, cross_origin

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    # a simple page that says hello
    @app.route('/pose', methods = ['POST'])
    @cross_origin(origin='*',headers=['Content-Type'])
    def pose():
        import sys
        print("SAVING IMAGE")
        base64_image = request.json['image']
        base64_decoded  = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(base64_decoded))
        image_np = np.array(image)

        im = Image.fromarray(image_np)
        im.save("image.jpg")
        
        if isinstance(image_np, np.ndarray):
            print("OK", file=sys.stderr)
            return {"Status": "OK"}

        print("NOT OK", file=sys.stderr)
        return{"Status": "NOT OK"}

    return app