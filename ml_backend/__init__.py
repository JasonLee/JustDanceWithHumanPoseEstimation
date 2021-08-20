import os
import numpy as np
import base64
import io
from PIL import Image
from flask import Flask, request
from flask_cors import CORS, cross_origin
from .OpenPose import OpenPose
from .utils import crop_resize_image
import cv2
import pandas as pd

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    op = OpenPose()

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
        timestamp = request.json['timestamp']
        print("timestamp", timestamp)
        print("SAVING IMAGE")
        base64_image = request.json['image']
        base64_decoded  = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(base64_decoded))
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        data = op.process_image_keypoints(cv2_image)
        data = data[0, :15]
        data = crop_resize_image(data)

        data = data.tolist()
  
        connection = op.get_joint_mapping()
        imagecv = op.get_cv_image(cv2_image)
        # cv2.imwrite("image.jpg", imagecv)

        return {
                "Status": "OK",
                "points": data,
                "mapping": connection
                }

        
    return app