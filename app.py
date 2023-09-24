import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

Retainal = ['Mild', 'Moderate',  'Proliferate_DR', 'Severe']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("images",filename)

@app.route("/upload",methods=["POST","GET"])
def upload():
    if request.method=='POST':
        print("hdgkj")
        m = int(request.form["alg"])

        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join("images/", fn)
        myfile.save(mypath)

        print("{} is the file name", fn)
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)

        if m == 1:
            print("bv1")
            new_model = load_model('models/CNN.h5')
            test_image = image.load_img(mypath, target_size=(128, 128))
            test_image = image.img_to_array(test_image)

        else:
            print("bv3")
            new_model = load_model('models/MobileNet.h5')
            test_image = image.load_img(mypath, target_size=(128, 128))
            test_image = image.img_to_array(test_image)


        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        preds = Retainal[np.argmax(result)]

        return render_template("template.html", text=preds, image_name=fn)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)