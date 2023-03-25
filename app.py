from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import io
from PIL import Image
import tensorflow as tf
from keras_layers import CastToFloat32

CUSTOM_OBJECTS = {
    "CastToFloat32": CastToFloat32,
}
model = load_model('model', custom_objects=CUSTOM_OBJECTS)
print("Model Loaded")
app = Flask(__name__)

@app.route('/', )
def home():
   return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_tumor():
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))
    image.save('static/test_img.jpg')
    image = image.resize((64, 64))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    prob = model.predict(image)
    label = 1 if prob>0.5 else 0
    return render_template("results.html", label=label, probability=prob[0])

if __name__=='__main__':
    app.run(debug = True)
    