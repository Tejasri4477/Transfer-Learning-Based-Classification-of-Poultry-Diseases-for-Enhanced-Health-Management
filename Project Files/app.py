from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')

# Match with folder class names
classes = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        file_path = os.path.join('static/images', file.filename)
        file.save(file_path)

        img = load_img(file_path, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        prediction = classes[np.argmax(pred)]

    return render_template('predict.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
