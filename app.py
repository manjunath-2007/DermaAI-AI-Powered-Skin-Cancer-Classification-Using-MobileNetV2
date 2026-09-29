import os
import io
import base64
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
THRESHOLD = 0.23  # cancer probability above this => "Cancer Detected"

model = None


def load_model():
    """Load the MobileNetV2 model once at startup."""
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'skin_cancer_mobilenetv2.h5')
    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        return
    try:
        import tensorflow as tf
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception:
            # Fallback for models saved with newer Keras (batch_shape / optional args)
            from tensorflow.keras.layers import InputLayer

            class CompatInputLayer(InputLayer):
                def __init__(self, **kwargs):
                    kwargs.pop('batch_shape', None)
                    kwargs.pop('optional', None)
                    super().__init__(**kwargs)

            model = tf.keras.models.load_model(
                model_path, custom_objects={'InputLayer': CompatInputLayer}
            )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(img):
    img = img.convert('RGB').resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    cancer_prob = float(model.predict(arr, verbose=0)[0][0])
    detected = cancer_prob > THRESHOLD
    confidence = cancer_prob if detected else 1 - cancer_prob
    return {
        'cancer_detected': detected,
        'label': 'Cancer Detected' if detected else 'Cancer Not Detected',
        'confidence': round(confidence * 100, 2),
    }


@app.route('/')
def index():
    return render_template('index.html', model_ready=model is not None)


@app.route('/predict', methods=['POST'])
def predict_route():
    if model is None:
        return jsonify(error='AI model is not loaded on the server.'), 503

    file = request.files.get('image')
    if not file or file.filename == '' or not allowed_file(file.filename):
        return jsonify(error='Please upload a PNG, JPG, JPEG, BMP or TIFF image.'), 400

    try:
        data = file.read()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        return jsonify(error='That file could not be read as an image.'), 400

    try:
        result = predict(img)
    except Exception as e:
        return jsonify(error=f'Prediction failed: {e}'), 500

    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='Image is too large (max 16 MB).'), 413


# Load at import time so it works with both `python app.py` and gunicorn/flask run.
load_model()

if __name__ == '__main__':
    app.run(debug=False, port=5000)
