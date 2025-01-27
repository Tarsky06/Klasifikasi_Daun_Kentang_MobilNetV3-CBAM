from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Layer, Activation, Multiply, Reshape, Conv2D
from tensorflow.keras import backend as K
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# CBAM Layer
@tf.keras.utils.register_keras_serializable()
class CBAM(tf.keras.layers.Layer):
    def __init__(self, filters, reduction=16, name=None, **kwargs):
        super(CBAM, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.reduction = reduction

        # Channel Attention
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.fc1 = tf.keras.layers.Dense(filters // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(filters, activation='sigmoid')

        # Spatial Attention
        self.conv2d = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        # Channel Attention
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)
        avg_out = self.fc2(self.fc1(avg_out))
        max_out = self.fc2(self.fc1(max_out))
        channel_out = tf.keras.layers.Reshape((1, 1, self.filters))(avg_out + max_out)
        x = inputs * channel_out

        # Spatial Attention
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_out = tf.concat([avg_out, max_out], axis=-1)
        spatial_out = self.conv2d(spatial_out)
        return x * spatial_out

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({
            'filters': self.filters,
            'reduction': self.reduction
        })
        return config

    @classmethod
    def from_config(cls, config):
        if 'name' not in config:
            config['name'] = None
        return cls(**config)


# Flask App Initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load Pretrained Model
model_path = 'Model_MobilenetV3_large_with_CBAM.keras'
try:
    model = load_model(model_path, custom_objects={'CBAM': CBAM})
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_LABELS = {0: 'Early_Blight', 1: 'Healthy', 2: 'Late_Blight'}

# Function to Classify Images
def classify_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model is not None:
        predictions = model.predict(img_array)
        print(f"Predictions: {predictions}")  # Debug log

        class_index = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][class_index]

        # Thresholding based on confidence
        threshold = 0.7
        if confidence < threshold:
            return "Kelas Tidak Dikenal", confidence, predictions[0]
        
        return CLASS_LABELS.get(class_index, "Kelas Tidak Dikenal"), confidence, predictions[0]
    else:
        return "Model tidak tersedia", 0.0, [0]*len(CLASS_LABELS)  # Mengembalikan nol untuk semua kelas

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)

        # Validasi ekstensi file
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            flash("Invalid file type. Please upload a PNG or JPG image.", "danger")
            return redirect(request.url)

        # Simpan file
        image_path = os.path.join('static/images', file.filename)
        file.save(image_path)

        # Lakukan klasifikasi gambar
        label, confidence, probabilities = classify_image(image_path)
        uploaded_image_url = url_for('uploaded_file', filename=file.filename)  # Ambil URL gambar

        return render_template(
            'classify.html',
            filename=file.filename,
            result=label,
            confidence=f"{confidence * 100:.2f}%",
            probabilities=probabilities,
            class_labels=CLASS_LABELS,
            uploaded_image_url=uploaded_image_url  # Tambahkan URL gambar ke konteks
        )
    return render_template('classify.html', class_labels=CLASS_LABELS)

@app.route('/info')
def apk_info():
    return render_template('info.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    os.makedirs('static/images', exist_ok=True)
    app.run(debug=True)