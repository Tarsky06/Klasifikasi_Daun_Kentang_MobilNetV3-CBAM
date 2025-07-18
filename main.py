import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Conv2D, Activation, Multiply
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.graphics.texture import Texture
from kivy.clock import Clock

# Definisikan layer CBAM
@tf.keras.utils.register_keras_serializable()
class CBAM(Layer):
    def __init__(self, filters, ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.channel_avg_pool = GlobalAveragePooling2D()
        self.channel_max_pool = GlobalMaxPooling2D()
        self.shared_dense_1 = Dense(self.filters // self.ratio, activation='relu', use_bias=False)
        self.shared_dense_2 = Dense(self.filters, activation='sigmoid', use_bias=False)
        self.conv = Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid', use_bias=False)
        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        avg_pool = self.channel_avg_pool(inputs)
        max_pool = self.channel_max_pool(inputs)
        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))
        channel_attention = Activation('sigmoid')(avg_out + max_out)
        channel_refined = Multiply()([inputs, Reshape((1, 1, self.filters))(channel_attention)])

        avg_pool_spatial = K.mean(channel_refined, axis=-1, keepdims=True)
        max_pool_spatial = K.max(channel_refined, axis=-1, keepdims=True)
        spatial_attention = self.conv(K.concatenate([avg_pool_spatial, max_pool_spatial], axis=-1))
        refined_output = Multiply()([channel_refined, spatial_attention])

        return refined_output

class MainMenu(BoxLayout):
    def __init__(self, **kwargs):
        super(MainMenu, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10
        self.spacing = 10

        self.upload_btn = Button(text='Upload Image', size_hint=(1, 0.1))
        self.upload_btn.bind(on_press=self.upload_image)
        self.add_widget(self.upload_btn)

        self.camera_btn = Button(text='open Camera', size_hint=(1, 0.1))
        self.camera_btn.bind(on_press=self.open_camera)
        self.add_widget(self.camera_btn)

        self.camera_image = KivyImage(size_hint=(1, 0.6))
        self.add_widget(self.camera_image)

        self.capture_btn = Button(text='Capture Image', size_hint=(1, 0.1))
        self.capture_btn.bind(on_press=self.capture_image)
        self.add_widget(self.capture_btn)

        self.cap = None  # Variabel kamera
        self.current_frame = None  # Frame kamera saat ini

    def open_camera(self, instance):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            Clock.schedule_interval(self.update_camera, 1.0 / 30.0)

    def update_camera(self, dt):
        ret, frame = self.cap.read()
        if ret:
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_image.texture = texture

    def capture_image(self, instance):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Simpan gambar dengan nama yang berbeda setiap kali
                img_count = len([name for name in os.listdir('.') if name.startswith('captured_image')])
                img_name = f"captured_image_{img_count + 1}.jpg"
                cv2.imwrite(img_name, frame)
                self.display_image(img_name)

    def upload_image(self, instance):
        filechooser = FileChooserIconView()
        filechooser.bind(on_submit=self.load_image)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        layout.add_widget(filechooser)

        btn_layout = BoxLayout(size_hint_y=None, height=50)
        ok_btn = Button(text='OK')
        cancel_btn = Button(text='Cancel')
        popup = Popup(title='Select Image', content=layout, size_hint=(0.9, 0.9))
        cancel_btn.bind(on_press=popup.dismiss)
        ok_btn.bind(on_press=popup.dismiss)  # Tambahkan aksi untuk menutup popup
        btn_layout.add_widget(ok_btn)
        btn_layout.add_widget(cancel_btn)
        layout.add_widget(btn_layout)


        popup.open()

    def load_image(self, instance, selection, touch):
        if selection:
            self.selected_image = selection[0]
            self.display_image(self.selected_image)

    def display_image(self, img_path):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        image_widget = KivyImage(source=img_path, size_hint_y=None, height=400)
        layout.add_widget(image_widget)

        classify_btn = Button(text='Show Classification Results', size_hint=(1, 0.1))
        classify_btn.bind(on_press=lambda x: self.classify_image(img_path))
        layout.add_widget(classify_btn)

        btn_layout = BoxLayout(size_hint_y=None, height=50)
        ok_btn = Button(text='OK')
        cancel_btn = Button(text='Cancel')
        popup = Popup(title='Image Results', content=layout, size_hint=(0.9, 0.9))

        # Tambahkan aksi untuk tombol OK
        ok_btn.bind(on_press=popup.dismiss)
        cancel_btn.bind(on_press=popup.dismiss)

        ok_btn.bind(on_press=popup.dismiss)  # Tambahkan aksi untuk menutup popup
        btn_layout.add_widget(ok_btn)
        btn_layout.add_widget(cancel_btn)
        layout.add_widget(btn_layout)


        popup.open()


    def load_and_prepare_image(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalisasi
            return img_array
        except Exception as e:
            self.display_classification_result(f'Terjadi kesalahan saat memuat gambar: {str(e)}')
            return None


    def classify_image(self, img_path):
        try:
            model = tf.keras.models.load_model('Model_InceptionResNetV2_CBAM.keras', custom_objects={'CBAM': CBAM})
            prepared_image = self.load_and_prepare_image(img_path)
            prediction = model.predict(prepared_image)
            predicted_probabilities = prediction[0]  # Ambil array probabilitas
            predicted_class = np.argmax(predicted_probabilities)  # Ambil indeks kelas dengan probabilitas tertinggi
            max_probability = predicted_probabilities[predicted_class] * 100  # Konversi ke persen

            disease_labels = {0: "Blackspot", 1: "Canker", 2: "Greening", 3: "Young_Healthy", 4: "Healthy"}
            disease_name = disease_labels.get(predicted_class, "Unknown Disease")

            self.display_classification_result(f"{disease_name} ({max_probability:.2f}%)")

        except Exception as e:
            self.display_classification_result(f'Terjadi kesalahan: {str(e)}')

    def display_classification_result(self, result_text):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        label = Label(text=f'Result: {result_text}')
        layout.add_widget(label)

        btn_layout = BoxLayout(size_hint_y=None, height=50)
        ok_btn = Button(text='OK')
        cancel_btn = Button(text='Cancel')
        popup = Popup(title='Classification Results', content=layout, size_hint=(0.7, 0.3))

        # Tambahkan aksi untuk tombol OK
        ok_btn.bind(on_press=popup.dismiss)
        cancel_btn.bind(on_press=popup.dismiss)

        btn_layout.add_widget(ok_btn)
        btn_layout.add_widget(cancel_btn)
        layout.add_widget(btn_layout)

        popup.open()


class MyApp(App):
    def build(self):
        return MainMenu()

if __name__ == '__main__':
    MyApp().run()