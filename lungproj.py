from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from datetime import datetime

DIAGNOSIS_DB = 'diagnosis_history.json'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'dcm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_diagnosis_history():
    try:
        with open(DIAGNOSIS_DB, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_diagnosis_history(history):
    with open(DIAGNOSIS_DB, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

diagnosis_history = load_diagnosis_history()

try:
    model = tf.keras.models.load_model('models/best_lung_model.h5')
    print("Модель загружена успешно!")
except:
    print("Создаем временную модель...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save('models/best_lung_model.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0) 
        print(f"Изображение обработано. Форма: {img_array.shape}")
        return img_array  
    except Exception as e:
        print(f"Ошибка обработки изображения: {e}")
        return None
    
def predict_disease(image_path):
    try:
        img_array = preprocess_image(image_path)
        if img_array is None:
            return "Ошибка обработки изображения", 0.0
        prediction = model.predict(img_array, verbose=0)
        print(f"Raw prediction: {prediction}")
        if prediction[0][0] > 0.5:
            diagnosis = "Пневмония"
            probability = float(prediction[0][0])
        else:
            diagnosis = "Здоров"
            probability = float(1 - prediction[0][0])
        probability_percent = round(probability * 100, 2)
        return diagnosis, probability_percent
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return "Ошибка предсказания", 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def view_history():
    return render_template('index.html', history=diagnosis_history)

@app.route('/clear_history')
def clear_history():
    global diagnosis_history
    diagnosis_history = []
    save_diagnosis_history(diagnosis_history)
    return render_template('index.html', message='История очищена!', history=[])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file selected')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No file selected')
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        diagnosis, probability = predict_disease(filepath)
        history_entry = {
            'id': len(diagnosis_history) + 1,
            'filename': filename,
            'diagnosis': diagnosis,
            'probability': probability,
            'timestamp': datetime.now().strftime('%d.%m.%Y %H:%M'),
            'image_path': filename 
        }
        diagnosis_history.append(history_entry)
        save_diagnosis_history(diagnosis_history)
        return render_template('index.html', 
                             message='File uploaded successfully!',
                             prediction=f"{diagnosis} (вероятность: {probability}%)",
                             image_path=filepath,
                             history=diagnosis_history[-5:])
    return render_template('index.html', message='Invalid file type')

if __name__ == '__main__':
    app.run(debug=True)

