import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import os
import shutil

# ==================== НАСТРОЙКИ ====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  
EPOCHS = 5 
TRAIN_DIR = 'chest_xray/train'
VAL_DIR = 'chest_xray/val_balanced'
TEST_DIR = 'chest_xray/test'

def create_balanced_validation():
    # Создаем новые папки
    os.makedirs('chest_xray/val_balanced/NORMAL', exist_ok=True)
    os.makedirs('chest_xray/val_balanced/PNEUMONIA', exist_ok=True)
    
    # Берем 20% из train для validation
    for class_name in ['NORMAL', 'PNEUMONIA']:
        train_dir = f'chest_xray/train/{class_name}'
        images = os.listdir(train_dir)
        
        # Разделяем 80/20
        train_imgs, val_imgs = train_test_split(
            images, test_size=0.2, random_state=42
        )
        
        # Копируем validation изображения
        for img_name in val_imgs:
            src = os.path.join(train_dir, img_name)
            dst = f'chest_xray/val_balanced/{class_name}/{img_name}'
            shutil.copy2(src, dst)
            
        print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

create_balanced_validation()

# ==================== ПРОВЕРКА ДАННЫХ ====================
print("\nПроверка структуры данных")

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.exists(folder):
        print(f"Ошибка: Папка {folder} не существует!")
        exit()

def analyze_dataset():
    print("\n📊 Анализ датасета:")
    for split in ['train', 'val', 'test']:
        split_dir = f'chest_xray/{split}'
        if os.path.exists(split_dir):
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    count = len(os.listdir(class_dir))
                    print(f"   {split}/{class_name}: {count} изображений")

analyze_dataset()

# ==================== УЛУЧШЕННАЯ АУГМЕНТАЦИЯ ====================
print("\nНастройка аугментации")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,           
    width_shift_range=0.3,       
    height_shift_range=0.3,        
    shear_range=0.3,            
    zoom_range=0.3,              
    horizontal_flip=True,
    vertical_flip=True,          
    brightness_range=[0.8, 1.2], 
    fill_mode='constant',        
    cval=0.0                     
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# ==================== ЗАГРУЗКА ДАННЫХ С БАЛАНСИРОВКОЙ ====================
print("\nЗагрузка данных")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    color_mode='rgb'
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, 
    class_mode='binary',
    shuffle=False,
    color_mode='rgb'
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
    color_mode='rgb'
)

print(f"\nКлассы: {train_generator.class_indices}")
print(f"\nОбучающие примеры: {train_generator.samples}")
print(f"\nВалидационные примеры: {val_generator.samples}")

# ==================== ВЫЧИСЛЕНИЕ ВЕСОВ КЛАССОВ ====================

print("\n Вычисление весов классов...")

train_labels = train_generator.classes
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print(f"Веса классов: {class_weight_dict}")

# ==================== УЛУЧШЕННАЯ АРХИТЕКТУРА МОДЕЛИ ====================
def create_improved_model():
    print("\n Создание модели...")
    
    model = Sequential([
        # Первый блок
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=(224, 224, 3), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Второй блок
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Третий блок
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Четвертый блок
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Классификация
        GlobalAveragePooling2D(),  # Лучше чем Flatten!
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Улучшенная компиляция
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Меньше learning rate!
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
        tf.keras.metrics.Recall(name='recall'), 
        tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Создаем модель
model = create_improved_model()
model.summary()

# ==================== УЛУЧШЕННЫЕ CALLBACKS ====================
print("\nНастройка callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,           # Увеличил терпение
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'models/best_lung_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,            # Уменьшаем learning rate вдвое
        patience=8,            # Ждем 8 эпох без улучшений
        min_lr=0.000001,       # Минимальный learning rate
        verbose=1
    )
]

# ==================== ОБУЧЕНИЕ С ВЕСАМИ КЛАССОВ ====================
print("\nНачало обучения...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weight_dict,  
    verbose=1
)

# ==================== СОХРАНЕНИЕ И ОЦЕНКА ====================
print("\nСохранение модели...")

model.save('models/lung_diagnosis_model.h5')
print("Модель сохранена!")

print("\n📊 Оценка на тестовых данных:")
test_results = model.evaluate(test_generator, verbose=0)

print(f"📈 Тестовые метрики:")
print(f"   - Loss: {test_results[0]:.4f}")
print(f"   - Accuracy: {test_results[1]:.4f}")
print(f"   - Precision: {test_results[2]:.4f}") 
print(f"   - Recall: {test_results[3]:.4f}")
print(f"   - AUC: {test_results[4]:.4f}")

# ==================== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nСохранение графиков обучения...")
plot_training_history(history)

print("\n🎉 Обучение завершено успешно!")
print("✨ Используйте models/best_lung_model.h5 в вашем приложении!")