import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -------------------------
# 參數設定
# -------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
TRAIN_DIR = 'train'
VALID_DIR = 'valid'
NUM_CLASSES = len(os.listdir(TRAIN_DIR))  # 自動抓類別數

# -------------------------
# 資料增強與生成器
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,           # 垂直翻轉（需視任務而定）
    zoom_range=0.2,              # 縮放範圍
    shear_range=0.2,             # 剪切變換
    fill_mode='nearest'          # 填充模式
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse'  # single-label 整數標籤
)

print(train_generator.class_indices)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

# -------------------------
# 建立 CNN 模型
# -------------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    
    Conv2D(512, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Conv2D(512, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax') 
])

# -------------------------
# 編譯模型
# -------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # single-label
    metrics=['accuracy']
)

# -------------------------
# 訓練模型
# -------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator
)

# -------------------------
# 儲存模型
# -------------------------
model.save('circle_or_oval_v1.h5')
print("Model saved as circle_or_oval_v1.h5")
