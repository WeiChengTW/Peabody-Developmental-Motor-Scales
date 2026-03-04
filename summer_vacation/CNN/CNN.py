import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers  # 改為從 tensorflow.keras 導入
import numpy as np
import matplotlib.pyplot as plt
# GPU 設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"使用 GPU: {gpus}")
    except RuntimeError as e:
        print(f"GPU 設定失敗: {e}")
        
# 檢查是否有可用的 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("✅ 使用 GPU：", physical_devices[0])
else:
    print("⚠️ 使用 CPU")

# 參數
img_size = (224, 224)  # ResNet50預設尺寸
batch_size = 16
epochs = 30

# 資料增強＆標準化
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    r"train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)
val_gen = val_datagen.flow_from_directory(
    r"train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)
num_classes = train_gen.num_classes

# 載入ResNet50模型，不含頂層分類層
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 只做特徵萃取，先不微調

# 設計自訂分類層
model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# 編譯模型
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# callbacks
earlystop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)

# 訓練
history = model.fit(
    train_gen, epochs=epochs, validation_data=val_gen, callbacks=[earlystop, reduce_lr]
)

# 儲存類別名稱與模型
class_names = list(train_gen.class_indices.keys())
np.save("class_names.npy", class_names)
model.save("triangle_resnet50_model.keras")

# 畫 acc 曲線
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.show()
