import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models, regularizers
from TrainingHistory import TrainingHistory as TH
from ConfusionMatrices import ConfusionMatrices as CM
from LoadLabel import LoadLabel as LL
from sklearn.metrics import classification_report
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#========================= 定義模型 =========================#
def build_model(nc, PX=100):

    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(PX, PX, 3)),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.4),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.4),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.4),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.4),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.4),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(2e-2)),
        layers.Dropout(0.3),
        layers.Dense(nc, activation='sigmoid')
    ])
    return model

#========================= 主程式 =========================#
def main():
    PX = 100
    img_size = (PX, PX)

    train_df = pd.read_csv(r"train/_classes.csv")
    valid_df = pd.read_csv(r"valid/_classes.csv")
    test_df = pd.read_csv(r"test/_classes.csv")

    train_df.columns = train_df.columns.str.strip()
    valid_df.columns = valid_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    label_columns = train_df.columns[1:].tolist()

    X_train, y_train = LL.load_images_and_labels(train_df, img_size, label_columns, img_dir=r'train')
    X_val, y_val = LL.load_images_and_labels(valid_df, img_size, label_columns, img_dir=r'valid')
    X_test, y_test = LL.load_images_and_labels(test_df, img_size, label_columns, img_dir=r'test')

    os.makedirs("models", exist_ok=True)

    nc = len(label_columns)
    model = build_model(nc, PX)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 資料增強設定
    datagen = ImageDataGenerator(
        rotation_range=20,       # 隨機旋轉角度
        width_shift_range=0.1,   # 水平平移
        height_shift_range=0.1,  # 垂直平移
        zoom_range=0.1,          # 隨機縮放
        horizontal_flip=True,    # 水平翻轉
        fill_mode='nearest'      # 填補方式
    )

    datagen.fit(X_train)

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=5e-5)
    checkpoint = ModelCheckpoint(
        filepath=f'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[lr_scheduler, checkpoint]
    )

    TH.plot_training_history(history)

if __name__ == "__main__":
    main()