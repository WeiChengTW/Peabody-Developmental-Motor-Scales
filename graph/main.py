import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models, regularizers
from TrainingHistory import TrainingHistory as TH
from ConfusionMatrices import ConfusionMatrices as CM
from LoadLabel import LoadLabel as LL
from sklearn.metrics import classification_report
import os

#========================= 定義模型 =========================#
def build_model(PX=100):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(PX, PX, 3)),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.22),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.22),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.22),
        layers.MaxPooling2D(2, 2),

        # layers.Conv2D(64, (3, 3), activation='relu'),
        # layers.BatchNormalization(),
        # layers.SpatialDropout2D(0.4),
        # layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(3e-2)),
        layers.Dropout(0.7),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

#========================= 主程式 =========================#
def main():
    PX = 100
    img_size = (PX, PX)

    train_df = pd.read_csv(r"Data/train/_classes.csv")
    valid_df = pd.read_csv(r"Data/valid/_classes.csv")
    test_df = pd.read_csv(r"Data/test/_classes.csv")

    train_df.columns = train_df.columns.str.strip()
    valid_df.columns = valid_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    label_columns = train_df.columns[1:].tolist()

    X_train, y_train = LL.load_images_and_labels(train_df, img_size, label_columns, img_dir=r'Data/train')
    X_val, y_val = LL.load_images_and_labels(valid_df, img_size, label_columns, img_dir=r'Data/valid')
    X_test, y_test = LL.load_images_and_labels(test_df, img_size, label_columns, img_dir=r'Data/test')

    os.makedirs("models", exist_ok=True)
    
    #每個特徵訓練一個模型
    for i, label in enumerate(label_columns):
        print(f"\nTraining model for: {label}")

        y_train_single = y_train[:, i]
        y_val_single = y_val[:, i]
        y_test_single = y_test[:, i]

        model = build_model(PX)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=5e-5)

        checkpoint = ModelCheckpoint(
            filepath=f'models/{label}_best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )

        history = model.fit(
            X_train, y_train_single,
            validation_data=(X_val, y_val_single),
            batch_size=32,
            epochs=50,
            callbacks=[lr_scheduler, checkpoint]
        )

        TH.plot_training_history(history)

        y_pred = model.predict(X_test)
        y_pred_bin = (y_pred > 0.5).astype(int)

        # CM.plot_confusion_matrices(y_test_single.reshape(-1, 1), y_pred_bin, [label])
        # report = classification_report(y_test_single, y_pred_bin, target_names=[label], zero_division=0)
        # print(report)

if __name__ == "__main__":
    main()
