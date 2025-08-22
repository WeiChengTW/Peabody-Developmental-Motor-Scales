import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models, regularizers
from TrainingHistory import TrainingHistory as TH
from ConfusionMatrices import ConfusionMatrices as CM
from LoadLabel import LoadLabel as LL
from sklearn.metrics import classification_report
import os
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def extract_macro_f1(y_true, y_pred_bin):
    report = classification_report(y_true, y_pred_bin, output_dict=True, zero_division=0)
    return report['accuracy'], report["macro avg"]["f1-score"]

seeds = [2, 16, 22, 36, 45]

#========================= 定義模型 =========================#
def build_model(PX=100):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(PX, PX, 3)),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.5),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.5),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.5),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(2e-2)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def binary_focal_loss(gamma, alpha):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        modulating = tf.where(tf.equal(y_true, 1), (1 - y_pred) ** gamma, y_pred ** gamma)
        alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(alpha_weight * modulating * bce)
    return focal_loss

#========================= 主程式 =========================#
def main(save):
    PX = 100
    img_size = (PX, PX)

    label = 'upside-down'
    idx = 3

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
    
    print(f"\nTraining model for: {label}")

    
    y_train_single = y_train[:, idx]
    y_val_single = y_val[:, idx]
    y_test_single = y_test[:, idx]


    model = build_model(PX)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=binary_focal_loss(gamma=1.75, alpha=0.55),
        metrics=['accuracy']
    )

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=5e-5)

    checkpoint = ModelCheckpoint(
        filepath=f'pre_models/{label}_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    callback = [lr_scheduler]
    if save:
        checkpoint = ModelCheckpoint(
            filepath=f'pre_models/{label}_best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callback.append(checkpoint)
    
    history = model.fit(
        X_train, y_train_single,
        validation_data=(X_val, y_val_single),
        batch_size=16,
        epochs=50,
        callbacks=callback
    )

    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int)

    acc, result = extract_macro_f1(y_test_single, y_pred_bin)
    print(f"acc : {acc}, f1_score : {result}")
    TH.plot_training_history(history)
    CM.plot_confusion_matrices(y_test_single.reshape(-1, 1), y_pred_bin, [label])
    
    # return f1_score


    
if __name__ == "__main__":
    main(True)
#     f1_scores = []
#     save = False
#     for i, seed in enumerate(seeds):
#         print(f"============================ 第 {i + 1} 次 訓練 ============================")
#         set_seed(seed)
#         f1_scores.append(main(save))
# print(f"\n平均 Macro F1: {np.mean(f1_scores):.4f}")
# print(f"標準差: {np.std(f1_scores):.4f}")
