import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.TrainingHistory import TrainingHistory as TH
from lib.ConfusionMatrices import ConfusionMatrices as CM
from lib.LoadLabel import LoadLabel as LL
from sklearn.metrics import (
    accuracy_score,
    f1_score as sklearn_f1_score,
    classification_report,
)


def binary_focal_loss(gamma, alpha):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        modulating = tf.where(tf.equal(y_true, 1), (1 - y_pred) ** gamma, y_pred**gamma)
        alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(alpha_weight * modulating * bce)

    return focal_loss


def main():
    # 載入 CSV
    train_df = pd.read_csv(r"../train/_classes.csv")
    train_df.columns = train_df.columns.str.strip()
    valid_df = pd.read_csv(r"../valid/_classes.csv")
    valid_df.columns = valid_df.columns.str.strip()
    test_df = pd.read_csv(r"../test/_classes.csv")
    test_df.columns = test_df.columns.str.strip()

    # 只使用 no_line 特徵
    tag = "no line"
    label_columns = [tag]

    PX = 100
    # 圖片大小
    img_size = (PX, PX)

    # 分割訓練集與驗證集
    X_train, y_train = LL.load_images_and_labels(
        train_df, img_size, label_columns, img_dir=r"../train"
    )
    X_val, y_val = LL.load_images_and_labels(
        valid_df, img_size, label_columns, img_dir=r"../valid"
    )
    X_test, y_test = LL.load_images_and_labels(
        test_df, img_size, label_columns, img_dir=r"../test"
    )

    model = models.Sequential(
        [
            layers.Conv2D(16, (3, 3), activation="relu", input_shape=(PX, PX, 3)),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.4),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.4),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.4),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(
                128, activation="relu", kernel_regularizer=regularizers.l2(1e-2)
            ),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),  # 只有一個輸出節點for Asymmetry
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=binary_focal_loss(gamma=2.5, alpha=0.7),
        metrics=["accuracy"],
    )

    model.summary()

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5
    )

    # file_count = len([f for f in os.listdir('model') if os.path.isfile(os.path.join('model', f))])
    # lr_scheduler = ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    # )

    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    # )
    checkpoint = ModelCheckpoint(
        filepath=f"../raw_models/{tag}_best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=50,
        callbacks=[lr_scheduler, checkpoint],
    )

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 為了配合 ConfusionMatrices，確保 y_test 和 y_pred_binary 都是正確的形狀
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        # 只取 Asymmetry 列 (第一列)
        y_test_asymmetry = y_test[:, 0:1]
    else:
        y_test_asymmetry = y_test

    # CM.plot_confusion_matrices(y_test_asymmetry, y_pred_binary, label_columns)
    TH.plot_training_history(history, tag)

    print("Classification Report:")
    print(
        classification_report(
            y_test_asymmetry.flatten(),
            y_pred_binary.flatten(),
            target_names=["否", "是"],
        )
    )


if __name__ == "__main__":
    main()
