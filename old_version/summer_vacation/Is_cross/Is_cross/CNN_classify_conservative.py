import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
import os

# 檢查圖片目錄
image_dir = "practice" if os.path.exists("practice") else "test"
print(f"使用圖片目錄: {image_dir}")

# 讀取標註
df = pd.read_csv("labels.csv")

# 分析標籤分佈
print("標籤分佈統計:")
label_counts = df[["asymmetric", "nonstraight", "nonrightangle"]].sum()
print(label_counts)
print("\n標籤比例:")
label_ratios = df[["asymmetric", "nonstraight", "nonrightangle"]].mean()
print(label_ratios)

# 簡單的隨機切分（避免分層切分可能的問題）
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

print(f"\n訓練集大小: {len(df_train)}, 驗證集大小: {len(df_val)}")

# 檢查驗證集的標籤分佈
print("\n驗證集標籤分佈:")
val_label_counts = df_val[["asymmetric", "nonstraight", "nonrightangle"]].sum()
print(val_label_counts)

# 最小化的資料增強
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,  # 極小的旋轉
    width_shift_range=0.05,  # 極小的位移
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 小batch size
batch_size = 8

try:
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=image_dir,
        x_col="filename",
        y_col=["asymmetric", "nonstraight", "nonrightangle"],
        target_size=(100, 100),  # 較小的圖片尺寸
        class_mode="raw",
        batch_size=batch_size,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        directory=image_dir,
        x_col="filename",
        y_col=["asymmetric", "nonstraight", "nonrightangle"],
        target_size=(100, 100),
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"資料生成器創建成功")

except Exception as e:
    print(f"創建資料生成器時出錯: {e}")
    exit()

# 非常簡單的模型架構 - 防止過擬合
model = Sequential(
    [
        # 第一個卷積塊
        Conv2D(16, (5, 5), activation="relu", input_shape=(100, 100, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        # 第二個卷積塊
        Conv2D(32, (5, 5), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        # 第三個卷積塊
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        # 全連接層 - 非常簡單
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(3, activation="sigmoid"),  # 多標籤輸出
    ]
)

# 較大的學習率，更簡單的優化
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"],  # 只用accuracy，避免其他指標的複雜性
)

print("\n簡化模型架構:")
model.summary()

# 更積極的早停策略
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,  # 更小的patience
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,  # 更小的patience
        min_lr=1e-6,
        verbose=1,
    ),
]

# 訓練模型
print("\n開始訓練...")
try:
    history = model.fit(
        train_gen,
        epochs=30,  # 減少epochs
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # 保存模型
    model.save("cross_multilabel_simple.h5")
    print("\n模型已保存為 'cross_multilabel_simple.h5'")

    # 顯示訓練結果
    print(f"\n訓練完成!")
    if len(history.history["loss"]) > 0:
        print(f"最終訓練損失: {history.history['loss'][-1]:.4f}")
        print(f"最終驗證損失: {history.history['val_loss'][-1]:.4f}")
        print(f"最終訓練準確率: {history.history['accuracy'][-1]:.4f}")
        print(f"最終驗證準確率: {history.history['val_accuracy'][-1]:.4f}")

    # 手動測試幾個樣本
    print("\n手動測試驗證集前5個樣本:")
    val_sample = df_val.head()
    for idx, row in val_sample.iterrows():
        filename = row["filename"]
        true_labels = [row["asymmetric"], row["nonstraight"], row["nonrightangle"]]

        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array

            img_path = os.path.join(image_dir, filename)
            if os.path.exists(img_path):
                img = load_img(img_path, target_size=(100, 100))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                pred = model.predict(img_array, verbose=0)[0]
                pred_binary = (pred > 0.5).astype(int)

                print(
                    f"{filename}: 真實={true_labels}, 預測={pred_binary}, 信心度={pred}"
                )
            else:
                print(f"{filename}: 文件不存在")
        except Exception as e:
            print(f"{filename}: 預測錯誤 - {e}")

except Exception as e:
    print(f"訓練過程中出錯: {e}")
    import traceback

    traceback.print_exc()
