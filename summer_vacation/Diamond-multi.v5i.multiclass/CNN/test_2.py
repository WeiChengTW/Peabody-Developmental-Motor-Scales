import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score as sklearn_f1_score,
    classification_report,
)

# 載入模型時不編譯，避免自定義函數問題
models = {
    # "gap": load_model("../models/gap_best_model.keras", compile=False),
    "Unlabeled": load_model("../models/unlabeled_best_model.keras", compile=False),
}

# 只處理有可用模型的特徵
available_features = list(models.keys())
# 特徵名稱對應的中文說明
feature_names = {
    "Sharp corner": "尖角",
    "Unlabeled": "未標記",
    "gap": "GAP",
}
print(f"可用的特徵模型: {[feature_names[f] for f in available_features]}")
print("=" * 60)

test_folder = "valid"  # 測試資料夾名稱
extensions = (".jpg", ".jpeg", ".png")

df = pd.read_csv(rf"../{test_folder}/_classes.csv")
label_columns = df.columns[1:].tolist()

print(f"CSV中的所有標籤列: {label_columns}")
print(f"將只對有模型的特徵進行預測: {available_features}")
print("=" * 60)


# 用於收集所有預測和真實標籤的字典
all_predictions = {feature: [] for feature in available_features}
all_true_labels = {feature: [] for feature in available_features}
all_confidences = {feature: [] for feature in available_features}

# 預載入所有圖片以提高效率
print("載入所有圖片...")
all_images = []
valid_rows = []

for idx, row in df.iterrows():
    img_path = os.path.join(f"../{test_folder}", row[0])
    if not img_path.lower().endswith(extensions):
        continue

    try:
        # 載入並預處理圖片
        img = image.load_img(img_path, target_size=(100, 100))
        x = image.img_to_array(img) / 255.0
        all_images.append(x)
        valid_rows.append(row)
    except Exception as e:
        print(f"無法載入圖片 {img_path}: {e}")
        continue

# 轉換為批次格式
X_batch = np.array(all_images)

print(f"成功載入 {len(all_images)} 張圖片")
print("開始進行批次預測...")
print("=" * 60)

# 對每個特徵模型進行批次預測
batch_predictions = {}
for feature in available_features:
    print(f"預測 {feature_names[feature]}...")
    model = models[feature]
    predictions = model.predict(X_batch, verbose=0)
    batch_predictions[feature] = predictions

# 處理預測結果
for i, row in enumerate(valid_rows):
    print(f"Image: {row[0]}")
    print("-" * 50)

    # 對每個特徵顯示結果
    for feature in available_features:
        feature_index = label_columns.index(feature)
        pred_confidence = batch_predictions[feature][i][0]
        pred_binary = 1 if pred_confidence >= 0.5 else 0
        true_label = row[feature_index + 1]

        # 收集預測結果和真實標籤
        all_predictions[feature].append(pred_binary)
        all_true_labels[feature].append(true_label)
        all_confidences[feature].append(pred_confidence)

        status = "✓" if pred_binary == true_label else "✗"
        print(
            f"{feature_names[feature]:8} - 預測: {'是' if pred_binary else '否'} (信心度: {pred_confidence:.3f}) | 實際: {'是' if true_label else '否'} {status}"
        )

    print()

# 計算並顯示每個特徵的正確率和 F1 分數
print("\n" + "=" * 60)
print("各特徵預測結果統計")
print("=" * 60)

overall_accuracy = []
overall_f1 = []
valid_f1_features = []

for feature in available_features:
    y_true = all_true_labels[feature]
    y_pred = all_predictions[feature]
    confidences = all_confidences[feature]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = sklearn_f1_score(y_true, y_pred, zero_division=0)

    # 計算正樣本數量和平均信心度
    positive_count = sum(y_true)
    total_count = len(y_true)
    avg_confidence = np.mean(confidences)

    overall_accuracy.append(accuracy)
    overall_f1.append(f1)

    # 只有當正樣本數量合理時才計入F1平均值
    if positive_count >= 3 and f1 > 0:
        valid_f1_features.append(f1)

    print(
        f"{feature_names[feature]:8} - 正確率: {accuracy:.3f} ({accuracy*100:.1f}%) | F1: {f1:.3f} | 正樣本: {positive_count:2d}/{total_count:2d} | 平均信心度: {avg_confidence:.3f}"
    )

# 計算平均值
avg_accuracy = np.mean(overall_accuracy)
avg_f1_all = np.mean(overall_f1)
avg_f1_valid = np.mean(valid_f1_features) if valid_f1_features else 0

print("-" * 60)
print(f"平均正確率: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
print(f"平均F1分數 (全部): {avg_f1_all:.3f}")
print(f"平均F1分數 (有效): {avg_f1_valid:.3f} (排除樣本不足或F1=0的特徵)")
print("-" * 60)

# 數據平衡情況分析
print("\n數據平衡情況分析:")
print("=" * 60)
for feature in available_features:
    y_true = all_true_labels[feature]
    positive_count = sum(y_true)
    negative_count = len(y_true) - positive_count
    balance_ratio = positive_count / len(y_true) * 100

    if balance_ratio < 20:
        status = "⚠️  嚴重不平衡"
    elif balance_ratio < 35:
        status = "⚠️  輕度不平衡"
    else:
        status = "✅ 相對平衡"

    print(
        f"{feature_names[feature]:8} - 是:{positive_count:2d} 否:{negative_count:2d} ({balance_ratio:.1f}%) {status}"
    )

print("=" * 60)

# 顯示詳細的分類報告
print("\n詳細分類報告:")
print("=" * 60)
for feature in available_features:
    y_true = all_true_labels[feature]
    y_pred = all_predictions[feature]

    print(f"\n{feature_names[feature]} ({feature}):")
    print(
        classification_report(
            y_true, y_pred, target_names=["否", "是"], zero_division=0
        )
    )
