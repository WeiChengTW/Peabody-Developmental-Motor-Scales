import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 從混淆矩陣計算準確率
# -------------------------

# 你的混淆矩陣數據
confusion_matrix_data = np.array([
    [33, 1],   # 圓形：33個正確，1個錯誤
    [4, 81]    # 其他：4個錯誤，81個正確
])

print("混淆矩陣：")
print(confusion_matrix_data)
print()

# 提取數值
TP = confusion_matrix_data[0, 0]  # True Positive (圓形->圓形)
FN = confusion_matrix_data[0, 1]  # False Negative (圓形->其他)
FP = confusion_matrix_data[1, 0]  # False Positive (其他->圓形)
TN = confusion_matrix_data[1, 1]  # True Negative (其他->其他)

total_samples = TP + TN + FP + FN

print(f"TP (True Positive): {TP}")
print(f"TN (True Negative): {TN}")
print(f"FP (False Positive): {FP}")
print(f"FN (False Negative): {FN}")
print(f"總樣本數: {total_samples}")
print()

# -------------------------
# 計算各種評估指標
# -------------------------

# 1. 整體準確率 (Overall Accuracy)
accuracy = (TP + TN) / total_samples
print(f"整體準確率 (Overall Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

# 2. 圓形類別的精確率 (Precision for Circle)
precision_circle = TP / (TP + FP) if (TP + FP) > 0 else 0
print(f"圓形精確率 (Circle Precision): {precision_circle:.4f} ({precision_circle*100:.2f}%)")

# 3. 圓形類別的召回率 (Recall for Circle)
recall_circle = TP / (TP + FN) if (TP + FN) > 0 else 0
print(f"圓形召回率 (Circle Recall): {recall_circle:.4f} ({recall_circle*100:.2f}%)")

# 4. 圓形類別的F1分數
f1_circle = 2 * (precision_circle * recall_circle) / (precision_circle + recall_circle) if (precision_circle + recall_circle) > 0 else 0
print(f"圓形F1分數 (Circle F1-Score): {f1_circle:.4f} ({f1_circle*100:.2f}%)")

print("\n" + "="*50)

# 5. 其他類別的精確率
precision_other = TN / (TN + FN) if (TN + FN) > 0 else 0
print(f"其他精確率 (Other Precision): {precision_other:.4f} ({precision_other*100:.2f}%)")

# 6. 其他類別的召回率
recall_other = TN / (TN + FP) if (TN + FP) > 0 else 0
print(f"其他召回率 (Other Recall): {recall_other:.4f} ({recall_other*100:.2f}%)")

# 7. 其他類別的F1分數
f1_other = 2 * (precision_other * recall_other) / (precision_other + recall_other) if (precision_other + recall_other) > 0 else 0
print(f"其他F1分數 (Other F1-Score): {f1_other:.4f} ({f1_other*100:.2f}%)")

# -------------------------
# 繪製混淆矩陣熱力圖
# -------------------------
def plot_confusion_matrix(cm, class_names=['Circle', 'Other']):
    """繪製混淆矩陣熱力圖"""
    plt.figure(figsize=(8, 6))
    
    # 計算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 創建標籤（數量 + 百分比）
    labels = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                       for j in range(cm.shape[1])] 
                       for i in range(cm.shape[0])])
    
    # 繪製熱力圖
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # 添加統計信息
    plt.figtext(0.02, 0.02, 
                f'Total Samples: {total_samples}\n'
                f'Correct Predictions: {TP + TN}\n'
                f'Wrong Predictions: {FP + FN}',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(confusion_matrix_data)

# -------------------------
# 模型性能評估報告
# -------------------------
print("\n" + "="*60)
print("模型性能評估報告")
print("="*60)

print(f"數據集大小: {total_samples} 張圖片")
print(f"圓形圖片: {TP + FN} 張")
print(f"其他圖片: {TN + FP} 張")
print()

print("分類結果:")
print(f"• 正確預測: {TP + TN} 張 ({((TP + TN)/total_samples)*100:.2f}%)")
print(f"• 錯誤預測: {FP + FN} 張 ({((FP + FN)/total_samples)*100:.2f}%)")
print()

print("錯誤分析:")
print(f"• 圓形被誤判為其他: {FN} 張")
print(f"• 其他被誤判為圓形: {FP} 張")
print()

# 判斷模型性能
if accuracy >= 0.95:
    performance_level = "優秀"
    color_code = "🟢"
elif accuracy >= 0.90:
    performance_level = "良好"
    color_code = "🟡"
elif accuracy >= 0.80:
    performance_level = "一般"
    color_code = "🟠"
else:
    performance_level = "需要改進"
    color_code = "🔴"

print(f"模型性能等級: {color_code} {performance_level}")
print(f"整體準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# -------------------------
# 使用模型進行預測的範例程式碼
# -------------------------
print("\n" + "="*60)
print("在實際使用中計算準確率的程式碼範例:")
print("="*60)

example_code = '''
# 載入訓練好的模型
model = tf.keras.models.load_model('circle_or_oval_resnet50.h5')

# 在測試集上進行預測
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# 獲取真實標籤
true_classes = test_generator.classes

# 計算準確率
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(true_classes, predicted_classes)
print(f"測試集準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 詳細分類報告
report = classification_report(true_classes, predicted_classes)
print("分類報告:")
print(report)
'''

print(example_code)