import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, 
    Dropout, 
    GlobalAveragePooling2D,
    BatchNormalization,
    Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 參數設定
# -------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15  # 第一階段
EPOCHS_STAGE2 = 20  # 第二階段
TRAIN_DIR = 'train'
VALID_DIR = 'valid'
NUM_CLASSES = len(os.listdir(TRAIN_DIR))

print(f"檢測到 {NUM_CLASSES} 個類別")

# -------------------------
# 改進的資料增強策略
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,           # 降低旋轉角度
    width_shift_range=0.2,       # 降低平移範圍
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,              # 降低縮放範圍
    shear_range=0.15,            # 降低剪切變換
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]  # 減少亮度變化
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

print("類別對應：", train_generator.class_indices)
print(f"訓練樣本數: {train_generator.samples}")

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

print(f"驗證樣本數: {valid_generator.samples}")

# -------------------------
# 改進的 ResNet50 模型（更簡潔）
# -------------------------
def create_improved_resnet50_model(num_classes, input_shape=(224, 224, 3)):
    """
    建立改進的 ResNet50 模型
    - 簡化分類頭（防止過擬合）
    - 適當的正則化
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    inputs = base_model.input
    x = base_model.output
    
    # 使用單一池化層
    x = GlobalAveragePooling2D()(x)
    
    # 簡化的分類頭（減少層數）
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    # 輸出層
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model

# 建立模型
model, base_model = create_improved_resnet50_model(NUM_CLASSES)

print("ResNet50 基礎模型層數:", len(base_model.layers))
print("完整模型層數:", len(model.layers))
model.summary()

# -------------------------
# 第一階段：訓練分類頭
# -------------------------
print("\n" + "="*70)
print("第一階段：訓練分類頭（凍結 ResNet50）")
print("="*70)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop_stage1 = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_stage1 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

checkpoint_stage1 = ModelCheckpoint(
    'best_model_stage1.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print(f"可訓練參數: {model.count_params():,}")

history_1 = model.fit(
    train_generator,
    epochs=EPOCHS_STAGE1,
    validation_data=valid_generator,
    callbacks=[early_stop_stage1, reduce_lr_stage1, checkpoint_stage1],
    verbose=1
)

print(f"\n第一階段完成！最佳驗證準確率: {max(history_1.history['val_accuracy']):.4f}")

# -------------------------
# 第二階段：微調 ResNet50
# -------------------------
print("\n" + "="*70)
print("第二階段：微調 ResNet50")
print("="*70)

# 載入第一階段最佳權重
model.load_weights('best_model_stage1.h5')

# 解凍 ResNet50 後半部分
base_model.trainable = True

# 只微調最後 50 層（ResNet50 共 175 層）
fine_tune_at = len(base_model.layers) - 50

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"解凍層數: {len(base_model.layers) - fine_tune_at}")
print(f"可訓練參數: {model.count_params():,}")

# 使用更小的學習率
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 降低學習率
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 第二階段 Callbacks
early_stop_stage2 = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_stage2 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-8,
    verbose=1
)

checkpoint_stage2 = ModelCheckpoint(
    'best_model_final.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history_2 = model.fit(
    train_generator,
    epochs=EPOCHS_STAGE2,
    validation_data=valid_generator,
    callbacks=[early_stop_stage2, reduce_lr_stage2, checkpoint_stage2],
    verbose=1,
    initial_epoch=len(history_1.history['loss'])
)

print(f"\n第二階段完成！最佳驗證準確率: {max(history_2.history['val_accuracy']):.4f}")

# 載入最佳模型
model.load_weights('best_model_final.h5')

# -------------------------
# 合併訓練歷史
# -------------------------
total_history = {}
for key in history_1.history.keys():
    total_history[key] = history_1.history[key] + history_2.history[key]

# -------------------------
# 儲存最終模型
# -------------------------
model.save('q_or_other_final.h5')
print("\n最終模型已儲存為 q_or_other_final.h5")

# -------------------------
# 繪製訓練結果
# -------------------------
def plot_resnet_training_history(history, phase_1_epochs, save_plots=True):
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    train_loss = history['loss']
    train_acc = history['accuracy']
    val_loss = history['val_loss']
    val_acc = history['val_accuracy']
    epochs_range = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss 曲線
    ax1.plot(epochs_range, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.axvline(x=phase_1_epochs, color='gray', linestyle='--', alpha=0.7, 
                label=f'Fine-tuning starts (Epoch {phase_1_epochs})')
    ax1.set_title('ResNet50 Model Loss', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    min_val_loss_epoch = np.argmin(val_loss) + 1
    min_val_loss = np.min(val_loss)
    ax1.scatter(min_val_loss_epoch, min_val_loss, color='red', s=100, zorder=5)
    
    # Accuracy 曲線
    ax2.plot(epochs_range, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.axvline(x=phase_1_epochs, color='gray', linestyle='--', alpha=0.7,
                label=f'Fine-tuning starts (Epoch {phase_1_epochs})')
    ax2.set_title('ResNet50 Model Accuracy', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    max_val_acc_epoch = np.argmax(val_acc) + 1
    max_val_acc = np.max(val_acc)
    ax2.scatter(max_val_acc_epoch, max_val_acc, color='red', s=100, zorder=5)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('resnet50_training_history.png', dpi=300, bbox_inches='tight')
        print("\n訓練歷史圖表已儲存")
    
    plt.show()
    
    # 訓練總結
    print("\n" + "="*70)
    print("ResNet50 遷移學習訓練總結")
    print("="*70)
    print(f"第一階段 (凍結基礎模型): {phase_1_epochs} epochs")
    print(f"第二階段 (微調): {len(train_loss) - phase_1_epochs} epochs")
    print(f"總訓練週期: {len(train_loss)}")
    print()
    print(f"最終訓練 Loss: {train_loss[-1]:.4f}")
    print(f"最終訓練 Accuracy: {train_acc[-1]:.4f}")
    print(f"最終驗證 Loss: {val_loss[-1]:.4f}")
    print(f"最終驗證 Accuracy: {val_acc[-1]:.4f}")
    print()
    print(f"最佳驗證 Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch})")
    print(f"最佳驗證 Accuracy: {max_val_acc:.4f} (Epoch {max_val_acc_epoch})")
    
    phase1_end_val_acc = val_acc[phase_1_epochs-1] if phase_1_epochs <= len(val_acc) else val_acc[-1]
    final_val_acc = val_acc[-1]
    improvement = final_val_acc - phase1_end_val_acc
    
    print()
    print(f"第一階段結束時驗證準確率: {phase1_end_val_acc:.4f}")
    print(f"微調後改善程度: {improvement:+.4f}")
    
    acc_gap = train_acc[-1] - val_acc[-1]
    
    print()
    if acc_gap > 0.15:
        print("⚠️  嚴重過擬合 (訓練準確率比驗證準確率高 >15%)")
        print("   建議: 增加 Dropout、增加訓練資料、或減少模型複雜度")
    elif acc_gap > 0.1:
        print("⚠️  中度過擬合 (訓練準確率比驗證準確率高 10-15%)")
        print("   建議: 調整正則化參數、增加數據增強")
    elif acc_gap > 0.05:
        print("✓  輕微過擬合 (訓練準確率比驗證準確率高 5-10%)")
        print("   建議: 可接受範圍，可微調正則化參數")
    else:
        print("✓  模型表現良好，沒有明顯過擬合")
    
    print("="*70)

plot_resnet_training_history(total_history, phase_1_epochs=len(history_1.history['loss']))

# -------------------------
# 模型評估
# -------------------------
print("\n" + "="*50)
print("最終模型評估")
print("="*50)

val_loss, val_accuracy = model.evaluate(valid_generator, verbose=1)
print(f"\n驗證集 Loss: {val_loss:.4f}")
print(f"驗證集 Accuracy: {val_accuracy:.4f}")

# -------------------------
# 預測範例
# -------------------------
print("\n" + "="*50)
print("進行預測測試")
print("="*50)

# 取一批驗證數據
x_val, y_val = next(valid_generator)
predictions = model.predict(x_val, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# 顯示前 5 個預測結果
print("\n前 5 個預測結果:")
class_names = list(train_generator.class_indices.keys())
for i in range(min(5, len(predicted_classes))):
    true_class = int(y_val[i])
    pred_class = predicted_classes[i]
    confidence = predictions[i][pred_class]
    
    print(f"樣本 {i+1}:")
    print(f"  真實類別: {class_names[true_class]}")
    print(f"  預測類別: {class_names[pred_class]}")
    print(f"  信心度: {confidence:.2%}")
    print(f"  {'✓ 正確' if true_class == pred_class else '✗ 錯誤'}")
    print()

print("="*50)
print("訓練完成！")
print("="*50)