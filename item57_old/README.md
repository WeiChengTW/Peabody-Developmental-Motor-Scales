# 剪紙評分系統使用說明

## 概述

這個系統可以自動分析剪紙照片並給出評分：

- **評分 2**: 把紙剪成均分的 2 等份
- **評分 1**: 只剪到紙的 1/4 或更少
- **評分 0**: 只動動剪刀未剪下去

## 文件結構

```
ch3-t3/
├── cut_paper.py                 # 主要的完整分析器
├── score_2_complete_cut.py      # 專門檢測完全剪切（評分2）
├── score_1_partial_cut.py       # 專門檢測部分剪切（評分1）
├── score_0_no_cut.py           # 專門檢測無實際剪切（評分0）
├── main_scorer.py              # 整合所有檢測器的主程式
├── requirements.txt            # 依賴套件列表
└── README.md                   # 本說明文件
```

## 安裝依賴

首先安裝必要的 Python 套件：

```bash
pip install -r requirements.txt
```

或者手動安裝：

```bash
pip install opencv-python numpy pillow
```

## 使用方法

### 1. 使用整合主程式（推薦）

```python
python main_scorer.py
```

這會啟動互動式介面，您可以：

- 分析單張圖像
- 批量分析整個資料夾
- 生成詳細報告和視覺化結果

### 2. 程式化使用

```python
from main_scorer import PaperCuttingScorer

# 創建評分器
scorer = PaperCuttingScorer()

# 分析單張圖像
score = scorer.score_image("path/to/your/image.jpg")
print(f"評分: {score}")

# 批量分析
results = scorer.batch_score("path/to/image/folder")
for filename, score in results.items():
    print(f"{filename}: {score}")
```

### 3. 使用個別檢測器

如果您只想檢測特定類型的剪切：

```python
# 檢測完全剪切（評分2）
from score_2_complete_cut import CompleteCutDetector
detector2 = CompleteCutDetector()
is_complete = detector2.detect_complete_cut("image.jpg")

# 檢測部分剪切（評分1）
from score_1_partial_cut import PartialCutDetector
detector1 = PartialCutDetector()
is_partial = detector1.detect_partial_cut("image.jpg")

# 檢測無實際剪切（評分0）
from score_0_no_cut import NoActualCutDetector
detector0 = NoActualCutDetector()
is_no_cut = detector0.detect_no_actual_cut("image.jpg")
```

## 評分標準詳解

### 評分 2 - 完全剪切

**特徵識別：**

- 紙張被完全分離成兩個獨立的部分
- 兩部分的面積大致相等（允許 30%誤差）
- 使用連通區域分析檢測分離的部分

**技術實現：**

- 連通區域檢測
- 面積比例計算
- 分離度驗證

### 評分 1 - 部分剪切

**特徵識別：**

- 檢測到剪切線但未完全分離
- 剪切深度在 5%-25%之間
- 剪切從紙張邊緣開始

**技術實現：**

- 霍夫直線檢測找到剪切線
- 計算剪切深度相對紙張尺寸的比例
- 邊緣起始點檢測

### 評分 0 - 無實際剪切

**特徵識別：**

- 紙張保持基本完整
- 輪廓相對簡單（類矩形）
- 只有最小程度的表面干擾

**技術實現：**

- 輪廓簡化度分析
- 形態學變化檢測
- 紙張完整性評估

## 輸出結果

### 1. 控制台輸出

程式會在控制台顯示詳細的分析過程和結果：

```
正在分析圖像: test_image.jpg
==================================================
✓ 檢測到部分剪切：test_image.jpg
  - 剪切深度比例: 0.180
  - 檢測到的剪切線數量: 2
  - 從邊緣開始剪切: True
最終評分: 1 (部分剪切 - 剪到1/4或更少)
```

### 2. 視覺化結果

程式會生成帶有分析標記的圖像：

- 綠色輪廓：紙張邊界
- 紅色直線：檢測到的剪切線
- 藍色文字：評分和統計信息

### 3. 詳細報告

批量分析後會生成文字報告：

```
剪紙評分詳細報告
========================================

總圖像數量: 10

評分分布:
- 評分 2 (完全剪切): 3 張 (30.0%)
- 評分 1 (部分剪切): 4 張 (40.0%)
- 評分 0 (無實際剪切): 3 張 (30.0%)
```

## 參數調整

如果需要調整檢測靈敏度，可以修改各檢測器的參數：

### CompleteCutDetector 參數

```python
self.min_separation_ratio = 0.8  # 最小分離比例
self.min_area_ratio = 0.1        # 每部分最小面積比例
```

### PartialCutDetector 參數

```python
self.max_cut_ratio = 0.25        # 最大剪切比例（1/4）
self.min_cut_length = 20         # 最小剪切長度（像素）
```

### NoActualCutDetector 參數

```python
self.motion_threshold = 0.02     # 動作檢測閾值
self.min_cut_evidence = 5        # 最小剪切證據像素數
```

## 故障排除

### 常見問題

1. **無法檢測到紙張**

   - 確保照片中紙張與背景有足夠對比度
   - 檢查照片是否清晰，避免模糊

2. **評分不準確**

   - 調整檢測閾值參數
   - 確保照片角度適當，避免嚴重傾斜

3. **處理速度慢**
   - 降低圖像解析度
   - 批量處理時使用較小的圖像

### 支援的圖像格式

- JPG/JPEG
- PNG
- BMP
- TIFF/TIF

## 技術細節

### 主要使用的 OpenCV 技術

- **邊緣檢測**: Canny 邊緣檢測
- **輪廓分析**: findContours + 輪廓特徵分析
- **直線檢測**: 霍夫變換 (HoughLinesP)
- **形態學操作**: 開運算、閉運算
- **連通區域分析**: connectedComponentsWithStats

### 性能考量

- 圖像預處理：高斯模糊 + 二值化
- 多尺度分析：適應不同解析度的圖像
- 魯棒性設計：多重驗證機制避免誤判

## 聯絡資訊

如有問題或建議，請聯絡開發團隊。
