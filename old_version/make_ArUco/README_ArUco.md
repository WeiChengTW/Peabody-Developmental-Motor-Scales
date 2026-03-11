# ArUco 偵測與 1/4 A4 長方形繪製程式

## 功能說明

- 偵測圖片中的 ArUco 標記 (使用 DICT_4X4_50 字典)
- 根據 2.8cm 的 ArUco 標記尺寸自動計算比例尺
- 繪製 1/4 A4 紙張尺寸的長方形 (105mm x 148.5mm)
- **長方形會自動旋轉以與 ArUco 標記保持平行**
- 顯示詳細的尺寸、比例尺和旋轉角度資訊

## 目錄結構

```
item60/
├── detect_aruco_and_draw_quarter_a4.py  # 主程式
├── make_ArUco.py                        # 生成 ArUco 標記
├── test_result_directory.py             # 測試程式
├── requirements.txt                     # 依賴套件
├── img/                                 # 輸入圖片目錄
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── result/                              # 結果輸出目錄 (自動創建)
    ├── 0_quarter_a4_detected.jpg
    ├── 1_quarter_a4_detected.jpg
    └── ...
```

## 使用方法

### 1. 安裝依賴套件

```bash
pip install opencv-python matplotlib numpy reportlab pillow
```

### 2. 執行主程式

```bash
python detect_aruco_and_draw_quarter_a4.py
```

### 3. 測試功能

```bash
python test_result_directory.py
```

## 重要設定

- **ArUco 標記尺寸**: 2.8cm (28mm)
- **1/4 A4 尺寸**: 105mm x 148.5mm
- **結果保存位置**: `result/` 目錄
- **檔案命名格式**: `[原檔名]_quarter_a4_detected.jpg`

## 輸出資訊

程式會在圖片上標註：

- ArUco 標記 ID 和偵測框
- 1/4 A4 長方形 (綠色邊框，與 ArUco 平行)
- 長方形四個角點 (藍色圓點)
- 尺寸資訊 (105x149mm)
- 比例尺資訊 (Scale: 1mm=X.XXpx)
- ArUco 標記尺寸 (ArUco: 2.8cm)
- 旋轉角度 (Angle: X.X°)

## 主要特色

✅ **自動旋轉**: 長方形會根據 ArUco 標記的角度自動旋轉保持平行  
✅ **精確比例尺**: 基於 2.8cm ArUco 標記計算精確的像素/毫米比例  
✅ **詳細資訊**: 提供完整的偵測和計算資訊  
✅ **自動保存**: 結果自動保存到 result 目錄  
✅ **批次處理**: 支援處理整個目錄的圖片

## 注意事項

- 確保 ArUco 標記清晰可見
- ArUco 標記實際尺寸必須為 2.8cm
- 圖片品質會影響偵測精度
- result 目錄會自動創建
