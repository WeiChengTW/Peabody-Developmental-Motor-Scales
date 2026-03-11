import pandas as pd
labels = pd.read_csv('labels.csv')  # 或讀你label檔案
print(labels['asymmetric'].sum(), labels['nonstraight'].sum(), labels['nonrightangle'].sum())
