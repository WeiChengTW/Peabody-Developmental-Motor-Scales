import pandas as pd

labels_test = pd.read_csv("test\_classes.csv")
labels_valid = pd.read_csv("valid\_classes.csv")
labels_train = pd.read_csv("train\_classes.csv")
labels = pd.concat([labels_test, labels_valid, labels_train], ignore_index=True)

label_columns = labels.columns[1:].tolist()


for col in label_columns:
    print(f"{col}={labels[col].sum()}")
