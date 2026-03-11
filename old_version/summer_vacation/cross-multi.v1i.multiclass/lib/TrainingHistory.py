import matplotlib.pyplot as plt
import os


class TrainingHistory:

    @staticmethod
    def plot_training_history(history, label_name=None):
        plt.figure(figsize=(12, 5))

        # 準確率
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        # 損失值
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # 自動檢測 result 資料夾的位置
        if os.path.exists("result"):
            result_path = "result"
        elif os.path.exists("../result"):
            result_path = "../result"
        else:
            # 如果都不存在，創建 result 資料夾
            result_path = "result"
            os.makedirs(result_path, exist_ok=True)

        plt.savefig(rf"{result_path}/{label_name}_training_history.png")
        # plt.show()
