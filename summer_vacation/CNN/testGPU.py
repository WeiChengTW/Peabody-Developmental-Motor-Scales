import tensorflow as tf
from tensorflow.python.client import device_lib

print("=== TensorFlow GPU 檢測 ===")
print(f"TensorFlow 版本: {tf.__version__}")
print(f"GPU 是否可用: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA 建置: {tf.test.is_built_with_cuda()}")

print("\n=== 可用設備列表 ===")
print(device_lib.list_local_devices())

print("\n=== GPU 記憶體資訊 ===")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"GPU: {gpu}")