from ultralytics import YOLO


def train():
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=r"C:\Users\chang\Downloads\circle\graph.v2i.yolov8\data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
    )


if __name__ == "__main__":
    train()
