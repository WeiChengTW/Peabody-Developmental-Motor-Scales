from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  
    model.train(
        data="check_cap\\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="model"
    )

if __name__ == '__main__':
    train()
