from ultralytics import YOLO

def train():
    model = YOLO('yolov8n.pt')
    model.train(
        data = 'data.yaml', 
        epochs = 150,
        imgsz = 640,
        batch = 16
    )

if __name__ == "__main__":
    train()