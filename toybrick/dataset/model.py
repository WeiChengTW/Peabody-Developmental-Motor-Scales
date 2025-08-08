from ultralytics import YOLO

def train():
    model = YOLO('yolov8n-seg.pt')
    model.train(
        data = 'data.yaml',
        epochs = 100,
        batch = 32,
        imgsz = 640
    )

if __name__ == '__main__':
    train()