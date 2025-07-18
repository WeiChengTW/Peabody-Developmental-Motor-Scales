from ultralytics import YOLO

def train():
    model = YOLO('yolov8n-seg.pt')
    model.train(
        data = 'dataset\\data.yaml',
        epochs = 200,
        batch = 16,
        imgsz = 640
    )

if __name__ == '__main__':
    train()