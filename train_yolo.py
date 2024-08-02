import argparse
from pathlib import Path
from typing import NamedTuple
from ultralytics import YOLO

# DONE: Figure out what we need to pass in from the command line (from loop.py)
class YoloRunSettings(NamedTuple):
    model: str
    data: str
    epochs: int
    learning_rate: float

# DONE: Parse the command line arguments into a YoloRunSettings object
def parse_args() -> YoloRunSettings:
    parser = argparse.ArgumentParser(description='YOLOv8 Parser')
    
    parser.add_argument('--model', type=str, help='pretrained(.pt) or custom(.yaml)')
    parser.add_argument('--data', type=str, help='training data(.yaml)')
    parser.add_argument('--epochs', type=int, help='epochs')
    parser.add_argument('--lr0', type=float, help='learning rate')
    args = parser.parse_args()
    model = args.model
    data = args.data
    epochs = int(args.epochs)
    lr = float(args.lr0)
    
    return YoloRunSettings(model=model, data=data, epochs=epochs, learning_rate=lr)

# DONE: Train YOLO using the given settings
# Refer to https://docs.ultralytics.com/modes/train/
def train_yolo(settings: YoloRunSettings):
    model = YOLO(settings.model)
    results = model.train(
        data=settings.data, 
        epochs=settings.epochs, 
        lr0=settings.learning_rate,
        batch=64,
        workers=8,
        project='runs',
        name=Path(settings.data).parent.stem
    )

def main():
    args = parse_args()
    print("Running YOLO: ", args)
    train_yolo(args)

if __name__ == "__main__":
    main()
