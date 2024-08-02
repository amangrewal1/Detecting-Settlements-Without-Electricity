from pathlib import Path


### YOLO Configuration ###
YOLO_MODEL = "yolov8n-seg.pt"


### Data Configuration ###
RAW_DATA_DIR = Path("data/raw/").resolve()
PROCESSED_DATA_DIR = Path("data/processed/").resolve()

### Other Configuration ###
SEED = 42