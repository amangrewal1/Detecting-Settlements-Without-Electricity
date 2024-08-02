from typing import Iterable
import cv2
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt

from pathlib import Path

from data_utils.dataset import SatDataset
from sat_types import BandSelection


class YoloSegmenter:
    """
    Wrapper class for inference using a YOLO model.
    """
    def __init__(self, model_path: Path, batch_size: int = 32) -> None:
        self.model = YOLO(str(model_path))
        self.batch_size = batch_size
        
    def predict(self, images: np.ndarray):
        return self.model(images)
    
    @staticmethod
    def create_visualization(image: np.ndarray, result) -> np.ndarray:
        original_image = image.copy()
        
        for mask in result.masks.xy:
            points = np.array(mask).reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(image, [points], color=(0, 255, 0))
            
        # Blend the two images
        alpha = 0.5
        cv2.addWeighted(image, alpha, original_image, 1 - alpha, 0, image)
        
        return np.array(image)

def visualize_prediction(image: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray):
    visualization = YoloSegmenter.create_visualization(image, prediction)
    
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(visualization)
    plt.title("Prediction")
    
    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth)
    plt.title("Ground Truth")
    
    plt.show()
    
def example():
    model_path = Path("yolov8n-seg.pt")
    segmenter = YoloSegmenter(model_path)
    
    # Example usage
    band_selection = BandSelection(
        sentinel2=('02', '03', '04')
    )
    
    dataset = SatDataset(band_selection, Path(f"data/processed/{str(band_selection)}"))
    
    image, ground_truth = dataset[0]
    prediction = segmenter.predict(image)
    
    visualize_prediction(image, prediction, ground_truth)

if __name__ == "__main__":
    example()