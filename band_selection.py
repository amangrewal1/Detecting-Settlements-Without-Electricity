import argparse
from itertools import product
import json
from pathlib import Path
import shutil
from typing import NamedTuple

import numpy as np
import cv2
from tqdm import tqdm

from config import SEED
from data_utils.dataset import SatDataset
from sat_types import BandSelection

class Settings(NamedTuple):
    input_path: Path
    output_path: Path
    bands: BandSelection
    
    def __str__(self):
        return f"Settings(input_path={self.input_path}, output_path={self.output_path}, bands={self.bands.__repr__()})"
    

def normalize_path(path: str) -> Path:
    return Path(path).resolve()


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('input_path', type=str, help='Path to the input file')
    parser.add_argument('output_path', type=str, help='Path to the output file')
    parser.add_argument('band_selection_location', type=str, help='Path to the json file containing the band selection')

    args = parser.parse_args()
    
    input_path = normalize_path(args.input_path)
    output_path = normalize_path(args.output_path)
    band_selection_location = normalize_path(args.band_selection_location)

    with open(band_selection_location, "r") as f:
        band_selection = BandSelection.from_dict(json.load(f))
        
    return Settings(input_path, output_path, band_selection)


def data_already_converted(settings: Settings) -> bool:
    """
    Check if the data has already been converted to YOLO format.
    
    This should be more than just checking if the output folder exists.
    E.g., you can write the .yaml file at the end of the conversion process,
    then you'd know if the last conversion was successful.
    """
    output_path = settings.output_path
    return output_path.exists() and Path.exists(normalize_path(f"{output_path}/data.yaml"))
   

def write_label(gt_mask: np.ndarray, label_path: Path):
    """
    Write the label file for the YOLO format.
    """
    # Convert to binary mask. Only the class 1 (settlement w/out electrivity) is relevant.
    bin_mask = (gt_mask == 1).astype(np.uint8)
    
    # output_path / f"{subdir}"/ "labels" / f"{tile_id}.txt"
    with open(label_path, "w") as f:
        # Convert each pixel of the relevant to the YOLO format
        for i, j in product(range(bin_mask.shape[0]), range(bin_mask.shape[1])):
            if bin_mask[i, j] == 0: continue

            x_coord = j/bin_mask.shape[1]
            y_coord = 1 - i/bin_mask.shape[0]
            x_step = 1/bin_mask.shape[1]
            y_step = 1/bin_mask.shape[0]

            yolo_tile = f"0 {x_coord} {y_coord} {x_coord+x_step} {y_coord} {x_coord+x_step} {y_coord-y_step} {x_coord} {y_coord-y_step}\n"

            f.write(yolo_tile)
            
            # YOLO format: <object-class> <x1> <y1> <x2> <y2> ...
            # The coordinates are normalized to the image width and height


def create_yaml(output_path: Path):
    data_yaml = [
        f"path: {output_path}\n",
        "train: train/images\n",
        "val: val/images\n\n",
        "names: \n",
        "   0: roi\n",
    ]
    with open(f"{output_path}/data.yaml", 'w') as f:
        f.writelines(data_yaml)

def save_output(name: str, image: np.ndarray, gt_mask: np.ndarray, output_path: Path, is_val: bool):
    if len(image.flat) == 0: return
    
    subdir = "val" if is_val else "train"
    
    image_path = normalize_path(f"{output_path}/{subdir}/images/{name}.png")
    label_path = normalize_path(f"{output_path}/{subdir}/labels/{name}.txt")
    
    # Need to convert to CV2 format
    cv2_image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
    cv2.imwrite(str(image_path), cv2_image)
    
    write_label(gt_mask, label_path)

# TODO: Implement this function
# Use helper functions liberally
def convert_to_yolo(settings: Settings):
    """
    Convert data from the tile folders to YOLO format: https://docs.ultralytics.com/datasets/segment/

    Refer to issue: https://github.com/cs175cv-w2024/final-project-settlement-stalkers/issues/4
    """
    input_path, output_path, bands = settings
    
    if not Path.exists(input_path):
        raise ValueError(f"Input path {input_path} does not exist")
    
    # make directories
    if not output_path.exists():
        Path.mkdir(output_path, parents=True, exist_ok=True)
    else:
        # Delete the existing files
        for path in output_path.iterdir():
            shutil.rmtree(path)
            
    Path.mkdir(normalize_path(f"{output_path}/train"))
    Path.mkdir(normalize_path(f"{output_path}/val"))
    Path.mkdir(normalize_path(f"{output_path}/train/images"))
    Path.mkdir(normalize_path(f"{output_path}/val/images"))
    Path.mkdir(normalize_path(f"{output_path}/train/labels"))
    Path.mkdir(normalize_path(f"{output_path}/val/labels"))
    
    subtile_scale = 2
    dataset = SatDataset(bands, input_path, subtile_scale)
    
    generator = np.random.default_rng(SEED)
    val_idx_mask: np.ndarray = generator.choice([True, False], len(dataset), replace=True)
    
    for tile_id, subtile_id, x, y in dataset:
        is_val: bool = bool(val_idx_mask[tile_id * subtile_scale ** 2 + subtile_id])

        save_output(f"{tile_id}_{subtile_id}", x, y, output_path, is_val)

    create_yaml(output_path)


def main():
    """
    Entry point for parsing the command line arguments and running the formatting.
    """
    settings: Settings = parse_args()
    print(settings)
    
    if data_already_converted(settings):
        print("Data already converted in ", settings.output_path)
        print("Remove the output folder to re-convert the data.")
        return
    
    convert_to_yolo(settings)


if __name__ == "__main__" :
    main()
