import itertools
from pathlib import Path
from typing import Callable, Generator, Iterable
import numpy as np
from torch.utils.data import Dataset

from config import RAW_DATA_DIR
from sat_types import BandSelection, GroundTruth, Satellite, SatelliteType, Sentinel1, Viirs


class SatDataset(Dataset):
    def __init__(
        self, 
        band_selection: BandSelection,
        data_dir: Path = RAW_DATA_DIR, 
        subtile_scale: int = 1,
        x_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        y_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    ):
        self.band_selection = band_selection
        self.data_dir = data_dir
        self.subtile_scale = subtile_scale
        self.x_transform = x_transform
        self.y_transform = y_transform
        
    def __len__(self) -> int:
        return len(list(self.data_dir.glob("Tile*"))) * self.subtile_scale ** 2
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        tile_id, subtile_id = divmod(index, self.subtile_scale ** 2)
        tile_id += 1 # Tile dirs are 1-indexed
        
        tile_dir = self.data_dir / f"Tile{tile_id}"
        
        for sat_type in self.band_selection.satellite_types():
            sat = Satellite.create(sat_type, tile_dir)
            bands = self.band_selection[sat_type]
            
            bands_encountered = set()
            images = []
            
            for _, band, data in sat.get_data(set(bands)):
                # For now, ignore subsequent occurrences of the same band
                if band in bands_encountered: continue
                bands_encountered.add(band)
                
                images.append(data)
                
            if sat_type == SatelliteType.VIIRS and "maxproj" in bands:
                images.append(Viirs(tile_dir).load_maxproj())
                
            if sat_type == SatelliteType.SENTINEL1 and "VV-VH" in bands:
                images.append(Sentinel1(tile_dir).load_vv_vh())

        # Vertical and horizontal indices of the subtile
        i, j = divmod(subtile_id, self.subtile_scale)
        x: np.ndarray = __class__.subtile(np.stack(images), self.subtile_scale, i, j)
        y: np.ndarray = next(iter(GroundTruth(tile_dir).get_data()))[2]
        
        return self.x_transform(x), self.y_transform(y)
    
    def __iter__(self) -> Generator[tuple[int, int, np.ndarray, np.ndarray], None, None]:
        """
        Yields the tile index, subtile index, and the subtile images.
        """
        for tile_id in range(len(list(self.data_dir.glob("Tile*")))):
            for subtile_id in range(self.subtile_scale ** 2):
                yield tile_id, subtile_id, *self[tile_id * self.subtile_scale ** 2 + subtile_id]
    
    @staticmethod
    def subtiles(image: np.ndarray, subtile_scale: int) -> Iterable[np.ndarray]:
        """
        Yield subtile images of the given image.
        """
        for i, j in itertools.product(range(subtile_scale), range(subtile_scale)):
            yield __class__.subtile(image, subtile_scale, i, j)
                
    @staticmethod
    def subtile(image: np.ndarray, subtile_scale: int, i: int, j: int) -> np.ndarray:
        """
        Retrieve the subtile image of the given image.
        """
        *_, height, width = image.shape
        sub_height, sub_width = height // subtile_scale, width // subtile_scale
        
        return image[i*sub_height:(i+1)*sub_height, j*sub_width:(j+1)*sub_width]
        