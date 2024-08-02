from pathlib import Path

import tifffile
from config import RAW_DATA_DIR
from data_utils.preprocess_utils import gammacorr, minmax_scale, per_band_gaussian_filter, quantile_clip

import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, NamedTuple, Literal
import re


class SatelliteType(Enum):
    SENTINEL1 = "sentinel1"
    SENTINEL2 = "sentinel2"
    LANDSAT = "landsat"
    VIIRS = "viirs"
    GROUND_TRUTH = "gt"


# TODO: Check if the literals match up with the actual band names
class BandSelection(NamedTuple):
    """
    Data class to store the selected bands for each satellite type. This ensures that we have type safety.
    
    See example in the __main__ statement. Try to use the wrong band; the IDE should give you an error.
    """
    sentinel1: tuple[Literal["VV", "VH", "VV-VH"], ...] = ()
    sentinel2: tuple[Literal["01", "02", "03", "04", "05", "06", "07", "08", "08A", "09", "10", "10", "11", "12"], ...] = ()
    landsat: tuple[Literal["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"], ...] = ()
    viirs: tuple[Literal["NIR", "maxproj"], ...] = ()
    ground_truth: tuple[Literal["gt"], ...] = ()
        
    def satellite_types(self):
        """
        Yields the satellite types that have bands selected
        """
        for sat in SatelliteType:
            if len(self[sat]) > 0:
                yield sat
        
    @staticmethod 
    def all():
        """
        Creates a BandSelection with all the bands selected
        """
        return BandSelection(
            sentinel1=("VV", "VH", "VV-VH"),
            sentinel2=("01", "02", "03", "04", "05", "06", "07", "08", "08A", "09", "10", "10", "11", "12"),
            landsat=("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"),
            viirs=("NIR", "maxproj"),
            ground_truth=("gt",)
        )
        
    @staticmethod
    def from_dict(d: dict[str, list[str]]):
        """
        Creates a BandSelection from a dictionary
        """
        return BandSelection(
            sentinel1=tuple(d.get(SatelliteType.SENTINEL1.value, tuple())), # type: ignore
            sentinel2=tuple(d.get(SatelliteType.SENTINEL2.value, tuple())), # type: ignore
            landsat=tuple(d.get(SatelliteType.LANDSAT.value, tuple())), # type: ignore
            viirs=tuple(d.get(SatelliteType.VIIRS.value, tuple())), # type: ignore
            ground_truth=tuple(d.get(SatelliteType.GROUND_TRUTH.value, tuple())) # type: ignore
        )
        
    def to_dict(self) -> dict[SatelliteType, tuple[str, ...]]:
        """
        For use in serialization to pass between processes.
        """
        return {
            SatelliteType.SENTINEL1: self.sentinel1,
            SatelliteType.SENTINEL2: self.sentinel2,
            SatelliteType.LANDSAT: self.landsat,
            SatelliteType.VIIRS: self.viirs,
            SatelliteType.GROUND_TRUTH: self.ground_truth
        }
        
    def to_json(self):
        return {
            sat_type.value: bands
            for sat_type, bands in self.to_dict().items()
        }
        
    def __getitem__(self, satellite: SatelliteType|str) -> tuple[str, ...]:
        sat = SatelliteType(satellite)
        match sat:
            case SatelliteType.SENTINEL1:
                return self.sentinel1
            case SatelliteType.SENTINEL2:
                return self.sentinel2
            case SatelliteType.LANDSAT:
                return self.landsat
            case SatelliteType.VIIRS:
                return self.viirs

            case SatelliteType.GROUND_TRUTH:
                return self.ground_truth
            case _:
                raise ValueError(f"Invalid satellite type: {satellite}")
            
    def __str__(self):
        return "+".join(
            f"{satellite.value}({' '.join(self[satellite])})"
            for satellite in self.satellite_types()
        )


class Satellite(ABC):
    """
    Parent class for wrappers around different satellite types.
    
    Provides interface for extracting date and band from filenames and preprocessing the data.
    """
    sat_type: SatelliteType
    file_pattern: str
    
    def __init__(self, tile_dir: Path = RAW_DATA_DIR) -> None:
        self.tile_dir = tile_dir
    
    def get_files(self) -> Iterable[Path]:
        """
        Retrieve all satellite files matching the satellite type pattern.
        """
        return self.tile_dir.glob(self.file_pattern)
    
    def get_data(self, bands: set[str]|None = None) -> Iterable[tuple[str, str, np.ndarray]]:
        """
        Yields the date, band, and pre-processed data of a satellite in a tile.
        """
        for file in self.get_files():
            date, band = self.extract_date_band(file.name)
            
            if bands and not band in bands: continue
            
            array = self.preprocess(tifffile.imread(file))
            yield date, band, array
    
    @staticmethod
    def create(sat_type: SatelliteType, tile_dir: Path) -> "Satellite":
        """
        Factory method to create a Satellite object based on the SatelliteType.
        """
        match sat_type:
            case SatelliteType.SENTINEL1:
                return Sentinel1(tile_dir)
            case SatelliteType.SENTINEL2:
                return Sentinel2(tile_dir)
            case SatelliteType.LANDSAT:
                return Landsat(tile_dir)
            case SatelliteType.VIIRS:
                return Viirs(tile_dir)
            case SatelliteType.GROUND_TRUTH:
                return GroundTruth(tile_dir)
    
    @staticmethod
    @abstractmethod
    def extract_date_band(filename: str) -> tuple[str, str]:
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def preprocess(stack: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        stack : np.ndarray
            A stack of satellite images in the shape (time, bands, height, width)
        """
        raise NotImplemented

class Viirs(Satellite):
    sat_type = SatelliteType.VIIRS
    file_pattern = 'DNB_VNP46A1_*'
    
    @staticmethod
    def extract_date_band(filename: str) -> tuple[str, str]:
        """
        This function takes in the filename of a VIIRS file and outputs
        a tuple containin two strings, in the format (date, band)

        Example input: DNB_VNP46A1_A2020221.tif
        Example output: ("2020221", "0")

        Parameters
        ----------
        filename : str
            The filename of the VIIRS file.

        Returns
        -------
        tuple[str, str]
            A tuple containing the date and band.
        """
        match = re.match(r".*DNB_VNP46A1_A(.*)\.tif", str(filename))
        if not match:
            raise ValueError(f"Filename {filename} does not match pattern.")
        date = match.groups()[0]
        return (date, "NIR")
    
    @staticmethod
    def preprocess(stack: np.ndarray) -> np.ndarray:
        """
        - Clip higher and lower quantile outliers per band per timestep
        - Minmax scale
        """
        stack = quantile_clip(stack, clip_quantile=0.05)
        stack = minmax_scale(stack)
        return stack
    
    @staticmethod
    def maxprojection(stack: np.ndarray) -> np.ndarray:
        """
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a max projection
        - Minmax scale
        """
        stack = quantile_clip(stack, clip_quantile=0.05)
        stack = np.max(stack, axis=0)
        stack = minmax_scale(stack)
        return stack
    
    def load_maxproj(self):
        files = self.get_files()
        raw = [tifffile.imread(file).astype(np.float32) for file in files]
        
        height, width = raw[0].shape
        
        data_stack = self.maxprojection(
            # VIIRS only has one band, and we're projecting across time
            # NOTE: Stacks are of shape (time, band, height, width)
            np.array(raw).reshape((len(raw), 1, height, width))
        ).reshape((height, width))
        
        return data_stack
    

class Sentinel1(Satellite):
    sat_type = SatelliteType.SENTINEL1
    file_pattern = 'S1A_IW_GRDH_*'
    
    @staticmethod
    def extract_date_band(filename: str) -> tuple[str, str]:
        """
        This function takes in the filename of a Sentinel-1 file and outputs
        a tuple containin two strings, in the format (date, band)

        Example input: S1A_IW_GRDH_20200804_VV.tif
        Example output: ("20200804", "VV")

        Parameters
        ----------
        filename : str
            The filename of the Sentinel-1 file.

        Returns
        -------
        tuple[str, str]
            A tuple containing the date and band.
        """
        match = re.match(r".*S1A_IW_GRDH_(.*)_(.*)\.tif", str(filename))
        if not match:
            raise ValueError(f"Filename {filename} does not match pattern.")
        date = match.groups()[0]
        band = match.groups()[1]
                
        return (date, band)
    
    @staticmethod
    def preprocess(stack: np.ndarray) -> np.ndarray:
        """
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gaussian filter
        - Minmax scale
        """

        # convert data to dB
        sentinel1_stack = np.log10(stack)

        # clip outliers
        stack = quantile_clip(stack, clip_quantile=0.01)
        stack = per_band_gaussian_filter(stack, sigma=2)
        stack = minmax_scale(stack)

        return stack
    
    def load_vv_vh(self):
        """
        Loads the first instances of the VV and VH bands from the Sentinel-1
        and returns their difference
        """
        files = self.get_files()
        vv_files = set()
        
        # Find the first instance of each band. Their dates should match.
        for file in files:
            _, band = self.extract_date_band(file.name)
            
            if band == "VV":
                vv_files.add(file.stem)
                
            elif band == "VH" and file.stem[:-1] + "V" in vv_files:
                vv = tifffile.imread(file)
                vh_str = str(file.with_suffix(''))[:-1] + "H.tif"
                vh = tifffile.imread(Path(vh_str).resolve())
                return vv - vh
                
        raise ValueError("Could not find a matching VV and VH file.")
    

class Sentinel2(Satellite):
    sat_type = SatelliteType.SENTINEL2
    file_pattern = 'L2A_*'
    
    @staticmethod
    def extract_date_band(filename: str) -> tuple[str, str]:
        """
        This function takes in the filename of a Sentinel-2 file and outputs
        a tuple containin two strings, in the format (date, band)

        Example input: L2A_20200816_B01.tif
        Example output: ("20200804", "01")

        Parameters
        ----------
        filename : str
            The filename of the Sentinel-2 file.

        Returns
        -------
        tuple[str, str]
        """
        match = re.match(r".*L2A_(.*)_B(.*)\.tif", str(filename))
        if not match:
            raise ValueError(f"Filename {filename} does not match pattern.")
        date = match.groups()[0]
        band = match.groups()[1]
        # change to fit BandSelection, i.e. 8A -> 08A
        if band == "8A":
            band = '0' + band
                
        return (date, band)
    
    @staticmethod
    def preprocess(stack: np.ndarray) -> np.ndarray:
        """
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
        """
        stack = quantile_clip(stack, clip_quantile=0.1)
        stack = gammacorr(stack, gamma=2.2)
        stack = minmax_scale(stack)

        return stack
    

class Landsat(Satellite):
    sat_type = SatelliteType.LANDSAT
    file_pattern = 'LC08_L1TP_*'
    
    @staticmethod
    def extract_date_band(filename: str) -> tuple[str, str]:
        """
        This function takes in the filename of a Landsat file and outputs
        a tuple containing two strings, in the format (date, band)

        Example input: LC08_L1TP_2020-08-30_B9.tif
        Example output: ("2020-08-30", "9")

        Parameters
        ----------
        filename : str
            The filename of the Landsat file.

        Returns
        -------
        tuple[str, str]
            A tuple containing the date and band.
        """
        match = re.match(r".*LC08_L1TP_(.*)_B(.*)\.tif", str(filename))
        if not match:
            raise ValueError(f"Filename {filename} does not match pattern.")
        date = match.groups()[0]
        band = match.groups()[1]
        # change to fit BandSelection, i.e. 8 -> 08 etc.
        if len(band) == 1:
            band = '0' + band
        
        return (date, band)
    
    @staticmethod
    def preprocess(stack: np.ndarray) -> np.ndarray:
        """
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
        """
        stack = quantile_clip(
            stack,
            clip_quantile=0.05,
            )
        stack = gammacorr(stack, gamma=2.2)
        stack = minmax_scale(stack)

        return stack
    

class GroundTruth(Satellite):
    sat_type = SatelliteType.GROUND_TRUTH
    file_pattern = 'groundTruth.tif'
    
    @staticmethod
    def extract_date_band(filename: str) -> tuple[str, str]:
        """
        This function takes in the filename of the ground truth file and returns
        ("0", "0"), as there is only one ground truth file.

        Example input: groundTruth.tif
        Example output: ("0", "0")

        Parameters
        ----------
        filename: str
            The filename of the ground truth file though we will ignore it.

        Returns
        -------
        tuple[str, str]
            A tuple containing the date and band.
        """
        return ("0", "0")
    
    @staticmethod
    def preprocess(stack: np.ndarray) -> np.ndarray:
        return stack
