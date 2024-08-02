## Overview
Team Settlement Stalkers' final project for CS175.

Approach: Train YOLO on combinations of bands to directly detect settlements without electricity.

### Pipeline
1. Data Preprocessing
   - Load the raw data.
   - Preprocess the data (color correction, normalization, band selection, and subtiling).
   - Create YOLO labels.
   - Save the processed data and labels.
2. Training
   - Run YOLO on formatted data.

### Development Process
1. Choose bands to use.
2. Preprocess and format the data for YOLO.
3. Train and validate model on the formatted data.
4. Choose the best-performing model.

### Notes for Grading
- `file_utils` and `sat_preprocess` have been refactored into the `Satellite` class in `sat_types.py` and `preprocess_utils.py` to follow OOP.
- Transformations are not implemented here because YOLO does it internally.
- `Dataset` is implemented in `data_utils/dataset` because it is used in formatting data for YOLO. `Dataloader` is not implemented because it is not necessary for our data pipeline.


## Setup
- Create a [conda](https://docs.anaconda.com/free/miniconda/) environment using `conda create -n <env_name> python`. (Optional)
- Install the required packages using `pip install -r requirements.txt`
- Download the dataset and extract it to the `data` folder.
  - By default, the tile directories should in `data/raw/`, but this is configurable in `config.py`.
    ```
    data/
    ├── raw/
    │   ├── Tile1/
    │   │   ├── DNB_VN...tif
    │   │   ├── groudTruth.tif
    │   │   ├── ...
    ```


## Running
- To train, run the main script using `python main.py`.
- Change settings such as raw data directory, processed data directory, model in `config.py`.
- Inferencing code and an example is procided in `inference.py`.
  - This is not entirely stable because we don't need it for validation. YOLO handles the validation and visualization automatically.


## Results
- Run results are in the `runs` directory.
- The best results in `best/`.
  - The directory contains performance and training graphs.
  - The model weights are in `best/weights/best.pt`.


## Dataset
This section was copied from HW01's readme.

### Description
** The following description is taken directly from the IEEE GRSS 2021 Challenge [website](https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/).

The IEEE GRSS 2021 ESD dataset is composed of 98 tiles of 800×800 pixels, distributed respectively across the training, validation and test sets as follows: 60, 19, and 19 tiles. Each tile includes 98 channels from the below listed satellite images. Please note that all the images have been resampled to a Ground Sampling Distance (GSD) of 10 m. Thus each tile corresponds to a 64km2 area.

### Satellite Data
#### Sentinel-1 polarimetric SAR dataset

2 channels corresponding to intensity values for VV and VH polarization at a 5×20 m spatial resolution scaled to a 10×10m spatial resolution.

File name prefix: “S1A_IW_GRDH_*.tif”
Size : 2.1 GB (float32)
Number of images : 4
Acquisition mode : Interferometric Wide Swath
Native resolution : 5x20m
User guide : [link](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/interferometric-wide-swath)

#### Sentinel-2 multispectral dataset

12 channels of reflectance data covering VNIR and SWIR ranges at a GSD of 10 m, 20 m, and 60 m. The cirrus band 10 is omitted, as it does not contain ground information.

File name prefix: “L2A_*.tif”
Size : 6,2 GB (uint16)
Number of images : 4
Level of processing : 2A
Native Resolution : [link](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument)
User guide : [link](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2)


#### Landsat 8 multispectral dataset

11 channels of reflectance data covering VNIR, SWIR, and TIR ranges at a GSD of 30m and 100 m, and a Panchromatic band at a GSD of 15m.

File name prefix: “LC08_L1TP_*.tif”
Size : 8,5 GB (float32)
Number of images : 3
Sensor used : OLI and TIRS
Native Resolution : [link](https://landsat.gsfc.nasa.gov/landsat-8/landsat-8-overview/)
User guide : [link](https://www.usgs.gov/core-science-systems/nli/landsat/landsat-8-data-users-handbook)

#### The Suomi Visible Infrared Imaging Radiometer Suite (VIIRS) night time dataset

The Day-Night Band (DNB) sensor of the VIIRS (Visible Infrared Imaging Radiometer Suite) provides on 1 channel, the global daily measurements of nocturnal visible and near-infrared (NIR) light at a GSD of 750 m. The VNP46A1 product is a corrected version of the original DNB data, and is at a 500m GSD resolution.

File name prefix: “DNB_VNP46A1_*.tif”
Size : 1,2 GB(uint16)
Number of images : 9
Product Name : VNP46A1’s 500x500m sensor radiance dataset
Native resolution : 750m (raw resolution)
User sheet : [link](https://viirsland.gsfc.nasa.gov/PDF/VIIRS_BlackMarble_UserGuide.pdf)

### Semantic Labels
The provided training data is split across 60 folders named TileX, X being the tile number. Each folder includes 100 files. 98 files correspond to the satellite images listed earlier.

We also provide reference information (‘groundTruth.tif’ file) for each tile. Please note that the labeling has been performed as follows:

#### Human settlement: 
If a building is present in a patch of 500×500m, this area is considered to have human settlement

#### Presence of Electricity: 
If a fraction of a patch of 500×500m is illuminated, this area is considered to be illuminated regardless of the size of the illuminated area fraction.
The reference file (‘groundTruth.tif’) is 16×16 pixels large, with a resolution of 500m corresponding to the labelling strategy described above. The pixel values (1, 2, 3 and 4) correspond to the four following classes:

1: Human settlements without electricity (Region of Interest): Color ff0000
2: No human settlements without electricity: Color 0000ff
3: Human settlements with electricity: Color ffff00
4: No human settlements with electricity: Color b266ff
An additional reference file ( ‘groundTruthRGB.png’ ) is provided at 10m resolution in RGB for easier visualization.