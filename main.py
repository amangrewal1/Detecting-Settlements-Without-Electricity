import itertools
import json
from pathlib import Path
import subprocess
from band_selection import BandSelection
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, YOLO_MODEL

def run_formatter(
    run_index: int, band_selection: BandSelection, input_path: str, output_path: str
) -> subprocess.Popen:
    """
    Run the formatter on the given band selection.
    
    Args:
        band_selection: The band selection to use.
        input_path: The path to the input file.
        output_path: The path to the output file.
        
    Returns:
        The subprocess `Popen` object. 
        Call `wait()` to wait for the process to finish.
        Check `returncode` to see if it was successful.
    """
    # Write the band selection to a temporary file
    band_selection_path = f"band_selection_{run_index}.json"
    with open(band_selection_path, "w") as f:
        json.dump(band_selection.to_json(), f)
    
    return subprocess.Popen([
        "python", 
        "band_selection.py", 
        input_path, 
        output_path, 
        band_selection_path
    ])
    

def run_yolo(model: str, data: str, epochs: int, learning_rate: float):
    """
    Run the YOLO training process.
    
    Returns:
        The subprocess `Popen` object. 
        Call `wait()` to wait for the process to finish.
        Check `returncode` to see if it was successful.
    """
    args = ["python", "train_yolo.py", "--model", model,"--data", data, "--epochs", epochs, "--lr0", learning_rate]
    return subprocess.Popen(list(map(str, args)))


# DONE: Train loop, according to https://github.com/cs175cv-w2024/final-project-settlement-stalkers/issues/5
def main():
    """
    For each band combination, invoke the formatter, 
    then call YOLO's train method to train the model using the converted data.
    
    Use subprocess for concurrency, not asyncio, threading, or multiprocessing. 
    They don't work for CPU-bound tasks because of the GIL.
    
    A good approach is probably to loop through band combinations to process,then use 
    subprocess to call the formatter on the next set and train YOLO on the current one.
    
    Remember to check if the process was successful, and if not, 
    log the error and skip to the next band combination.
    """
    
    sentinel2_combos: list[BandSelection] = [
        BandSelection(sentinel2=('04','03','02')),
        BandSelection(sentinel2=('08','04','03')),
        BandSelection(sentinel2=('12','08A','04')),
        BandSelection(sentinel2=('11','08','02')),
        BandSelection(sentinel2=('12','11','02')),
        BandSelection(sentinel2=('04','03','01'))
    ]
    
    landsat_combos: list[BandSelection] = [
        BandSelection(landsat=('04','03','02')),
        BandSelection(landsat=('07','06','04')),
        BandSelection(landsat=('05','04','03')),
        BandSelection(landsat=('06','05','02')),
        BandSelection(landsat=('07','06','05')),
        BandSelection(landsat=('05','06','02')),
        BandSelection(landsat=('05','06','04')),
        BandSelection(landsat=('07','05','03')),
        BandSelection(landsat=('07','05','04'))
    ]
    
    BAND_SELECTIONS = itertools.chain(sentinel2_combos, landsat_combos)
    
    train_process: subprocess.Popen|None = None
    for run_index, bands in enumerate(BAND_SELECTIONS):
        processed_path = PROCESSED_DATA_DIR / str(bands)
        format_process = run_formatter(run_index, bands, str(RAW_DATA_DIR), str(processed_path))   
        format_process.wait()
        
        if format_process.returncode != 0:
            print(f"Error processing {bands}")
            return
        
        # Wait for the previous training process to finish
        if train_process is not None:
            train_process.wait()
            if train_process.returncode != 0:
                print(f"Error training {bands}")
                return
            
        train_process = run_yolo(YOLO_MODEL, str(processed_path / "data.yaml"), epochs=150, learning_rate=0.01)

if __name__ == "__main__":
    main()
