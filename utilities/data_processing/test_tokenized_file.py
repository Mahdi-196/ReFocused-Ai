import numpy as np
from pathlib import Path
import os

def test_tokenized_file():
    # Get the first .npz file - using Windows path format
    data_dir = Path(r'C:\Users\mahdi\Downloads\Documents\Desktop\data_tokenized_production')
    
    # List first few files to see what we have
    files = list(data_dir.glob('*.npz'))
    print(f"Found {len(files)} .npz files")
    
    if files:
        first_file = files[0]
        print(f"Testing file: {first_file.name}")
        print(f"File size: {first_file.stat().st_size / (1024*1024):.2f} MB")
        
        try:
            data = np.load(str(first_file), allow_pickle=True)
            print(f"Keys in file: {list(data.keys())}")
            
            for key in data.keys():
                print(f"Key '{key}': shape={data[key].shape}, dtype={data[key].dtype}")
                if hasattr(data[key], '__len__') and len(data[key]) > 0:
                    print(f"  First element type: {type(data[key][0])}")
                    if hasattr(data[key][0], '__len__'):
                        print(f"  First element length: {len(data[key][0])}")
                
        except Exception as e:
            print(f"Error loading file: {e}")
    else:
        print("No .npz files found!")

if __name__ == "__main__":
    test_tokenized_file() 