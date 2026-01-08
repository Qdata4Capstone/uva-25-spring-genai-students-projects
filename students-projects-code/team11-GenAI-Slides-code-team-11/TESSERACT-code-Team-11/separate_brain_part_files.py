import os
import shutil
import numpy as np

def copy_brain_part_files(data_dir, num_brain_parts, splits=None):
    """
    Copy files with specific brain parts to new directories.
    
    Args:
        data_dir (str): Path to the main data directory (e.g., 'oasis-dataset').
        num_brain_parts (int): Number of brain parts to filter by (2 or 4).
        splits (list, optional): List of splits to process. Defaults to ['train', 'val', 'test'].
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    
    # Map allowable brain-part selections
    part_map = {
        2: ["Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter"],
        4: [
            "Left-Cerebral-White-Matter",
            "Right-Cerebral-White-Matter",
            "Left-Thalamus",
            "Right-Thalamus",
        ],
    }
    
    brain_parts = part_map.get(num_brain_parts)
    if brain_parts is None:
        raise ValueError(f"Invalid num_brain_parts: {num_brain_parts}. Must be 2 or 4.")
    
    for split in splits:
        # Create source and destination directories
        split_dir = os.path.join(data_dir, split)
        dest_dir = os.path.join(data_dir, f"{split}-{num_brain_parts}")
        
        # Check if source directory exists
        if not os.path.exists(split_dir):
            print(f"Warning: Directory {split_dir} not found. Skipping.")
            continue
            
        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Created directory: {dest_dir}")
        
        # Get all .npy files in the split directory
        npy_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
        
        if len(npy_files) == 0:
            print(f"No .npy files found in {split_dir}. Skipping.")
            continue
        
        copied_count = 0
        for file in npy_files:
            file_path = os.path.join(split_dir, file)
            
            try:
                # Load the data to check if it contains required brain parts
                data = np.load(file_path, allow_pickle=True).item()
                text_prompt = data.get("text_prompt", "")
                
                # Check if any of the required brain parts are in the text prompt
                if any(part in text_prompt for part in brain_parts):
                    # Copy the file to the destination directory
                    dest_path = os.path.join(dest_dir, file)
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Copied {copied_count} files from '{split}' to '{split}-{num_brain_parts}'")

# Example usage:
if __name__ == "__main__":
    # Copy files for brain parts=2 for all splits
    copy_brain_part_files(data_dir="oasis-redefined", num_brain_parts=4, splits=["test"])
    
    # Or just for test split
    # copy_brain_part_files("oasis-dataset", 2, splits=["test"])