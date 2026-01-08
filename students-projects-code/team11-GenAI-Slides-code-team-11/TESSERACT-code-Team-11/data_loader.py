import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class OASISDataset(Dataset):
    def __init__(self, data_dir, split, num_brain_parts):
        """
        Initialize the dataset by loading the .npy files from the specified directory and split.
        Only includes samples where text_prompt contains "Left-Cerebral-White-Matter".
        
        Args:
            data_dir (str): Path to the main data directory (e.g., 'oasis-dataset').
            split (str): The split name ('train', 'val', or 'test').
        """
        # Construct the path to the split directory (train, val, test)
        split_dir = os.path.join(data_dir, split)
        
        # Check if the split directory exists
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} not found.")
        
        # Get all .npy files in the split directory
        npy_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
        self.data_files = []
        for file in npy_files:
            file_path = os.path.join(split_dir, file)
            self.data_files.append(file_path)


        print(f"Found {len(self.data_files)} files in '{split}' split")

    # def __init__(self, data_dir, split, num_brain_parts):
    #     """
    #     Initialize the dataset by loading the .npy files from the specified directory and split.
    #     Only includes samples where text_prompt contains "Left-Cerebral-White-Matter".
        
    #     Args:
    #         data_dir (str): Path to the main data directory (e.g., 'oasis-dataset').
    #         split (str): The split name ('train', 'val', or 'test').
    #     """
    #     # Construct the path to the split directory (train, val, test)
    #     split_dir = os.path.join(data_dir, split)
        
    #     # Check if the split directory exists
    #     if not os.path.exists(split_dir):
    #         raise FileNotFoundError(f"Directory {split_dir} not found.")
        
    #     # Get all .npy files in the split directory
    #     npy_files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
        
    #     if len(npy_files) == 0:
    #         raise FileNotFoundError(f"No .npy files found in {split_dir}.")
        
    #     # Map allowable brain‑part selections
    #     part_map = {
    #         2: ["Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter"],
    #         4: [
    #             "Left-Cerebral-White-Matter",
    #             "Right-Cerebral-White-Matter",
    #             "Left-Thalamus",
    #             "Right-Thalamus",
    #         ],
    #     }

    #     brain_parts = part_map.get(num_brain_parts)
    #     self.data_files = []

    #     for file in npy_files:
    #         file_path = os.path.join(split_dir, file)

    #         if brain_parts is None:
    #             # No filtering — keep every file
    #             self.data_files.append(file_path)
    #             continue

    #         # Otherwise, keep only those whose text_prompt mentions one of the target parts
    #         try:
    #             data = np.load(file_path, allow_pickle=True).item()
    #             if any(part in data.get("text_prompt", "") for part in brain_parts):
    #                 self.data_files.append(file_path)
    #         except Exception as e:
    #             print(f"Skipping {file_path}: {e}")

    #     print(f"Found {len(self.data_files)} files in '{split}' split")


    def __len__(self):
        """Return the total number of filtered .npy files in the split."""
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        Fetch the data at the specified index from the filtered .npy files.
        
        Args:
            idx (int): Index of the data to retrieve.
        
        Returns:
            tuple: Tuple containing (voxal_video, filtered_mask, text_prompt).
        """
        # Load the .npy file corresponding to the idx
        npy_file = self.data_files[idx]
        
        # Load the .npy file which contains the dictionary of data
        data = np.load(npy_file, allow_pickle=True).item()
        
        # Extract the data
        voxal_video = data['voxal_video']  # numpy array
        filtered_mask = data['filtered_mask']  # numpy array
        text_prompt = data['text_prompt']  # string
        
        # Convert the numpy arrays to torch tensors
        voxal_video = torch.tensor(voxal_video, dtype=torch.float32)
        filtered_mask = torch.tensor(filtered_mask, dtype=torch.float32)
        
        # Normalize the filtered mask
        filtered_mask = (filtered_mask - filtered_mask.min()) / (filtered_mask.max() - filtered_mask.min() + 1e-8)  # Added small epsilon to avoid division by zero

        # Return the sliced tensors and text prompt
        return voxal_video[:, 80:120, :], filtered_mask[:, 80:120, :], text_prompt
