import skimage
import torch
import pandas as pd
import PIL
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class COCO_dataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None, landmark_bool=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.landmark_bool = landmark_bool

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = PIL.Image.open(img_name)
        
        if self.landmark_bool:
          landmarks = self.landmarks_frame.iloc[idx, 1:]
          landmarks = np.array([landmarks])
          landmarks = landmarks.astype('float').reshape(-1, 3)
        else:
          landmarks = 0
        sample = (image, landmarks)

        if self.transform:
            sample = self.transform(sample)

        return sample




