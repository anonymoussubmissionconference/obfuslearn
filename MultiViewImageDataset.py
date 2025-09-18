import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from torchvision.transforms import ToPILImage

class MultiViewImageDataset(Dataset):
    def __init__(self, annotation_file, image_dirs, weights_file=None, transform=None, target_transform=None):
        assert len(image_dirs) == 2, "Expected three image directories for two views"
        self.img_labels = pd.read_csv(annotation_file)
        self.image_dirs = image_dirs
        self.transform = transform
        self.target_transform = target_transform

        self.weights_file = weights_file
        self.weights = None
        if weights_file:
            self.weights = pd.read_csv(weights_file).values.reshape(-1)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        images = []
        img_name = self.img_labels.iloc[idx, 0]
        to_pil = ToPILImage()

        for i, img_dir in enumerate(self.image_dirs):
            if i == 1 or i==2:  # Suppose the second view is .npy files
                npy_name = img_name.replace(".png", ".npy")
                img_path = os.path.join(img_dir, npy_name)
                array = np.load(img_path)
                #print(array.shape)
                # Convert to 3-channel (C, H, W) torch.Tensor
                if array.ndim == 1:
                    array=array.reshape((32, 16))
                    array = np.stack([array] * 3, axis=0)  # Shape: (3, H, W)

                if array.ndim == 2:
                    array = np.stack([array] * 3, axis=0)  # Shape: (3, H, W)

                elif array.ndim == 3 and array.shape[-1] == 3:
                    array = array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

                array = array.astype(np.float32)
                array = (array - array.min()) / (array.max() - array.min() + 1e-5)

                #print(array.shape)
                image = to_pil(torch.from_numpy(array))
            else:
                img_path = os.path.join(img_dir, img_name)
                image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)
            images.append(image)

        label = self.img_labels.iloc[idx, 1]
        if self.target_transform:
            label = self.target_transform(label)

        return tuple(images), label

    def get_weights_tensor(self):

        if self.weights is not None:
            return torch.tensor(self.weights, dtype=torch.float32)
        return None
