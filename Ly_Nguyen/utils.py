import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]  # Assuming the first column contains file paths
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        label = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)  # Assuming the second column contains labels

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Function to calculate output shape after convolution
def calculate_output_shape(image_size, kernel, stride, padding):
    assert len(image_size) == len(kernel) == len(stride) == len(padding), "All values should have the same length"
    output_shape = []
    for dim in range(len(image_size)):
        output = 1 + (image_size[dim] + (2 * padding[dim]) - kernel[dim]) / stride[dim]
        assert output.is_integer(), "Change kernel size, padding, and stride to get an integer value for the output"
        output_shape.append(int(output))
    return output_shape