import os
import numpy as np
import torchvision.transforms as T

from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class DatasetColorization(Dataset):

    def __init__(self, data_path, image_size=256, training_mode=True):
        self.data_path = data_path
        self.image_size = image_size
        self.images_list = os.listdir(self.data_path)

        if training_mode:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), T.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size), T.InterpolationMode.BICUBIC),
            ])

    def __getitem__(self, img_idx):
        # Load RGB image and apply data transformations
        img_rgb = Image.open(os.path.join(self.data_path, self.images_list[img_idx])).convert("RGB")
        img_rgb = self.transforms(img_rgb)

        # Convert to numpy array and move the color channel from the first to the last dim
        img_rgb = np.array(img_rgb)
        img_rgb = np.moveaxis(img_rgb, 0, -1)

        # Converting RGB to L*a*b
        img_lab = rgb2lab(img_rgb).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)

        # Extract input and target images and normalize their values to the range [-1, +1]
        L_channel = img_lab[[0], ...] / 50 - 1.
        ab_channel = img_lab[[1, 2], ...] / 110

        return {"input": L_channel, "target": ab_channel, "image_name": self.images_list[img_idx]}

    def __len__(self):
        return len(self.images_list)


def get_dataloader(data_path, image_size=256, batch_size=16, training_mode=True):
    print("Training mode is:", training_mode)
    dataset = DatasetColorization(data_path, image_size=image_size, training_mode=training_mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training_mode)
    return dataloader


get_dataloader(data_path="../Dataset/training/")

