import os
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class DatasetColorization(Dataset):

    """A class used to represent the preprocessing of the image data.

    Attributes
    ----------
    data_path : <class 'str'>
        the path with the images to be processed
    image_size : <class 'int'>
        the size of an image
    images_list : <class 'list'>
        a list with the names of all images under the path

    Methods
    -------
    __getitem__()
        Loads and preprocesses all images
    """

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
        """Loads and preprocesses all images.

        Parameters
        ----------
        img_idx : <class 'int'>
            The index with the image from the self.images_list to be processed

        Returns
        -------
        Dictionary with the preprocessed input and target images to be used later in the model
        """

        # Load RGB image and apply data transformations
        img_rgb = Image.open(os.path.join(self.data_path, self.images_list[img_idx])).convert("RGB")
        img_rgb = self.transforms(img_rgb)

        # Convert to numpy array and move the color channel from the first to the last dim
        img_rgb = np.array(img_rgb)
        img_rgb = np.moveaxis(img_rgb, 0, -1)

        # Converting RGB color space to L*a*b
        img_lab = rgb2lab(img_rgb).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)

        # Get input and target images and normalize their values to the range [-1, +1]
        L_channel = img_lab[[0], ...] / 50 - 1.
        ab_channel = img_lab[[1, 2], ...] / 110

        return {"input": L_channel, "target": ab_channel, "image_name": self.images_list[img_idx]}

    def __len__(self):
        return len(self.images_list)


def get_dataloader(data_path, image_size=256, batch_size=16, training_mode=True):
    """Creates a DataLoader object with preprocessed images.

    Parameters
    ----------
    data_path : <class 'str'>
        the path with the images to be processed
    image_size : <class 'int'>
        the size of an image
    batch_size : <class 'int'>
        the batch size for processing the images by the model
    training_mode : <class 'bool'>
        boolean to control whether to generate a train or validation set object

    Returns
    -------
    DataLoader object
    """

    print("Training mode is:", training_mode)
    dataset = DatasetColorization(data_path, image_size=image_size, training_mode=training_mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=training_mode)
    return dataloader


def setup_input(data_batch):
    """Separates the grayscale input images from the target images for training the networks.

    Parameters
    ----------
    data_batch : <class 'dict'>
        a batch of 16 images

    Returns
    -------
    The grayscale input images of size [16, 1, 256, 256] and the target images of size [16, 2, 256, 256]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_real = data_batch['input'].to(device)
    target_real = data_batch['target'].to(device)
    return input_real, target_real


dataloader = get_dataloader(data_path="../Dataset/training/")

for data in dataloader:
    setup_input(data)
    break

