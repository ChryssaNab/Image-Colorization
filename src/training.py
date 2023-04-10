import torch

from dataset import get_dataloader


def setup_input(data_batch, device):
    """ Separates the grayscale input images from the color target images for training the networks.

    Parameters
    ----------
    data_batch : <class 'dict'>
        a batch of 16 images
    device : <class 'torch.device'>
        CPU or CUDA selected device

    Returns
    -------
    The batches of the grayscale input images of size [16, 1, 256, 256] and the target images of size [16, 2, 256, 256]
    """

    input_real = data_batch['input'].to(device)
    target_real = data_batch['target'].to(device)
    return input_real, target_real


def train_model(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataloader = get_dataloader(args.train_data_path, args.image_size, args.batch_size)

    for epoch in range(args.num_epochs):
        for data in train_dataloader:
            input_real, target_real = setup_input(data, device)
            print(input_real.shape, target_real.shape)
            break
        break