import shutil

import torch

from GANModel import ColorizationGAN
from dataset import get_dataloader
from utilities import *
import os
import pandas as pd

# Path for saving model checkpoints
PATH = os.path.join(os.pardir, "saved_models")

# Save checkpoint
if not (os.path.exists(PATH)):
    os.mkdir(PATH)

# Track losses
losses = {
    'loss_D_fake': [],
    'loss_D_real': [],
    'loss_D': [],
    'loss_G': []
}


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


def print_losses(loss_meter_dict, save=True):
    """ Displays current losses for each module and stores loss logs. """
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

    if save:
        for loss_name, loss_meter in loss_meter_dict.items():
            losses[loss_name].append(loss_meter.avg)
        print(losses)
        print(type(losses))
        print(losses.values())
        # Save loss values for each epoch as csv (saving happens in every epoch)
        output = pd.DataFrame.from_dict(losses)
        output.to_csv('output.csv', index=False)


def train_model(args, checkpoint=1):
    """ Performs a complete training process of the GAN model.

    Parameters
    ----------
    args : <class 'argparse.Namespace'>
        training arguments parsed in main
    checkpoint : <class 'int'>
        save checkpoints for the model
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loader
    train_dataloader = get_dataloader(args.train_data_path, args.image_size, args.batch_size, args.pretrained, training_mode=True)

    # Instantiate GAN model
    model = ColorizationGAN(device, pretrained=args.pretrained)

    # Begin training
    epochs = args.num_epochs
    for epoch in range(epochs):
        start_time = time.time()
        # Create a dict containing the loss of each module
        loss_meter_dict = create_loss_dict()
        for data in train_dataloader:
            # Separate L and ab channels
            input_real, target_real = setup_input(data, device)
            # Perform one round of training on the current batch
            model.optimize(input_real, target_real)
            loss_meter_dict = update_losses(model, loss_meter_dict, count=data['input'].size(0))
        end_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Elapsed time:", end_time-start_time)
        # Display loss in every epoch
        print_losses(loss_meter_dict)

        if epoch % checkpoint == 0:
            save_file_path = f"{PATH}/{get_timestamp()}/"
            if os.path.exists(save_file_path):
                os.rmdir(save_file_path)
            os.mkdir(save_file_path)
            save_path = os.path.join(save_file_path, f'checkpoint_{str(epoch//checkpoint)}.pth')

            torch.save({
                'epoch': epoch,
                'generator_state_dict': model.gen.state_dict(),
                'discriminator_state_dict': model.disc.state_dict(),
                'generator_optim_state_dict': model.opt_G.state_dict(),
                'discriminator_optim_state_dict': model.opt_D.state_dict()
                }, save_path)