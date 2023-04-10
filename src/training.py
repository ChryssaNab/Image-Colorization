import torch
from tqdm.notebook import tqdm
from dataset import get_dataloader
from utilities import *
import os
import csv
import pandas as pd

# Path for saving model checkpoints
PATH = os.path.join(os.pardir, "saved_models")

# Track losses
losses = {
    'loss_D_fake': [],
    'loss_D_real': [],
    'loss_D': [],
    'loss_G_GAN': [],
    'loss_G_L1': [],
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

    input_real = data_batch['input'].to(device) # !! .to(device) is called in optimize as well, remove one of the two !!
    target_real = data_batch['target'].to(device)
    return input_real, target_real

def print_losses(loss_meter_dict, save=True):
    """ Displays current losses for each module and saves loss logs.

    """
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

    if save:
        for loss_name, loss_meter in loss_meter_dict.items():
            losses[loss_name].append(loss_meter)
        # Save loss values for each epoch as csv (saving happens in every epoch)
        output = pd.DataFrame.from_dict(losses)
        output.to_csv('output.csv', index=False)



def train_model(args, model, checkpoint=25):
    """ Performs a complete training process of the GAN model.

        Parameters
        ----------
        args : <class 'argparse.Namespace'>
            training arguments parsed in main
        model : <class 'ColorizationGAN'>
            an instance of the ColorizationGAN() GAN model
        checkpoint : <class 'int'>
            save checkpoints for the model

    """
    epochs = args.num_epochs

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create data loader
    train_dataloader = get_dataloader(args.train_data_path, args.image_size, args.batch_size, args.pretrained, training_mode=True)

    # Begin training
    # Epoch loop
    for epoch in range(epochs):
        loss_meter_dict = create_loss_dict()  # create a dict containing the loss of each module
        # Batch loop
        for data in train_dataloader:
            input_real, target_real = setup_input(data, device) # separate L and ab channels
            model.optimize(input_real,target_real) # perform one round of training on the current batch
            loss_meter_dict = update_losses(model, loss_meter_dict, count=data['input'].size(0)) # ~~ assigning here/return in update_losses() might not be needed ~~
            print(input_real.shape, target_real.shape)
        print(f"\nEpoch {epoch+1}/{epochs}")
        print_losses(loss_meter_dict) # display loss in every epoch

        # Save checkpoint
        if not(os.path.exists(PATH)):
            os.mkdir(PATH)
        if epoch % checkpoint == 0:
            path = f"{PATH}/{get_timestamp()}/checkpoint_{str(epoch//checkpoint)}"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gen_optimizer_state_dict': model.get_opt()[0].state_dict(),   # ~~ assuming that model has a getter for the two optimizers that returns them in a tuple (?) ~~
                'disc_optimizer_state_dict': model.get_opt()[1].state_dict(),
                'loss_D_fake': loss_meter_dict["loss_D_fake"],
                'loss_D_real': loss_meter_dict["loss_D_real"],
                'loss_D': loss_meter_dict["loss_D"],
                'loss_G_GAN': loss_meter_dict["loss_G_GAN"],
                'loss_G_L1': loss_meter_dict["loss_G_L1"],
                'loss_G': loss_meter_dict["loss_G"]
                }, path)