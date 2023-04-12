import os

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.color import lab2rgb

from GANModel import ColorizationGAN
from dataset import get_dataloader


def load_checkpoints(generator, discriminator, ckeckpoint_path):
    checkpoint = torch.load(ckeckpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    return generator, discriminator


def plot_losses(path):
    """ Plots the logged losses saved as a csv file in 'path'.

    Arguments
    ----------
    path : <class 'str'>
        path to the csv file
    """

    loss_df = pd.read_csv(path, index_col=False)
    # for loss_name, loss_array in loss_dict.items():
    #     print(f"{loss_name}: {loss_meter.avg:.5f}")
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle('Losses')
    i = 1
    for loss_name in loss_df:
        ax = plt.subplot(3, 2, i)
        loss_df[loss_name].plot(ax=ax)
        ax.set_title(loss_name)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        i += 1
    plt.show()


def lab_space(image):
    """ Plots the separate Lab channels of the input image.

    Arguments
    ----------
    image : <class 'torch.Tensor'>
        input Lab image

    Returns
    -------
    The plt object of the plotted color channels
    """

    # Scale the lab image
    lab_im = np.array(image)
    lab_im = np.moveaxis(lab_im, 0, -1)
    image_lab_scaled = (lab_im + [0, 128, 128]) / [100, 255, 255]
    fig, ax = plt.subplots(1, 4, figsize=(11, 8))
    ax[0].imshow(image_lab_scaled)
    ax[0].axis('off')
    ax[0].set_title('Lab scaled')
    ax[1].imshow(image_lab_scaled[:, :, 0], cmap='gray')  # , ax=ax[1])
    ax[1].axis('off')
    ax[1].set_title('L')
    ax[2].imshow(image_lab_scaled[:, :, 1], cmap='RdYlGn_r')
    ax[2].axis('off')
    ax[2].set_title('a')
    ax[3].imshow(image_lab_scaled[:, :, 2], cmap='YlGnBu_r')
    ax[3].axis('off')
    ax[3].set_title('b')
    return plt


def colorize(model, data):
    """ Outputs/displays the predicted Lab images from the given model, the target images, their L color channel
        and a grid with their L, a and b color channels separated.

    Arguments
    ----------
    model : <class 'ColorizationGAN'>
        the GAN model
    data : <class 'dict'>
        a batch of images
    """

    # Create output dir
    save_path = '../Results/output_images'
    if not (os.path.exists(save_path)):
        os.makedirs(save_path)

    # Split input images to L and ab channels
    L = data['input']
    ab = data['target']

    # Feed L channel to the model for inference
    fake_colours = model.generate(L)
    i = 1
    # Save/display fake images
    for image in fake_colours:
        # Convert to numpy array and move the color channel from the first to the last dim
        rgb_im = np.array(image.detach().cpu())*120
        rgb_im = np.moveaxis(rgb_im, 0, -1)
        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_LAB2BGR)
        rgb_im = cv2.resize(rgb_im, (320, 320))  # resize for bigger output
        cv2.imwrite(save_path + '/' + f'fake{i}' + '.jpg', rgb_im * 255)  # must scale back to 0-255
        i += 1

    # Rescale L and ab
    L = (L + 1.) * 50.
    ab = ab * 110.

    i = 1
    # Save/display grayscale images
    for image in L:
        # Convert to numpy array and move the color channel from the first to the last dim
        gray_im = np.array(image)
        gray_im = np.moveaxis(gray_im, 0, -1)
        gray_im = (gray_im + [0, 128, 128]) / [100, 255, 255]  # scale Lab image
        gray_im = cv2.resize(gray_im, (320, 320))  # resize for bigger output
        plt.imsave(save_path + '/' + f'gray{i}' + '.jpg', gray_im[:, :, 0], cmap='gray')
        i += 1

    # Concat real L and ab
    L = torch.reshape(L[:, 0, :, :], (L.shape[0], 1, L.shape[2], L.shape[3]))
    real_colours = torch.cat([L, ab], dim=1)

    i = 1
    # Save/display real images
    for image in real_colours:
        # Convert to numpy array and move the color channel from the first to the last dim
        rgb_im = np.array(image)
        rgb_im = np.moveaxis(rgb_im, 0, -1)
        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_LAB2BGR)
        # cv2.imshow("Real", rgb_im)  # show numpy array
        # cv2.waitKey(0)  # wait for ay key to exit window
        rgb_im = cv2.resize(rgb_im, (320, 320))  # resize for bigger output
        cv2.imwrite(save_path + '/' + f'real{i}' + '.jpg', rgb_im * 255)  # must scale back to 0-255
        i += 1

    # Save/display separate Lab color channels
    i = 1
    for image in real_colours:
        figure = lab_space(image)
        figure.savefig(save_path + '/' + f'space{i}' + '.jpg')
        i += 1


if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = "../Dataset/test/"
    checkpoint_path = "../Results/saved_models/checkpoint_99.pth"
    image_size = 256
    pretrained = False
    num_of_im = 6

    # Plot losses of Generator and Discriminator during training
    #plot_losses('../Results/output_losses.csv')

    # Create dataloader
    test_dataloader = get_dataloader(image_path, image_size, num_of_im, pretrained, training_mode=False)

    # Instantiate GAN model
    model = ColorizationGAN(device, pretrained=pretrained)
    model.gen, model.disc = load_checkpoints(model.gen, model.disc, checkpoint_path)

    # If num_of_im is set to the number of images present in the input folder (e.g., 'test_images'), then all images
    # will be plotted. Otherwise, only the last num_of_im will.
    for data in test_dataloader:
        colorize(model, data)
        break
