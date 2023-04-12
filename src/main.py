import argparse

from training import train_model


def main():
    image_size = 256
    batch_size = 16
    num_epochs = 2
    train_data_path = "../Dataset/training/"
    test_data_path = "../Dataset/test/"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--train_data_path", default=train_data_path,
                        type=str, help="The path with the training images"),
    parser.add_argument("--test_data_path", default=test_data_path,
                        type=str, help="The path with the test images"),
    parser.add_argument("--image_size", default=image_size,
                        type=int, help="Size of images to be fed in the model")
    parser.add_argument("--batch_size", default=batch_size,
                        type=int, help="batch size")
    parser.add_argument("--num_epochs", default=num_epochs,
                        type=int, help="Number of epochs to train the model")
    parser.add_argument("--pretrained", default=True,
                        type=bool, help="Enable pre-training")
    parser.add_argument("--lr_g", default=0.0002,
                        type=float, help="Generator learning rate")
    parser.add_argument("--lr_d", default=0.0002,
                        type=float, help="Discriminator learning rate")

    args = parser.parse_args()
    # Train the GAN model for the given args
    train_model(args)


if __name__ == "__main__":
    main()
