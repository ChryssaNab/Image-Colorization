import argparse

from training import train_model


def main():
    image_size = 256
    batch_size = 16
    num_epochs = 100
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
    parser.add_argument("--pretraining_mode", default=True,
                        type=bool, help="Enable pre-training")

    args = parser.parse_args()
    # Run the pipeline given the args
    train_model(args)


if __name__ == "__main__":
    main()
