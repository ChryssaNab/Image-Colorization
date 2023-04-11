import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_losses(path):
    """ Plots the logged losses saved as a csv file in 'path'.

    Arguments
    ----------
    path : <class 'str'>
        path to the csv file

    """
    loss_df = pd.read_csv(path,index_col=False)
    # for loss_name, loss_array in loss_dict.items():
    #     print(f"{loss_name}: {loss_meter.avg:.5f}")
    epoch_range = np.arange(1,len(loss_df.iloc[:,1])+1)
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle('Losses')
    i = 1
    for loss_name in loss_df:
        ax = plt.subplot(3,2,i)
        loss_df[loss_name].plot(ax=ax)
        ax.set_title(loss_name)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        i += 1
    plt.show()


if __name__ == "__main__":
    plot_losses('output.csv')