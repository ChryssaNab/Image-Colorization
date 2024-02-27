# Image colorization with Conditional Generative Adversarial Networks

### [**Contents**](#)
1. [Project Description](#descr)
1. [Project setup](#setup)
2. [Dataset](#dataset)
3. [Execution](#execution)
4. [Output](#output)
5. [Visualization](#visualization)
6. [Trained Models](#trained)
7. [Team](#team)

---

### [**Project Description**](#) <a name="descr"></a>

In this project we apply Conditional GANs on the task of image colorization - adding color to grayscale images. We implement the framework proposed by Isola et al (https://arxiv.org/abs/1611.07004) and add our own twist by replacing the generator with one based on a ResNet architecture with pre-trained weights on the ImageNet dataset.

---

### [**Project setup**](#) <a name="setup"></a>

In order to prepare the environment follow the following commands in a new empty directory:

**1.** Clone this repository:

``` shell
$ git clone https://github.com/ChryssaNab/Deep_Learning-RUG.git
$ cd Deep_Learning-RUG
```

**2.** Create a new python environment and activate it:

``` shell
$ python -m venv env
$ source env/bin/activate
```

**3.** Modify the requirements.txt if necessary. If your machine does not support CUDA, add the following line at the top of the *requirements.txt* file:

>--extra-index-url https://download.pytorch.org/whl/cpu

If your machine does support CUDA, add the following line instead, replacing **117** with the CUDA version your machine supports:

>--extra-index-url https://download.pytorch.org/whl/cu117

**4.** Install the requirements:

``` shell
$ pip install -r requirements.txt
```
---

### [**Dataset**](#) <a name="dataset"></a>

Create two new folder under the parent directory called *initData/MS_COCO/training_set* and *initData/MS_COCO/test_set*.

Download the MSCOCO dataset (2014 version) from the official website (https://cocodataset.org/#download), both the training and testing subsets. Unzip them and copy the output folders in the respective directories we created in the previous step. 

Create a new folder under the parent directory called *Dataset* and under two more directories called *training* and *test*. Navigate to the *src* directory and execute

`python helper_functions/data_sampling.py`

This will populate the training and test folder in *Dataset* with 8000 images and 2000 images from the original dataset respectively. Alternatively, if you want to train the model on the whole dataset, you can just copy all the images from the extracted dataset.

---

### [**Execution**](#) <a name="execution"></a>

To start the training, navigate to the *src* diretory and execute 

`python main.py <args>`

The following arguments can be provided to tune the settings of the training:

- --batch_size
- --num_epochs
- --pretrained : If set to true, the ResNet based cGAN will be used, if set to false the UNet based cGAN will be used.
- --lr_g: The generator's learning rate
- --lr_d: The discriminator's learning rate

---

### [**Output**](#) <a name="output"></a>

After the model is done training, a folder *Results* will be created and under it the folder *Unet* or *Resnet* depending on the training configuration. In them, we can find the checkpoints of the model for each epoch and a csv with the losses over all the epochs.

---
### [**Visualization**](#) <a name="visualization"></a>

The script *visualize.py* loads a checkpoint, creates the corresponding model, plots the loss curves and of course passes images from test set through the model.

To execute this script, edit the following variables from the code itself:
- base_results_path: points the parent results directory. For example "<path_to_project>/Deep_Learning_RUG/Results/U_net"
- checkpoint_path: points to the model we want to load
- pretrained: True for ResNet GAN and false for UNet GAN
- batch_size: if necessary increase or decrease
  
Executing this script will show a grid with the progression of the losses based on the *output.csv* file and create a new directory under *Results* with the real, grayscale and generated images.

---

### [**Trained Models**](#) <a name="trained"></a>

You can download our trained models from the following links:
- ResNet cGAN: 
  - Checkpoint: https://drive.google.com/file/d/1Sh8A9pZc8-isVd7ZL4lwDdnhUZQ6Bwgp/view?usp=share_link
  - Loss csv: https://drive.google.com/file/d/1eEVHeMD2ZJDAmf5nYYjnT_9ZDM9QNTNV/view?usp=sharing
- UNet cGAN: 
  - Checkpoint: https://drive.google.com/file/d/1zk9hPP9b5ZNSTLfAenukymkPt_gLvK4A/view?usp=sharing
  - Loss csv: https://drive.google.com/file/d/1gCKI4EAH56c49FvYdiQl-IKKj6TtkHUz/view?usp=sharing


