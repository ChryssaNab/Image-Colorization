# Image colorization with <br /> Conditional Generative Adversarial Networks

### [**Contents**](#)
1. [Project Description](#descr)
1. [Setup](#setup)
2. [Dataset Configuration](#dataset)
3. [Execution](#execution)
4. [Output](#output)
5. [Visualization](#visualization)
6. [Trained Models](#trained)
7. [Team](#team)

---

### [**Project Description**](#) <a name="descr"></a>

In this project we apply Conditional GANs on the task of image colorization - adding color to grayscale images. We implement the framework proposed by Isola et al (https://arxiv.org/abs/1611.07004) and add our own twist by replacing the generator with one based on a ResNet architecture with pre-trained weights on the ImageNet dataset.

---

### [**Setup**](#) <a name="setup"></a>

**1.** We assume that Python3 is already installed in the system.


**2.** Clone this repository:

``` shell
$ git clone https://github.com/ChryssaNab/Deep_Learning-RUG.git
$ cd Deep_Learning-RUG
```

**3.** Create a new Python environment and activate it:

``` shell
$ python3 -m venv env
$ source env/bin/activate
```

**4.** Modify the *requirements.txt* file if necessary: 

  - If your machine does not support CUDA, add the following line at the top of the *requirements.txt* file:

  >--extra-index-url https://download.pytorch.org/whl/cpu

  - If your machine does support CUDA, add the following line instead, replacing **117** with the CUDA version your machine supports:

>--extra-index-url https://download.pytorch.org/whl/cu117

**5.** Install necessary requirements:

``` shell
$ pip install -r requirements.txt
```

---

### [**Dataset Configuration**](#) <a name="dataset"></a>

**1.** Under the parent directory, create the following two folders *initData/MS_COCO/training_set* & *initData/MS_COCO/test_set*:

``` shell
$ mkdir -p initData/MS_COCO/training_set
$ mkdir -p initData/MS_COCO/test_set
```

**2.** Download the original training and test subsets of the MS-COCO dataset (2014 version) from the official website (https://cocodataset.org/) using the `wget` command:

``` shell
$ wget -P initData/MS_COCO/training_set http://images.cocodataset.org/zips/train2014.zip
$ wget -P initData/MS_COCO/test_set http://images.cocodataset.org/zips/test2014.zip
```

**3.** Unzip the files:
``` shell
$ unzip initData/MS_COCO/training_set/train2014.zip -d initData/MS_COCO/training_set
$ unzip initData/MS_COCO/test_set/test2014.zip -d initData/MS_COCO/test_set
```

**4.** Under the parent directory, create also the following two folders *dataset/training* & *dataset/test*:

``` shell
$ mkdir -p dataset/training
$ mkdir -p dataset/test
```

**5.** Run the following command to randomly sample 8K images for the training set and 2K images for the test set from their respective entire subsets:

``` shell
$ python3 src/helper_functions/data_sampling.py
```

If you intend to train the model using the entire dataset instead, you can simply copy all the images from the extracted dataset.

---

### [**Execution**](#) <a name="execution"></a>

To start the training, run the following command: 

``` shell
$ python3 src/main.py <args>
```

The following arguments can be provided to tune the settings of the training:

- --batch_size
- --num_epochs
- --lr_g: The generator's learning rate
- --lr_d: The discriminator's learning rate
- --pretrained : If set to ${\color{green}TRUE}$, the ResNet based cGAN will be used, if set to ${\color{green}FALSE}$ the UNet based cGAN will be used.


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

### [**Team**](#) <a name="team"></a>

- [Chryssa Nampouri](https://github.com/ChryssaNab)
- [Philip Andreadis](https://github.com/philip-andreadis)
- Christodoulos Hadjichristodoulou
