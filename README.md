# Image ${\color{red}C}$ ${\color{orange}o}$ ${\color{yellow}l}$ ${\color{green}o}$ ${\color{blue}r}$ ${\color{purple}i}$ ${\color{red}z}$ ${\color{orange}a}$ ${\color{yellow}t}$ ${\color{green}i}$ ${\color{blue}o}$ ${\color{purple}n}$ with <br /> Conditional Generative Adversarial Networks

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

**4.** Modify the *requirements.txt* file: 

> If your machine **does NOT support** CUDA, add the following line at the top of the *requirements.txt* file:
>> extra-index-url https://download.pytorch.org/whl/cpu
>
> If your machine **does support** CUDA, add the following line instead, replacing **117** with the CUDA version your machine supports:
>> extra-index-url https://download.pytorch.org/whl/cu117

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

**4.** Under the parent directory, create also the following two folders *dataset/training* & *dataset/training*:

``` shell
$ mkdir -p dataset/training
$ mkdir -p dataset/test
```

**5.** Run the following command to randomly sample 8*K* images for the training set and 2*K* images for the test set from their respective entire subsets:

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

> --batch_size (default=16)
> 
> --num_epochs (num_epochs=100)
> 
> --lr_g: The generator's learning rate (default=0.0002)
> 
> --lr_d: The discriminator's learning rate (default=0.0002)
> 
> --pretrained : If set to TRUE, the pre-trained ResNet is instantiated; otherwise, if FALSE, the U-Net with random initialization is utilized (default=False).


---

### [**Output**](#) <a name="output"></a>

During model training, a folder named *results/* is automatically generated. Within this folder, a sub-directory named either *U_Net/* or *ResNet/* is created based on the training configuration. This directory contains the model checkpoints for each epoch and a CSV file documenting the losses across all epochs.

---
### [**Visualization**](#) <a name="visualization"></a>

The script *visualize.py* loads a checkpoint, initializes the corresponding model, plots the loss curves, and finally evaluates the model on the test set.

To execute this script, first modify the following variables from the code itself:
- *base_results_path:*  Indicates the directory for results (default="./results/U_net")
- *checkpoint_path:* Indicates the model we intend to load.
- *pretrained:* Set to True for ResNet cGAN and False for UNet cGAN.
  
Running this script will display a grid illustrating the progression of losses based on the *output.csv* file. Additionally, it will generate a new directory *output_images/* within *./results/$MODEL/* containing the real, grayscale, and generated images.

---

### [**Trained Models**](#) <a name="trained"></a>

You can download our trained models for 300 epochs from the following links:
- ResNet cGAN: 
  - Last checkpoint: https://drive.google.com/file/d/1Sh8A9pZc8-isVd7ZL4lwDdnhUZQ6Bwgp/view?usp=share_link
  - Loss CSV: https://drive.google.com/file/d/1eEVHeMD2ZJDAmf5nYYjnT_9ZDM9QNTNV/view?usp=sharing
- U-Net cGAN: 
  - Last checkpoint: https://drive.google.com/file/d/1zk9hPP9b5ZNSTLfAenukymkPt_gLvK4A/view?usp=sharing
  - Loss CSV: https://drive.google.com/file/d/1gCKI4EAH56c49FvYdiQl-IKKj6TtkHUz/view?usp=sharing

### [**Team**](#) <a name="team"></a>

- [Chryssa Nampouri](https://github.com/ChryssaNab)
- [Philip Andreadis](https://github.com/philip-andreadis)
- Christodoulos Hadjichristodoulou
