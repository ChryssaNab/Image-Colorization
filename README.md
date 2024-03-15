# Image ${\color{red}C}$ ${\color{orange}o}$ ${\color{yellow}l}$ ${\color{green}o}$ ${\color{blue}r}$ ${\color{purple}i}$ ${\color{red}z}$ ${\color{orange}a}$ ${\color{yellow}t}$ ${\color{green}i}$ ${\color{blue}o}$ ${\color{purple}n}$ with <br /> Conditional Generative Adversarial Networks

### [**Contents**](#)
1. [Project Description](#descr)
1. [Setup](#setup)
2. [Data Configuration](#dataset)
3. [Execution](#execution)
4. [Output](#output)
5. [Visualization](#visualization)
6. [Saved Models](#trained)
7. [Team](#team)

---

### [**Project Description**](#) <a name="descr"></a>

In this project, we employ Conditional Generative Adversarial Networks (cGANs) for the task of image colorization, which involves adding color to grayscale images. Initially, we adopt the framework proposed by Isola et al. [[1]](#1), where the generator follows a U-Net-like architecture [[2]](#2) trained from scratch. Subsequently, we introduce our own modification by replacing the generator with one based on a ResNet architecture pre-trained on the ImageNet dataset [[3]](#3). The aim is to conduct a comparative analysis between the two different generator modules of the final network and assess the impact of transfer learning and pre-trained knowledge. 

All experiments are conducted on a subset of the MS-COCO dataset [[4]](#4) due to computational constraints. For a comprehensive overview of the methodology and final results, please refer to the [Report](https://github.com/ChryssaNab/Deep_Learning-RUG/blob/main/report/Image_Colorization_with_CGANs.pdf). 

The current project was implemented in the context of the course "Deep Learning" taught by Professors [Matias Valdenegro](https://mvaldenegro.github.io/) and [Matthia Sabatelli](https://scholar.google.it/citations?hl=it&user=YO2Php8AAAAJ&view_op=list_works&sortby=pubdate) at [University of Groningen](https://www.rug.nl/?lang=en).

---

### [**Setup**](#) <a name="setup"></a>

**1.** We assume that Python3 is already installed on the system.


**2.** Clone this repository:

``` shell
$ git clone https://github.com/ChryssaNab/Image-Colorization.git
$ cd Image-Colorization
```

**3.** Create a new Python environment and activate it:

``` shell
$ python3 -m venv env
$ source env/bin/activate
```

**4.** Modify the *requirements.txt* file: 

> If your machine **does NOT support** CUDA, add the following line at the top of the *requirements.txt* file:
>> --extra-index-url https://download.pytorch.org/whl/cpu
>
> If your machine **does support** CUDA, add the following line instead, replacing **115** with the CUDA version your machine supports:
>> --extra-index-url https://download.pytorch.org/whl/cu115

**5.** Install necessary requirements:

``` shell
$ pip install -r requirements.txt
```

---

### [**Data Configuration**](#) <a name="dataset"></a>

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

**5.** Run the following command to randomly sample 8*K* images for the training set and 2*K* images for the test set from their respective entire sets:

``` shell
$ python3 src/helper_functions/data_sampling.py
```

If you intend to train the model using the entire dataset instead, you can simply copy all the images from the extracted dataset.

---

### [**Execution**](#) <a name="execution"></a>

To start the training, call the *main.py*. 

``` shell
$ python3 src/main.py -h
usage: main.py [-h] [--train_data_path TRAIN_DATA_PATH]
               [--test_data_path TEST_DATA_PATH] [--image_size IMAGE_SIZE]
               [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
               [--pretrained PRETRAINED] [--lr_g LR_G] [--lr_d LR_D]

options:
  -h, --help            show this help message and exit
  --train_data_path TRAIN_DATA_PATH
                        The path with the training images (default: ./dataset/training/)
  --test_data_path TEST_DATA_PATH
                        The path with the test images (default: ./dataset/test/)
  --image_size IMAGE_SIZE
                        Size of images to be fed in the model (default: 256)
  --batch_size BATCH_SIZE
                        The batch size (default: 16)
  --num_epochs NUM_EPOCHS
                        The number of epochs to train the model (default: 100)
  --pretrained PRETRAINED
                        If set to TRUE, the pre-trained ResNet model is instantiated; otherwise,
                        if FALSE, the U-Net with random initialization is utilized (default: False)
  --lr_g LR_G           Generator learning rate (default: 0.0002)
  --lr_d LR_D           Discriminator learning rate (default: 0.0002)

```

<!---
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
-->

---

### [**Output**](#) <a name="output"></a>

During model training, a folder named *results/* is automatically generated. Within this folder, a sub-directory named either *U_Net/* or *ResNet/* is created based on the training configuration. This directory contains the model checkpoints for each epoch and a CSV file documenting the losses across all epochs.

---
### [**Visualization**](#) <a name="visualization"></a>

The script *visualize.py* loads a checkpoint, initializes the corresponding model, plots the loss curves, and evaluates the model on the test set.

To execute the script, first modify the following variables from the code itself:
- *base_results_path:*  Indicates the directory for results (default="./results/U_net")
- *checkpoint_path:* Indicates the path of the model we intend to load.
- *pretrained:* Set to True for ResNet cGAN and False for UNet cGAN.

``` shell
$ python3 src/visualize.py
```
  
Running this script will display a grid illustrating the progression of losses based on the *output.csv* file. Additionally, it will generate a new directory *output_images/* within *./results/$MODEL/* containing the real, grayscale, and generated images.

---

### [**Saved Models**](#) <a name="trained"></a>

You can download our trained models for 300 epochs from the following links:
- ResNet cGAN: 
  - Last checkpoint: https://drive.google.com/file/d/1Sh8A9pZc8-isVd7ZL4lwDdnhUZQ6Bwgp/view?usp=share_link
- U-Net cGAN: 
  - Last checkpoint: https://drive.google.com/file/d/1zk9hPP9b5ZNSTLfAenukymkPt_gLvK4A/view?usp=sharing
    
---

## References
<a id="1">[1]</a> 
Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).*

<a id="2">[2]</a> 
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 (pp. 234-241).* Springer International Publishing.

<a id="3">[3]</a> 
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. *In 2009 IEEE conference on computer vision and pattern recognition (pp. 248–255).*

<a id="4">[4]</a> 
Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014). Microsoft coco: Common objects in context. *In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13 (pp. 740-755).* Springer International Publishing.

---

### [**Team**](#) <a name="team"></a>

- [Chryssa Nampouri](https://github.com/ChryssaNab)
- [Philip Andreadis](https://github.com/philip-andreadis)
- Christodoulos Hadjichristodoulou
