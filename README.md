# SGSN

Supervised Generative Segmentation Network for road segmentation for all-day outdoor robot navigation.

<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/sgsn.png" width="480" height="450" alt="SGSN Outputs"/>

## Major Requirements

Python 2.7.12

Numpy 1.13.1

Tensorflow 1.2.0

cuda 8.0

cudnn 5.1.10

OpenCV 2.4.13

## Running codes

### 1. Build six folders

(YOUR_PATH)/SGSN/datasets/train/X/images/

(YOUR_PATH)/SGSN/datasets/train/X/labels/

(YOUR_PATH)/SGSN/datasets/train/Y/images/

(YOUR_PATH)/SGSN/datasets/train/Y/labels/

(YOUR_PATH)/SGSN/datasets/test/X/images/

(YOUR_PATH)/SGSN/datasets/test/X/labels/

### 2. Download UAS dataset

Please follow the links in the Download links.

### 3. Copy images and labels

To build training set, copy the training images and the labels (not label graphs) of dusk_sight, night_sight, rain_sight and sun_sight to X domain (./datasets/train/X/images/, ./datasets/train/X/labels), and copy the images and the labels of dusk_sight to Y domain (./datasets/train/Y/images/, ./datasets/train/Y/labels).

To build test set, copy the test images and the labels (not label graphs) of dusk_sight, night_sight, rain_sight and sun_sight to X domian (./datasets/test/X/images/, ./datasets/test/X/labels).

### 4. Duplicate images and labels of night in training set

cd (YOUR_PATH)/SGSN/datasets

python2 ./copy_night.py

### 5. Train SGSN

cd (YOUR_PATH)/SGSN

python2 ./train.py

### 6. Evaluate SGSN

cd (YOUR_PATH)/SGSN

python2 ./evaluate.py

## Citation

# UAS (UESTC All-Day Scenery)

## Introduction

The dataset for road segmentation for the outdoor images in any time. 

<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/DuskSight58.jpg" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/DuskLabelGraph58.png" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/DuskSight161.jpg" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/DuskLabelGraph161.png" width="160" height="90">

<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/NightSight248.jpg" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/NightLabelGraph248.png" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/NightSight716.jpg" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/NightLabelGraph716.png" width="160" height="90">

<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/RainSight752.jpg" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/RainLabelGraph752.png" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/RainSight808.jpg" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/RainLabelGraph808.png" width="160" height="90">

<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/SunSight1169.jpg" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/SunLabelGraph1169.png" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/SunSight1304.jpg" width="160" height="90">&nbsp;<img src="https://github.com/yuxiaoz/SGSN/blob/master/images/SunLabelGraph1304.png" width="160" height="90">

A total of 6380 images of different time and weather conditions are contained in UAS, including 1399 samples captured at dusk, 2167 samples captured at night, 819 samples captured in the rain and 1995 samples captured in the sunshine.
For each image, a precise binary semantic label is made by manual annotation.

## Download links

https://pan.baidu.com/s/1IWSVKYBrYwxaRThPfDsDGg

## Citation

  @article{zhang2018road,
    title={Road segmentation for all-day outdoor robot navigation},
    author={Zhang, Yuxiao and Chen, Haiqiang and He, Yiran and Ye, Mao and Cai, Xi and Zhang, Dan},
    journal={Neurocomputing},
    volume={314},
    pages={316--325},
    year={2018},
    publisher={Elsevier}
  }