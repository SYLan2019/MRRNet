# MRRNet
## 1. Introduction
This repository contains PyTorch implementation of the following paper: **Face Super-Resolution with Spatial Attention Guided by Multiscale Receptive-Field Features**.

Face super-resolution (FSR) is dedicated to the restoration of high-resolution (HR) face images from their low-resolution (LR) counterparts. Many deep FSR methods exploit facial prior knowledge (e.g., facial landmark and parsing map) related to facial structure information to generate HR face images. However, directly training a facial prior estimation network with deep FSR model requires manually labeled data, and is often computationally expensive. In addition, inaccurate facial priors may degrade super-resolution performance. In this paper, we propose a residual FSR method with spatial attention mechanism guided by multiscale receptive-field features (MRF) for converting LR face images (i.e., 16 × 16) to HR face images (i.e., 128 × 128). With our spatial attention mechanism, we can recover local details in face images without explicitly learning the prior knowledge. Quantitative and qualitative experiments show that our method outperforms state-of-the-art FSR method.
## 2. Requirements and Installation
We recommended the following dependencies.
* Python>=3.7
* torch>=1.5.1
* opencv-python
* scikit-image
* tqdm
* imgaug

For installation, clone this repository:
```
  git clone https://github.com/SYLan2019/MRRNet.git
```
## 3.Preparing Data for training
### Getting the CelebA Dataset
* Download the dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* ...



## 3. Citing MRRNet

If you use this repository or would like to refer the paper, please use the following BibTeX entry
```
@INPROCEEDINGS{MRRNet,
  author={Huang, Weikang and Lan, Shiyong and Wang, Wenwu and Yuan, Xuedong and Yang, Hongyu and Li, Piaoyang and Ma, Wei},
  booktitle={International Conference on Artificial Neural Networks,(ICANN2022)}, 
  title={Face Super-Resolution with Spatial Attention Guided by Multiscale Receptive-Field Features}, 
  year={2022},
  volume={},
  number={},
  pages={},
  doi={}}
```
