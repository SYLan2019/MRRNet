# MRRNet
## 1. Introduction
This repository contains PyTorch implementation of the following paper: **Face Super-Resolution with Spatial Attention Guided by Multiscale Receptive-Field Features**.

Face super-resolution (FSR) is dedicated to the restoration of high-resolution (HR) face images from their low-resolution (LR) counterparts. Many deep FSR methods exploit facial prior knowledge (e.g., facial landmark and parsing map) related to facial structure information to generate HR face images. However, directly training a facial prior estimation network with deep FSR model requires manually labeled data, and is often computationally expensive. In addition, inaccurate facial priors may degrade super-resolution performance. In this paper, we propose a residual FSR method with spatial attention mechanism guided by multiscale receptive-field features (MRF) for converting LR face images (i.e., 16 × 16) to HR face images (i.e., 128 × 128). With our spatial attention mechanism, we can recover local details in face images without explicitly learning the prior knowledge. Quantitative and qualitative experiments show that our method outperforms state-of-the-art FSR method.

![comparision](./figures/128ImageComV2.pdf)
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
  cd MRRNet
```
## 3.Preparing Data for training
### Getting the CelebA Dataset
* Download the dataset from [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
* Extract the data.
### Image preprocessing
* Use OpenCV to crop face images roughly from CelebA. 
* Remove the excess background in the rough cropped face images.
* resize the cropped images to 128×128 as HR images. Downsample the HR images to 16×16 as LR face images.
We proviede ```crop.py``` to crop face images.

## 4.Training
To train MRRGAN:
```
python train.py --gpus 1 --name MRRNet_m6d16_10middleBlock_SEblock_allattention --model mrrnet \
    --Gnorm "in" --lr 0.0001 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot ../progressiveFSR/crop_celeba --dataset_name celeba --batch_size 12 --total_epochs 20 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500 #--continue_train 
```

## Testing
To test MRRGAN:
```
python test.py --gpus 1 --model mrrnet --name MRRNet_m6d16_10middleBlock_SEblock_allattention \
    --load_size 128 --dataset_name single --dataroot test_dirs/CelebA_test_DIC/LR \
    --pretrain_model_path ./check_points/MRRNet_m6d16_10middleBlock_SEblock_allattention/latest_net_G.pth \
    --save_as_dir results_CelebA/MRRNet_m6d16_10middleBlock_SEblock_allattention/
```



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
