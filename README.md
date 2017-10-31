# PyramidNet
This repository contains the code for the paper:

Dongyoon Han*, Jiwhan Kim*, and Junmo Kim, "Deep Pyramidal Residual Networks", CVPR 2017 (* equal contribution).

Arxiv: https://arxiv.org/abs/1610.02915. 

The code is based on Facebook's implementation of ResNet (https://github.com/facebook/fb.resnet.torch).

### Caffe implementation of PyramidNet: [site](https://github.com/jhkim89/PyramidNet-caffe)
### PyTorch implementation of PyramidNet: [site](https://github.com/dyhan0920/PyramidNet-PyTorch)

## Abstract
 Deep convolutional neural networks (DCNNs) have shown remarkable performance in image classification tasks in recent years. Generally, deep neural network architectures are stacks consisting of a large number of convolution layers, and they perform downsampling along the spatial dimension via pooling to reduce memory usage. At the same time, the feature map dimension (i.e., the number of channels) is sharply increased at downsampling locations, which is essential to ensure effective performance because it increases the capability of high-level attributes. Moreover, this also applies to residual networks and is very closely related to their performance. In this research, instead of using downsampling to achieve a sharp increase at each residual unit, we gradually increase the feature map dimension at all the units to involve as many locations as possible. This is discussed in depth together with our new insights as it has proven to be an effective design to improve the generalization ability. Furthermore, we propose a novel residual unit capable of further improving the classification accuracy with our new network architecture. Experiments on benchmark CIFAR datasets have shown that our network architecture has a superior generalization ability compared to the original residual networks.

<p align="center"><img src="https://cloud.githubusercontent.com/assets/22743125/19235579/7e7e33c6-8f2d-11e6-9397-1b505688e92a.png" width="960"></p>

Figure 1: Schematic illustration of (a) basic residual units, (b) bottleneck, (c) wide residual units, and (d) our pyramidal residual units.

<p align="center"><img src="https://cloud.githubusercontent.com/assets/22743125/19235610/bb3d5fd0-8f2d-11e6-84bd-46c9b7a4797a.png" width="640"></p>

Figure 2: Visual illustrations of (a) additive PyramidNet, (b) multiplicative PyramidNet, and (c) comparison of (a) and (b).

## Usage

1. Install Torch (http://torch.ch) and ResNet (https://github.com/facebook/fb.resnet.torch).
2. Add the files addpyramidnet.lua and mulpyramidnet.lua to the folder "models".
3. Manually set the parameter "alpha" in the files addpyramidnet.lua and mulpyramidnet.lua (Line 28).
4. Change the learning rate schedule in the file train.lua: "decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0" to "decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0".
5. Train our PyramidNet, by running main.lua as below:

To train additive PyramidNet-164 (alpha=48) on CIFAR-10 dataset:
```bash
th main.lua -dataset cifar10 -depth 164 -nEpochs 300 -LR 0.1 -netType addpyramidnet -batchSize 128 -shareGradInput true
```
To train additive PyramidNet-164 (alpha=48) with 4 GPUs on CIFAR-100 dataset:
```bash
th main.lua -dataset cifar100 -depth 164 -nEpochs 300 -LR 0.5 -nGPU 4 -nThreads 8 -netType addpyramidNet -batchSize 128 -shareGradInput true
```

## Results

#### CIFAR

Top-1 error rates on CIFAR-10 and CIFAR-100 datasets.  "alpha" denotes the widening factor; "add" and "mul" denote the results obtained with additive and multiplicative pyramidal networks, respectively.

| Network                           | # of parameters | Output feat. dimension | CIFAR-10    |  CIFAR-100  |
| --------------------------------- | --------------- | ---------------------- | ----------- | ----------- |
| PyramidNet-110 (mul), alpha=4.75  | 1.7M            |  76                    | 4.62        | 23.16       |
| PyramidNet-110 (add), alpha=48    | 1.7M            |  **64**                | 4.62        | 23.31       |
| PyramidNet-110 (mul), alpha=8     | 3.8M            |  128                   | 4.50        | 20.94       |
| PyramidNet-110 (add), alpha=84    | 3.8M            |  **100**               | 4.27        | 20.21       |
| PyramidNet-110 (mul), alpha=27    | 28.3M           |  432                   | 4.06        | 18.79       |
| PyramidNet-110 (add), alpha=270   | 28.3M           |  **286**               | **3.73**    | **18.25**   |

Top-1 error rates of our model with the **bottleneck architecture** on CIFAR-10 and CIFAR-100 datasets.  We use the additive pyramidal networks.

| Network                           | # of parameters | Output feat. dimension | CIFAR-10    |  CIFAR-100  |
| --------------------------------- | --------------- | ---------------------- | ----------- | ----------- |
| PyramidNet-164 (add), alpha=48    | 1.7M            |  256                   | 4.21        | 19.52       |
| PyramidNet-164 (add), alpha=84    | 3.8M            |  400                   | 3.96        | 18.32       |
| PyramidNet-164 (add), alpha=270   | 27.0M           |  1144                  | **3.48**    | **17.01**   |
| PyramidNet-200 (add), alpha=240   | 26.6M           |  1024                  | **3.44**    | **16.51**   |
| PyramidNet-236 (add), alpha=220   | 26.8M           |  944                   | **3.40**    | **16.37**   |
| PyramidNet-272 (add), alpha=200   | 26.0M           |  864                   | **3.31**    | **16.35**   |

![cifar](https://user-images.githubusercontent.com/22743125/28292795-c058f7dc-6b8b-11e7-9d3a-280ed49a4191.png)

Figure 3: Performance distribution according to number of parameters on CIFAR-10 (left) and CIFAR-100 (right).

#### ImageNet

Top-1 and Top-5 error rates of single-model, single-crop (224*224) on ImageNet dataset.  We use the additive PyramidNet for our results. 

| Network                                   | # of parameters | Output feat. dimension | Top-1 error | Top-5 error |
| ----------------------------------------- | --------------- | ---------------------- | ----------- | ----------- |
| PreResNet-200                             | 64.5M           |  2048                  | 21.66       | 5.79        |
| PyramidNet-200, alpha=300                 | 62.1M           |  1456                  | 20.47       | 5.29        |
| PyramidNet-200, alpha=450, Dropout (0.5)  | 116.4M          |  2056                  | 20.11       | 5.43        |

Model files download: [link](https://1drv.ms/f/s!AmNvwgeB0n4GsiDFDNJWZkEbajJf)


## Notes

1. The parameter "alpha" can only be changed in the files addpyramidnet.lua and mulpyramidnet.lua (Line 28).
2. We recommend to use multi-GPU when training additive PyramidNet with alpha=270 or multiplicative PyramidNet with alpha=27.  Otherwise you may get "out of memory" error.
3. We are currently testing our code in the ImageNet dataset.  We will upload the result when the training is completed.

## Updates

07/17/2017:

1. Caffe implementation of PyramidNet is released.

02/12/2017:

1. Results of the bottleneck architecture on CIFAR datasets are updated.

01/23/2017:

1. Added Imagenet pretrained models.

## Contact
Jiwhan Kim (jhkim89@kaist.ac.kr),
Dongyoon Han (dyhan@kaist.ac.kr),
Junmo Kim (junmo.kim@kaist.ac.kr)
