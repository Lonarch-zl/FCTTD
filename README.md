## FCTTD

This is a pytorch implementation of the paper 'FCTTD: Towards Fine-grained Coupling Tensor Tucker Decomposition for Deep Learning Model Compression'. 

In short, FCTTD achieves fine-grained tensor decomposition of the fully connected layer of the CNN model, which is more compact than the Tucker decomposition structure, and achieves greater compression ratio and performance of the linear layer.

![FCTTD](https://github.com/Lonarch-zl/FCTTD/images/FCTTD.png)

## CNN architecture decomposed

- VGG-16

## Dataset

- Cifar-10

## Pretrained model

 Pretrained model pretrained_vgg16.pt is included in the models directory.

## Results

Performance of the model decomposed by FCTTD method in cifar-10 data set. Rank represents the rank of Tucker decomposition and the approximate rank of kruskal in the experiment, #params represents the number of parameters in the linear layer, CR represents the compression ratio of parameters, and Top-1 ACC represents the accuracy. 

Experiments show that FCTTD has more advantages in compression ratio and accuracy than Tucker, TT and TR.

![lab](https://github.com/Lonarch-zl/FCTTD/images\lab.png)

## Contacts

 This code was written by [Zhang Lei](https://github.com/Lonarch-zl)([zhanglei0301@hnu.edu.cn](zhanglei0301@hnu.edu.cn)) .
