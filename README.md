# PyTorch Implementation of GLFC

This is the implementation code of CVPR 2022 paper 'Federated Class-Incremental Learning'.

![overview](https://github.com/conditionWang/FCIL/tree/main/fig/overview.png)


## requirement

python == 3.6

torch == 1.2.0

numpy

PIL

torchvision == 0.4.0

cv2

scipy == 1.5.2

sklearn == 0.24.1


## pre-preparations

### CIFAR100

You don't need to do anything before running the experiments of CIFAR100.

### Mini-Imagenet (Imagenet-Subset)

You need to download the Mini-Imagenet from [here](https://github.com/yaoyao-liu/mini-imagenet-tools) and place it in './train'.

### Tiny-Imagenet

You need to download the Tiny-Imagenet from [here](https://github.com/seshuad/IMagenet) and place it in './tiny-imagenet-200'.


## run

```shell
python fl_main.py
```

The detailed arguments can be found in './src/option.py'.

## performance

![cifar](https://github.com/conditionWang/FCIL/tree/main/fig/cifar_result.png)

![imagenet-subset](https://github.com/conditionWang/FCIL/tree/main/fig/imagenet_subset_result.png)


## cite

If you find our work is helpful to your research, please consider to cite.




