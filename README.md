# Feature Map Denosing

Using feature map denosing method to imporve network's robustness against adversarial attack by adding denoising blocks (in my implementation it is a [non-local means block](https://arxiv.org/abs/1711.07971)) into the network. Tested on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Trained with a GTX-1080 gpu and got the following result:

| model | accuracy(%) |
| ---- | ----|
| origin, clean train, clean test | 97.5 |
| origin, clean train, adv test | 0.01 |
| origin, adv train, clean test | 84 |
| origin, adv train, adv test | 63 |
| modified, clean train, clean test | 95 |
| modified, clean train, adv test | 0.01 |
| modified, adv train, clean test | 83 |
| modified, adv train, adv test | 71 |

## Requirements

[pytorch](https://github.com/pytorch/pytorch) 1.0.0

[advertorch](https://github.com/BorealisAI/advertorch) 0.1.4

## Usage

```
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e]
               [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
               [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
               [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
               [--advtrain]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | inception_v3
                        | resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn | FD_resnet101 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --advtrain            use adversarial training

```

## Example

### Basic Training

Train with pretrained:

```
python main.py --arch resnet101 --epochs 10 --lr 0.005 -b 60 --pretrained ./dataset 
```

Train begin with checkpoint:

```
python main.py --arch resnet101 --epochs 18 --lr 0.0005 -b 60 --resume path/to/your/checkpoint ./dataset 
```

### Adversarial Training

Train with pretrained:

```
python main.py --arch resnet101 --epochs 10 --lr 0.005 -b 60 --pretrained --advtrain ./dataset 
```

Train begin with checkpoint:

```
python main.py --arch resnet101 --epochs 18 --lr 0.0005 -b 60 --resume path/to/your/checkpoint --advtrain ./dataset 
```

### Feature Denoising

To test the effect of feature denoising, simply change `--arch resnet101` to `--arch FD_resnet101`. It is a modified resnet with a non-local mean block appended.

## License

This project is MIT licensed, as found in the LICENSE file.
