#!/usr/bin/env python3
import argparse
from xmlrpc.client import boolean
from numpy.core.fromnumeric import argmax
import torch
from distutils import util
import os
import os.path
from utils import calculate_fid
from torchvision.datasets import celeba
from model.celeba import MobileNet
from model.mnist import mnist
from model.fashionmnist import FashionCNN
from model.svhn import svhn
from model.cifar10_models.cifar10 import resnet20
from model.cifar10_models.vgg import VGG
from model.cifar10_models.dla import DLA
from model.cifar10_models.densenet import densenet_cifar
from model.cifar10_models.mobilenetv2 import MobileNetV2
from data import TRAIN_DATASETS, DATASET_CONFIGS, TEST_DATASETS
from train import train_model
from eval_ga import eval_ga
from eval_ss import eval_ss
from baseline import baseline
from simple_mc import simple_mc
import utils


parser = argparse.ArgumentParser('Interpretation Testing Pytorch Implementation')
parser.add_argument('--dataset', default='mnist', choices=list(TRAIN_DATASETS.keys()))
parser.add_argument('--train', default = False, dest='train',type=util.strtobool)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--sample-size', type=int, default=20)
parser.add_argument('--lr', type=float, default = 1e-03)
parser.add_argument('--weight-decay', type=float, default = 5e-04)
parser.add_argument('--resume', default = False,type=util.strtobool)
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

# mnist 
# FashionMnist 
# svhn 
# cifar10 
# celeba 


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = args.cuda and torch.cuda.is_available()
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_dataset = TRAIN_DATASETS[args.dataset]
    test_dataset = TEST_DATASETS[args.dataset]


    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)


    if args.dataset == 'celeba':
        
        model = MobileNet(num_classes=dataset_config['classes'], label = args.dataset)
        eps = 0.05
        # 0.05
        train_dataset = utils.filter_celeba(train_dataset)
        test_dataset = utils.filter_celeba(test_dataset)

    else:

        if args.dataset == 'mnist':
            eps = 0.3
            model = mnist(num_classes=dataset_config['classes'], label = args.dataset)

        if args.dataset == 'FashionMnist':
            eps = 0.08
            model = FashionCNN(num_classes=dataset_config['classes'], label = args.dataset)

        if args.dataset == 'svhn':
            eps = 0.03
            model = svhn(num_classes=dataset_config['classes'], data_name= args.dataset)

        if args.dataset == 'cifar10':
            eps = 0.03
            model = resnet20(num_classes=dataset_config['classes'], label = args.dataset)
            # model = VGG('VGG16',label = args.dataset)
            # model = MobileNetV2(label = args.dataset)

    # move the model parameters to the gpu if needed.
    if cuda:
        model.cuda()


    # run a test or a training process.
    if args.train:
        train_model(
            model, train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            cuda=cuda
        )

    else:
        utils.load_checkpoint(model, args.checkpoint_dir,cuda)

        # # eval the robustness of explanation by getetic algorithm
        # eval_ga(model, test_dataset, dataset_config, args.batch_size, eps, cuda)

        # # eval the robustness of explanation by subset simulation
        # eval_ss(model, test_dataset, dataset_config, args.batch_size, eps, cuda)

        ###############################################################################
        ###############################################################################
        # # simple monte carlo sampling 

        # # compare smc with ss
        # simple_mc(model, test_dataset, dataset_config, args.batch_size, eps, cuda)

        # # compare smc with ga
        # baseline(model, test_dataset, dataset_config, args.batch_size, eps, cuda)




        
        

        
       
        



       