from ast import arg
from cgitb import grey
from unittest import result
import utils
import torch
import os
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal
import torch.distributions as dist
from torch.distributions.categorical import Categorical
import torchvision.utils as vutils
import numpy as np
from utils import get_nearest_oppo_dist, cal_robust,cal_attr, fitness_score, mse, mutation, min_max_scale
import numpy as np
from multi_level import multilevel_uniform, greyscale_multilevel_uniform
from utils import calculate_fid, pearsonr
import time
import matplotlib.pyplot as plt
from captum.attr import InputXGradient,IntegratedGradients,GuidedGradCam,LRP,DeepLift, GuidedBackprop


def baseline(model, dataset, data_config, batch_size, sigma, cuda):

    torch.manual_seed(1)
    model.eval()


    n_channel  = data_config['channels']
    img_size = data_config['size']
    n_class = data_config['classes']

    n = len(dataset)
    data_loader = utils.get_data_loader(dataset, batch_size, cuda = cuda)

    attr_test = []
    

    # Get data into arrays for convenience
    if cuda:
        x_test = torch.zeros(n, n_channel, img_size, img_size).cuda()
        y_test = torch.zeros(n, dtype = int).cuda()
        y_pred = torch.zeros(n, dtype = int).cuda()


    else:
        x_test = torch.zeros(n, n_channel, img_size, img_size)
        y_test = torch.zeros(n, dtype = int)
        y_pred = torch.zeros(n, dtype = int)


    
    for idx, (data, target) in enumerate(data_loader):
        if cuda:
            data, target = data.float().cuda(), target.long().cuda()
        else:
            data, target = data.float(), target.long()

        if len(target.size()) > 1:
            target = torch.argmax(target, dim=1)

        attr_batch,_ = cal_attr(model,data/2+0.5,target)
        attr_test.append(attr_batch)

        with torch.no_grad():
            target_pred = torch.argmax(model(data/2+0.5), dim=1)

            x_test[(idx * batch_size):((idx + 1) * batch_size), :, :, :] = data
            y_test[(idx * batch_size):((idx + 1) * batch_size)] = target
            y_pred[(idx * batch_size):((idx + 1) * batch_size)] = target_pred
           

    attr_test = torch.cat(attr_test, dim=0)
    indices = torch.where(y_pred==y_test)
    x_test = x_test[indices] 
    y_test = y_test[indices]
    attr_test = attr_test[indices]

    x_min = 0
    x_max = 1
    count_particles = 1000
    n = 500

    pcc_set = []
    lip_set = []
    sen_set = []

    for idx in range(0,505,5):
        print(idx)

        x_sample = x_test[idx] /2 + 0.5
        attr_sample = attr_test[idx]
        y_sample = y_test[idx]


        # Perturbation model
        if cuda:
            prior = dist.Uniform(low=torch.max(x_sample-sigma*(x_max-x_min), torch.tensor([x_min]).cuda()), high=torch.min(x_sample+sigma*(x_max-x_min), torch.tensor([x_max]).cuda()))
        else:
            prior = dist.Uniform(low=torch.max(x_sample-sigma*(x_max-x_min), torch.tensor([x_min])), high=torch.min(x_sample+sigma*(x_max-x_min), torch.tensor([x_max])))

        lip = 0
        sen = 0


        for i in range(n):
            x = prior.sample(torch.Size([count_particles]))
            # outputs = model(x)
            # labels = torch.argmax(outputs, dim=1)
            # x = x[torch.where(labels==y_sample)]
            
            # obj1 lipchiz
            # obj2 max sensitivity
            attr,_ = cal_attr(model,x,y_sample)
            x_dis = torch.norm(x-x_sample,p=2,dim=(1,2,3))
            int_dis = torch.norm(attr-attr_sample,p=2,dim=(1,2,3))
            obj1,idx1 = torch.max(int_dis/x_dis,dim = 0)
            obj2,idx2 = torch.max(int_dis,dim = 0)

            if obj1.item() > lip or lip == 0:
                lip = obj1.item()
                lip_attr = attr[idx1]
                lip_x = x[idx1]
            
            if obj2.item() > sen or sen == 0:
                sen = obj2.item()
                sen_attr = attr[idx2]
                sen_x = x[idx2]
        

        pcc =  torch.mean((sen_attr - attr_sample)**2)

        pcc_set.append(pcc.item())
        lip_set.append(lip)
        sen_set.append(sen)

        print('MSE:{}'.format(np.mean(pcc_set)))
        print('lip:{}'.format(np.mean(lip_set)))
        print('sen:{}'.format(np.mean(sen_set)))



