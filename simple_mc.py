from cgitb import grey
import utils
import torch
import os
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.categorical import Categorical
import torchvision.utils as vutils
import torch.distributions as dist
import numpy as np
from utils import get_nearest_oppo_dist, cal_robust, cal_attr, fitness_score, mutation, min_max_scale
import numpy as np
from multi_level import multilevel_uniform, greyscale_multilevel_uniform
import time, math
from utils import calculate_fid, pearsonr
import time
import matplotlib.pyplot as plt

def simple_mc(model, dataset, data_config, batch_size, sigma, cuda):

    torch.manual_seed(0)
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

        attr_batch, pred_ind = cal_attr(model,data/2+0.5,target)
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

    idx = 3
    x_min = 0
    x_max = 1
    count_particles = 5000
    n = 20000
    print('------------------------------------------------------')
    print('Sample ID: ',idx)

    x_seed = x_test[idx] /2 + 0.5
    y_seed = y_test[idx]
    attr_seed = attr_test[idx]

    n_f1 = np.zeros(10)
    n_f2 = np.zeros(10)
    count_total = 0

    for t in range(10):
        torch.manual_seed(t)
        # Perturbation model
        if cuda:
            prior = dist.Uniform(low=torch.max(x_seed-sigma*(x_max-x_min), torch.tensor([x_min]).cuda()), high=torch.min(x_seed+sigma*(x_max-x_min), torch.tensor([x_max]).cuda()))
        else:
            prior = dist.Uniform(low=torch.max(x_seed-sigma*(x_max-x_min), torch.tensor([x_min])), high=torch.min(x_seed+sigma*(x_max-x_min), torch.tensor([x_max])))

        start = time.time()
        
        for i in range(n):
            x = prior.sample(torch.Size([count_particles]))

            attr, pred_ind = cal_attr(model,x,y_seed)
            int_dis = pearsonr(attr,attr_seed)

            for itr in range(10):
                n_f1[itr] += int((pred_ind*1/int_dis >= 1/(0.2 + itr*0.01)).float().sum().item())
                n_f2[itr] += int((-pred_ind*int_dis >= (0.4+ itr*0.01)).float().sum().item())

            count_total += count_particles

        print('misinterpretation failure (classified correctly)):{}'.format(np.log(n_f1/(t+1)) - np.log(count_total/(t+1))))
        print('misinterpretation failure (classified incorrectly)):{}'.format(np.log(n_f2/(t+1)) - np.log(count_total/(t+1))))
        end = time.time()
        print('-----------------------------------------------------------')
        print(f'Took {(end - start) / 60} minutes...')
        print('-----------------------------------------------------------')
        