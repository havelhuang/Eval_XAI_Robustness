import os
import os.path
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from torchvision import transforms
import numpy as np
import time
from captum.attr import InputXGradient,IntegratedGradients,GuidedGradCam,LRP,DeepLift, GuidedBackprop,Lime,Deconvolution,ShapleyValueSampling
from multi_level import multilevel_uniform, greyscale_multilevel_uniform
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle = False,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def save_checkpoint_adv(model,mode,model_dir, epoch):
    path = os.path.join(model_dir, model.name+'_'+mode)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

def load_checkpoint_adv(model, model_dir,mode):
    path = os.path.join(model_dir, model.name+'_'+mode)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']


def load_checkpoint(model, model_dir,cuda):
    path = os.path.join(model_dir, model.name)

    
    # load the checkpoint.
    if cuda:
        checkpoint = torch.load(path,map_location=torch.device('cuda:0'))
    else:
        checkpoint = torch.load(path,map_location=torch.device('cpu'))

    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch

def get_nearest_oppo_dist(X, y, norm, n_jobs=10):
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    p = norm

    def helper(yi):
        return NearestNeighbors(n_neighbors=1,
                                metric='minkowski', p=p, n_jobs=12).fit(X[y != yi])

    nns = Parallel(n_jobs=n_jobs)(delayed(helper)(yi) for yi in np.unique(y))
    ret = np.zeros(len(X))
    for yi in np.unique(y):
        dist, _ = nns[yi].kneighbors(X[y == yi], n_neighbors=1)
        ret[np.where(y == yi)[0]] = dist[:, 0]

    return nns, ret

def filter_celeba(dataset):
    # drop unrelated attr
    attr = dataset.attr
    attr_names = dataset.attr_names[:40]
    new_attr_names = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    mask_attr = torch.tensor([True if x in new_attr_names else False for x in attr_names]) 
    dataset.attr = attr[:,mask_attr]
    dataset.attr_names = new_attr_names
    # keep only 1 attr instance and drop others
    mask_id = torch.sum(dataset.attr, dim = 1) == 1
    dataset = torch.utils.data.Subset(dataset, torch.where(mask_id)[0])
    return dataset

def cal_robust(x_sample, x_class, model, CUDA, grey_scale,sigma):

    if grey_scale:
        robustness_stat = greyscale_multilevel_uniform
    else:
        robustness_stat = multilevel_uniform

    # sigma = 0.1
    rho = 0.1
    debug= True
    stats=False
    count_particles = 1000
    count_mh_steps = 200

    print('rho', rho, 'count_particles', count_particles, 'count_mh_steps', count_mh_steps)

    def prop(x):
      y = model(x)
      y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
      y_diff, _ = y_diff.max(dim=1)
      return y_diff #.max(dim=1)

    start = time.time()
    with torch.no_grad():
      lg_p, max_val, _, l = robustness_stat(prop, x_sample, sigma, CUDA=CUDA, rho=rho, count_particles=count_particles,
                                              count_mh_steps=count_mh_steps, debug=debug, stats=stats)
    end = time.time()
    print(f'Took {(end - start) / 60} minutes...')

    if debug:
      print('lg_p', lg_p, 'max_val', max_val)
      print('---------------------------------')

    return lg_p

def cal_attr(model,images,seed_y):
    outputs = model(images)
    labels = torch.argmax(outputs, dim=1)
    pred_ind = torch.where(labels==seed_y,1.0,-1.0)
    if images.requires_grad == False:
        images.requires_grad = True
    # InputXGradient,IntegratedGradients,LRP,DeepLift,GuidedBackprop, GuidedGradCam 8 17 52, Lime,Deconvolution,ShapleyValues
    # only for gradcam to get internal layers
    # modulelist = list(model.features.modules())
    input_x_gradient = InputXGradient(model)
    attribution = input_x_gradient.attribute(images,target=labels).detach()
    attribution = torch.abs(attribution)
    return attribution, pred_ind

def mutation(x_seed, adv_images, eps, p):
    delta = torch.empty_like(adv_images).normal_(mean=0.0,std=0.003)
    mask = torch.empty_like(adv_images).uniform_() > p 
    delta[mask] = 0.0
    delta = adv_images + delta - x_seed
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_images = torch.clamp(x_seed + delta, min=0, max=1).detach()
    return adv_images

def pred_loss(x,x_class,model):
    with torch.no_grad():
      y = model(x)
      y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
      y_diff, _ = y_diff.max(dim=1)
      return y_diff 

def cal_dist(x, x_a, model):
    model.eval()
    act_a = model(x_a)[0]
    act = model(x)[0]
    act_a = torch.flatten(act_a, start_dim = 1)
    act = torch.flatten(act, start_dim = 1)
    mse = calculate_fid(act, act_a)
    return mse

def mse(x,x_a):
    loss = (x_a - x)**2
    return torch.mean(loss,dim=[1,2,3])

def psnr(x,x_a):
    mse_loss = torch.mean((x_a - x) ** 2, dim=[1,2,3])
    return 20 * torch.log10(1.0 / torch.sqrt(mse_loss))

def ms_ssim_module(x,x_a):
    x, x_a = torch.broadcast_tensors(x, x_a)
    ms_ssim_val = SSIM(data_range=1, size_average=False, channel=x.shape[-3])
    return ms_ssim_val(x,x_a)

def min_max_scale(x):
    return (x-x.min())/(x.max()-x.min())

def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    """
    x, y = torch.broadcast_tensors(x, y)
    x = torch.flatten(x, start_dim=1)
    y = torch.flatten(y, start_dim=1)
    mean_x = torch.mean(x,dim=1)
    mean_y = torch.mean(y,dim=1)
    xm = x-mean_x[:,None]
    ym = y-mean_y[:,None]
    r_num = torch.sum(xm*ym,dim=1)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1)
    r_val = r_num / r_den
    return r_val

# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*torch.log2(q[i]) for i in range(len(p))])

# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def fitness_score(x,y,x_a,model,adv_obj,attr_seed, alpha):
    loss = pred_loss(x_a,y,model)
    attr, pred_ind = cal_attr(model,x_a,y)
    int_dis = pearsonr(attr,attr_seed)

    if adv_obj == 'clas':
        obj = min_max_scale(loss)
        return obj,loss,int_dis,attr

    # first type adversarial explanation, classificatio correct
    elif adv_obj  == 'attr1':
        obj = pred_ind*torch.exp(pred_ind * min_max_scale(1/int_dis))
        return obj, loss, int_dis, attr

    # second type adversarial explanation, classificatio wrong
    elif adv_obj == 'attr2':

        # boost the adversarial samples in the test set
        if torch.sum(loss>0)/len(loss) < 0.5:
            obj = min_max_scale(loss)
            return obj,loss,int_dis,attr
        # condition on adversarial examples to find adversarial exaplanation
        else:
            obj = -pred_ind*torch.exp(pred_ind * min_max_scale(1/int_dis))
            return obj, loss, int_dis, attr

    else:
        raise Exception("Choose the support adv_obj from clas, attr!")    

def isnan(x):
    return x != x
    
def stats(v):
  print('min', torch.min(v).cpu().detach().numpy(), 'max', torch.max(v).cpu().detach().numpy(), 'mean', torch.mean(v).cpu().detach().numpy(), 'NaNs', torch.sum(isnan(v)).cpu().detach().numpy(), '-Inf', torch.sum(v==float("-Inf")).cpu().detach().numpy(), '+Inf', torch.sum(v==float("Inf")).cpu().detach().numpy() )


    
    

    




