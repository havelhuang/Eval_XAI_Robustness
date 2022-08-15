from cgitb import grey
import utils
import torch
import numpy as np
from utils import get_nearest_oppo_dist, cal_robust, cal_attr, fitness_score, mutation, min_max_scale
import numpy as np
from multi_level import multilevel_uniform, greyscale_multilevel_uniform
from utils import calculate_fid, pearsonr
import time


def eval_ss(model, dataset, data_config, batch_size, eps, cuda):

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


        with torch.no_grad():
            target_pred = torch.argmax(model(data/2+0.5), dim=1)

            x_test[(idx * batch_size):((idx + 1) * batch_size), :, :, :] = data
            y_test[(idx * batch_size):((idx + 1) * batch_size)] = target
            y_pred[(idx * batch_size):((idx + 1) * batch_size)] = target_pred
           

    indices = torch.where(y_pred==y_test)
    x_test = x_test[indices] 
    y_test = y_test[indices]


    grey_scale = True

    if grey_scale:
        robustness_stat = greyscale_multilevel_uniform
    else:
        robustness_stat = multilevel_uniform

    # sigma = 0.1
    rho = 0.1
    debug= False
    stats=False
    count_particles = 1000
    # 1000
    count_mh_steps = 250
    log_p_set = []
    log_p1_set = []
    log_p2_set = []

    log_std_set = []
    log_std1_set = []
    log_std2_set = []

    print('rho', rho, 'count_particles', count_particles, 'count_mh_steps', count_mh_steps)
    print('------------------------------------------------------')

    for idx in range(500):
        
        print('Sample ID: ',idx)
    
        x_seed = x_test[idx] /2 + 0.5
        y_seed = y_test[idx]
        attr_seed, pred_ind = cal_attr(model,x_seed.unsqueeze(0),y_seed.unsqueeze(0))
        attr_seed = attr_seed[0]


        def prop(x):
            y = model(x)
            y_diff = torch.cat((y[:,:y_seed], y[:,(y_seed+1):]),dim=1) - y[:,y_seed].unsqueeze(-1)
            y_diff, _ = y_diff.max(dim=1)
            return y_diff #.max(dim=1)

        def prop1(x): 
            attr, pred_ind = cal_attr(model,x,y_seed)
            int_dis = pearsonr(attr,attr_seed)
            obj = pred_ind*1/int_dis - 1/0.4
            return obj

        def prop2(x): 
            attr, pred_ind = cal_attr(model,x,y_seed)
            int_dis = pearsonr(attr,attr_seed)
            obj = -pred_ind*int_dis - 0.6
            return obj    


        start = time.time()
        with torch.no_grad():
            lg_p, cov_square, test_set, l = robustness_stat(prop1=prop, prop2=None, x_sample =x_seed, sigma = eps, CUDA=cuda, rho=rho, count_particles=count_particles,
                                                count_mh_steps=count_mh_steps, debug=debug, stats=stats)                                         
        end = time.time()
        print(f'Took {(end - start) / 60} minutes...')
        
        if cov_square == None:
            continue

        lg_std = (torch.log(cov_square)+2*lg_p)/2
        print('lg_p', lg_p, 'lg_std', lg_std.item())
        print('---------------------------------')


        start = time.time()
        with torch.no_grad():
            lg_p1, cov_square, test_set, l = robustness_stat(prop1=prop1, prop2=None, x_sample =x_seed, sigma = eps, CUDA=cuda, rho=rho, count_particles=count_particles,
                                                count_mh_steps=count_mh_steps, debug=debug, stats=stats)
        end = time.time()
        print(f'Took {(end - start) / 60} minutes...')

        if cov_square == None:
            continue
        
        lg_std1 = (torch.log(cov_square)+2*lg_p1)/2
        print('lg_p', lg_p1, 'lg_std', lg_std1.item())
        print('---------------------------------')


        start = time.time()
        with torch.no_grad():
            lg_p2, cov_square, test_set, l = robustness_stat(prop1=prop, prop2=prop2, x_sample =x_seed, sigma = eps, CUDA=cuda, rho=rho, count_particles=count_particles,
                                                count_mh_steps=count_mh_steps, debug=debug, stats=stats)
        end = time.time()
        print(f'Took {(end - start) / 60} minutes...')

        if cov_square == None:
            continue
        
        lg_std2 = (torch.log(cov_square)+2*lg_p2)/2
        print('lg_p', lg_p2, 'lg_std', lg_std2.item())
       



        log_p_set.append(lg_p)
        log_p1_set.append(lg_p1)
        log_p2_set.append(lg_p2)

        log_std_set.append(lg_std.item())
        log_std1_set.append(lg_std1.item())
        log_std2_set.append(lg_std2.item())


        print('-------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------')
        print('Avg. misclassfication log probaility:{} +- {}'.format(np.mean(log_p_set),np.mean(log_std_set)))
        print('Avg. misinterpretation log probaility (classified correctly)):{} +- {}'.format(np.mean(log_p1_set),np.mean(log_std1_set)))
        print('Avg. misinterpretation log probaility (classified incorrectly)):{} +- {}'.format(np.mean(log_p2_set),np.mean(log_std2_set)))
        print('-------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------')
    print('------------------------------------------END-------------------------------------------')
    
    