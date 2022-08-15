from cmath import nan
import utils
import torch
from torch.distributions.categorical import Categorical
import numpy as np
from utils import get_nearest_oppo_dist, cal_robust, cal_attr, fitness_score, mutation, min_max_scale, pred_loss
import numpy as np
from utils import calculate_fid



def ga_test_generation(model,x_seed,y_seed,n_particles,eps,max_itr,adv_obj, attr_seed,alpha,n_mate):
    pred_loss = []
    int_dis = []
    # initialize the population
    adv_images = x_seed.repeat(n_particles,1,1,1)
    delta = torch.empty_like(adv_images).normal_(mean=0.0,std=0.01)
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_images = torch.clamp(x_seed + delta, min=0, max=1).detach()

    for i in range(max_itr):
        obj,loss, dis, attr_adv = fitness_score(x_seed, y_seed, adv_images, model, adv_obj, attr_seed,alpha)
        sorted, indices = torch.sort(obj, dim=-1, descending=True)
        parents = adv_images[indices[:n_mate]]
        obj_parents = sorted[:n_mate]

        pred_loss.append(loss[indices[0]].item())
        int_dis.append(dis[indices[0]].item())

        if torch.isnan(obj_parents).any():
            return None, None, None, None


        # Generating next generation using crossover
        m = Categorical(logits=obj_parents)
        parents_list = m.sample(torch.Size([2*n_particles]))
        parents1 = parents[parents_list[:n_particles]]
        parents2 = parents[parents_list[n_particles:]]
        pp1 = obj_parents[parents_list[:n_particles]]
        pp2 = obj_parents[parents_list[n_particles:]]
        pp2 = pp2 / (pp1+pp2)
        pp2 = pp2[(..., ) + (None,)*3]

        mask_a = torch.empty_like(parents1).uniform_() > pp2
        mask_b = ~mask_a
        parents1[mask_a] = 0.0
        parents2[mask_b] = 0.0
        children = parents1 + parents2

        # add some mutations to children and genrate test set for next generation
        children = mutation(x_seed, children, eps, p=0.2)
        adv_images = torch.cat([children,parents], dim=0)

    obj, loss, dis, attr_adv = fitness_score(x_seed, y_seed, adv_images, model, adv_obj, attr_seed, alpha)
    sorted, indices = torch.sort(obj, dim=-1, descending=True)

    parents = adv_images[indices[0]]
    parents_attr = attr_adv[indices[0]]
    parents_dis = dis[indices[0]]
    parents_loss = loss[indices[0]]

    outputs = model(torch.stack([x_seed,parents]))
    labels = torch.argmax(outputs, dim=1)
    print('prediction labels',labels)
    print('interpretation dicrepancy',parents_dis)
    print('prediction loss:',parents_loss)

    return parents_loss, parents_dis, parents, parents_attr


def eval_ga(model, dataset, data_config, batch_size, eps, cuda):

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


    n_particles = 500
    n_mate = 20
    n_channel = 1
    max_itr = 500
    alpha = 1.0
    
    adv_loss = []
    adv_int1_dis = []
    adv_int2_dis = []
    print('------------------------------------------------------')

    for idx in range(500):
        print('Sample ID: ',idx)

        x_seed = x_test[idx] /2 + 0.5
        y_seed = y_test[idx]
        attr_seed, pred_ind = cal_attr(model,x_seed.unsqueeze(0),y_seed.unsqueeze(0))
        attr_seed = attr_seed[0]

        adv_obj = 'clas'
        pred_loss, _, _, _ = ga_test_generation(model,x_seed,y_seed,n_particles,eps,max_itr,adv_obj, attr_seed,alpha,n_mate)
        print('-------------------------------------------------------------------------------------')

        #################################################################################
        # f(x)=f(x'), g(x)=!g(x')
        adv_obj = 'attr1'
        _, int_dis1, _, _ = ga_test_generation(model,x_seed,y_seed,n_particles,eps,max_itr,adv_obj, attr_seed,alpha,n_mate)
        print('-------------------------------------------------------------------------------------')


        ############################################################################
        # f(x)!=f(x'), g(x)=g(x')
        adv_obj = 'attr2'
        _, int_dis2, _, _ = ga_test_generation(model,x_seed,y_seed,n_particles,eps,max_itr,adv_obj, attr_seed,alpha,n_mate)


        if int_dis1 == None or int_dis2 == None:
            continue
        
        adv_loss.append(pred_loss.item())
        adv_int1_dis.append(int_dis1.item())
        adv_int2_dis.append(int_dis2.item())

        print('-------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------')
        print('Avg. classfication robustness:{}'.format(np.mean(adv_loss)))
        print('Avg. interpretation robustness (classified correctly)):{}'.format(np.mean(adv_int1_dis)))
        print('Avg. interpretation robustness (classified incorrectly)):{}'.format(np.mean(adv_int2_dis)))
        print('-------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------')
    print('------------------------------------------END-------------------------------------------')
    
   
   




   
    # # plot adv figures
    # parents_attr = torch.stack([1-min_max_scale(parents_attr1),1-min_max_scale(parents_attr2)])
    # parents = torch.stack([parents1,parents2])

    # x_cat_seed = torch.stack([x_seed,1-min_max_scale(attr_seed)])

    # vutils.save_image(
    #     x_cat_seed,
    #     'seed.png',
    #     normalize=False,
    #     nrow=1
    # )

    # x_cat = torch.cat([parents, parents_attr],0)

    # vutils.save_image(
    #     x_cat,
    #     'demo.png',
    #     normalize=False,
    #     nrow=2
    # )

    




    

    