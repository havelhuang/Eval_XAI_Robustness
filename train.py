from torch import optim
import torch
import torch.nn as nn
import time
import copy
from torch.autograd import Variable
import torchvision.utils as vutils
from tqdm import tqdm
import utils

# hessian weight mnist 0.1 other 0.01

# gradient_regularization
def gradient_hessian_regularization(model,x,y):

    x = Variable(x.data, requires_grad=True)
    result = model(x)
    loss = nn.CrossEntropyLoss()(result, y)
    grad_x = torch.autograd.grad(loss, x,create_graph=True)[0]
    loss1 = torch.sum(grad_x*grad_x)
    v = torch.normal(0, 1, size=grad_x.shape).to('cuda')
    loss2 = torch.sum(grad_x*v)
    hess_x = torch.autograd.grad(loss2, x,create_graph=True)[0]
    loss3 = torch.sum(hess_x*hess_x)

    return result, loss+0.01*loss1/len(x)+0.1*loss3/len(x)

# gradient_regularization
def gradient_regularization(model,x,y):

    x = Variable(x.data, requires_grad=True)
    result = model(x)
    loss = nn.CrossEntropyLoss()(result, y)
    grad_x = torch.autograd.grad(loss, x,create_graph=True)[0]
    loss2 = torch.sum(grad_x*grad_x)

    return result, loss+0.01*loss2/len(x)

# adversarial training
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=0.1,
                  num_steps=10,
                  step_size=2/255):
    model.eval()
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to('cuda')
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss2 = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss2.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

# hessian_regularization
def hessian_regularization(model,x,y):

    x = Variable(x.data, requires_grad=True)
    result = model(x)
    loss = nn.CrossEntropyLoss()(result, y)
    grad_x = torch.autograd.grad(loss, x,create_graph=True)[0]
    v = torch.normal(0, 1, size=grad_x.shape).to('cuda')
    loss2 = torch.sum(grad_x*v)
    hess_x = torch.autograd.grad(loss2, x,create_graph=True)[0]
    loss3 = torch.sum(hess_x*hess_x)

    return result, loss+0.1*loss3/len(x)


def train(data_loader, model, optimizer, epoch, cuda):
    model.train()
    running_loss = 0.0
    correct = 0
    data_stream = tqdm(enumerate(data_loader))
    for batch_index, (x, y) in data_stream:

        # prepare data on gpu if needed
        if cuda:
            x = x.to('cuda')
            y = y.to('cuda')

        x = x/2 + 0.5
        
        if len(y.size()) > 1:
            real_labels = torch.argmax(y, dim=1)
        else:
            real_labels = y
        
        # flush gradients and run the model forward
        optimizer.zero_grad()


        ###### normal training #########
        result = model(x)
        loss = nn.CrossEntropyLoss()(result, real_labels)

        ###### adversarial training #################
        # with torch.no_grad():
        #     x, y = Variable(x, requires_grad=True), Variable(real_labels)
        #     x = _pgd_whitebox(copy.deepcopy(model), x, y)

        # result = model(x)
        # loss = nn.CrossEntropyLoss()(result, real_labels)

        ####### input gradient norm regularization ############
        # result, loss = gradient_regularization(model,x,real_labels)
        
        ######## input hessian regularization ##################
        # result, loss = hessian_regularization(model,x,real_labels)

        ####### input gradient norm + hessian regularization ############
        # result, loss = gradient_hessian_regularization(model,x,real_labels)

        # backprop gradients from the loss
        loss.backward()
        optimizer.step()

        pred_labels = torch.argmax(result, dim=1)
        correct += (pred_labels == real_labels).sum().item() 
        running_loss += loss.item()

        # update progress
        data_stream.set_description((
            'epoch: {epoch} | '
            'progress: [{trained}/{total}] ({progress:.0f}%) | '
            ' => '
            'loss: {loss:.7f} / '
        ).format(
            epoch=epoch,
            trained=batch_index * len(x),
            total=len(data_loader.dataset),
            progress=(100. * batch_index / len(data_loader)),
            loss = loss.data.item()
        ))
    
    acc = correct / len(data_loader.dataset)
    running_loss /= len(data_loader)
    print('\nTraining set: Epoch: %d, Loss: %.5f, Accuracy: %.2f %%' % (epoch, running_loss, 100. * acc))


def test(data_loader, model,cuda):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    data_stream = tqdm(enumerate(data_loader))
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        for batch_index, (x, y) in data_stream:
            # prepare data on gpu if needed
            if cuda:
                x = x.to('cuda')
                y = y.to('cuda')

            x = x/2 + 0.5

            if len(y.size()) > 1:
                real_labels = torch.argmax(y, dim=1)
            else:
                real_labels = y

            result = model(x)
            loss = criterion(result, real_labels)
            running_loss += loss.item()

            pred_labels = torch.argmax(result, dim=1)
            correct += (pred_labels == real_labels).sum().item()

            data_stream.set_description((
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                ' => '
                'loss: {total_loss:.7f} / '
            ).format(
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss= loss,
            ))

        acc = correct / len(data_loader.dataset)
        running_loss /= len(data_loader)

    print('\nTest set: Loss: %.5f, Accuracy: %.2f %%' % (running_loss, 100. * acc))

    return acc


def train_model(model, train_dataset, test_dataset, epochs=10,
                batch_size=32, sample_size=32,
                lr=3e-04, weight_decay=1e-5,
                checkpoint_dir='./checkpoints',
                resume=False,
                cuda=False):

    # prepare optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )

    if resume:
        epoch_start = utils.load_checkpoint(model, checkpoint_dir)
    else:
        epoch_start = 1

    train_data_loader = utils.get_data_loader(train_dataset, batch_size, cuda=cuda)
    test_data_loader = utils.get_data_loader(test_dataset, batch_size, cuda=cuda)

    BEST_acc = 0.0
    LAST_SAVED = -1

    for epoch in range(epoch_start, epochs+1):
        
        train(train_data_loader, model, optimizer, epoch, cuda)
        acc = test(test_data_loader, model, cuda)
            
        print()
        if acc >= BEST_acc:
            BEST_acc = acc
            LAST_SAVED = epoch
            print("Saving model!")
            utils.save_checkpoint(model, checkpoint_dir, epoch)
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))

