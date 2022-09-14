from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
from dataloader import dividemix_dataloader
import wandb
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_classes', default=0, type=int)
parser.add_argument('--task', default='2', type=str)
parser.add_argument('--datasets', default='5', type=str, help='path to dataset')
args = parser.parse_args()

total_data_szie = len(open('data/task{}/{}/data/label.txt'.format(args.task, args.datasets),'rU').readlines())
warm_up = 30
if args.task == '1' or args.task == '2' or args.task == '3':
    if args.datasets == '1' or args.datasets == '2':
        args.num_classes = 10
    elif args.datasets == '3' or args.datasets == '4':
        args.num_classes = 100
    elif args.datasets == '5' or args.datasets == '6':
        args.num_classes = 200
elif args.task == '4':
    if args.datasets == '1':
        args.num_classes = 50
    elif args.datasets == '2':
        args.num_classes = 100
    elif args.datasets == '3':
        args.num_classes = 50
loader = dividemix_dataloader(batch_size=args.batch_size,num_workers=4, task=int(args.task), sub_task=int(args.datasets))



# if args.task == '2':
#     if args.datasets == '3' or args.datasets == '4':
#         total_data_szie = 50000
#     if args.datasets == '5' or args.datasets == '6':
#         total_data_szie = 21600
# elif args.task == '1':
#     if args.datasets == '3' or args.datasets == '4':
#         total_data_szie = 50000
#     if args.datasets == '5' or args.datasets == '6':
#         total_data_szie = 100000

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)



wandb.init(project="task{}".format(args.task), entity="lifuguan", name="DivideMix_task{}_{}".format(args.task, args.datasets))
wandb.config.update(args)

def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.datasets, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        wandb.log({'Labeled loss':Lx.item(), 'Unlabeled loss':Lu.item()})


def val(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    sys.stdout.write("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    wandb.log({'Accuracy':acc})


def warmup(epoch,net,optimizer,dataloader, net_name):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.datasets, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        wandb.log({'{}-CE-loss'.format(net_name):loss.item()})

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(total_data_szie)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(evaltrain_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

def get_train_label(net1,net2, val_loader):
    net1.eval()
    net2.eval()
    end_pre = torch.zeros(len(val_loader.dataset))
    n = 0
    with torch.no_grad():
        for _, (inputs, _) in enumerate(val_loader):
            inputs= inputs.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1) 
            for b in range(inputs.size(0)):
                end_pre[n] = predicted[b]
                n += 1       
    return end_pre

def get_test_label(net1,net2, val_loader):
    net1.eval()
    net2.eval()
    end_pre = torch.zeros(len(val_loader.dataset))
    n = 0
    with torch.no_grad():
        for _, (inputs) in enumerate(val_loader):
            inputs= inputs.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1) 
            for b in range(inputs.size(0)):
                end_pre[n] = predicted[b]
                n += 1       
    return end_pre

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))



print('| Building net | ')
def create_model():
    model = ResNet18(num_classes=args.num_classes)
    model = model.cuda()
    return model
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks


for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 75:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    evaltrain_loader = loader.run('eval_train')   
    val_loader = loader.run('val')   
    test_loader = loader.run('test')
    if args.task == '4':
        web_test_loader = loader.run('web_test')


    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader, 'net1')    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader, 'net2') 
   
    else:        
        print('Eval Net1 & Net2') 
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2         

    val(epoch,net1,net2)  

    if (epoch+1) % 30 == 0:
        save_path = Path('results/result_a/task{}/{}/'.format(args.task, args.datasets))

        if save_path.exists() is False:
            save_path.mkdir(parents=True)
        # np.save(save_path.as_posix()+'/model.npy', net1.state_dict())
        end_pre_train = get_train_label(net1, net2, val_loader)
        end_pre_train = np.array(end_pre_train.cpu())
        np.save(save_path.as_posix()+'/label_train.npy', end_pre_train)


        if args.task == '1' or args.task == '2' or args.task == '3':
            end_pre_test = get_test_label(net1, net2, test_loader)
            end_pre_test = np.array(end_pre_test.cpu())
            np.save(save_path.as_posix()+'/label_test.npy', end_pre_test)
        elif args.task == '4':
            end_pre_test = get_test_label(net1, net2, test_loader)
            end_pre_test = np.array(end_pre_test.cpu())
            np.save(save_path.as_posix()+'/label_test_web.npy', end_pre_test)
            end_pre_web = get_test_label(net1, net2, web_test_loader)
            end_pre_web = np.array(end_pre_web.cpu())
            np.save(save_path.as_posix()+'/label_test_img.npy', end_pre_web)

