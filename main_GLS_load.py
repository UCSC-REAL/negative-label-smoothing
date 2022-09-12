# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10
from data.datasets import input_dataset
from models import *
import argparse, sys
import numpy as np
import datetime
import shutil
from random import sample 
from loss import loss_gls
from torch.utils.data import RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--loss', type = str, default = 'gls')
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--ideal', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--model', type = str, help = 'cnn,resnet', default = 'resnet')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--wa', type=float, default=0)
parser.add_argument('--wb', type=float, default=1)
parser.add_argument('--smooth_rate', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')


def adjust_learning_rate(optimizer, epoch, lr_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=lr_plan[epoch]
        

# Train the Model
def train(epoch, num_classes, train_loader, model, optimizer, smooth_rate, wa, wb):
    train_total=0
    train_correct=0
    
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        logits = model(images)
            
        loss = loss_gls(epoch,logits, labels, smooth_rate, wa, wb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, loss.data))


    train_acc=0.0
    return train_acc

# Evaluate the Model
def evaluate(test_loader,model,save=False,epoch=0,best_acc_=0,args=None):
    model.eval()    # Change model to 'eval' mode.
    print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total) 

    if save:
        if acc > best_acc_:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            torch.save(state,os.path.join(save_dir, 'GLS_load'+args.loss + args.noise_type + str(args.noise_rate)+'wa'+str(args.wa)+'wb'+str(args.wb)+'smooth_rate'+str(args.smooth_rate)+'best.pth.tar'))
            best_acc_ = acc
        if epoch == args.n_epoch -1:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            torch.save(state,os.path.join(save_dir,'GLS_load'+args.loss + args.noise_type + str(args.noise_rate)+'wa'+str(args.wa)+'wb'+str(args.wb)+'smooth_rate'+str(args.smooth_rate)+'last.pth.tar'))
    return acc, best_acc_



#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr

wa_val = args.wa
wb_val = args.wb
smooth_rate_val = args.smooth_rate
n_type = args.noise_type

# load dataset
train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type,args.noise_rate)


# load model
print('building model...')

if args.model == 'cnn':
    model = CNN(input_channel=3, n_outputs=num_classes)
else:
    model = ResNet34(num_classes)
print('building model done')


# Load pre-trained model
tmp_dir = args.result_dir +'/' +args.dataset + '/' + args.model 
state_dict = torch.load(f"{tmp_dir}/{args.loss}{args.noise_type}{str(args.noise_rate)}best.pth.tar", map_location = "cpu")
model.load_state_dict(state_dict['state_dict'])

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001, nesterov= True)


### save result and model checkpoint #######   
save_dir = args.result_dir +'/' +args.dataset + '/' + args.model
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128, 
                                   num_workers=args.num_workers,
                                   shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64, 
                                  num_workers=args.num_workers,
                                  shuffle=False)
lr_plan = [1e-6] * 100

model.cuda()
txtfile=save_dir + '/' +  'GLS_load'+args.loss + args.noise_type + str(args.noise_rate)+'wa'+str(args.wa)+'wb'+str(args.wb)+'smooth_rate'+str(args.smooth_rate)+ '.txt'
if os.path.exists(txtfile):
    os.system('rm %s' % txtfile)
with open(txtfile, "a") as myfile:
    myfile.write('epoch: test_acc \n')

epoch=0
train_acc = 0
best_acc_ = 0.0
# training
for epoch in range(args.n_epoch):
# train models
    adjust_learning_rate(optimizer, epoch, lr_plan)
    model.train()
    train_acc = train(epoch,num_classes,train_loader, model, optimizer, smooth_rate=smooth_rate_val, wa=wa_val, wb=wb_val)

# evaluate models
    test_acc, best_acc_ = evaluate(test_loader=test_loader, save=True, model=model,epoch=epoch,best_acc_=best_acc_,args=args)
    
    print('test acc on test images is ', test_acc)
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(test_acc) + "\n")
