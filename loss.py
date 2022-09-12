import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
# Loss functions


def det_acc(save_dir,best_ratio,args,loss_matrix, noise_or_not, epoch, sum_epoch = False):
    if sum_epoch is False:
        loss = loss_matrix[:,epoch]
    else:
        loss = np.sum(loss_matrix[:,:epoch+1],axis =1)
    loss_sort = np.argsort(loss)
    true_num_noise = sum(noise_or_not)
    num_noise_dectect = sum(noise_or_not[loss_sort[-true_num_noise:]])
    ratio = num_noise_dectect*1.0/true_num_noise
    if ratio > best_ratio[0]:
        best_ratio[0] = ratio
        np.save(save_dir+'/'+args.noise_type + str(args.noise_rate)+'detect.npy',loss_sort[-true_num_noise:])
    return ratio

def loss_gls(epoch, y, t, smooth_rate=0.1, wa=0, wb=1):
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(y, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=t.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = (wa + wb * confidence) * nll_loss + wb * smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch
 
