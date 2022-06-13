# To Smooth or Not? When Label Smoothing Meets Noisy Labels

This repository is the official Pytorch implementation of "[To Smooth or Not? When Label Smoothing Meets Noisy Labels](https://arxiv.org/abs/2106.04149)" accepted by ICML2022 (Oral). 


## Plug-in implementation of (Generalized) Label Smoothing in PyTorch
```python
import torch
import torch.nn.functional as F

def loss_gls(logits, labels, smooth_rate=0.1):
    # logits: model prediction logits before the soft-max, with size [batch_size, classes]
    # labels: the (noisy) labels for evaluation, with size [batch_size]
    # smooth_rate: could go either positive or negative, 
    # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch
```

## Required Packages & Environment
**TODO** 

## Experiments on synthetic noisy CIFAR dataset
**TODO** 

## Experiments on CIFAR-N dataset
**TODO** 

## Observations
**TODO** 


## Citation

If you use our code, please cite the following paper:

```
@article{wei2021understanding,
  title={Understanding Generalized Label Smoothing when Learning with Noisy Labels},
  author={Wei, Jiaheng and Liu, Hangyu and Liu, Tongliang and Niu, Gang and Liu, Yang},
  journal={arXiv preprint arXiv:2106.04149},
  year={2021}
}

```

## Thanks for watching!
