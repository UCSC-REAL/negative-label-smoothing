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
We recommend readers build an virtual environment and install required packages in ``requirements.txt``.

## Experiments on synthetic noisy CIFAR dataset

### Direct training on CIFAR-10
For Vanilla Loss and PLS, direct training works better when learning with symmetric noisy labels under noise rate 0.2. Run the code bellow to reproduce our results:

```
CUDA_VISIBLE_DEVICES=0 python3 main_GLS_direct_train.py --noise_type symmetric --noise_rate 0.2
```
### Warm-up with CE loss
When noise rates are large, warming up with CE loss makes PLS and NLS reaches a better performance. Run the code bellow to generate the warm-up model:

```
CUDA_VISIBLE_DEVICES=0 python3 main_warmup.py --noise_type symmetric --noise_rate 0.2
```

After the warming up, proceed with GLS:

```
CUDA_VISIBLE_DEVICES=0 python3 main_GLS_load.py --noise_type symmetric --noise_rate 0.2
```

## Experiments on CIFAR-N dataset
You may want to refer to "[CIFAR-N Github Page](https://github.com/UCSC-REAL/cifar-10-100n)", and modify the file ``loss.py`` by referring to the ``loss_gls`` plug-in implementation specified above.

### Details of key arguments:

In experiments, we formulate GLS as  ``wa * Vanilla Loss + wb * GLS``.

* --lr: learning rate
* --noise_rate: the error rate in symmetric noise model
* --n_epoch: number of epochs 
* --wa: the weight of Vanilla Loss (default is 0)
* --wb: the weight of GLS (default is 1)
* --smooth_rate: the smooth rate in GLS


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
