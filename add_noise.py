import torch
import torch.nn as nn
import torch.optim as optim

stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
# draw noise samples from std dev tensor
noise = torch.zeros_like(img_train)
noise = torch.normal(mean=noise, std=stdn.expand_as(noise))