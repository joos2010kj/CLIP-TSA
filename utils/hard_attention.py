
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbedTopK(nn.Module):
    def __init__(self, k: float, num_samples: int=1000, sigma: float=0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x, train_mode):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma, train_mode)

class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: float, num_samples:int=1000, sigma: float=0.05, train_mode: bool=True): # k = top-k
        b, d = x.shape #  x ?= score, where b ?= batch, d ?= dim

        k = int(d * k)

        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, # of samples , d

        if k > perturbed_x.shape[-1]:
            k = perturbed_x.shape[-1]
        elif k == 0:
            k = 1

        # k = max(3, k)
        if not train_mode:
            k = min(1000, k)

        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, # of samples , k
        indices = torch.sort(indices, dim=-1).values # b, # of samples , k
        
        # b, # of samples, k, d
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([ None ] * 5)

        grad_expected = torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, ctx.noise)
        grad_expected /= (ctx.num_samples * ctx.sigma)
        grad_input = torch.einsum("bkde,bke->bd", grad_expected, grad_output)
        return (grad_input,) + tuple([ None ] * 5)


class MLP(nn.Module):
    def __init__(self, input_dim=512):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(x))
        return out

class HardAttention(nn.Module):
    def __init__(self, k=1.0, num_samples=100, input_dim=512):
        super(HardAttention, self).__init__()
        self.scorer = MLP(input_dim)
        self.hard_att = PerturbedTopK(k=k, num_samples=num_samples)

    def forward(self, inputs):
        scores = self.scorer(inputs)
        b, t, d = scores.shape

        if b > 1:
            train_mode = True
        else:
            train_mode = False

        topk = self.hard_att(scores.squeeze(-1), train_mode)
        out = topk.unsqueeze(-1) * inputs.unsqueeze(1)
        out = torch.sum(out, dim=1)

        return out
