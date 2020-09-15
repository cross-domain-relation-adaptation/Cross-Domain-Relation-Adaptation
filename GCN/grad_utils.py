import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        dx = grads * (-1)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, x):
        return GradientReversalFunction.apply(x)


rev_grad = GradientReversalFunction.apply


class DontRequiresGrad:
    def __init__(self, model:torch.nn.Module):
        self.params = list(model.named_parameters())
        self.params_requires_grad = [p.requires_grad for _, p in self.params]

    def __enter__(self):
        for _, p in self.params:
            p.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for (_, p), requires_grad in zip(self.params, self.params_requires_grad):
            p.requires_grad = requires_grad

