import torch
from .utils import leaky_clamp


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class InstanceNorm1d(torch.nn.Module):
    """An implementation of InstanceNorm1d compatible with ONNX"""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x shape: (N, C, L)
        mean = x.mean(dim=[2], keepdim=True)
        var = x.var(dim=[2], keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            # reshape weight and bias for broadcasting
            weight = self.weight.view(1, -1, 1)
            bias = self.bias.view(1, -1, 1)
            x_normalized = x_normalized * weight + bias
        return x_normalized


class ClampedInstanceNorm1d(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ClampedInstanceNorm1d, self).__init__()
        self.norm = InstanceNorm1d(*args, **kwargs)

    def forward(self, x):
        return self.norm(leaky_clamp(x, -1e15, 1e15))


class ClampedInstanceNorm2d(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ClampedInstanceNorm2d, self).__init__()
        self.norm = torch.nn.InstanceNorm2d(*args, **kwargs)

    def forward(self, x):
        return self.norm(leaky_clamp(x, -1e10, 1e10, 0.0001))