import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from params.constants import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS

################################################################################


def _get_even(xs): return xs[:, 0::2]
def _get_odd(xs): return xs[:, 1::2]


def _interleave(first, second, order):
    """
    Given 2 rank-2 tensors with same batch dimension, interleave their columns.

    The tensors "first" and "second" are assumed to be of shape (B,M) and (B,N)
    where M = N or N+1, repsectively.
    """
    cols = []
    if order == 'even':
        for k in range(second.shape[1]):
            cols.append(first[:, k])
            cols.append(second[:, k])
        if first.shape[1] > second.shape[1]:
            cols.append(first[:, -1])
    else:
        for k in range(first.shape[1]):
            cols.append(second[:, k])
            cols.append(first[:, k])
        if second.shape[1] > first.shape[1]:
            cols.append(second[:, -1])
    return torch.stack(cols, dim=1)


class _BaseCouplingLayer(nn.Module):
    def __init__(self, dim, partition, nonlinearity):
        """
        Base coupling layer that handles the permutation of the inputs and wraps
        an instance of torch.nn.Module.
        Usage:
        >> layer = AdditiveCouplingLayer(1000, 'even', nn.Sequential(...))

        Args:
        * dim: dimension of the inputs.
        * partition: str, 'even' or 'odd'. If 'even', the even-valued columns are sent to
        pass through the activation module.
        * nonlinearity: an instance of torch.nn.Module.
        """
        super(_BaseCouplingLayer, self).__init__()
        # store input dimension of incoming values:
        self.dim = dim
        # store partition choice and make shorthands for 1st and second partitions:
        assert (partition in [
                'even', 'odd']), "[_BaseCouplingLayer] Partition type must be `even` or `odd`!"
        self.partition = partition
        if (partition == 'even'):
            self._first = _get_even
            self._second = _get_odd
        else:
            self._first = _get_odd
            self._second = _get_even
        # store nonlinear function module:
        # (n.b. this can be a complex instance of torch.nn.Module, for ex. a deep ReLU network)
        self.add_module('nonlinearity', nonlinearity)

    def forward(self, x):
        """Map an input through the partition and nonlinearity."""
        return _interleave(
            self._first(x),
            self.coupling_law(self._second(
                x), self.nonlinearity(self._first(x))),
            self.partition
        )

    def inverse(self, y):
        """Inverse mapping through the layer. Gradients should be turned off for this pass."""
        return _interleave(
            self._first(y),
            self.anticoupling_law(self._second(
                y), self.nonlinearity(self._first(y))),
            self.partition
        )

    def coupling_law(self, a, b):
        # (a,b) --> g(a,b)
        raise NotImplementedError(
            "[_BaseCouplingLayer] Don't call abstract base layer!")

    def anticoupling_law(self, a, b):
        # (a,b) --> g^{-1}(a,b)
        raise NotImplementedError(
            "[_BaseCouplingLayer] Don't call abstract base layer!")


class AdditiveCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a + b."""

    def coupling_law(self, a, b):
        return (a + b)

    def anticoupling_law(self, a, b):
        return (a - b)


class MultiplicativeCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b."""

    def coupling_law(self, a, b):
        return torch.mul(a, b)

    def anticoupling_law(self, a, b):
        return torch.mul(a, torch.reciprocal(b))


class AffineCouplingLayer(_BaseCouplingLayer):
    """Layer with coupling law g(a;b) := a .* b1 + b2, where (b1,b2) is a partition of b."""

    def coupling_law(self, a, b):
        return torch.mul(a, self._first(b)) + self._second(b)

    def anticoupling_law(self, a, b):
        # TODO
        raise NotImplementedError("TODO: AffineCouplingLayer (sorry!)")


def _build_relu_network(latent_dim, hidden_dim, num_layers):
    """Helper function to construct a ReLU network of varying number of layers."""
    _modules = [nn.Linear(latent_dim, hidden_dim)]
    for _ in range(num_layers):
        _modules.append(nn.Linear(hidden_dim, hidden_dim))
        _modules.append(nn.ReLU())
        _modules.append(nn.BatchNorm1d(hidden_dim))
    _modules.append(nn.Linear(hidden_dim, latent_dim))
    return nn.Sequential(*_modules)


class NICE(nn.Module):
    """
    Replication of model from the paper:
      "Nonlinear Independent Components Estimation",
      Laurent Dinh, David Krueger, Yoshua Bengio (2014)
      https://arxiv.org/abs/1410.8516
    Contains the following components:
    * four additive coupling layers with nonlinearity functions consisting of
      five-layer RELUs
    * a diagonal scaling matrix output layer
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NICE, self).__init__()
        assert (input_dim % 2 ==
                0), "[NICEModel] only even input dimensions supported for now"
        assert (num_layers > 2), "[NICEModel] num_layers must be at least 3"
        self.input_dim = input_dim
        half_dim = int(input_dim / 2)
        self.layer1 = AdditiveCouplingLayer(
            input_dim, 'odd', _build_relu_network(half_dim, hidden_dim, num_layers))
        self.layer2 = AdditiveCouplingLayer(
            input_dim, 'even', _build_relu_network(half_dim, hidden_dim, num_layers))
        self.layer3 = AdditiveCouplingLayer(
            input_dim, 'odd', _build_relu_network(half_dim, hidden_dim, num_layers))
        self.layer4 = AdditiveCouplingLayer(
            input_dim, 'even', _build_relu_network(half_dim, hidden_dim, num_layers))
        self.scaling_diag = nn.Parameter(torch.ones(input_dim))

        # randomly initialize weights:
        for p in self.layer1.parameters():
            if len(p.shape) > 1:
                init.kaiming_uniform_(p, nonlinearity='relu')
            else:
                init.normal_(p, mean=0., std=0.001)
        for p in self.layer2.parameters():
            if len(p.shape) > 1:
                init.kaiming_uniform_(p, nonlinearity='relu')
            else:
                init.normal_(p, mean=0., std=0.001)
        for p in self.layer3.parameters():
            if len(p.shape) > 1:
                init.kaiming_uniform_(p, nonlinearity='relu')
            else:
                init.normal_(p, mean=0., std=0.001)
        for p in self.layer4.parameters():
            if len(p.shape) > 1:
                init.kaiming_uniform_(p, nonlinearity='relu')
            else:
                init.normal_(p, mean=0., std=0.001)

    def forward(self, xs):
        """
        Forward pass through all invertible coupling layers.

        Args:
        * xs: float tensor of shape (B,dim).
        Returns:
        * ys: float tensor of shape (B,dim).
        """
        ys = self.layer1(xs)
        ys = self.layer2(ys)
        ys = self.layer3(ys)
        ys = self.layer4(ys)
        ys = torch.matmul(ys, torch.diag(torch.exp(self.scaling_diag)))
        return ys

    def inverse(self, ys):
        """Invert a set of draws from gaussians"""
        with torch.no_grad():
            xs = torch.matmul(ys, torch.diag(
                torch.reciprocal(torch.exp(self.scaling_diag))))
            xs = self.layer4.inverse(xs)
            xs = self.layer3.inverse(xs)
            xs = self.layer2.inverse(xs)
            xs = self.layer1.inverse(xs)
        return xs

################################################################################


class NICEWrapper:

    def __init__(self):
        self.model = NICE(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)

    def _get_nb_parameters(self):

        nb_params = 0
        for param in self.model.parameters():
            nb_params += param.numel()

        return nb_params

    def save(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict()},
                   os.path.join(path, f'checkpoint_{epoch}.pth'))

    def get_info(self):

        infos = str(self.model)
        infos += f'\nNumber of parameters: {self._get_nb_parameters()}'

        return infos
