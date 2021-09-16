import torch
from torch import nn
from torch import Tensor
from typing import List, Any

"""
Adapted from
https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
"""

class FlowBlock(nn.Module):
    """
    Abstract base class for any flow blocks.
    """

    def __init__(self, dimension):
        super(FlowBlock, self).__init__()
        self.dimension = dimension

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """
        When implemented, forward method will represent z = f(x) and log |det f'(x)/dx|
        x: (*, dimension), z: (*, dimension) and log_det: (*, 1)
        """
        raise NotImplementedError("Forward not implemented")

    def inverse(self, z: Tensor) -> (Tensor, Tensor):
        """
        When implemented, inverse method will represent x = f^-(z) and log |det f^-'(z)/dz|
        z: (*, dimension), x: (*, dimension) and log_det: (*, 1)
        """
        raise NotImplementedError("Inverse not implemented")


class GlowBlock(FlowBlock):
    """
    Adapted from
    https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    As introduced in Glow paper.
    """

    def __init__(self, dimension):
        super().__init__(dimension)
        Q = torch.nn.init.orthogonal_(torch.randn(dimension, dimension))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.D = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S
        self.I = nn.Parameter(torch.ones(self.dimension), requires_grad=False)

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, D) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(self.I)
        U = torch.triu(self.U, diagonal=+1) + torch.diag(self.D)
        W = self.P @ L @ U
        return W

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        W = self._assemble_W()
        z = x @ W  # z: (*, dimension)
        log_det = torch.sum(torch.log(torch.abs(self.D)))
        return z, log_det.repeat(x.shape[0], 1)

    def inverse(self, z: Tensor) -> (Tensor, Tensor):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.D)))
        return x, log_det.repeat(z.shape[0], 1)


class AffineConstantFlow(FlowBlock):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dimension, scale=True, shift=True):
        super().__init__(dimension)
        zeros = torch.zeros(size=(1, dimension))
        self.s = nn.Parameter(torch.randn(1, dimension, requires_grad=True)) if scale else zeros
        self.t = nn.Parameter(torch.randn(1, dimension, requires_grad=True)) if shift else zeros

    def forward(self, x) -> (Tensor, Tensor):
        z = x * torch.exp(self.s) + self.t
        log_det = torch.sum(self.s, dim=1)
        return z, log_det.repeat(x.shape[0], 1)

    def inverse(self, z) -> (Tensor, Tensor):
        x = (z - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s, dim=1)
        return x, log_det.repeat(z.shape[0], 1)


class ConstantModule(nn.Module):

    def __init__(self, output_dimension, initial_value):
        super(ConstantModule, self).__init__()
        self.constant_value = nn.Parameter(torch.Tensor([initial_value]*output_dimension))

    def forward(self, x):
        return self.constant_value.repeat(x.shape[0], 1)


class AffineCoupling(FlowBlock):
    """
    Additive coupling block
    """
    
    def __init__(self,
                 dimension,
                 identity_indices,
                 scaling_model_factory,
                 translating_model_factory):
        super(AffineCoupling, self).__init__(dimension)
        self.identity_indices = identity_indices
        self.transform_indices = [x for x in range(dimension) if x not in identity_indices]
        self.back_indices = [0]*dimension
        count = 0
        for x in self.identity_indices + self.transform_indices:
            self.back_indices[x] = count
            count += 1
        self.identity_indices = nn.Parameter(torch.LongTensor(self.identity_indices), requires_grad=False)
        self.transform_indices = nn.Parameter(torch.LongTensor(self.transform_indices), requires_grad=False)
        self.back_indices = nn.Parameter(torch.LongTensor(self.back_indices), requires_grad=False)
        self.scaling_model = scaling_model_factory(len(identity_indices), len(self.transform_indices))  # possibly a MLP
        self.translating_model = translating_model_factory(len(identity_indices), len(self.transform_indices))  # possibly a MLP

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        x0, x1 = torch.index_select(x, 1, self.identity_indices), torch.index_select(x, 1, self.transform_indices)
        s = self.scaling_model(x0)
        t = self.translating_model(x0)
        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1, keepdim=True)
        return z, log_det

    def inverse(self, z) -> (Tensor, Tensor):
        z0, z1 = z[:, :len(self.identity_indices)], z[:, len(self.identity_indices):self.dimension]
        s = self.scaling_model(z0)
        t = self.translating_model(z0)
        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        x = torch.cat([x0, x1], dim=1)
        x = torch.index_select(x, 1, self.back_indices)
        log_det = torch.sum(-s, dim=1, keepdim=True)
        return x, log_det


class NF(FlowBlock):

    def __init__(self, dimension, blocks: List[FlowBlock]):
        """
        A Normalizing Flow block is a flow block.
        blocks is a sequence of blocks.
        """
        super(NF, self).__init__(dimension)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x) -> (Tensor, Tensor):
        log_det = torch.zeros(size=(x.shape[0], 1)).to(x.device)
        z = x
        for flow in self.blocks:
            z, ld = flow.forward(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z) -> (Tensor, Tensor):
        log_det = torch.zeros(size=(z.shape[0], 1)).to(z.device)
        x = z
        for flow in self.blocks:
            x, ld = flow.forward(x)
            log_det = log_det + ld
        return x, log_det


class NormalizingFlowsModel(nn.Module):

    def __init__(self, dimension, blocks, prior: torch.distributions.Distribution):
        super(NormalizingFlowsModel, self).__init__()
        self.model = NF(dimension, blocks)
        self.prior = prior

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        z, log_det = self.model.forward(x)
        prior_ll = self.prior.log_prob(z).view(x.size(0), -1).sum(1)
        return z, prior_ll + log_det  # batch latent and log-likelihood

    def sample(self, num_samples: int) -> Tensor:
        z = self.prior.sample((num_samples,)).to(self.device)
        x, _ = self.flow.backward(z)
        return x

