import torch
import torch.nn as nn
import tensorly as tl
import numpy as np
from tensorly.decomposition import CP
from tensorly.decomposition import tucker


class TuckerLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, ranks):
        super(TuckerLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks

        # Initialize shapes for factor matrices
        self.factor_matrices_shapes = [
                                          (out_features[i], ranks[i]) for i in range(3)
                                      ] + [
                                          (in_features[i], ranks[i + 3]) for i in range(3)
                                      ]

        # Initialize Tucker core and factor matrices
        self.core = nn.Parameter(torch.rand(ranks) * 0.1)
        self.factor_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(shape) * 0.1) for shape in self.factor_matrices_shapes
        ])
        self.bias = nn.Parameter(torch.zeros(np.prod(out_features)))

    def set_weights(self, weight, bias):
        """
        Set weights using provided weight tensor and bias.
        """
        weight_reshape = weight.view(*self.out_features, *self.in_features)
        core, factors = tucker(weight_reshape, rank=self.ranks, init='svd')

        self.core.data = core
        for i, factor in enumerate(factors):
            self.factor_matrices[i].data = factor

        self.bias.data = bias

    def forward(self, x):
        """
        Forward pass of TuckerLinearLayer.
        """
        x_reshape = x.view(-1, *self.in_features)
        W_temp = torch.einsum('ijklmn,ai,bj,ck,dl,em,fn->abcdef',
                              self.core, *[mat for mat in self.factor_matrices])
        result = torch.einsum('abcdef,ndef->nabc', W_temp, x_reshape)
        y = result.view(-1, np.prod(self.out_features))
        return y + self.bias


class KruskalLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, ranks_tucker, rank_cp):
        super(KruskalLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks_tucker = ranks_tucker
        self.rank_cp = rank_cp

        # Initialize shapes for core and factor matrices
        self.core_matrices_shapes = [(rank, rank_cp) for rank in ranks_tucker]
        self.factor_matrices_shapes = [
                                          (out_features[i], ranks_tucker[i]) for i in range(3)
                                      ] + [
                                          (in_features[i], ranks_tucker[i + 3]) for i in range(3)
                                      ]

        # Initialize core and factor matrices
        self.core_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(shape) * 0.1) for shape in self.core_matrices_shapes
        ])
        self.factor_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(shape) * 0.1) for shape in self.factor_matrices_shapes
        ])
        self.bias = nn.Parameter(torch.zeros(np.prod(out_features)))

    def set_weights(self, core, factor_matrices, bias):
        """
        Set weights using provided core tensor, factor matrices, and bias.
        """
        with tl.context('pytorch'):
            cp_decomp = CP(rank=self.rank_cp, init='svd')
            weights, factors = cp_decomp.fit_transform(core)

        for i, matrix in enumerate(factors):
            self.core_matrices[i].data = torch.tensor(matrix, dtype=torch.float)

        for i, matrix in enumerate(factor_matrices):
            self.factor_matrices[i].data = matrix

        self.bias.data = bias

    def forward(self, x):
        """
        Forward pass of KruskalLinearLayer.
        """
        x_reshape = x.view(-1, *self.in_features)
        core = torch.einsum('ir,jr,kr,lr,mr,nr->ijklmn', *[mat for mat in self.core_matrices])
        W_temp = torch.einsum('ijklmn,ai,bj,ck,dl,em,fn->abcdef',
                              core, *[mat for mat in self.factor_matrices])
        result = torch.einsum('abcdef,ndef->nabc', W_temp, x_reshape)
        y = result.view(-1, np.prod(self.out_features))
        return y + self.bias
