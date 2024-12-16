import math
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ops

"""
Shape parameters: 
    B -> batch size, N -> number of database items
    M -> number of query items, F -> feature channels of database/query items
    E -> feature channels of embedding
    O -> number of potential neighbors per query item

    Potential neighbors:
    "In practice, for each query item, we confine the set of potential neighbors to a subset of all items, \eg all image patches in a certain local region. 
This allows our $\nnn$ block to scale linearly in the number of items instead of quadratically."
"""


def compute_distances(xe, ye):
    """
    Computes pairwise distances for all pairs of query items and
    potential neighbors.

    :param xe: Tensor of potential neighbor embeddings, shape (B*G, E, O)
    :param ye: Tensor of query item embeddings, shape (B*G, E, 1)
    :param train: Whether to use tensor comprehensions for inference (forward only)

    :return: A tensor of distances, shape (B*G, 1, O)
    """

    # xe -> B*G x E x O
    # ye -> B*G x E x 1
    b_g, e, o = xe.shape

    # Calculate the full distance matrix in training mode
    # D_full -> B*G x 1 x O
    D_full = ops.euclidean_distance(ye.transpose(1, 2), xe)
    D = D_full.view(b_g, 1, o)

    return -D


def aggregate_output(W, xe):
    """
    Calculates weighted averages for k nearest neighbor volumes.

    :param W: B*G x 1 x O x K matrix of weights
    :param xe: B*G x E x O tensor of potential neighbor embeddings
    :param train: Whether to use tensor comprehensions for inference (forward only)

    :return: B*G x E x K tensor of the k nearest neighbor volumes for each query item
    """

    # W -> b_g 1 o k
    # xe -> b_g e o
    b_g, e, o = xe.shape
    _, _, _, k = W.shape

    # Weighted aggregation
    W = W.squeeze(1)  # (B*G, O, K)
    z = torch.matmul(xe, W)  # (B*G, E, K)

    return z


def log1mexp(x, expm1_guard=1e-7):
    # See https://cran.r-project.org/package=Rmpfr/.../log1mexp-note.pdf
    t = x < math.log(0.5)
    y = torch.zeros_like(x)
    y[t] = torch.log1p(-x[t].exp())
    not_t = torch.logical_not(t)

    # for x close to 0 we need expm1 for numerically stable computation
    # we furtmermore modify the backward pass to avoid instable gradients,
    # ie situations where the incoming output gradient is close to 0 and the gradient of expm1 is very large
    expxm1 = torch.expm1(x[not_t])
    log1mexp_fw = (-expxm1).log()
    log1mexp_bw = (-expxm1 + expm1_guard).log()  # limits magnitude of gradient

    y[not_t] = log1mexp_fw.detach() + (log1mexp_bw - log1mexp_bw.detach())
    return y


class NeuralNearestNeighbors(nn.Module):
    """
    Computes neural nearest neighbor volumes based on pairwise distances
    """

    def __init__(self, k, temp_opt={}):
        r"""
        :param k: Number of neighbor volumes to compute
        :param temp_opt: temperature options:
            external_temp: Whether temperature is given as external input
                rather than fixed parameter
            temp_bias: A fixed bias to add to the log temperature
            distance_bn: Whether to put distances through a batchnorm layer
        """
        super(NeuralNearestNeighbors, self).__init__()
        self.external_temp = temp_opt.get("external_temp")
        self.log_temp_bias = log(temp_opt.get("temp_bias", 1))
        distance_bn = temp_opt.get("distance_bn")

        if not self.external_temp:
            self.log_temp = nn.Parameter(torch.FloatTensor(1).fill_(0.0))
        if distance_bn:
            self.bn = nn.BatchNorm1d(1)
        else:
            self.bn = None

        self.k = k

    def forward(self, D, log_temp=None):
        b_g, _, o = D.shape
        if self.bn is not None:
            D = self.bn(D.view(b_g, 1, o)).view(D.shape)

        if self.external_temp:
            log_temp = log_temp.view(b_g, 1, 1)
        else:
            log_temp = self.log_temp.view(1, 1, 1)

        log_temp = log_temp + self.log_temp_bias

        temperature = log_temp.exp()
        if self.training:
            M = D.data > -float("Inf")
            if len(temperature) > 1:
                D[M] /= temperature.expand_as(D)[M]
            else:
                D[M] = D[M] / temperature[0, 0, 0]
        else:
            D /= temperature

        logits = D.view(b_g, -1)

        samples_arr = []

        for r in range(self.k):
            weights = F.log_softmax(logits, dim=1)
            weights_exp = weights.exp()

            samples_arr.append(weights_exp.view(b_g, o))
            logits = logits + log1mexp(weights.view(*logits.shape))

        W = torch.stack(samples_arr, dim=2).unsqueeze(1)  # (B*G, 1, O, K)

        return W


"""
Shape parameters: 
    B -> batch size, N -> number of database items
    M -> number of query items, F -> feature channels of database/query items
    E -> feature channels of embedding
    O -> number of potential neighbors per query item
"""


class N3AggregationBase(nn.Module):
    """
    Domain agnostic base class for computing neural nearest neighbors
    """

    def __init__(self, k, temp_opt={}):
        """
        :param k: Number of neighbor volumes to compute
        :param temp_opt: options for handling temperatures, see `NeuralNearestNeighbors`
        """
        super(N3AggregationBase, self).__init__()
        self.k = k
        self.nnn = NeuralNearestNeighbors(k, temp_opt=temp_opt)
        self.learnable_temp = nn.Parameter(torch.zeros(k).float())
        self.log_temp_bias = nn.Parameter(torch.zeros(k).float())

    def forward(self, xe, ye):
        """
        :param xe: Embedding of potential neighbors, shape (B*G, E, O)
        :param ye: Embedding of query items, shape (B*G, E, 1)
        :param log_temp: Optional log temperature
        :return: Aggregated features, shape (B*G, E, k)
        """
        b_g, e, o = xe.shape
        _, _, _ = ye.shape

        # Compute distance
        # shape (B*G, 1, O)
        D = compute_distances(xe, ye)
        assert (b_g, 1, o) == D.shape

        # # Compute aggregation weights
        # tic = perf_counter()
        # W = self.nnn(D, log_temp=log_temp)  # shape (B*G, 1, O, K)
        # assert (b_g, 1, o, k) == W.shape
        # print_log(f"sample nn weights time: {perf_counter()-tic}", logger="soft_knn detail")

        # # Aggregate output
        # # shape (B*G, E, K)
        # tic = perf_counter()
        # z = aggregate_output(W, xe)
        # assert (b_g, e, k) == z.shape
        # print_log(f"aggregate_output time: {perf_counter()-tic}", logger="soft_knn detail")

        # shape (1, k, 1)
        temperature = (
            (self.learnable_temp.exp() + self.log_temp_bias).unsqueeze(0).unsqueeze(2)
        )

        # Compute softmax weights for each temperature
        D = D.unsqueeze(1)  # shape (B*G, 1, 1, O)
        W = torch.softmax(D / temperature, dim=-1)  # shape (B*G, 1, k, O)

        # Aggregate output
        xe = xe.unsqueeze(1).expand(-1, self.k, -1, -1)  # shape (B*G, k, E, O)
        # shape (B*G, E, k, 1)
        z = torch.einsum("biko,bkeo->beki", W, xe).squeeze()

        del D, W, xe
        return z
