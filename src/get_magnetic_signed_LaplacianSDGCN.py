from typing import Optional

import torch
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix


def get_magnetic_signed_Laplacian(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                                  normalization: Optional[str] = 'sym',
                                  dtype: Optional[int] = None,
                                  num_nodes: Optional[int] = None,
                                  q: Optional[float] = 0.25,
                                  return_lambda_max: bool = False,
                                  absolute_degree: bool = True):
    r""" Computes the magnetic signed Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **normalization** (str, optional) - The normalization scheme for the magnetic Laplacian (default: :obj:`sym`) -
            1. :obj:`None`: No normalization :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`

            2. :obj:`"sym"`: Symmetric normalization :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`

        * **dtype** (torch.dtype, optional) - The desired data type of returned tensor in case :obj:`edge_weight=None`. (default: :obj:`None`)
        * **num_nodes** (int, optional) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        * **q** (float, optional) - The value q in the paper for phase.
        * **return_lambda_max** (bool, optional) - Whether to return the maximum eigenvalue. (default: :obj:`False`)
        * **absolute_degree** (bool, optional) - Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)

    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic signed Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic signed Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic signed Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # 区分正边和负边
    pos_edges = edge_index[:, edge_weight > 0].cpu().numpy().T
    neg_edges = edge_index[:, edge_weight < 0].cpu().numpy().T
    pos_edge_weight = edge_weight[edge_weight > 0].cpu().numpy()
    neg_edge_weight = edge_weight[edge_weight < 0].cpu().numpy()
    edge_weight_all = np.concatenate([pos_edge_weight, neg_edge_weight])

    # 仿照hermitian_decomp_sparse的逻辑
    diag = coo_matrix((np.ones(num_nodes), (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes),
                      dtype=np.float32)

    if edge_weight_all is None:
        A = coo_matrix((np.ones(len(pos_edges) + len(neg_edges)),
                        (np.concatenate((pos_edges[:, 0], neg_edges[:, 0])),
                         np.concatenate((pos_edges[:, 1], neg_edges[:, 1])))),
                       shape=(num_nodes, num_nodes), dtype=np.float32)
    else:
        A = coo_matrix((edge_weight_all,
                        (np.concatenate((pos_edges[:, 0], neg_edges[:, 0])),
                         np.concatenate((pos_edges[:, 1], neg_edges[:, 1])))),
                       shape=(num_nodes, num_nodes), dtype=np.float32)

    A_sym = 0.5 * (A + A.T)

    if normalization == 'sym':
        d = np.array(np.abs(A_sym.sum(axis=0)))[0]
        """
        # 处理零值
        d[d <= 0] = 1e-10
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)
        """



        d[d == 0]= 1
        deg_inv_sqrt = np.power(d, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')]= 0
        D = coo_matrix((deg_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)




    phase_pos = coo_matrix((np.ones(len(pos_edges)), (pos_edges[:, 0], pos_edges[:, 1])), shape=(num_nodes, num_nodes),
                           dtype=np.float32)
    theta_pos = q * 1j * phase_pos
    theta_pos.data = np.exp(theta_pos.data)
    theta_pos_t = -q * 1j * phase_pos.T
    theta_pos_t.data = np.exp(theta_pos_t.data)

    phase_neg = coo_matrix((np.ones(len(neg_edges)), (neg_edges[:, 0], neg_edges[:, 1])), shape=(num_nodes, num_nodes),
                           dtype=np.float32)
    theta_neg = (np.pi + q) * 1j * phase_neg
    theta_neg.data = np.exp(theta_neg.data)
    theta_neg_t = (np.pi - q) * 1j * phase_neg.T
    theta_neg_t.data = np.exp(theta_neg_t.data)

    data = np.concatenate((theta_pos.data, theta_pos_t.data, theta_neg.data, theta_neg_t.data))
    theta_row = np.concatenate((theta_pos.row, theta_pos_t.row, theta_neg.row, theta_neg_t.row))
    theta_col = np.concatenate((theta_pos.col, theta_pos_t.col, theta_neg.col, theta_neg_t.col))
    phase = coo_matrix((data, (theta_row, theta_col)), shape=(num_nodes, num_nodes), dtype=np.complex64)
    Theta = phase

    if normalization == 'sym':
        D = diag
    else:
        d = np.sum(np.abs(A_sym), axis=0)
        D = coo_matrix((d, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)

    L = D - Theta.multiply(A_sym)

    if normalization == 'sym':
        max_eigen = 2
        L = (2.0 / max_eigen) * L - diag

    # 将稀疏矩阵转换为torch的稀疏张量
    L = L.tocoo()
    edge_index = torch.tensor(np.vstack((L.row, L.col)), dtype=torch.long)
    edge_weight = torch.tensor(L.data, dtype=torch.complex64)

    if not return_lambda_max:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L_scipy = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        lambda_max = eigsh(L_scipy, k=1, which='LM', return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index, edge_weight.real, edge_weight.imag, lambda_max