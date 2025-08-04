from typing import Optional
import torch
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import scipy

import flipping as flip
import antiparallel as anti


def get_magnetic_signed_Laplacian(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  q: Optional[float] = 0.25,
                  return_lambda_max: bool = False, 
                  absolute_degree: bool = True,
                  gcn: bool = False,
                  net_flow: bool = False):
    """
    Replacement: Uses SiMaNet style magnetic signed Laplacian construction
    but keeps MSGNN-style input/output interface.
    """

    device = edge_index.device

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index.cpu()
    size = num_nodes

    # ===== SiMaNet style: construct A matrix
    A = coo_matrix((edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32)
    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)

    if net_flow:
        A = flip.new_adj(A)
        A_double = 0
    else:
        A_double = anti.antiparalell(A)

    if gcn:
        A += diag

    A_sym = 0.5 * (A + A.T)
    operation = diag + A_double + (scipy.sparse.csr_matrix.sign(np.abs(A) - np.abs(A.T))) * 1j

    deg = np.array(np.abs(A_sym).sum(axis=0))[0]  # out degree

    if normalization is None:
        D = coo_matrix((deg, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - A_sym.multiply(operation)
    elif normalization == 'sym':
        deg[deg == 0] = 1
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D_inv = coo_matrix((deg_inv_sqrt, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D_inv.dot(A_sym).dot(D_inv)
        L = diag - A_sym.multiply(operation)

    # ====== Convert back to torch
    L = L.tocoo()
    indices = torch.tensor(np.vstack((L.row, L.col)), device=device, dtype=torch.long)
    values = torch.tensor(L.data, device=device, dtype=torch.complex64)
    edge_index_out, edge_attr_out = coalesce(indices, values, num_nodes, num_nodes, op="add")

    if not return_lambda_max:
        return edge_index_out, edge_attr_out.real, edge_attr_out.imag
    else:
        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index_out, edge_attr_out.real, edge_attr_out.imag, lambda_max
