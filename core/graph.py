import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

def build_knn_graph(X: np.ndarray, k: int = 10, metric: str = 'cosine', symmetric: bool = True) -> sp.csr_matrix:
    """
    Construct a k-NN graph from input embeddings.

    Args:
    - X (np.ndarray): (n_samples, n_features) embedding vectors.
    - k (int): Number of nearest neighbors.
    - metric (str): Distance metric （'cosine', 'euclidean', etc.）.
    - symmetric (bool): Whether to symmetrize the adjacency matrix.

    Returns:
    - sp.csr_matrix: (n_samples, n_samples) sparse adjacency matrix
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    distances, indices = nn.kneighbors(X)

    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].reshape(-1)
    weights = distances[:, 1:].reshape(-1)

    adj = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))

    if symmetric:
        adj = 0.5 * (adj + adj.T)

    return adj


if __name__ == "__main__":
    X = np.random.randn(100, 64)
    A = build_knn_graph(X, k=5, metric="cosine", symmetric=True)

    print("Adjacency matrix shape:", A.shape)
    print("Number of nonzero edges:", A.nnz)
    print("Sample row (nonzero entries):", A[0].nonzero()[1])
    print("Sample row values:", A[0].data)

