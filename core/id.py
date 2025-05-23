import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def estimate_id(X: np.ndarray, method: str = "twonn", local: bool = False, **kwargs):
    """
    Estimate intrinsic dimension using specified method.

    Args:
    - X (np.ndarray): (n_samples, n_features) embedding matrix.
    - method (str): 'twonn', 'mle', 'gride', 'pca'.
    - local (bool): Whether to return point-wise local ID.
    - **kwargs: Method-specific arguments.

    Returns:
    - float or np.ndarray: Global ID or local ID array.
    """
    if method == "pca":
        if local:
            raise ValueError("PCA does not support local ID estimation.")
        return estimate_id_pca(X, **kwargs)

    # Map user-friendly method names to ratio estimator parameters
    if method == "twonn":
        return estimate_id_ratio(X, n_ref=2, n_divs=[1], average=not local)
    elif method == "mle":
        k = kwargs.get("n_neighbors", 10)
        return estimate_id_ratio(X, n_ref=k, n_divs=list(range(1, k)), average=not local)
    elif method == "gride":
        if local:
            raise ValueError("GRIDE currently does not support local estimation.")
        n1 = kwargs.get("n1", 2)
        n2 = kwargs.get("n2", 20)
        return estimate_id_ratio(X, n_ref=n2, n_divs=[n1], average=True)
    else:
        raise ValueError(f"Unsupported ID estimation method: {method}")


# === Ratio-based ID Estimator (TwoNN / MLE / GRIDE) ===
def estimate_id_ratio(X: np.ndarray, n_ref: int, n_divs: list[int], average: bool = True) -> np.ndarray:
    """
    Unified neighbor-ratio-based ID estimator.

    Args:
    - X (np.ndarray): Data matrix.
    - n_ref (int): Index of reference neighbor.
    - n_divs (list[int]): Indices of dividing neighbors.
    - average (bool): Return global average if True, else return local values.

    Returns:
    - float or np.ndarray: Estimated global or local ID.
    """
    max_k = max(n_ref, max(n_divs))
    nbrs = NearestNeighbors(n_neighbors=max_k + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)

    rk = dists[:, n_ref].reshape(-1, 1)
    rj = dists[:, n_divs]
    logs = np.log(rk / (rj + 1e-8))
    id_values = len(n_divs) / (np.sum(logs, axis=1) + 1e-8)

    return float(np.mean(id_values)) if average else id_values


# === PCA-based ID Estimation ===
def estimate_id_pca(X: np.ndarray, energy_threshold: float = 0.95) -> int:
    """
    Estimate ID using number of principal components to reach given variance threshold.

    Args:
    - X (np.ndarray): Data matrix.
    - energy_threshold (float): Cumulative variance ratio threshold.

    Returns:
    - int: Number of components required to reach threshold.
    """
    pca = PCA().fit(X)
    explained = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(explained, energy_threshold)) + 1


if __name__ == "__main__":
    Z = np.random.randn(200, 10)
    W = np.random.randn(10, 768)
    X = Z @ W
    print("=== Global ID Estimates ===")
    for method in ["twonn", "mle", "pca", "gride"]:
        est = estimate_id(X, method=method)
        print(f"{method.upper()}:", est)

    print("\n=== Local ID (TwoNN, first 5 samples) ===")
    local_ids = estimate_id(X, method="twonn", local=True)
    print(local_ids[:5])
