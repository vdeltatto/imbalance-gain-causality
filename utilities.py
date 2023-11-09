import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform


def compute_rank_matrix(data, metric="euclidean"):
    """
    Computes the rank matrix of the input data.

    Args:
        data (np.ndarray or list(list(float))): dataset of shape (N,D), with N points and D features

        metric (str, default="euclidean"): name of the distance employed

    Returns:
        rank_matrix (np.ndarray): array of shape (N,N), with ij-th element corresponding to the rank
                                  order of point j with respect to i. Diagonal elements are set to
                                  np.infty; the remaining entries have integer values between 1 and N-1
    """
    pairwise_dist = squareform(pdist(data, metric=metric))
    pairwise_dist = pairwise_dist.astype('float')

    np.fill_diagonal(pairwise_dist, np.infty)
    rank_matrix = rankdata(pairwise_dist, method='average', axis=1)
    np.fill_diagonal(rank_matrix, np.infty)

    return rank_matrix


def nns_index_array(rank_matrix, k=1):
    """
    Computes the indices of the k nearest neighbors to each point.

    Args:
        rank_matrix (np.ndarray): array of shape (N,N), output of 'compute_rank_matrix'

        k (int, default=1): number of nearest neighbors considered in the Information Imbalance calculation

    Returns:
        NNs (np.ndarray): array of shape (N,k). The ij-th element is the index of the j-th nearest 
                          neighbor to point i.
    """
    N = rank_matrix.shape[0]
    NNs = np.zeros((N, k), dtype=int)
    for i in np.arange(N):
        NNs[i, :] = np.argpartition(rank_matrix[i], np.arange(k))[:k]
    return np.array(NNs)


def compute_info_imbalance(data_A, data_B, k_A=1, k_B=1, metric="euclidean"):
    """
    Computes the Information Imbalances Delta(A->B) and Delta(B->A)

    Args:
        data_A (np.ndarray): array of shape (N,D1) with N points and D1 features

        data_A (np.ndarray): array of shape (N,D2) with N points and D2 features

        k_A (int, default=1): number of nearest neighbor to compute Delta(A->B)

        k_B (int, default=1): number of nearest neighbor to compute Delta(B->A)

        metric (str, default="euclidean"): name of distance employed

    Returns:
        imb_A_to_B (float): Information Imbalance Delta(A->B)

        imb_B_to_A (float): Information Imbalance Delta(B->A)
    """
    if data_A.shape[0] != data_B.shape[0]:
        raise ValueError("Number of points must be the same in the two representations!")
    N = data_A.shape[0]
    rank_matrix_A = compute_rank_matrix(data_A, metric=metric)
    rank_matrix_B = compute_rank_matrix(data_B, metric=metric)

    # Find the nn indices in each space
    nns_A = nns_index_array(rank_matrix_A, k=k_A)
    nns_B = nns_index_array(rank_matrix_B, k=k_B)

    # Find conditional ranks in two spaces
    conditional_ranks_B = np.zeros((N, k_A))
    for i_point in range(N):
        rank_B = rank_matrix_B[i_point][nns_A[i_point]]
        conditional_ranks_B[i_point] = rank_B
    conditional_ranks_B = conditional_ranks_B.flatten()

    conditional_ranks_A = np.zeros((N, k_B))
    for i_point in range(N):
        rank_A = rank_matrix_A[i_point][nns_B[i_point]]
        conditional_ranks_A[i_point] = rank_A
    conditional_ranks_A = conditional_ranks_A.flatten()

    # The information imbalances:
    imb_A_to_B = 2/N * np.mean(conditional_ranks_B)
    imb_B_to_A = 2/N * np.mean(conditional_ranks_A)

    return imb_A_to_B, imb_B_to_A


def construct_time_delay_embedding(X, E, tau_e, sample_times=None):
    """
    Computes the time-delay embeddings of X, with embedding length E and embedding time tau_e

    Args:
        X (np.ndarray): one-dimensional array with N points

        E (int): embedding dimension

        tau_e (int): embedding time

        sample_times (np.ndarray or list(int), default=None):
            whether to construct one embedding for each point in X (if sample_times==None)
            or to use only the points in the array sample_times (if sample_times!=None)
    Returns:
        X_time_delay (np.ndarray): time-delay embedding of X, with shape (N,E)
    """

    if sample_times is None:
        N = len(X)
        start_time = E*tau_e

        X_time_delay = np.zeros((N-start_time, E))
        X_time_delay[:, 0] = X[start_time:]
        for i_dim in range(1, E):
            X_time_delay[:, i_dim] = X[start_time-i_dim*tau_e:-i_dim*tau_e]

    else:
        N = len(sample_times)
        X_time_delay = np.zeros((N, E))

        for i_sample in range(N):
            embedding_times = np.arange(sample_times[i_sample],
                                        sample_times[i_sample] - tau_e*E,
                                        -tau_e)

            X_time_delay[i_sample, :] = X[embedding_times]

    return X_time_delay


def compute_pearson_correlation(X, Y):
    """
    Computes the Pearson correlation coefficient between X and Y

    Args:
        X (np.ndarray): one-dimensional array

        Y (np.ndarray): one-dimensional array

    Returns:
        rho (float): Pearson correlation rho(X,Y)
    """
    X_standardized = (X - np.mean(X)) / np.std(X)
    Y_standardized = (Y - np.mean(Y)) / np.std(Y)
    rho = np.mean(X_standardized*Y_standardized)
    return rho
