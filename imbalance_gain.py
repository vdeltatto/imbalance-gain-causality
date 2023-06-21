import numpy as np
from joblib import Parallel, delayed
from utilities import compute_rank_matrix, nns_index_array


def compute_info_imbalance_causality(X0, Y0,
                                     rank_matrix_Ytau,
                                     alpha=1.0,
                                     k=1,
                                     metric="euclidean"):
    """
    Computes the Information Imbalance Delta( (alpha*X0, Y0) -> Ytau ).

    Args:
        X0 (np.ndarray or list(list(float))): array of shape (N,D1), defining the space
                                              of the putative driver system at time 0

        Y0 (np.ndarray or list(list(float))): array of shape (N,D2), defining the space
                                              of the putative driven system at time 0

        rank_matrix_Ytau (np.ndarray): array of shape (N,N), output of 'compute_rank_matrix'

        alpha (float, default=1.0): scaling parameter weighting the putative driver system

        k (int, default=1): number of nearest neighbors considered in the Information Imbalance calculation

        metric (str, default="euclidean"): name of the distance employed

    Returns:
        info_imbalance (float): value of the Information Imbalance
    """
    if X0.shape[0] != Y0.shape[0]:
        raise ValueError("Number of points must be the same in X(0) and Y(0)!")
    N = X0.shape[0]

    data_A = np.column_stack((alpha*X0, Y0))
    rank_matrix_A = compute_rank_matrix(data_A, metric=metric)

    # Find indices of nearest neighbors in space A ((alpha*X0, Y0))
    nns_A = nns_index_array(rank_matrix_A, k=k)

    # Find conditional ranks in space B (Ytau)
    conditional_ranks_B = np.zeros(N)
    for i_point in range(N):
        rank_B = np.mean(rank_matrix_Ytau[i_point][nns_A[i_point]])
        conditional_ranks_B[i_point] = rank_B

    # Compute the Information Imbalance:
    info_imbalance = 2/N * np.mean(conditional_ranks_B)

    return info_imbalance


def scan_alphas(cause_present, effect_present,
                rank_matrix_effect_future,
                alphas,
                k=1,
                n_jobs=1,
                metric="euclidean"):
    """
    Computes the Information Imbalance Delta( (alpha*cause_present, effect_present) -> effect_future ) parallelizing the 
    loop over different values of the scaling parameter.

    Args:
        cause_present (np.ndarray or list(list(float))): array of shape (N,D1), defining the space
                                                         of the putative driver system at time 0

        effect_present (np.ndarray or list(list(float))): array of shape (N,D2), defining the space
                                                          of the putative driven system at time 0

        rank_matrix_effect_future (np.ndarray): array of shape (N,N), output of 'compute_rank_matrix'

        alphas (np.ndarray or list(float)): scaling parameters scanned in the loop

        k (int, default=1): number of nearest neighbors considered in the Information Imbalance calculation

        n_jobs (int, default=1): the number of jobs to run in parallel

        metric (str, default="euclidean"): name of the distance employed

    Returns:
        info_imbalance (float): value of the Information Imbalance
    """

    info_imbalances = Parallel(n_jobs=n_jobs)(delayed(
        compute_info_imbalance_causality)(X0=cause_present, Y0=effect_present,
                                          rank_matrix_Ytau=rank_matrix_effect_future,
                                          alpha=alpha, k=k, metric=metric)
                                              for alpha in alphas)
    return info_imbalances


def compute_imbalance_gain(info_imbalances):
    """
    Computes the Imbalance Gain
        ( Delta(alpha=0) - min_alpha Delta(alpha) ) / Delta(alpha=0)

    Args:
        info_imbalances (np.ndarray): Delta(alpha) for the values of, output of 'scan_alphas'
                                              of the putative driver system at time 0
    Returns:
        imbalance_gain (float): value of the Imbalance Gain

        optimal_alpha_index (int): index of the scaling parameter minimizing Delta(alpha)
    """

    imbalance_gain = (info_imbalances[0] - np.min(info_imbalances)) / \
        info_imbalances[0]
    optimal_alpha_index = np.argmin(info_imbalances)

    return imbalance_gain, optimal_alpha_index
