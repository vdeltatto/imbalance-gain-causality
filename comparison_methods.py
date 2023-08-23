import sys
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

from utilities import compute_info_imbalance, compute_pearson_correlation, construct_time_delay_embedding


################# Convergent Cross Mapping ###################

def compute_cross_mapping(X_time_delay, Y_time_delay):
    """
    Computes the cross mapping coefficients X|Y and Y|X

    Args:
        M_X (np.ndarray): array of shape (N, E), containing N time-delay
                          embeddings of length E of the first variable X

        M_Y (np.ndarray): array of shape (N, E), containing N time-delay
                          embeddings of length E of the second variable Y

    Returns:
        CCM_coefficient_X_given_Y (float): CCM coefficient X|Y, quantifying the link X->Y
        CCM_coefficient_Y_given_X (float): CCM coefficient Y|X, quantifying the link Y->X

    """
    if X_time_delay.shape[1] != Y_time_delay.shape[1]:
        raise ValueError("Embedding dimensions must be the same for the two systems!")
    E = X_time_delay.shape[1]  # embedding dimension
    L = X_time_delay.shape[0]  # library length
    X = X_time_delay[:, 0]
    Y = Y_time_delay[:, 0]

    nbrs_x = NearestNeighbors(n_neighbors=E+2).fit(X_time_delay)
    nbrs_y = NearestNeighbors(n_neighbors=E+2).fit(Y_time_delay)

    distances_x, indices_x = nbrs_x.kneighbors(X_time_delay)
    distances_y, indices_y = nbrs_y.kneighbors(Y_time_delay)

    u_j_x = np.zeros((L, E+1))
    u_j_y = np.zeros((L, E+1))

    for neighbor in range(E+1):
        u_j_x[:, neighbor] = np.exp(-distances_x[:, neighbor+1] /
                                    distances_x[:, 1])
        u_j_y[:, neighbor] = np.exp(-distances_y[:, neighbor+1] /
                                    distances_y[:, 1])

    omega_j_x = np.zeros((L, E+1))
    omega_j_y = np.zeros((L, E+1))
    for neighbor in range(E+1):
        omega_j_x[:,neighbor] = u_j_x[:, neighbor] / np.sum(u_j_x, axis=-1)
        omega_j_y[:,neighbor] = u_j_y[:, neighbor] / np.sum(u_j_y, axis=-1)

    X_reconstructed = np.zeros(L)
    Y_reconstructed = np.zeros(L)
    for i_time in range(L):
        for neighbor in range(E+1):
            X_reconstructed[i_time] += omega_j_y[i_time, neighbor] * \
                                       X[indices_y[i_time, neighbor+1]]
            Y_reconstructed[i_time] += omega_j_x[i_time, neighbor] * \
                                       Y[indices_x[i_time, neighbor+1]]
    CCM_coefficient_X_given_Y = compute_pearson_correlation(X, X_reconstructed)
    CCM_coefficient_Y_given_X = compute_pearson_correlation(Y, Y_reconstructed)

    return CCM_coefficient_X_given_Y, CCM_coefficient_Y_given_X


################# Extended Granger Causality ###################

def compute_extended_granger_index(X_time_delay, Y_time_delay,
                                   n_neighborhoods=200,
                                   ks=[100],
                                   seed=1998):
    """
    Computes the Extended GC index (EGCI)

    Args:
        X_time_delay (np.ndarray): array of shape (N, E1+1), containing N time-delay embeddings of
                                   length E1+1 of the first variable X (the first E1 variables are used
                                   to predict the last one)

        Y_time_delay (np.ndarray): array of shape (N, E2+1), containing N time-delay embeddings of
                                   length E2+1 of the first variable X (the first E2 variables are used
                                   to predict the last one)

        n_neighborhoods (int, default=200): number of local regression

        ks (list(int), default=[100]): list of neighborhood sizes considered in len(ks)
                                       calculations of the EGCI, in terms of the number of
                                       nearest neighbors considered in each local regression

        seed (int, default=1998): seed for the numpy random generator

    Returns:
        EGCI_X_to_Y (np.ndarray): Extended GC index X->Y for the values of k in 'ks'

        EGCI_Y_to_X (np.ndarray): Extended GC index Y->X for the values of k in 'ks'

    """
    np.random.seed(seed)
    N = X_time_delay.shape[0]
    embedding_dim = X_time_delay.shape[1] - 1
    Z = np.append(X_time_delay[:,1:], Y_time_delay[:,1:], axis=-1)

    nbrs = NearestNeighbors(n_neighbors=N).fit(Z)
    distances, indices = nbrs.kneighbors(Z)  # 'distances[i,j]' is the distance of the j-th nn from i
                                             # 'indices[i,j]' is the index of the j-th nn from i
    EGCI_X_to_Y = np.zeros(len(ks))
    EGCI_Y_to_X = np.zeros(len(ks))
    for i_k in range(len(ks)):
        k = ks[i_k]

        var_residual_Y_given_XY = []
        var_residual_Y_given_Y = []
        var_residual_X_given_XY = []
        var_residual_X_given_X = []

        i_neighborhood = 0
        random_points_indices = np.random.choice(np.arange(N), size=n_neighborhoods, replace=False)
        for i_neighborhood in range(n_neighborhoods):
            random_point_index = random_points_indices[i_neighborhood]
            neighbor_indices = indices[random_point_index, :k]

            Xs = X_time_delay[neighbor_indices, :]
            Ys = Y_time_delay[neighbor_indices, :]

            # (1) X->Y test
            # VAR to predict y using both past of x and y
            lr = LinearRegression()
            lr.fit(np.append(Xs[:, 1:], Ys[:, 1:], axis=-1), Ys[:, 0])
            var_residual_Y_given_XY.append(
                    np.mean((Ys[:, 0] -
                             lr.predict(np.append(Xs[:, 1:], Ys[:, 1:], axis=-1))
                             )**2
                            )
                    )

            # single autoregression to predict y using only the past of y
            lr = LinearRegression()
            lr.fit(Ys[:, 1:], Ys[:, 0])
            var_residual_Y_given_Y.append(
                        np.mean((Ys[:, 0] - lr.predict(Ys[:, 1:]))**2)
                    )

            # (2) Y->X test
            # VAR to predict x using both past of x and y
            lr = LinearRegression()
            lr.fit(np.append(Xs[:, 1:], Ys[:, 1:], axis=-1), Xs[:, 0])
            var_residual_X_given_XY.append(
                    np.mean((Xs[:, 0] -
                             lr.predict(np.append(Xs[:, 1:], Ys[:, 1:], axis=-1))
                             )**2
                            )
                    )

            # single autoregression to predict x using only the past of x
            lr = LinearRegression()
            lr.fit(Xs[:, 1:], Xs[:, 0])
            var_residual_X_given_X.append(
                    np.mean((Xs[:, 0] - lr.predict(Xs[:, 1:]))**2)
                    )

        EGCI_X_to_Y[i_k] = 1. - np.mean(np.array(var_residual_Y_given_XY) /
                                        np.array(var_residual_Y_given_Y))
        EGCI_Y_to_X[i_k] = 1. - np.mean(np.array(var_residual_X_given_XY) /
                                        np.array(var_residual_X_given_X))

    return np.array([EGCI_X_to_Y, EGCI_Y_to_X])


################# Measure L ###################

def compute_measure_L(X_time_delay, Y_time_delay, k=5):
    """
    Computes the the causality measures L(Y|X) and L(X|Y), using the equivalence
    with the Information Imbalance measures

    Args:
        X_time_delay (np.ndarray): array of shape (N, E1), containing N time-delay
                                   embeddings of length E1 of the first variable X

        Y_time_delay (np.ndarray): array of shape (N, E2), containing N time-delay
                                   embeddings of length E2 of the second variable Y

        k (int, default=5): number of nearest neighbors considered

    Returns:
        measure_L_X_given_Y (float): value of L(X|Y)

        measure_L_Y_given_X (float): value of L(Y|X)
    """
    if X_time_delay.shape[0] != Y_time_delay.shape[0]:
        raise ValueError("Number of samples must be the same in the two systems!")
    N = X_time_delay.shape[0]

    imbalance_X_to_Y, imbalance_Y_to_X = compute_info_imbalance(X_time_delay,
                                                                Y_time_delay,
                                                                k_A=k,
                                                                k_B=k,
                                                                metric="euclidean")

    measure_L_X_given_Y = N / (N-k-1) * (1. - imbalance_Y_to_X)
    measure_L_Y_given_X = N / (N-k-1) * (1. - imbalance_X_to_Y)

    return measure_L_X_given_Y, measure_L_Y_given_X

