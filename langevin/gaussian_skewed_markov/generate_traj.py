import numpy as np
import pickle
import argparse
from tqdm import tqdm

import numpy as np

def metropolis_hastings_sampling(prob_dist, n_samples, initial_position):
    """
    Perform Metropolis-Hastings sampling from a 2D probability distribution with periodic boundary conditions.

    Parameters:
    - prob_dist: 2D numpy array, the probability distribution over a grid.
    - n_samples: int, the number of samples to draw.
    - initial_position: tuple of ints, optional initial position in the grid.
    - seed: int, optional random seed for reproducibility.

    Returns:
    - samples: List of tuples, the (i, j) grid indices of the sampled points.
    """
    assert prob_dist.sum() == 1, "Probability distribution is not normalized!"
    
    n_rows, n_cols = prob_dist.shape
    current_position = initial_position

    samples = np.zeros((n_samples,2))
    for istep in range(n_samples):
        # Propose a new position by moving one step in a random direction
        i, j = current_position
        direction = np.random.choice(['up', 'down', 'left', 'right'], p=[0.1,0.1,0.4,0.4])
        if direction == 'left':
            new_position = [(i - 1) % n_rows, j]
        elif direction == 'right':
            new_position = [(i + 1) % n_rows, j]
        elif direction == 'down':
            new_position = [i, (j - 1) % n_cols]
        elif direction == 'up':
            new_position = [i, (j + 1) % n_cols]

        # Calculate the acceptance ratio
        current_prob = prob_dist[i, j]
        new_prob = prob_dist[new_position[0],new_position[1]]
        acceptance_ratio = new_prob / current_prob

        # Accept the new position with the acceptance ratio
        if np.random.rand() < acceptance_ratio:
            current_position = new_position
        
        # Record the current position
        samples[istep] = current_position

    return np.array(samples)

def target_prob(bins=100):
    cov = np.array([[1,0.6],[0.6,1]])
    inv_cov = np.linalg.inv(cov)
    xdata, ydata = np.meshgrid(np.linspace(-2.5,2.5,bins), np.linspace(-2.5,2.5,bins), copy=True, sparse=False, indexing='xy')
    prod = xdata * xdata * inv_cov[0,0] + 2 * xdata * ydata * inv_cov[0,1] + ydata * ydata * inv_cov[1,1]
    return 1/np.sqrt(2*np.pi) * np.exp(-0.5*prod)


def main():

    # read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_steps", "--n_steps", dest="n_steps",
                        default=1000000, type=int, help="Number of steps")
    parser.add_argument("-temperature", "--temperature", dest="temperature",
                        default=1, type=float, help="Temperature")
    parser.add_argument("-seed", "--seed", dest="seed",
                        default=1998, type=int, help="Random seed")
    parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                        default=1, type=int, help="Sampling stride")
    args = parser.parse_args()

    # construct empirical probability distribution over which MC will be carried out
    bins = 50
    prob = target_prob(bins=bins)

    # perform MC with PBC over the grid
    np.random.seed(args.seed)
    n = prob.shape[0]
    assert bins == n, "Error: grid is not created with desired number of bins "
    prob_flatten = prob.flatten()
    index_flattened = np.random.choice(np.arange(prob.shape[0]*prob.shape[1]), p=prob_flatten)
    initial_position = [(index_flattened +1 ) // n, (index_flattened +1 ) % n - 1]
    traj = metropolis_hastings_sampling(prob, args.n_steps, initial_position)

    # save trajectory
    with open(f"./pickles_traj/seed{args.seed}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}.p","wb") as f:
        pickle.dump(np.array(traj), f)


if __name__ == '__main__':
    main()
