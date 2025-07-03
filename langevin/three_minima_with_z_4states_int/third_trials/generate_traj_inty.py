import numpy as np
import pickle
import argparse
from tqdm import tqdm
import torch
from scipy.stats import multivariate_normal

device = 'cpu'
dtype = torch.float32

# Overdamped Langevin step
def ULA_update_faster(states_old, target_energy, n_steps, dt=0.01, temperature=1, friction=1, seed=0, sampling_stride=1):
    
    '''states_old = states at last update if I use 40 random walkers it has to be made like that:
    states_old = torch.tensor([[pos_rw_1],[pos_rw_2], ..., [pos_rw_40]]),
    target_energy = target energy function,
    n_steps = # of steps to perform, will define the size of the batch,
    dt = time step'''
    
    states_old = states_old.requires_grad_()
    optimizer = torch.optim.SGD([states_old], lr=dt/friction)
    
    # noisy part of Langevin update
    torch.manual_seed(seed)
    def add_noise(grad):
        noise = np.sqrt(2*friction*temperature/dt) * torch.randn_like(grad)
        return grad + noise
    
    states_old.register_hook(add_noise) # adds something to each gradient computed
    
    states_updates = np.zeros((n_steps//sampling_stride,states_old.shape[0],states_old.shape[1]))#[]
    for t in tqdm(range(n_steps)):
        optimizer.zero_grad()
        loss = target_energy(states_old).sum()
        loss.backward()
        optimizer.step()
        if t % sampling_stride == 0:
            states_updates[t//sampling_stride] = states_old.clone().detach()
        #states_updates.append(states_old.clone().detach())
    return torch.tensor(states_updates)


# energy function
def target_energy_python(states):
    gauss1 = 1/(2*np.pi)**(3/2) * torch.exp(-0.5*(states[0]**2 + states[1]**2 + states[2]**2))              # (0,0,0)
    gauss2 = 1/(2*np.pi)**(3/2) * torch.exp(-0.5*((states[0]-5)**2 + (states[1]-5)**2 + (states[2]-5)**2))  # (5,5,5)
    gauss3 = 1/(2*np.pi)**(3/2) * torch.exp(-0.5*((states[0]-12)**2 + states[1]**2 + (states[2]-12)**2))    # (12,0,12)
    gauss4 = 1/(2*np.pi)**(3/2) * torch.exp(-0.5*((states[0]-12)**2 + states[1]**2 + (states[2]-25)**2))    # (12,0,25)
    energy = -torch.log(1/3*gauss1 + 1/3*gauss2 + 1/6*gauss3 + 1/6*gauss4)
    return energy
# vectorization using torch.vmap
target_energy = torch.vmap(target_energy_python, in_dims=0, out_dims=0)

# sample Gaussian misxture (to initialize states with right populations)
def sample_gaussian_mixture(weights, means, covariances, n_samples):
    """
    Samples data points from a Gaussian Mixture Model.
    
    Args:
        weights (list): Mixing weights for the Gaussian components. Should sum to 1.
        means (list): List of mean vectors for each component.
        covariances (list): List of covariance matrices for each component.
        n_samples (int): Number of samples to generate.
    
    Returns:
        np.ndarray: Samples from the GMM, shape (n_samples, n_features).
        np.ndarray: Labels indicating the component of each sample, shape (n_samples,).
    """
    assert len(weights) == len(means) == len(covariances), "All inputs must have the same length"
    assert np.isclose(sum(weights), 1), "Weights must sum to 1"
    
    n_components = len(weights)
    n_features = len(means[0])
    
    # Determine the component each sample belongs to
    component_choices = np.random.choice(range(n_components), size=n_samples, p=weights)
    
    samples = np.zeros((n_samples, n_features))
    
    for i, component in enumerate(component_choices):
        samples[i] = np.random.multivariate_normal(mean=means[component], cov=covariances[component])
    
    return samples


def main():

    # read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--dt", dest="dt",
                        default=0.01, type=float, help="Timestep of integrator")
    parser.add_argument("-n_steps", "--n_steps", dest="n_steps",
                        default=1000000, type=int, help="Number of steps")
    parser.add_argument("-temperature", "--temperature", dest="temperature",
                        default=1, type=float, help="Temperature")
    parser.add_argument("-friction", "--friction", dest="friction",
                        default=1, type=float, help="Friction")
    parser.add_argument("-seed", "--seed", dest="seed",
                        default=1998, type=int, help="Random seed")
    parser.add_argument("-N", "--N", dest="N",
                        default=2500, type=int, help="Number of realizations")
    parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                        default=1000, type=int, help="Sampling stride")
    parser.add_argument("-int_value", "--int_value", dest="int_value",
                        default=None, type=float, help="Interventional value")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # initialize states
    weights = [1/3, 1/3, 1/6, 1/6]  # Mixture weights
    means = [[0, 0, 0], [5, 5, 5], [12, 0, 12], [12, 0, 25]]  # Means of the components
    covariances = [np.eye(3), np.eye(3), np.eye(3),  np.eye(3)]  # Covariance matrices
    n_samples = args.N  # Number of samples to generate


    states_start = sample_gaussian_mixture(weights, means, covariances, n_samples)
    assert states_start.shape == (args.N,3), "Something wrong with initial states!\n"
    states_start = torch.tensor(states_start)

    # integrate overdamped Langevin equation (only thermalization)
    traj = ULA_update_faster(states_old=states_start, 
                             target_energy=target_energy, 
                             n_steps=100*1000,
                             dt=args.dt, 
                             temperature=args.temperature, 
                             friction=args.friction,
                             seed=args.seed,
                             sampling_stride=1000)
    thermalized_states = np.array(traj[-1])

    # intervention do(Y=args.int_value)
    states_int = +thermalized_states
    states_int[:,1] = args.int_value
    states_int = torch.tensor(states_int)
    traj_int = ULA_update_faster(states_old=states_int, 
                             target_energy=target_energy, 
                             n_steps=args.n_steps, 
                             dt=args.dt, 
                             temperature=args.temperature, 
                             friction=args.friction,
                             seed=args.seed,
                             sampling_stride=args.sampling_stride)
    traj_int = np.array(traj_int)

    # save trajectory
    with open(f"./pickles_traj_inty/intvalue{args.int_value}_seed{args.seed}_N{args.N}_dt{args.dt}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}_frict{args.friction}_temp{args.temperature}.p","wb") as f:
        pickle.dump(traj_int, f)


if __name__ == '__main__':
    main()
