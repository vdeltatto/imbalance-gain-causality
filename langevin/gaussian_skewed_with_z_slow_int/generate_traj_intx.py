import numpy as np
import pickle
import argparse
from tqdm import tqdm
import torch
from scipy.stats import multivariate_normal

device = 'cpu'
dtype = torch.float32

# Overdamped Langevin step
# Overdamped Langevin step
def ULA_update_faster(states_old, target_energy, n_steps, dt=0.01, temperature=1, friction=torch.tensor([1,0.1,0.1]), seed=0, sampling_stride=1):
    '''states_old = states at last update if I use 40 random walkers it has to be made like that:
    states_old = torch.tensor([[pos_rw_1],[pos_rw_2], ..., [pos_rw_40]]),
    target_energy = target energy function,
    n_steps = # of steps to perform, will define the size of the batch,
    dt = time step'''
    states_old = states_old.requires_grad_()
    optimizer = torch.optim.SGD([states_old], lr=dt)

    # noisy part of Langevin update
    torch.manual_seed(seed)
    def add_noise(grad):
        noise = np.sqrt(2*temperature/dt) * torch.randn_like(grad)
        return grad*(friction**(-1)) + noise*(friction**(-1/2))
    states_old.register_hook(add_noise) # adds something to each gradient computed

    states_updates = np.zeros((n_steps//sampling_stride,states_old.shape[0],states_old.shape[1]))#[]
    for t in tqdm(range(n_steps)):
        optimizer.zero_grad()
        loss = target_energy(states_old).sum()
        loss.backward()
        optimizer.step()
        if t % sampling_stride == 0:
            states_updates[t//sampling_stride] = states_old.clone().detach()
        
    return torch.tensor(states_updates)


# energy function
def target_energy_python(states):
    cov = np.array([[1,0.6,0.6],[0.6,1,0.6],[0.6,0.6,1]]) # covariance matrix
    inv_cov = np.linalg.inv(cov)
    prod = states[0] * states[0] * inv_cov[0,0] + states[1] * states[1] * inv_cov[1,1] + states[2] * states[2] * inv_cov[2,2] \
            + 2 * states[0] * states[1] * inv_cov[0,1] + 2 * states[0] * states[2] * inv_cov[0,2] + 2 * states[1] * states[2] * inv_cov[1,2] 
    prob = 1/((2*np.pi)**(3.0/2.0) * np.sqrt(np.linalg.det(cov))) * torch.exp(-0.5*prod)
    energy = -torch.log(prob)
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
                        default=10000, type=int, help="Number of steps")
    parser.add_argument("-temperature", "--temperature", dest="temperature",
                        default=1, type=float, help="Temperature")
    parser.add_argument("-friction_x", "--friction_x", dest="friction_x",
                        default=1, type=float, help="Friction of X")
    parser.add_argument("-friction_y", "--friction_y", dest="friction_y",
                        default=0.1, type=float, help="Friction of Y")
    parser.add_argument("-friction_z", "--friction_z", dest="friction_z",
                        default=10, type=float, help="Friction of Z")
    parser.add_argument("-seed", "--seed", dest="seed",
                        default=1998, type=int, help="Random seed")
    parser.add_argument("-N", "--N", dest="N",
                        default=2500, type=int, help="Number of realizations")
    parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                        default=1, type=int, help="Sampling stride")
    parser.add_argument("-int_value", "--int_value", dest="int_value",
                        default=None, type=float, help="Interventional value")
    args = parser.parse_args()

    np.random.seed(args.seed)
    mean = np.array([0,0,0])
    cov = np.array([[1,0.6,0.6],[0.6,1,0.6],[0.6,0.6,1]]) # covariance matrix
    states_start = np.random.multivariate_normal(
        mean=mean, cov=cov, size=(args.N,)
    )
    assert states_start.shape == (args.N,3), "Something wrong with initial states!\n"
    states_start = torch.tensor(states_start)

    # integrate overdamped Langevin equation (only thermalization)
    # integrate overdamped Langevin equation
    friction_tensor = torch.tensor([args.friction_x, args.friction_y, args.friction_z])
    traj = ULA_update_faster(states_old=states_start, 
                             target_energy=target_energy,
                             n_steps=1000, 
                             dt=args.dt, 
                             temperature=args.temperature, 
                             friction=friction_tensor,
                             seed=args.seed,
                             sampling_stride=1)
    thermalized_states = np.array(traj[-1])

    # intervention do(X=args.int_value)
    states_int = +thermalized_states
    states_int[:,0] = args.int_value
    states_int = torch.tensor(states_int)
    traj_int = ULA_update_faster(states_old=states_int, 
                             target_energy=target_energy, 
                             n_steps=args.n_steps, 
                             dt=args.dt, 
                             temperature=args.temperature, 
                             friction=friction_tensor,
                             seed=args.seed,
                             sampling_stride=args.sampling_stride)
    traj_int = np.array(traj_int)

    # save trajectory
    with (open(f"./pickles_traj_intx/intvalue{args.int_value}_seed{args.seed}_N{args.N}_dt{args.dt}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}" 
            +f"_frictx{args.friction_x}_fricty{args.friction_y}_frictz{args.friction_z}_temp{args.temperature}.p","wb")) as f:
        pickle.dump(traj_int, f)


if __name__ == '__main__':
    main()
