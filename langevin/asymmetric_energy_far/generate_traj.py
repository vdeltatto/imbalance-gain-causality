import numpy as np
import pickle
import argparse
from tqdm import tqdm
import torch

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
    gauss1 = 1/np.sqrt(2*np.pi) * torch.exp(-0.5*((states[0])**2 + (states[1])**2))
    gauss2 = 1/np.sqrt(2*np.pi) * torch.exp(-0.5*((states[0]-5)**2 + (states[1]-5)**2))
    gauss3 = 1/np.sqrt(2*np.pi) * torch.exp(-0.5*((states[0]-12)**2 + (states[1])**2)) # asymmetry along x!
    energy = -torch.log(1/3*gauss1 + 1/3*gauss2 + 1/3*gauss3)
    return energy
# vectorization using torch.vmap
target_energy = torch.vmap(target_energy_python, in_dims=0, out_dims=0)


def main():

    # read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dt", "--dt", dest="dt",
                        default=0.01, type=float, help="Timestep of integrator")
    parser.add_argument("-n_steps", "--n_steps", dest="n_steps",
                        default=10**4, type=int, help="Number of steps")
    parser.add_argument("-temperature", "--temperature", dest="temperature",
                        default=1, type=float, help="Temperature")
    parser.add_argument("-friction", "--friction", dest="friction",
                        default=1, type=float, help="Friction")
    parser.add_argument("-seed", "--seed", dest="seed",
                        default=1998, type=int, help="Random seed")
    parser.add_argument("-N", "--N", dest="N",
                        default=2500, type=int, help="Number of realizations")
    parser.add_argument("-sampling_stride", "--sampling_stride", dest="sampling_stride",
                        default=1, type=int, help="Sampling stride")
    args = parser.parse_args()

    np.random.seed(args.seed)
    #states_start = torch.tensor(np.random.normal(loc=3,scale=3,size=(2,)).reshape(1,-1))
    states_start = np.column_stack((
        np.random.uniform(-3,15,size=(args.N,)),
        np.random.uniform(-3,8,size=(args.N,))
    ))
    assert states_start.shape == (args.N,2), "Something wrong with initial states!\n"
    states_start = torch.tensor(states_start)

    # integrate overdamped Langevin equation
    traj = ULA_update_faster(states_old=states_start, 
                             target_energy=target_energy, 
                             n_steps=args.n_steps, 
                             dt=args.dt, 
                             temperature=args.temperature, 
                             friction=args.friction,
                             seed=args.seed,
                             sampling_stride=args.sampling_stride)

    # save trajectory
    traj = np.array(traj)
    with open(f"./pickles_traj/seed{args.seed}_N{args.N}_dt{args.dt}_samplingdt{args.sampling_stride}_nsteps{args.n_steps}_frict{args.friction}_temp{args.temperature}.p","wb") as f:
        pickle.dump(np.array(traj), f)


if __name__ == '__main__':
    main()
