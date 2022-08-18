from .mdp import MDP, AbstractMDP

import numpy as onp
import jax.numpy as np
from jax.config import config
from tqdm import tqdm

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

def memory_cross_product(amdp, T_mem):
    """
    Returns AMDP resulting from cross product of the underlying MDP with given memory function

    :param amdp:  AMDP
    :param T_mem: memory transition function
    """
    T = amdp.T
    R = amdp.R
    phi = amdp.phi
    n_states = T.shape[-1]
    n_states_m = T_mem.shape[-1]
    n_states_x = n_states * n_states_m # cross (x) product MDP
    T_x = np.zeros((T.shape[0], n_states_x, n_states_x))
    R_x = np.zeros((R.shape[0], n_states_x, n_states_x))
    phi_x = np.zeros((n_states_x, phi.shape[-1] * n_states_m))

    # Rewards only depend on MDP (not memory function)
    R_x = R.repeat(n_states_m, axis=1).repeat(n_states_m, axis=2)

    # T_mem_phi is like T_pi
    # It is SxAxMxM
    T_mem_phi = np.tensordot(phi, T_mem.swapaxes(0, 1), axes=1)

    # Outer product that compacts the two i dimensions and the two l dimensions
    # (SxAxMxM, AxSxS -> AxSMxSM), where SM=x
    T_x = np.einsum('iljk,lim->lijmk', T_mem_phi, T).reshape(T.shape[0], n_states_x, n_states_x)

    # The new obs_x are the original obs times memory states
    # E.g. obs={r,b} and mem={0,1} -> obs_x={r0,r1,b0,b1}
    phi_x = np.kron(phi, np.eye(n_states_m))

    # Assuming memory starts with all 0s
    p0_x = np.zeros(n_states_x)
    # p0_x[::n_states_m] = amdp.p0
    p0_x = p0_x.at[::n_states_m].set(amdp.p0)
    mdp_x = MDP(T_x, R_x, p0_x, amdp.gamma)
    return AbstractMDP(mdp_x, phi_x)

def generate_1bit_mem_fns(n_obs, n_actions):
    """
    Generates all possible deterministic 1 bit memory functions with given number of obs and actions.
    There are M^(MZA) memory functions.
    1 bit means M=2.

    Example:
    For 2 obs (r, b) and 2 actions (up, down), binary_mp=10011000 looks like:

    m o_a    mp
    -----------
    0 r_up   1
    0 r_down 0
    0 b_up   0
    0 b_down 1
    1 r_up   1
    1 r_down 0
    1 b_up   0
    1 b_down 0

    """
    # TODO: add tests
    n_mem_states = 2
    fns = []

    MZA = n_mem_states * n_obs * n_actions
    for i in tqdm(range(n_mem_states**(MZA))):
        binary_mp = format(i, 'b').zfill(MZA)
        T_mem = onp.zeros((n_actions, n_obs, n_mem_states, n_mem_states))
        for m in range(n_mem_states):
            for ob in range(n_obs):
                for a in range(n_actions):
                    mp = int(binary_mp[m * n_obs * n_actions + ob * n_actions + a])
                    T_mem[a, ob, m, mp] = 1

        fns.append(T_mem)

    return fns
