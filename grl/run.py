import argparse
import logging
import pathlib
import time

import numpy as np
import jax.numpy as jnp
from jax import grad
from jax.config import config
config.update("jax_debug_nans", True)
import torch

from .environment import *
from .mdp import MDP, AbstractMDP
from .mc import mc
from .policy_eval import PolicyEval

def run_algos(spec, no_gamma, n_random_policies, grad, n_steps, max_rollout_steps):
    mdp = MDP(spec['T'], spec['R'], spec['gamma'])
    amdp = AbstractMDP(mdp, spec['phi'], p0=spec['p0'])

    # Policy Eval
    logging.info('\n===== Policy Eval =====')
    policies = spec['Pi_phi']
    if n_random_policies > 0:
        policies = amdp.generate_random_policies(n_random_policies)

    discrepancy_ids = []
    for i, pi in enumerate(policies):
        logging.info(f'\nid: {i}')
        logging.info(f'\npi: {pi}')
        pe = PolicyEval(amdp, pi)
        mdp_vals, amdp_vals, td_vals = pe.run(no_gamma, pi)
        logging.info(f'\nmdp: {mdp_vals}')
        logging.info(f'mc*: {amdp_vals}')
        logging.info(f'td: {td_vals}')
        logging.info('\n-----------')

        if not np.allclose(amdp_vals, td_vals):
            discrepancy_ids.append(i)

            if grad:
                do_grad(pe, no_gamma, pi)

    logging.info('\nTD-MC* Discrepancy ids:')
    if len(discrepancy_ids) > 0:
        logging.info(f'{discrepancy_ids}')
    else:
        logging.info('None')

    # Sampling
    # logging.info('\n\n===== Sampling =====')
    # for pi in spec['Pi_phi']:
    #     logging.info(f'\npi: {pi}')

    #     # MC*
    #     # MDP
    #     v, q, pi = mc(mdp, pi, p0=spec['p0'], alpha=0.01, epsilon=0, mc_states='all', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
    #     logging.info("\n- mc_states: all")
    #     logging.info(f'mdp: {v}')

    #     # AMDP
    #     v, q, pi = mc(amdp, pi, p0=spec['p0'], alpha=0.001, epsilon=0, mc_states='all', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
    #     logging.info(f'amdp: {v}')

    #     # MC1
    #     # ADMP
    #     logging.info("\n- mc_states: first")
    #     v, q, pi = mc(amdp, pi, p0=spec['p0'], alpha=0.01, epsilon=0, mc_states='first', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
    #     logging.info(f'amdp: {v}')

    #     logging.info('\n-----------')

def do_gradj(pe, no_gamma, pi_abs, lr=1):
    def mse_loss(pi):
        # pe.pi_abs = pi_abs
        # print('peeeeeee', pe.pi_abs)
        _, amdp_vals, td_vals = pe.run(no_gamma, pi)
        # print('valsslsls', amdp_vals, td_vals)
        diff = amdp_vals - td_vals
        # print('difffff', diff)
        return (diff**2).mean()

    print('start pi', pi_abs)
    pe.verbose = False
    old_pi = pi_abs
    i = 0
    done_count = 0
    while done_count < 5:
        i += 1
        if i % 10 == 0:
            print('iteration', i)

        pi_grad = grad(mse_loss)(pi_abs)
        # print('pi_graddd', pi_grad)

        old_pi = pi_abs
        pi_abs -= lr * pi_grad

        if np.allclose(old_pi, pi_abs):
            done_count += 1
        else:
            done_count = 0

    print('final pi', pi_abs)

    pe.verbose = True
    mdp_vals, amdp_vals, td_vals = pe.run(no_gamma, pi_abs)
    print('final vals')
    print(f'\nmdp: {mdp_vals}')
    print(f'mc*: {amdp_vals}')
    print(f'td: {td_vals}')


def do_grad(pe, no_gamma, pi_abs):
    def mse_loss(vals1, vals2):
        diff = torch.tensor(vals1 - vals2, requires_grad=True)
        return (diff**2).mean()

    pe.verbose = False
    pi_abs = torch.tensor(pi_abs)
    print('pi_abs original', pi_abs)
    optimizer = torch.optim.SGD([pi_abs], lr=0.01, momentum=0.9)

    for i in range(5):
        _, amdp_vals, td_vals = pe.run(no_gamma, pi_abs)
        loss = mse_loss(amdp_vals, td_vals)
        print('lossssss', loss)

        optimizer.zero_grad()
        loss.backward()
        print('graddddd', pi_abs.grad)
        optimizer.step()

    print('final pi', pi_abs)

if __name__ == '__main__':
    # Usage: python -m grl.run --spec example_11 --log

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='example_11', type=str)
    parser.add_argument('--no_gamma', action='store_true',
                        help='do not discount the weighted average value expectation in policy eval')
    parser.add_argument('--n_random_policies', default=0, type=int,
                        help='number of random policies to run--if not set, then use specified Pi_phi instead')
    parser.add_argument('--grad', action='store_true')
    parser.add_argument('--n_steps', default=20000, type=int,
                        help='number of rollouts to run')
    parser.add_argument('--max_rollout_steps', default=None, type=int,
                        help='max steps for mc rollouts')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('-f', '--fool-ipython')# hack to allow running in ipython notebooks
    parser.add_argument('--seed', default=None, type=int)

    args = parser.parse_args()
    del args.fool_ipython

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        rootLogger.addHandler(logging.FileHandler(f'logs/{args.spec}-{time.time()}.log'))

    if args.seed:
        np.random.seed(args.seed)

    # Get POMDP definition
    spec = environment.load_spec(args.spec)
    logging.info(f'spec:\n {args.spec}\n')
    logging.info(f'T:\n {spec["T"]}')
    logging.info(f'R:\n {spec["R"]}')
    logging.info(f'gamma: {spec["gamma"]}')
    logging.info(f'p0:\n {spec["p0"]}')
    logging.info(f'phi:\n {spec["phi"]}')
    logging.info(f'Pi_phi:\n {spec["Pi_phi"]}')
    logging.info(f'n_steps:\n {args.n_steps}')
    logging.info(f'max_rollout_steps:\n {args.max_rollout_steps}')


    # Run algos
    run_algos(spec, args.no_gamma, args.n_random_policies, args.grad, args.n_steps, args.max_rollout_steps)
