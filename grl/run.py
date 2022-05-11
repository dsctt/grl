import argparse
import logging
import pathlib
import time

import numpy as np

import environment
from mdp import MDP, AbstractMDP
from mc import mc


def run_algos(T, R, gamma, p0, phi, Pi_phi, n_steps, max_rollout_steps):
    mdp = MDP(T, R, gamma)

    amdp = None
    if phi is not None:
        amdp = AbstractMDP(mdp, phi)

    # MDP
    logging.info('\n===== MDP =====')
    for pi in Pi_phi:
        v, q, pi = mc(mdp, pi, p0=p0, alpha=0.01, epsilon=0, mc_states='all', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
        logging.info("\nmc_states: all")
        logging.info(f'v: {v}')
        logging.info(f'pi: {pi}')

    # AMDP
    logging.info('\n===== AMDP =====')
    if amdp:
        for pi in Pi_phi:
            v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
            logging.info("\nmc_states: all")
            logging.info(f'v: {v}')
            logging.info(f'pi: {pi}')

        for pi in Pi_phi:
            v, q, pi = mc(amdp, pi, p0=p0, alpha=0.01, epsilon=0, mc_states='first', n_steps=n_steps, max_rollout_steps=max_rollout_steps)
            logging.info("\nmc_states: first")
            logging.info(f'v: {v}')
            logging.info(f'pi: {pi}')

if __name__ == '__main__':
    # Usage: python run.py --spec example_11 --log

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='example_11', type=str)
    parser.add_argument('--n_steps', default=20000, type=int,
                        help='number of rollouts to run')
    parser.add_argument('--max_rollout_steps', default=None, type=int,
                        help='max steps for mc rollouts (useful for POMDPs with no terminal state)')
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

    # Get (PO)MDP definition
    T, R, gamma, p0, phi, Pi_phi = environment.load(args.spec)


    # Run algos
    run_algos(T, R, gamma, p0, phi, Pi_phi, args.n_steps, args.max_rollout_steps)
