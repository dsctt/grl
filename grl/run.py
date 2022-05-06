import argparse
import logging
import pathlib
import time

import examples_lib
from mdp import MDP, AbstractMDP
from mc import mc


def run_algos(T, R, gamma, p0, phi, Pi_phi):
    mdp = MDP(T, R, gamma)
    
    amdp = None
    if phi is not None:
        amdp = AbstractMDP(mdp, phi)

    # MDP
    logging.info('\n===== MDP =====')
    for pi in Pi_phi:
        v, q, pi = mc(mdp, pi, p0=p0, alpha=0.01, epsilon=0, mc_states='all', n_steps=20000)
        logging.info("\nmc_states: all")
        logging.info(f'v: {v}')
        logging.info(f'pi: {pi}')

    # AMDP
    logging.info('\n===== AMDP =====')
    if amdp:
        for pi in Pi_phi:
            v, q, pi = mc(amdp, pi, p0=p0, alpha=0.001, epsilon=0, mc_states='all', n_steps=20000)
            logging.info("\nmc_states: all")
            logging.info(f'v: {v}')
            logging.info(f'pi: {pi}')

        for pi in Pi_phi:
            v, q, pi = mc(amdp, pi, p0=p0, alpha=0.01, epsilon=0, mc_states='first', n_steps=20000)
            logging.info("\nmc_states: first")
            logging.info(f'v: {v}')
            logging.info(f'pi: {pi}')

if __name__ == '__main__':
    # Usage: python run.py --spec example_11 --log

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', default='example_11', type=str)
    parser.add_argument('--log', action='store_true')

    args = parser.parse_args()
    if args.spec not in dir(examples_lib):
        raise NotImplementedError(f'{args.spec} not defined in examples_lib')

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    if args.log:
        pathlib.Path('logs').mkdir(exist_ok=True)
        rootLogger = logging.getLogger()
        rootLogger.addHandler(logging.FileHandler(f'logs/{args.spec}-{time.time()}.log'))

    # Get (PO)MDP definition
    T, R, gamma, p0, phi, Pi_phi = examples_lib.load(args.spec)
    logging.info(f'T:\n {T}')
    logging.info(f'R:\n {R}')
    logging.info(f'gamma: {gamma}')
    logging.info(f'p0:\n {p0}')
    logging.info(f'phi:\n {phi}')
    logging.info(f'Pi_phi:\n {Pi_phi}')

    # Run algos
    run_algos(T, R, gamma, p0, phi, Pi_phi)