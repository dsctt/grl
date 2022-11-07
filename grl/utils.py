import numpy as np
import jax.numpy as jnp
from pathlib import Path
from time import time, ctime
from argparse import Namespace

from pprint import pformat
from typing import Sequence, Union, Tuple
from definitions import ROOT_DIR

RTOL = 1e-3

def pformat_vals(vals):
    """
    :param vals: dict
    """

    for k in vals.keys():
        vals[k] = np.array(vals[k])

    return pformat(vals)

def mi_results_path(args: Namespace):
    results_dir = Path(ROOT_DIR, 'results')
    results_dir.mkdir(exist_ok=True)
    if args.experiment_name is not None:
        results_dir /= args.experiment_name
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{args.spec}_{args.algo}_pi({args.policy_optim_alg})_miit({args.mi_iterations})_s({args.seed})_{ctime(time())}.npy"
    return results_path

def pe_results_path(args: Namespace):
    results_dir = Path(ROOT_DIR, 'results')
    results_dir.mkdir(exist_ok=True)
    if args.experiment_name is not None:
        results_dir /= args.experiment_name
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{args.spec}_{args.algo}_method({args.method})_grad({args.use_grad})_s({args.seed})_{ctime(time())}.npy"
    return results_path

def results_path(args: Namespace):
    results_dir = Path(ROOT_DIR, 'results')
    results_path = results_dir / f"{args.spec}_{args.algo}_s{args.seed}_{ctime(time())}.npy"
    return results_path

def glorot_init(shape: Sequence[int], scale: float = 0.5) -> jnp.ndarray:
    return np.random.normal(size=shape) * scale

def numpyify_dict(info: Union[dict, jnp.ndarray, np.ndarray, list, tuple]):
    """
    Converts all jax.numpy arrays to numpy arrays in a nested dictionary.
    """
    if isinstance(info, jnp.ndarray):
        return np.array(info)
    elif isinstance(info, dict):
        return {k: numpyify_dict(v) for k, v in info.items()}
    elif isinstance(info, list):
        return [numpyify_dict(i) for i in info]
    elif isinstance(info, tuple):
        return (numpyify_dict(i) for i in info)

    return info

def numpyify_and_save(path: Path, info: dict):
    numpy_dict = numpyify_dict(info)
    np.save(path, numpy_dict)

def load_info(results_path: Path):
    return np.load(results_path, allow_pickle=True).item()
