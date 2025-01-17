import numpy as np

hparams = {
    'file_name':
        'runs_tmaze_sweep_junction_pi.txt',
    'args': [{
        'algo': 'pe',
        'spec': 'tmaze_hyperparams',
        'tmaze_corridor_length': 5,
        'tmaze_discount': 0.9,
        'tmaze_junction_up_pi': np.linspace(0, 1, num=50),
        'method': 'a',
        'use_memory': 0,
        'use_grad': 'm',
        'lr': 1,
        'seed': [2020 + i for i in range(1, 10)],
    }]
}
