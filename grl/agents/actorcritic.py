import copy

# from jax.nn import softmax
from scipy.special import softmax
import numpy as np
import optuna

from grl.utils.math import glorot_init
from grl.agents.td_lambda import TDLambdaQFunction
from grl.agents.replaymemory import ReplayMemory

class ActorCritic:
    def __init__(self,
                 n_obs: int,
                 n_actions: int,
                 gamma: float,
                 n_mem_entries: int = 0,
                 n_mem_values: int = 2,
                 learning_rate: float = 0.001,
                 trace_type: str = 'accumulating') -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_mem_values = n_mem_values
        self.n_mem_entries = n_mem_entries
        self.n_mem_states = n_mem_values**n_mem_entries
        self.policy_params = glorot_init((n_obs, n_actions), scale=0.2)
        self.cached_policy = softmax(self.policy_params, axis=-1)
        self.memory_params = glorot_init((n_obs, n_actions, self.n_mem_states, self.n_mem_states))
        self.cached_memory_fn = softmax(self.memory_params, axis=-1)
        q_fn_kwargs = {
            'n_obs': n_obs,
            'n_actions': n_actions,
            'gamma': gamma,
            'learning_rate': learning_rate,
            'trace_type': trace_type
        }
        self.q_td = TDLambdaQFunction(lambda_=0, **q_fn_kwargs)
        self.q_mc = TDLambdaQFunction(lambda_=0.99, **q_fn_kwargs)
        self.replay = ReplayMemory(capacity=1000000)
        self.reset()

    def reset(self):
        self.memory = 0
        self.prev_action = None

    def act(self, obs):
        obs = self.augment_obs(obs)
        # policy_probs = softmax(self.policy_params[obs], axis=-1)
        action = np.random.choice(self.n_actions, p=self.cached_policy[obs])

        # memory_probs = softmax(self.memory_params[obs, action, self.memory])
        next_memory = np.random.choice(self.n_mem_states,
                                       p=self.cached_memory_fn[obs, action, self.memory])

        self.prev_memory = self.memory
        self.memory = next_memory

        return action

    def store(self, experience):
        self.replay.push(experience)

    def update_critic(self, experience: dict):
        augmented_experience = self.augment_experience(experience)
        self.q_td.update(**augmented_experience)
        self.q_mc.update(**augmented_experience)

    def optimize_memory(self):
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.CmaEsSampler)
        n_params = np.prod(self.memory_params.shape)

        def suggest_param(trial: optuna.Trial, name: str):
            return trial.suggest_float(name, low=-1e2, high=1e2, log=True)

        def objective(trial: optuna.Trial):
            params = np.asarray([suggest_param(trial, str(i))
                                 for i in range(n_params)]).reshape(self.memory_params.shape)
            self.memory_params = params
            self.cached_memory_fn = softmax(self.memory_params, axis=-1)
            return self.evaluate_memory()

        study.optimize(objective, n_trials=100)
        study.best_params

    def evaluate_memory(self):
        self.q_mc.reset()
        self.q_td.reset()
        for experience in self.replay.memory:
            e = experience.copy()
            del e['_index_']
            self.update_critic(experience)
            if experience['terminal']:
                self.reset()
        return np.abs(self.q_mc.q - self.q_td.q).mean()

    def augment_obs(self, obs: int, memory: int = None) -> int:
        if memory is None:
            memory = self.memory
        # augment last dim with mem states
        ob_augmented = self.n_mem_states * obs + memory
        return ob_augmented

    def augment_experience(self, experience: dict) -> dict:
        augmented_experience = copy.deepcopy(experience)
        augmented_experience['obs'] = self.augment_obs(experience['obs'], self.prev_memory)
        augmented_experience['next_obs'] = self.augment_obs(experience['next_obs'], self.memory)
        return experience

    def augment_policy(self, n_mem_states: int):
        """
        Expand π (O x A) => OM x A to include memory states
        """
        # augment last dim with input mem states
        self.n_obs *= n_mem_states
        pi_augmented = np.expand_dims(self.policy_params, 1).repeat(n_mem_states, 1) # O x M x A
        self.policy_params = pi_augmented.reshape(self.n_obs, self.n_actions) # OM x A
        self.cached_policy = softmax(self.policy_params, axis=-1)

    def add_memory(self, n_mem_entries=1):
        mem_increase_multiplier = (self.n_mem_values**n_mem_entries)
        self.q_td.augment_with_memory(mem_increase_multiplier)
        self.q_mc.augment_with_memory(mem_increase_multiplier)
        self.augment_policy(mem_increase_multiplier)