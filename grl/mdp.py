import copy
from gmpy2 import mpz
import numpy as np
import jax.numpy as jnp

def normalize(M, axis=-1):
    M = M.astype(float)
    if M.ndim > 1:
        denoms = M.sum(axis=axis, keepdims=True)
    else:
        denoms = M.sum()
    M = np.divide(M, denoms.astype(float), out=np.zeros_like(M), where=(denoms != 0))
    return M

def is_stochastic(M):
    return jnp.allclose(M, normalize(M))

def random_sparse_mask(size, sparsity):
    n_rows, n_cols = size
    p = (1 - sparsity) # probability of 1
    q = (n_cols * p - 1) / (n_cols - 1) # get remaining probability after mandatory 1s
    if 0 < q <= 1:
        some_ones = np.random.choice([0, 1], size=(n_rows, n_cols - 1), p=[1 - q, q])
        mask = np.concatenate([np.ones((n_rows, 1)), some_ones], axis=1)
    else:
        mask = np.concatenate([np.ones((n_rows, 1)), np.zeros((n_rows, n_cols - 1))], axis=1)
    for row in mask:
        np.random.shuffle(row)
    return mask

def random_stochastic_matrix(size):
    alpha_size = size[-1]
    out_size = size[:-1] if len(size) > 1 else None
    return np.random.dirichlet(np.ones(alpha_size), out_size)

def random_reward_matrix(Rmin, Rmax, size):
    R = np.random.uniform(Rmin, Rmax, size)
    R = np.round(R, 2)
    return R

def random_observation_fn(n_states, n_obs_per_block):
    all_state_splits = [
        random_stochastic_matrix(size=(1, n_obs_per_block)) for _ in range(n_states)
    ]
    all_state_splits = jnp.stack(all_state_splits).squeeze()
    #e.g.[[p, 1-p],
    #     [q, 1-q],
    #     ...]

    obs_fn_mask = jnp.kron(jnp.eye(n_states), jnp.ones((1, n_obs_per_block)))
    #e.g.[[1, 1, 0, 0, 0, 0, ...],
    #     [0, 0, 1, 1, 0, 0, ...],
    #     ...]

    tiled_split_probs = jnp.kron(jnp.ones((1, n_states)), all_state_splits)
    #e.g.[[p, 1-p, p, 1-p, p, 1-p, ...],
    #     [q, 1-q, q, 1-q, q, 1-q, ...],
    #     ...]

    observation_fn = obs_fn_mask * tiled_split_probs
    return observation_fn

def one_hot(x, n):
    return jnp.eye(n)[x]

class MDP:
    def __init__(self, T, R, p0, gamma=0.9):
        self.n_states = len(T[0])
        self.n_obs = self.n_states
        self.n_actions = len(T)
        self.gamma = gamma
        if isinstance(T, np.ndarray):
            self.T = np.stack(T).copy().astype(float)
            self.R = np.stack(R).copy().astype(float)
        else:
            self.T = jnp.stack(T).copy().astype(float)
            self.R = jnp.stack(R).copy().astype(float)

        self.R_min = np.min(self.R)
        self.R_max = np.max(self.R)
        self.p0 = p0

    def __repr__(self):
        return repr(self.T) + '\n' + repr(self.R)

    def get_policy(self, i):
        assert i < self.n_actions**self.n_states
        if not (2 <= self.n_actions <= 62):
            raise ValueError(f'gmpy2.mpz.digits only supports integer bases in the'
                             'range [2, 62], but n_actions = {self.n_actions}')
        policy_str = mpz(str(i)).digits(mpz(str(self.n_actions))).zfill(self.n_states)
        policy = np.array([int(x) for x in reversed(policy_str)])
        return policy

    def all_policies(self):
        policies = []
        n_policies = self.n_actions**self.n_states
        for i in range(n_policies):
            pi = self.get_policy(i)
            policies.append(pi)
        return policies

    def stationary_distribution(self, pi=None, p0=None, max_steps=200):
        if p0 is None:
            state_distr = np.ones(self.n_states) / self.n_states
        else:
            state_distr = p0
        old_distr = state_distr
        for t in range(max_steps):
            state_distr = self.image(state_distr, pi)
            if np.allclose(state_distr, old_distr):
                break
            old_distr = state_distr
        return state_distr

    def step(self, s, a, gamma):
        pr_next_s = self.T[a, s, :]
        sp = np.random.choice(self.n_states, p=pr_next_s)
        r = self.R[a][s][sp]
        # Check if sp is terminal state
        sp_is_absorbing = (self.T[:, sp, sp] == 1)
        done = sp_is_absorbing.all()
        # Discounting
        # End episode with probability 1-gamma
        if np.random.uniform() < (1 - gamma):
            done = True

        return sp, r, done

    def observe(self, s):
        return s

    def image(self, pr_x, pi=None):
        T = self.T_pi(pi)
        pr_next_x = pr_x @ T
        return pr_next_x

    def T_pi(self, pi):
        if pi is None:
            T_pi = np.mean(self.T, axis=0)
        else:
            T_pi = self.T[pi, np.arange(self.n_states), :]
        return T_pi

    def get_N(self, pi):
        return self.T_pi(pi)

    def get_I(self, pi):
        pi_one_hot = one_hot(pi, self.n_actions).transpose()[:, :, None]
        N = self.get_N(pi)[None, :, :]
        I = np.divide(self.T * pi_one_hot, N, out=np.zeros_like(self.T), where=N != 0)
        return I

    @classmethod
    def generate(cls, n_states, n_actions, sparsity=0, gamma=0.9, Rmin=-1, Rmax=1):
        T = [] # List of s -> s transition matrices, one for each action
        R = [] # List of s -> s reward matrices, one for each action
        for a in range(n_actions):
            T_a = random_stochastic_matrix(size=(n_states, n_states))
            R_a = random_reward_matrix(Rmin, Rmax, (n_states, n_states))
            if sparsity > 0:
                mask = random_sparse_mask((n_states, n_states), sparsity)
                T_a = normalize(T_a * mask)
                R_a = R_a * mask
            T.append(T_a)
            R.append(R_a)
        p0 = random_stochastic_matrix(size=[n_states])
        mdp = cls(T, R, p0, gamma)
        return mdp

class BlockMDP(MDP):
    def __init__(self, base_mdp, n_obs_per_block=2, obs_fn=None):
        super().__init__(base_mdp.T, base_mdp.R, base_mdp.gamma)
        self.base_mdp = copy.deepcopy(base_mdp)
        self.n_states = base_mdp.n_states * n_obs_per_block

        if obs_fn is None:
            obs_fn = random_observation_fn(base_mdp.n_states, n_obs_per_block)
        else:
            n_obs_per_block = obs_fn.shape[1]

        obs_mask = (obs_fn > 0).astype(int)

        self.T = [] # List of x -> x transition matrices, one for each action
        self.R = [] # List of x -> x reward matrices, one for each action
        for a in range(self.n_actions):
            Ta, Ra = base_mdp.T[a], base_mdp.R[a]
            Tx_a = obs_mask.transpose() @ Ta @ obs_fn
            Rx_a = obs_mask.transpose() @ Ra @ obs_mask
            self.T.append(Tx_a)
            self.R.append(Rx_a)
        self.T = jnp.stack(self.T)
        self.R = jnp.stack(self.R)
        self.obs_fn = obs_fn

class AbstractMDP(MDP):
    def __init__(self, base_mdp, phi, pi=None, t=200):
        super().__init__(base_mdp.T, base_mdp.R, base_mdp.p0, base_mdp.gamma)
        self.base_mdp = copy.deepcopy(base_mdp)
        self.phi = phi # array: base_mdp.n_states, n_abstract_states
        self.n_obs = phi.shape[-1]

        # self.belief = self.B(pi, t=t)
        # self.T = [self.compute_Tz(self.belief,T_a)
        #             for T_a in base_mdp.T]
        # self.R = [self.compute_Rz(self.belief,Rx_a,Tx_a,Tz_a)
        #             for (Rx_a, Tx_a, Tz_a) in zip(base_mdp.R, base_mdp.T, self.T)]
        # self.T = jnp.stack(self.T)
        # self.R = jnp.stack(self.R)

    def __repr__(self):
        base_str = super().__repr__()
        return base_str + '\n' + repr(self.phi)

    def observe(self, s):
        return np.random.choice(self.n_obs, p=self.phi[s])

    # def B(self, pi, t=200):
    #     p = self.base_mdp.stationary_distribution(pi=pi, p0=self.p0, max_steps=t)
    #     return normalize(p*self.phi.transpose())
    #
    # def compute_Tz(self, belief, Tx):
    #     return belief @ Tx @ self.phi
    #
    # def compute_Rz(self, belief, Rx, Tx, Tz):
    #     return jnp.divide( (belief@(Rx*Tx)@self.phi), Tz,
    #                      out=jnp.zeros_like(Tz), where=(Tz!=0) )

    def is_abstract_policy(self, pi):
        agg_states = (self.phi.sum(axis=0) > 1)
        for idx, is_agg in enumerate(agg_states):
            agg_cluster = (one_hot(idx, self.n_obs) @ self.phi.transpose()).astype(bool)
            if not np.all(pi[agg_cluster] == pi[agg_cluster][0]):
                return False
        return True

    def piecewise_constant_policies(self):
        return [pi for pi in self.base_mdp.all_policies() if self.is_abstract_policy(pi)]

    def get_abstract_policy(self, pi):
        assert self.is_abstract_policy(pi)
        mask = self.phi.transpose()
        obs_fn = normalize(mask)
        return (pi @ obs_fn.transpose()).astype(int)

    def get_ground_policy(self, pi):
        return self.phi @ pi

    def abstract_policies(self):
        pi_list = self.piecewise_constant_policies()
        return [self.get_abstract_policy(pi) for pi in pi_list]

    def generate_random_policies(self, n):
        policies = []
        for _ in range(n):
            policies.append(random_stochastic_matrix((self.n_obs, self.n_actions)))

        return policies

    @classmethod
    def generate(cls, n_states, n_actions, n_obs, sparsity=0, gamma=0.9, Rmin=-1, Rmax=1):
        mdp = MDP.generate(n_states, n_actions, sparsity, gamma, Rmin, Rmax)
        phi = random_stochastic_matrix(size=(n_states, n_obs))
        return cls(mdp, phi)

class UniformAbstractMDP(AbstractMDP):
    def __init__(self, base_mdp, phi, pi=None, p0=None):
        super().__init__(base_mdp, phi, pi, p0)

    def B(self, pi, t=200):
        p = self._replace_stationary_distribution(pi=pi, p0=self.p0, max_steps=t)
        return normalize(p * self.phi.transpose())

    def _replace_stationary_distribution(self, pi=None, p0=None, max_steps=200):
        return jnp.ones(self.base_mdp.n_states) / self.base_mdp.n_states

def test():
    # Generate a random base MDP
    mdp1 = MDP.generate(n_states=5, n_actions=3, sparsity=0.5)
    assert all([is_stochastic(mdp1.T[a]) for a in range(mdp1.n_actions)])

    # Add block structure to the base MDP
    mdp2 = BlockMDP(mdp1, n_obs_per_block=2)
    assert all([np.allclose(mdp2.base_mdp.T[a], mdp1.T[a]) for a in range(mdp1.n_actions)])
    assert all([np.allclose(mdp2.base_mdp.R[a], mdp1.R[a]) for a in range(mdp1.n_actions)])

    # Construct abstract MDP of the block MDP using perfect abstraction function
    phi = (mdp2.obs_fn.transpose() > 0).astype(int)
    mdp3 = AbstractMDP(mdp2, phi)
    assert np.allclose(mdp1.T, mdp3.T)
    assert np.allclose(mdp1.R, mdp3.R)
    print('All tests passed.')

if __name__ == '__main__':
    test()
