import logging

import numpy as np
import torch

from .mdp import MDP

class PolicyEval:
    def __init__(self, amdp, pi, verbose=True):
        """
        :param amdp:   AMDP
        :param pi:     A policy
        """
        self.amdp = amdp
        self.pi_abs = pi
        self.pi_ground = self.amdp.get_ground_policy(pi)
        self.verbose = verbose

    def run(self, no_gamma, pi_abs):
        """
        :param no_gamma: if True, do not discount the weighted average value expectation
        """
        self.pi_ground - self.amdp.get_ground_policy(pi_abs)

        # MC*
        mdp_vals = self.solve_mdp(self.amdp)
        occupancy = self.get_weights(no_gamma)
        amdp_vals = self.solve_amdp(mdp_vals, occupancy)
        if self.verbose:
            logging.info(f'occupancy:\n {occupancy}')

        # TD
        td_vals = self.solve_mdp(self.create_td_model(occupancy))

        return mdp_vals, amdp_vals, td_vals

    def solve_mdp(self, mdp):
        """
        Solves for V using linear equations.
        For all s, V_pi(s) = sum_s'[T(s,pi(s),s') * (R(s,pi(s),s') + gamma * V_pi(s'))]
        """
        # Each index of these lists corresponds to one linear equation 
        # b = A*V_pi(s)
        A = [] # Each index will contain a list of |S| coefficients for that equation (list of lists of floats)
        b = [] # Each index will contain the sum of the constants for that equation   (list of floats)
        for s in range(mdp.n_states):
            a_t = np.zeros(mdp.n_states)
            a_t[s] = -1 # subtract V_pi(s) to right side
            b_t = 0
            T_pi = np.tensordot(self.pi_ground[s], mdp.T, axes=1)
            R_pi = np.tensordot(self.pi_ground[s], mdp.R, axes=1)
            for next_s in range(mdp.n_states):
                t = T_pi[s,next_s]
                r = R_pi[s,next_s]
                a_t[next_s] += t * mdp.gamma # add V_pi(s') to right side
                b_t -= t * r # subtract constants to left side

            A.append(a_t)
            b.append(b_t)

        return np.linalg.solve(A, b)

    def get_weights(self, no_gamma):
        """
        Finds the likelihood, P_pi(s), of reaching each state.
        For all s, P_pi(s) = p0(s) + sum_s"[P_pi(s") * gamma * T(s",pi(s"),s)],
          where s" is the prev state
        """
        a = []
        for s in range(self.amdp.n_states):
            a_t = np.zeros(self.amdp.n_states)
            a_t[s] = -1 # subtract P_pi(s) to right side
            for prev_s in range(self.amdp.n_states):
                T_pi = np.tensordot(self.pi_ground[prev_s], self.amdp.T, axes=1)
                t = T_pi[prev_s,s]
                if not no_gamma:
                    t *= self.amdp.gamma
                a_t[prev_s] += t

            a.append(a_t)

        b = -1 * self.amdp.p0 # subtract p0(s) to left side
        return np.linalg.solve(a, b)

    def solve_amdp(self, mdp_vals, weights):
        """
        Weights the value contribution of each state to each observation for the amdp
        """
        amdp_vals = np.zeros(self.amdp.n_obs)
        for i in range(self.amdp.n_obs):
            col = self.amdp.phi[:,i].clone().detach()#copy().astype('float')
            col *= weights
            col /= col.sum()
            v = torch.tensor(mdp_vals) * col
            amdp_vals[i] += v.sum()

        return amdp_vals

    def create_td_model(self, occupancy):
        """
        Generates effective TD(0) model
        """
        T_obs_obs = np.zeros((len(self.amdp.T), self.amdp.n_obs, self.amdp.n_obs))
        R_obs_obs = np.zeros((len(self.amdp.R), self.amdp.n_obs, self.amdp.n_obs))
        for curr_ob in range(self.amdp.n_obs):
            # phi is |S|x|O|
            ###### curr_a = self.pi[curr_ob]
            # compute p_π(o|s) for all s
            p_π_of_o_given_s = self.amdp.phi[:, curr_ob].clone().detach()#copy().astype('float')
            # want p_π(s|o) ∝ p_π(o|s)p(s) = p_π_of_o_given_s * occupancy
            w = torch.tensor(occupancy) * p_π_of_o_given_s # Prob of being in each state * prob of it emitting curr obs i
            p_π_of_s_given_o = (w / w.sum())[:,None]

            for next_ob in range(self.amdp.n_obs):
                # Q: what action should this be? [self.pi[i]]
                p_π_of_op_given_sp = self.amdp.phi[:,next_ob].clone().detach()#copy().astype('float')

                # T
                T_contributions = (torch.tensor(self.amdp.T) * p_π_of_s_given_o * p_π_of_op_given_sp)
                # sum over s', then over s
                T_obs_obs[:,curr_ob,next_ob] = T_contributions.sum(2).sum(1)

                # R
                R_contributions = torch.tensor(self.amdp.R) * T_contributions 
                denom = T_obs_obs[:,curr_ob,next_ob][:, None, None]
                denom = np.where(denom == 0, 1, denom) # Avoid divide by zero
                R_contributions /= denom

                R_obs_obs[:,curr_ob,next_ob] = R_contributions.sum(2).sum(1)

        if self.verbose:
            logging.info(f'T_bar:\n {T_obs_obs}')
            logging.info(f'R_bar:\n {R_obs_obs}')
        return MDP(T_obs_obs, R_obs_obs, self.amdp.gamma)
