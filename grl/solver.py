import pathlib

import numpy as np
import pomdp_py
from pomdp_py import vi_pruning
from pomdp_py.utils import TreeDebugger
from pomdp_problems.tiger.tiger_problem import TigerProblem, TigerState

class State(pomdp_py.State):
    def __init__(self, id, terminal=False):
        self.id = id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.id == other.id
        return False

    def __str__(self):
        return f'S-{self.id}'

class Action(pomdp_py.Action):
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f'A-{self.id}'

class Observation(pomdp_py.Observation):
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.id == other.id
        return False

    def __str__(self):
        return f'Ob-{self.id}'

class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, phi):
        self.phi = phi

    def probability(self, observation, next_state, action):
        return self.phi[next_state.id, observation.id]

    def sample(self, next_state, action):
        distribution = self.phi[next_state.id].flatten().astype('float')
        distribution /= distribution.sum()
        id = np.random.choice(len(distribution), 1, p=distribution)[0]
        return Observation(id)

    def get_all_observations(self):
        return [Observation(id) for id in range(self.phi.shape[1])]

class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, T, p0):
        # Convert from our terminal state methodology (all 0's in the row) to
        # sending terminal states back to the start
        T[:, -1] = p0
        self.T = T

    def probability(self, next_state, state, action):
        return self.T[action.id, state.id, next_state.id]

    def sample(self, state, action):
        distribution = self.T[action.id, state.id].flatten().astype('float')
        distribution /= distribution.sum()

        next_id = np.random.choice(len(distribution), 1, p=distribution)[0]
        next_state = State(next_id)
        # if np.all(self.T[0, next_state.id].flatten() == 0):
        # Transitioned to terminal state
        # Terminal states have all 0's for every action
        # next_state.terminal = True
        return next_state

    def get_all_states(self):
        return [State(id) for id in range(self.T.shape[1])]

class RewardModel(pomdp_py.RewardModel):
    def __init__(self, R):
        self.R = R

    def sample(self, state, action, next_state):
        return self.R[action.id, state.id, next_state.id]

class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, pi):
        self.pi = pi

    def rollout(self, observation, history=None):
        distribution = self.pi[observation.id].astype('float')
        distribution /= distribution.sum()
        id = np.random.choice(len(distribution), 1, p=distribution)[0]
        return Action(id)

    def get_all_actions(self, state=None, history=None):
        return [Action(a) for a in range(self.pi.shape[1])]

class Problem(pomdp_py.POMDP):
    def __init__(self, spec, init_true_state, init_belief):
        agent = pomdp_py.Agent(init_belief, PolicyModel(spec['Pi_phi'][0]),
                               TransitionModel(spec['T'], spec['p0']),
                               ObservationModel(spec['phi']), RewardModel(spec['R']))
        env = pomdp_py.Environment(init_true_state, TransitionModel(spec['T'], spec['p0']),
                                   RewardModel(spec['R']))

        self.init_belief = init_belief
        super().__init__(agent, env, name=spec['name'])

    @staticmethod
    def create(spec):
        num_states = len(spec['p0'])
        init_true_state = State(np.random.choice(num_states, 1, p=spec['p0'])[0])
        # Uniform initial belief
        # prob = 1 / num_states
        # init_belief = pomdp_py.Histogram({State(i): prob for i in range(num_states)})
        init_belief = pomdp_py.Histogram(
            {State(i): prob
             for i, prob in zip(list(range(num_states)), spec['p0'])})

        problem = Problem(spec, init_true_state, init_belief)
        problem.agent.set_belief(init_belief, prior=True)
        return problem

def test_planner(problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of a given POMDP
    Args:
        problem (Problem): a problem instance
        lanner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    for i in range(nsteps):
        # if problem.env.state.terminal:
        #     break

        action = planner.plan(problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(problem.agent.tree)
            import pdb
            pdb.set_trace()

        print("==== Step %d ====" % (i + 1))
        print("True state:", problem.env.state)
        print("Belief:", problem.agent.cur_belief)
        print("Action:", action)
        reward = problem.env.state_transition(action, execute=True)
        print("Reward:", reward)

        real_observation = problem.agent.observation_model.sample(problem.env.state, action)
        print(">> Observation:", real_observation)
        problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        planner.update(problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(problem.agent.cur_belief, action,
                                                          real_observation,
                                                          problem.agent.observation_model,
                                                          problem.agent.transition_model)
            problem.agent.set_belief(new_belief)

def solve(spec, solver=None):

    if solver == 'vi':
        print("** Testing value iteration **")
        problem = Problem.create(spec)
        vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
        test_planner(problem, vi, nsteps=10)

    elif solver == 'pouct':
        print("\n** Testing POUCT **")
        problem = Problem.create(spec)
        pouct = pomdp_py.POUCT(max_depth=3,
                               discount_factor=0.95,
                               num_sims=4096,
                               exploration_const=50,
                               rollout_policy=problem.agent.policy_model,
                               show_progress=True)
        test_planner(problem, pouct, nsteps=10)
        TreeDebugger(problem.agent.tree).pp

    elif solver == 'pomcp':
        print("** Testing POMCP **")
        problem = Problem.create(spec)
        problem.agent.set_belief(pomdp_py.Particles.from_histogram(problem.init_belief,
                                                                   num_particles=100),
                                 prior=True)
        pomcp = pomdp_py.POMCP(max_depth=3,
                               discount_factor=0.95,
                               num_sims=1000,
                               exploration_const=50,
                               rollout_policy=problem.agent.policy_model,
                               show_progress=True,
                               pbar_update_interval=500)
        test_planner(problem, pomcp, nsteps=10)
        TreeDebugger(problem.agent.tree).pp

    else:
        # Use Cassandra's pomdp-solve

        # Initialize problem
        if False:
            init_state = "tiger-left"
            problem = TigerProblem(
                0.15, TigerState(init_state),
                pomdp_py.Histogram({
                    TigerState("tiger-left"): 0.5,
                    TigerState("tiger-right"): 0.5
                }))

        else:
            problem = Problem.create(spec)

        # Compute policy
        pathlib.Path('logs/solver').mkdir(exist_ok=True)
        pomdp_solve_path = "grl/pomdp-solve.bin"
        policy = vi_pruning(
            problem.agent,
            pomdp_solve_path,
            pomdp_name=f'logs/solver/{spec["name"]}-{solver}',
            discount_factor=.5,
            # options=["-horizon", "100"],
            remove_generated_files=False,
            return_policy_graph=False)

        # Simulate the POMDP using the policy
        for step in range(10):
            action = policy.plan(problem.agent)
            reward = problem.env.state_transition(action, execute=True)
            observation = problem.agent.observation_model.sample(problem.env.state, action)
            print(problem.agent.cur_belief, action, observation, reward)

            if isinstance(policy, pomdp_py.PolicyGraph):
                # No belief update needed. Just update the policy graph
                policy.update(problem.agent, action, observation)
            else:
                # belief update is needed for AlphaVectorPolicy
                new_belief = pomdp_py.update_histogram_belief(problem.agent.cur_belief, action,
                                                              observation,
                                                              problem.agent.observation_model,
                                                              problem.agent.transition_model)
                problem.agent.set_belief(new_belief)