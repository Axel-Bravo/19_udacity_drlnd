import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, num_actions:int = 6, td_algorithm: str = 'sarsa'):
        """ Initialize agent.

        Params
        ======
        - num_actions: number of actions available to the agent
        - policy: Temporal Difference algorithm implemented {'sarsa', 'sarsa_max', 'expect_sarsa'}
        """
        self.nA = num_actions
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.num_steps = 3500

        # Learning parameters
        self.alpha = 0.15
        self.epsilon = 0.1
        self.gamma = 0.8

        # Temporal Difference Algorithm
        self.td = td_algorithm

    def action_prob(self, state, epsilon) -> np.ndarray:
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment
        - epsilon: e-greedy  parameter value
        Returns
        =======
        - action_probabilities: action probabilities (based on e policy) for a given state
        """
        # Select maximum V(S,A)
        max_action_val_pos = self.Q[state].argmax()
        # Create policy pi(S,A)
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[max_action_val_pos] += (1 - epsilon)
        return policy_s

    def action(self, action_prob: np.ndarray)-> int:
        """ Given the actions probabilities values, select an action.

        Params
        ======
        - action_prob: action policy probabilities based on e-greedy algorithm
        Returns
        =======
        - action: action
        """
        return np.random.choice(np.arange(self.nA), p=action_prob)

    def step(self, state, action, reward, next_state, next_action, next_action_prob):
        """ Update the agent's knowledge, using the most recently sampled tuple.
            Step answer depends on the temporal difference algorithm employed

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if self.td is 'sarsa':
            self.Q[state][action] += self.alpha *\
                                     (reward + self.gamma *
                                      self.Q[next_state][next_action] - self.Q[state][action])
        elif self.td is 'sarsa_max':
            self.Q[state][action] += self.alpha *\
                                     (reward + self.gamma *
                                      self.Q[next_state].argmax() - self.Q[state][action])
        else:  # Expected Sarsa
            self.Q[state][action] += self.alpha *\
                                     (reward + self.gamma *
                                      sum(self.Q[next_state] * next_action_prob) - self.Q[state][action])