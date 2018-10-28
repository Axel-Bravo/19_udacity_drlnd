#%% Imports
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import check_test
from plot_utils import plot_values


#%% Environement load & description
env = gym.make('CliffWalking-v0')

print(env.action_space)
print(env.observation_space)

# define the optimal state-value function
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

plot_values(V_opt)

#%% Part 1: TD Control Sarsa


def e_greedy_action(s_action_values: np.ndarray, epsilon: float) -> int:
    """
    Given an Q (state-action table) returns an action based on "e-greedy" policy
    :param s_action_values: Q[state] is the estimated action value corresponding to a state
    :param state: Actual state where we are.
    :param epsilon:
    :return: Action to execute
    """
    # Select maximum V(S,A)
    max_action_val_pos = s_action_values.argmax()
    # Create policy pi(S,A)
    policy_s = np.ones(env.nA) * epsilon / env.nA
    policy_s[max_action_val_pos] += (1 - epsilon)
    # Select action based on policy pi(S,A)
    action = np.random.choice(np.arange(env.nA), p=policy_s)

    return action


def sarsa(env, num_episodes, alpha, gamma=1.0):
    """
    Implementation of the Sarsa control algorithm
    :param env: Instance of an OpenAi Gym environment
    :param num_episodes: Number of episodes that are generated through agent-environment interaction
    :param alpha: Step-size parameter for the update step
    :param gamma: Discount rate. It must be a value between 0 and 1, inclusive (default value:1)
    :return: Q (dict): where Q[s][a] is the estimated action value corresponding to state "s" and action "a"
    """
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)

    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # Initialize parameters
        score = 0
        state = env.reset()
        # Update epsilon
        epsilon = 1 / i_episode
        # Choose action
        action = e_greedy_action(s_action_values=Q[state], epsilon=epsilon)

        for t_step in np.arange(300):
            next_state, reward, done, info = env.step(action)
            # add reward to score
            score += reward

            if not done:
                next_action = e_greedy_action(s_action_values=Q[next_state], epsilon=epsilon)
                # Update Q table
                Q[state][action] += alpha*(reward + gamma*Q[next_state][next_action] - Q[state][action])
                # Update values
                state = next_state
                action = next_action

            if done:
                # Update Q table
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                # append score
                tmp_scores.append(score)
                break

        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores))

    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(scores), endpoint=False), np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))

    return Q

#%% Sarsa Testing
# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 5000, .02)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)

