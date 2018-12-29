#%% Import and function declaration
from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

'''
#%% 1| Agent Test Load
"""
Initial agent load for testing purposes, no learning involved.
"""
from dqn_agent import Agent

agent = Agent(state_size=37, action_size=4, seed=0)

# watch an untrained agent
env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score

while True:
    action = agent.act(state)  # select an action
    env_info = env.step(action)[brain_name]  # send the action to the environment

    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    score += reward  # update the score
    state = next_state  # roll over the state to next time step

    if done:  # exit loop if episode finished
        break

print("Score: {}".format(score))

env.close()
'''

#%% 2| Agent DQN
import torch
import matplotlib.pyplot as plt
from _001_dqn_agent import Agent
from collections import deque


def dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    agent = Agent(state_size=37, action_size=4, seed=0)

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        while True:
            # Agent decision and interaction
            action = agent.act(state)
            env_info = env.step(action)[brain_name]

            # Feedback on action
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # Update values
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward


            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return scores


scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

