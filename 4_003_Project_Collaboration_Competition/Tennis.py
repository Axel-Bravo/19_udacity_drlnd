#%% Imports & Function declaration
# Standard imports
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment

# Developed imports
from ddpg import DDGP

# Parameters
model_dir = 'models/'


def execute_maddpg(state_size, action_size, random_seed, n_episodes=8000, min_req_exp=200,
                   buffer_size=int(1e4),batch_size=1024, consec_learn_iter=5, learn_every=20,
                   lr_actor=1e-4, lr_critic=1e-3):
    """
    MADDPG - Executiong Algorithm Implementation:
      - Each agent will be independent from the others
      - The Critic is trained with full "state" information
      - No shared "ReplayBuffer" for each agent
    :param state_size: number of dimensions the state observed by the agent
    :param action_size: number of dimensions for the action executed by the agent
    :param random_seed: random seed number, for reproducibility
    :param n_episodes: number of episodes the algorithm will train
    :param min_req_exp: minimum number of episodes the algorithm needs to experiment before starting to learn
    :param buffer_size: sie of the experience replay buffer
    :param batch_size: number of samples to be trained the networks on at each step
    :param consec_learn_iter: number of consecutive learning iterations
    :param learn_every: steps in a game before triggering the learning procedure
    :param lr_actor: actor's learning rate
    :param lr_critic: critic's learning rate
    :return: results obtained during the training procedure
    """
    # 1| Initialization
    # 1.1 Agents
    agent_0 = DDGP(name='agent_0', state_size=state_size, action_size=action_size, random_seed=random_seed,
                   buffer_size=buffer_size, batch_size=batch_size, consec_learn_iter=consec_learn_iter,
                   learn_every=learn_every, lr_actor=lr_actor, lr_critic=lr_critic)
    agent_1 = DDGP(name='agent_1', state_size=state_size, action_size=action_size, random_seed=random_seed + 1,
                   buffer_size=buffer_size, batch_size=batch_size, consec_learn_iter=consec_learn_iter,
                   learn_every=learn_every, lr_actor=lr_actor, lr_critic=lr_critic)
    agents = [agent_0, agent_1]

    # 1.2| Scoring
    scores = []
    scores_deque = deque(maxlen=100)

    # 2| Episode run
    for i_episode in range(1, n_episodes+1):

        # 2.0| Initialization of episode
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        full_states = [states.ravel(-1), states.ravel(-1)]
        i_score = np.zeros(2)

        for agent in agents:
            agent.reset()

        # 2.1| Episode Run
        while True:
            # 2.1.1| Agent decision and interaction
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            env_info = env.step(actions)[brain_name]

            # 2.1.2| Feedback on action
            next_states = env_info.vector_observations
            next_full_states = [next_states.ravel(-1), next_states.ravel(-1)]
            rewards = env_info.rewards
            dones = env_info.local_done

            # 2.1.3| Experience saving
            for state, full_state, action, reward, next_state, next_full_state, done, agent in zip(
                    states, full_states, actions, rewards, next_states, next_full_states, dones, agents):
                agent.memorize(state, full_state, action, reward, next_state, next_full_state, done)
                agent.update_counter()

            # 2.1.4| Update values
            for agent in agents:
                if agent.train:
                    for learn_iter in range(agent.consec_learn_iter):
                        agent.trigger_learn()

            states, full_states = next_states, next_full_states
            i_score += rewards

            # 2.1.5| Episode ending
            if np.any(dones):
                break

        # 2.2| Episode post-processing
        # 2.2.1| Scoring
        scores.append(np.max(i_score))
        scores_deque.append(np.max(i_score))

        print('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode,
                                                                        np.mean(scores_deque), np.max(i_score)))
        # 2.2.2| Saving models
        if i_episode % 500 == 0:
            for agent in agents:
                agent.save(save_path=model_dir, iteration=i_episode)

        # 2.2.3| Completion condition
        if np.mean(scores_deque) > 0.5:
            for agent in agents:
                agent.save(save_path=model_dir, iteration=i_episode)
            print('\rEpisode employed for completing the challenge {}'.format(i_episode))

            break

    return scores, scores_deque


#%% Load Tennis environment
env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86")

# Get brain information
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size

# Environment information
env_info = env.reset(train_mode=False)[brain_name]
state_size = env_info.vector_observations.shape[1]
num_agents = len(env_info.agents)


#%% MADDPG - Agent Training
score, score_episodes_deque = execute_maddpg(state_size=state_size, action_size=action_size, random_seed=10)

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score)+1), score)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#%% Environment- Close
env.close()
