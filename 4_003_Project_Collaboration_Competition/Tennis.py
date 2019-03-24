#%% Imports & Function declaration
# Standard imports
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

# Developed imports
from ddpg import DDGP

# Parameters
model_dir = 'models/'


def execute_ddpg(state_size, action_size, num_agents, random_seed=7, n_episodes=1000, max_episode_t=2000,
                 buffer_size=int(1e5), batch_size=512, consec_learn_iter=10, learn_every=5,
                 lr_actor=1e-4, lr_critic=3e-4):
    """
    DDPG - Execution Algorithm Implementation
    :param state_size: number of dimensions the state observed by the agent
    :param action_size: number of dimensions for the action executed by the agent
    :param num_agents: number of agents present in the environment
    :param random_seed: random seed number, for reproducibility
    :param n_episodes: number of episodes the algorithm will train
    :param max_episode_t: maximum number of time steps to play at each episode
    :param buffer_size: sie of the experience replay buffer
    :param batch_size: number of samples to be trained the networks on at each step
    :param consec_learn_iter: number of consecutive learning iterations
    :param learn_every: steps in a game before triggering the learning procedure
    :param lr_actor: actor's learning rate
    :param lr_critic: critic's learning rate
    :return: results obtained during the training procedure
    """
    # 1| Initialization
    agent = DDGP(state_size=state_size, action_size=action_size, random_seed=random_seed, buffer_size=buffer_size,
                 batch_size=batch_size, lr_actor=lr_actor, lr_critic=lr_critic)

    global_score = []
    global_score_deque = deque(maxlen=100)

    # 2| Episode run
    for i_episode in range(1, n_episodes+1):

        # 2.0| Initialization of episode
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        i_score = np.zeros(num_agents)
        agent.reset()

        # 2.1| Episode Run
        for t_step in range(max_episode_t):
            # 2.1.1| Agent decision and interaction
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]

            # 2.1.2| Feedback on action
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # 2.1.3| Experience saving
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.memorize(state, action, reward, next_state, done)

            # 2.1.4| Update values
            states = next_states
            i_score += env_info.rewards

            # 2.1.4| Update values
            if t_step % learn_every == 0:
                for _ in range(consec_learn_iter):
                    agent.trigger_learn()

            # 2.1.5| Episode ending
            if np.any(dones):
                break

        # 2.2| Episode post-processing
        # 2.2.1| Scoring
        global_score.append(np.max(i_score))
        global_score_deque.append(np.max(i_score))

        if i_episode % 5 == 0:
            print('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
                i_episode, np.mean(global_score_deque), np.max(i_score)))

        # 2.2.2| Saving models
        if i_episode % 50 == 0:
            agent.save(save_path=model_dir, iteration=i_episode)

        # 2.2.3| Completion condition
        if np.mean(global_score_deque) >= 0.5:
            agent.save(save_path=model_dir, iteration=i_episode)
            print('\rEpisode employed for completing the challenge {}'.format(i_episode))
            break

    return global_score


#%% Load Tennis environment
env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")

# Get brain information
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size

# Environment information
env_info = env.reset(train_mode=False)[brain_name]
state_size = env_info.vector_observations.shape[1]
num_agents = len(env_info.agents)


#%% DDPG - Agent Training
score = execute_ddpg(state_size=state_size, action_size=action_size, num_agents=num_agents)

# Plot results
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(np.arange(1, len(score)+1), score)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#%% Environment- Close
env.close()
