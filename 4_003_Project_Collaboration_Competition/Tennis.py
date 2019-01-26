#%% Imports & Function declaration
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment

from _003_maddpg import MADDPG
from _buffer import ReplayBuffer


def run_maddpg(agent, n_episodes=2000, max_t=800, num_agents=2, consec_learn_iter=5):
    """ MADDPG - Algorithm implementation"""

    scores_episodes = []
    scores_episodes_deque = deque(maxlen=100)

    buffer = {'agent_1': ReplayBuffer(int(500 * max_t)),
              'agent_2': ReplayBuffer(int(500 * max_t))}

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)

        for t in range(max_t):
            # Agent decision and interaction
            actions =agent.act(states)
            env_info = env.step(actions)[brain_name]

            # Feedback on action
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # Experience saving
            for enum, experience in enumerate(zip(states, actions, rewards, next_states, dones)):
                buffer['agent_' + str(enum + 1)].push(experience)

            # Update values
            if len(buffer[0]) >= 128:
                agent.learn()

            states = next_states
            scores += rewards

            if np.any(dones):
                break

        # Scoring & Terminal information
        scores_episodes.append(np.max(scores))
        scores_episodes_deque.append(np.max(scores))

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_episodes_deque),
                                                                          np.max(scores)), end="")

        if i_episode % 100 == 0:
            agent.save_actors(checkpoint_name='checkpoint')
            agent.save_critics(checkpoint_name='checkpoint')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_episodes_deque)))

        if np.mean(scores_episodes_deque) > 0.5:
            agent.save_actors(checkpoint_name='final_model')
            agent.save_critics(checkpoint_name='final_model')
            print('\rEpisode employed for completing the challenge {}'.format(i_episode))

            break

    return scores_episodes


#%% Load Tennis environment
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86")

# Get brain information
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size

# Environment information
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
state_size = states.shape[1]
num_agents = len(env_info.agents)


#%% MA-DDPG - Agent Training
# Initialize Agent
agent = MADDPG()

# Execute MA-DDPG - Learning
score = run_maddpg(agent)

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score)+1), score)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#%% Actions for adapting code
# TODO: use and delete, TEMPORAL
for i in range(1, 6):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

#%% Environement- Close
env.close()
