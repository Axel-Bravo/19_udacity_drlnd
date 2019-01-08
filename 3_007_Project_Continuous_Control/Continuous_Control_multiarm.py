#%% Imports
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment
from ddpg_agent_multiarm import Agent

#%% Environment load
env = UnityEnvironment(file_name="Reacher_20_arms.app")

# Get brain information
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size

# Environment information
env_info = env.reset(train_mode=False)[brain_name]
state_size = env_info.vector_observations.shape[1]
num_agents = len(env_info.agents)


#%% Trainning DDPG -Agent

# Initialize Agent Band
agents = []
for agent_id in range(num_agents):
    agents.append(Agent(state_size=state_size, action_size=action_size, random_seed=agent_id))


def ddpg(n_episodes=2000, max_t=700, train=True):
    scores_episodes = np.zeros(num_agents)
    scores_episodes_deque = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        [agent.reset() for agent in agents]

        for t in range(max_t):
            # Agent decision and interaction
            actions = np.array([agent.act(state) for state, agent in zip(states, agents)])
            env_info = env.step(actions)[brain_name]

            # Feedback on action
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # Update values
            if train:
                [agent.step(state, action, reward, next_state, done)
                 for state, action, reward, next_state, done, agent
                 in zip(states, actions, rewards, next_states, dones, agents)]

            states = next_states
            scores += rewards

            if np.any(dones):
                break

        # Scoring & Terminal information
        scores_episodes = np.vstack((scores_episodes, scores))
        scores_episodes_deque.append(scores)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_episodes_deque),
                                                                          scores.mean()), end="")

        if i_episode % 100 == 0:
            torch.save(agents[0].actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agents[0].critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_episodes_deque)))
    return scores_episodes


scores_results = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_results)+1), scores_results.mean(axis=1))
plt.ylabel('Score - Agents mean')
plt.xlabel('Episode #')
plt.show()


#%% Environement- Close
env.close()