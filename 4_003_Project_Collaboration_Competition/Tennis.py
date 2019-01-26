#%% Imports & Function declaration
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment
from ddpg import Agent


def maddpg(agent, n_episodes=5000, max_t=1200, num_agents=2):
    """ DDPG - Algorithm implementation"""
    scores_episodes = []
    scores_episodes_deque = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = np.zeros(num_agents)
        agent.reset()

        for t in range(max_t):
            # Agent decision and interaction
            actions = agent.act(state)
            env_info = env.step(actions)[brain_name]

            # Feedback on action
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.memorize(state, actions, rewards, next_state, dones)
            agent.learn()

            state = next_state
            score += rewards

            if np.any(dones):
                break

        # Scoring & Terminal information
        scores_episodes.append(np.max(score))
        scores_episodes_deque.append(np.max(score))

        print('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
            i_episode, np.mean(scores_episodes_deque), np.max(score)))

        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_tennis.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_tennis.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_episodes_deque)))

        if np.mean(scores_episodes_deque) > 0.5:
            torch.save(agent.actor_local.state_dict(), 'model_actor_tennis.pth')
            torch.save(agent.critic_local.state_dict(), 'model_critic_tennis.pth')
            print('\rEpisode employed for completing the challenge {}'.format(i_episode))

            break

    return scores_episodes


#%% Load Reacher environment
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86")

# Get brain information
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size

# Environment information
env_info = env.reset(train_mode=False)[brain_name]
state_size = env_info.vector_observations.shape[1]
num_agents = len(env_info.agents)


#%% DDPG - Agent Training

# Initialize Agent
agent = Agent(state_size=state_size, action_size=action_size, num_agents=2, random_seed=10)

# Execute DDPG - Learning
score = maddpg(agent)

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score)+1), score)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#%% Environement- Close
env.close()
