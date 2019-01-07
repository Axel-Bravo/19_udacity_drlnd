#%% Imports
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment
from ddpg_agent import Agent

#%% Environment load
env = UnityEnvironment(file_name="Reacher 1_arm.app")

# Get brain information
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size

# Environment information
env_info = env.reset(train_mode=False)[brain_name]
state_size = env_info.vector_observations.shape[1]
#env.close()

#%% Trainning DDPG -Agent

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)


def ddpg(n_episodes=2000, max_t=700, train=True):
    scores = []
    scores_deque = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        agent.reset()

        for t in range(max_t):
            # Agent decision and interaction
            action = agent.act(state)
            env_info = env.step(action)[brain_name]

            # Feedback on action
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # Update values
            if train:
                agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        # Scoring & Terminal information
        scores.append(score)
        scores_deque.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score),
              end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return scores


scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


#%% Training DDPG - Agents
num_agents = len(env_info.agents)
agents = []

for agent_num in range(num_agents):
    agents.append(Agent(state_size=33, action_size=4, random_seed=agent_num))


