#%% Imports & Function declaration
# Standard imports
import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment

# Developed imports
from maddpg import MADDPG
from buffer import ReplayBuffer

# Parameters
model_dir = 'models/'


def execute_maddpg(n_episodes=8000, batch_size=512, n_update_learn=2, noise=2, noise_reduction=0.9999):
    """
    MADDPG - Executiong Algorithm Implementation
    :param n_episodes: number of episodes the algorithm will be trained
    :param batch_size: batch size of each learning iteration
    :param n_update_learn: number of episodes between each learning phase
    :param noise: noise initial value
    :param noise_reduction: noise reduction coefficient
    :return: the training process results
    """
    # 1| Initialize
    maddpg = MADDPG()
    scores = []
    scores_deque = deque(maxlen=100)
    buffer = ReplayBuffer(5e4)

    for i_episode in range(1, n_episodes+1):

        # 0| Initialization of episode
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations # TODO: create a "state_full" method
        i_score = np.zeros(num_agents)
        maddpg.reset()  # TODO: implement "reset" method for MADDPG

        # 1| Episode Run
        while True:
            # 1.1| Agent decision and interaction
            actions = maddpg.act(state, noise=noise)
            noise *= noise_reduction
            env_info = env.step(actions)[brain_name]

            # 1.2| Feedback on action
            next_state = env_info.vector_observations  # TODO: ojo que next_State no genera "next_obs_full" NEED
            rewards = env_info.rewards
            dones = env_info.local_done

            # 1.3| Experience saving
            transition = (state, state_full, actions, rewards, next_state, next_state_full, dones)
            buffer.push(transition)

            # 1.4| Update values
            i_score += rewards
            state, state_full = next_state, next_state_full

            # 1.5| Update agents
            # TODO: convertir en funcion
            if len(buffer) > batch_size and i_episode % n_update_learn == 0:
                for i_agent in range(2):
                    samples = buffer.sample(batch_size)
                    maddpg.update(samples, i_agent)
                maddpg.update_targets()  # soft update the target network towards the actual networks

            # 1.6| Episode ending
            if np.any(dones):
                break

        # 2| Episode post-processing
        # 2.1| Scoring
        scores.append(np.max(score))
        scores_deque.append(np.max(score))

        print('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode,
                                                                        np.mean(scores_deque), np.max(score)))
        # 2.2| Saving models
        if i_episode % 100 == 0:
            # Todo: convertir en funcion
            save_dict_list = []
            for i_agent in range(2):
                save_dict = {'actor_params': maddpg.maddpg_agent[i_agent].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i_agent].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i_agent].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i_agent].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, os.path.join(model_dir, 'i_episode-{}.pt'.format(i_episode)))

        # 3| Completion condition
            if np.mean(scores_deque) > 0.5:
                save_dict_list = []
                for i_agent in range(2):
                    save_dict = {'actor_params': maddpg.maddpg_agent[i_agent].actor.state_dict(),
                                 'actor_optim_params': maddpg.maddpg_agent[i_agent].actor_optimizer.state_dict(),
                                 'critic_params': maddpg.maddpg_agent[i_agent].critic.state_dict(),
                                 'critic_optim_params': maddpg.maddpg_agent[i_agent].critic_optimizer.state_dict()}
                    save_dict_list.append(save_dict)

                    torch.save(save_dict_list, os.path.join(model_dir, 'i_episode-{}.pt'.format(i_episode)))
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
score, score_episodes_deque = execute_maddpg()

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score)+1), score)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

#%% Environment- Close
env.close()
