#%% Imports and functions declaration
import pong_utils
import gym

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from parallelEnv import parallelEnv

import time
import matplotlib
import matplotlib.pyplot as plt

device = pong_utils.device
env = gym.make('PongDeterministic-v4') # PongDeterministic does not contain random frameskip so is faster to train


#%% Policy Implementation
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # CNN
        self.conv_1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # MLP
        self.size=1*16*16 # 1 fully connected layer
        self.fc = nn.Linear(9248, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Convolutional layer - 1
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Convolutional layer - 2
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # MLP        
        x = x.view(-1,9248) # flatten the tensor
        
        return self.sig(self.fc(x))  # P(left) = 1-P(right)


policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

#%% Trajectories rollout
envs = pong_utils.parallelEnv('PongDeterministic-v4', n=8, seed=12345)
prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)


#%% Function Definitions
def surrogate(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    # Normalize rewards
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10
    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)
    
    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)

    # include a regularization term
    # this steers new_policy towards 0.5
    # which prevents policy to become exactly 0 or 1
    # this helps with exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+         (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(torch.log(new_probs)*rewards + beta*entropy)

Lsur= surrogate(policy, prob, state, action, reward)

print(Lsur)

#%% Training

episode = 800
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
beta = .01
tmax = 320

# keep track of progress
mean_rewards = []

for e in range(episode):

    # collect trajectories
    old_probs, states, actions, rewards =         pong_utils.collect_trajectories(envs, policy, tmax=tmax)
        
    total_rewards = np.sum(rewards, axis=0)

    # this is the SOLUTION!
    # use your own surrogate function
    # L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)
    
    L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    del L
        
    # the regulation term also reduces
    # this reduces exploration in later runs
    beta*=.995
    
    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))


#%% Play
# play game after training!
pong_utils.play(env, policy, time=2000) 
plt.plot(mean_rewards)

# save your policy!
torch.save(policy, 'REINFORCE.policy')

# load your policy if needed
policy = torch.load('REINFORCE.policy')
