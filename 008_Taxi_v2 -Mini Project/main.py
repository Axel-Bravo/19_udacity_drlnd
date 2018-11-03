from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

# Plot results
plt.figure(figsize=(20, 10))
plt.plot(np.exp(avg_rewards))
plt.show()