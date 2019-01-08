
from unityagents import UnityEnvironment
import numpy as np

# In[ ]:


env = UnityEnvironment(file_name="Reacher_20_arms.app")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[ ]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# In[ ]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


#%% ### 3. Take Random Actions in the Environment
'''
ach step:

the agent adds its experience to the replay buffer, and
the (local) actor and critic networks are updated, using a sample from the replay buffer.
So, in order to make the code work with 20 agents, we modified the code so that after each step:

each agent adds its experience to a replay buffer that is shared by all agents, and
the (local) actor and critic networks are updated 20 times in a row (one for each agent), using 20 different samples
from the replay buffer.
'''

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
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# When finished, you can close the environment.

# In[ ]:


env.close()
