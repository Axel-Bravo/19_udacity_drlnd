from _002_ddpg import DDPGAgent
import torch

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

class MADDPG(object):
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # Agents | Critical input = Full observation + Actions: 24 + 2*2 = 28
        self.maddpg_agent = [DDPGAgent(24, 32, 16, 2, 28, 64, 32), DDPGAgent(24, 32, 16, 2, 28, 64, 32)]

        # Parameters
        self.discount_factor = discount_factor
        self.tau = tau

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_actors_target(self):
        """get target_actors of all the agents in the MADDPG object"""
        actor_target = [ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
        return actor_target

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def learn(self, experiences, agent):
        """get all the agents in the MADDPG object to learn """
        self.maddpg_agent[agent].learn(experiences, self.discount_factor)

    def save_actors(self, checkpoint_name: str):
        """Saves actors on the current working directory"""
        for pos, actor_local in enumerate(self.get_actors()):
            torch.save(actor_local.state_dict(), checkpoint_name + '_actor_tennis_' + str(pos+1) + '.pth')

    def save_critics(self, checkpoint_name: str):
        """Saves critics on the current working directory"""
        for pos, critic_local in enumerate([ddpg_agent.critic_local for ddpg_agent in self.maddpg_agent]):
            torch.save(critic_local.state_dict(), checkpoint_name + '_critic_tennis_' + str(pos+1) + '.pth')
