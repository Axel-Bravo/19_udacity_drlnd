from _001_network import Network
from _noise import OUNoise

import torch
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent(object):
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic,
                 hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2, tau=0.02):
        """
        DDPG algorithm implementation for a single agent
        :param in_actor: State space dimension
        :param hidden_in_actor: Hidden input layer dimension
        :param hidden_out_actor: Hidden output layer dimension
        :param out_actor: Number of actions dimension
        :param in_critic: Full observation dimension + all actions dimension
        :param hidden_in_critic: Hidden input layer dimension
        :param hidden_out_critic: Hidden output layer dimension
        :param lr_actor: learning rate actor network
        :param lr_critic: learning rate critic network
        :param tau: Soft network update coefficient
        """
        super(DDPGAgent, self).__init__()

        # Actor
        self.actor_local = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.actor_target = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic
        self.critic_local = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.critic_target = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=1.e-5)

        # OU noise for exploration
        self.noise = OUNoise(out_actor)

        # Parameters
        self.tau = tau
        self.exploration = 1.0

    def act(self, obs, noise=1.0, reduce_exploration=0.99):
        """
        Outputs action, based on a local actor policy
        :param obs: space current state, in which we want to execute an
        :param noise: quantity of noise to add to actions
        :param reduce_exploration: coefficient to move from a "exploratory" -> "exploiting" action selection
        :return: action
        """

        obs = torch.tensor(data=obs, dtype=torch.double, device=device)
        action = self.actor_local(obs).detach() # + noise*self.exploration*self.noise.noise()
        self.exploration *= reduce_exploration
        return action.numpy()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)  # Gradient clipping
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
