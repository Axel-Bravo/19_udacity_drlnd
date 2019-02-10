import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units_1=64, fc_units_2=63):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.batch1 = nn.LayerNorm(fc_units_1)
        self.batch2 = nn.LayerNorm(fc_units_2)
        self.batch3 = nn.LayerNorm(action_size)

        self.fc1 = nn.Linear(state_size, fc_units_1)
        self.fc2 = nn.Linear(fc_units_1, fc_units_2)
        self.fc3 = nn.Linear(fc_units_2, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.batch1(F.relu(self.fc1(state)))
        x = self.batch2(F.relu(self.fc2(x)))
        x = self.batch3(F.relu(self.fc3(x)))

        return F.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc_units_2=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.batch1 = nn.LayerNorm(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc_units_2)
        self.fc3 = nn.Linear(fc_units_2, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.batch1(state)
        xs = F.relu(self.fcs1(x))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)

