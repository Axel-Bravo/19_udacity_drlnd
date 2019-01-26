import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units_1=256, fc_units_2=128, fc_units_3=64):
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

        self.fc1 = nn.Linear(state_size, fc_units_1)
        self.ln_1 = nn.LayerNorm(fc_units_1)
        self.fc2 = nn.Linear(fc_units_1, fc_units_2)
        self.ln_2 = nn.LayerNorm(fc_units_2)
        self.fc3 = nn.Linear(fc_units_2, fc_units_3)
        self.ln_3 = nn.LayerNorm(fc_units_3)
        self.fc4 = nn.Linear(fc_units_3, action_size)
        self.ln_4 = nn.LayerNorm(action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = self.ln_1(F.relu(x))
        x = self.fc2(x)
        x = self.ln_2(F.relu(x))
        x = self.fc3(x)
        x = self.ln_3(F.relu(x))
        x = self.fc4(x)
        x = self.ln_4(F.relu(x))

        return F.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc_units_2=128, fc_units_3=64, fc_units_4=64):
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

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.ln_1 = nn.LayerNorm(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc_units_2)
        self.ln_2 = nn.LayerNorm(fc_units_2)
        self.fc3 = nn.Linear(fc_units_2, fc_units_3)
        self.ln_3 = nn.LayerNorm(fc_units_3)
        self.fc4 = nn.Linear(fc_units_3, fc_units_4)
        self.ln_4 = nn.LayerNorm(fc_units_4)
        self.fc5 = nn.Linear(fc_units_4, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fcs1(state)
        xs = self.ln_1(F.relu(xs))
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = self.ln_2(F.relu(x))
        x = self.fc3(x)
        x = self.ln_3(F.relu(x))
        x = self.fc4(x)
        x = self.ln_4(F.relu(x))
        return self.fc5(x)
