# Standard imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def intialize_nn_layers(layer: nn.Linear) -> tuple:
    """
    Initalize neural network parameters
    :param layer: neural network fully connected layer
    :return: initialization limits
    """
    layer_dimension = layer.weight.data.size()[0]
    limit = 1. / np.sqrt(layer_dimension)
    return -limit, limit


class Actor(nn.Module):

    def __init__(self, state_size: int, action_size: int, seed: int, units_fc1: int = 512, units_fc2: int = 256):
        """
        Initialize actor's network parameters and build model.
        :param state_size: number of state's dimensions
        :param action_size: number of action's dimensions
        :param seed: random seed value
        :param units_fc1: number of neurons units in first layer
        :param units_fc2: number of neurons units in second layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Network architecture
        self.fc1_layer = nn.Linear(state_size, units_fc1)
        self.fc2_layer = nn.Linear(units_fc1, units_fc2)
        self.fc3_layer = nn.Linear(units_fc2, action_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset networks parameters
        :return: None
        """
        self.fc1_layer.weight.data.uniform_(*intialize_nn_layers(self.fc1_layer))
        self.fc2_layer.weight.data.uniform_(*intialize_nn_layers(self.fc2_layer))
        self.fc3_layer.weight.data.uniform_(-2.75e-3, 2.75e-3)

    def forward(self, state: list) -> list:
        """
        Actor policy network mapping states to actions
        :param state: state current description
        :return: actions to be performed by agents controlling each tennis shovel
        """
        x = F.relu(self.fc1_layer(state))
        x = F.relu(self.fc2_layer(x))
        x = F.tanh(self.fc3_layer(x))

        return x


class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, units_fcs1=512, units_fc2=256):
        """
        Initialize critic's network parameters and build model.
        :param state_size: number of state's dimensions
        :param action_size: number of action's dimensions
        :param seed: random seed value
        :param units_fcs1: number of neurons units in first layer
        :param units_fc2: number of neurons units in second layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dropout = nn.Dropout(p=0.2)

        # Network architecture
        self.fcs1_layer = nn.Linear(state_size, units_fcs1)
        self.fc2_layer = nn.Linear(units_fcs1 + action_size, units_fc2)
        self.fc3_layer = nn.Linear(units_fc2, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
         Reset networks parameters
         :return: None
         """
        self.fcs1_layer.weight.data.uniform_(*intialize_nn_layers(self.fcs1_layer))
        self.fc2_layer.weight.data.uniform_(*intialize_nn_layers(self.fc2_layer))
        self.fc3_layer.weight.data.uniform_(-2.75e-3, 2.75e-3)

    def forward(self, state, action) -> float:
        """
        Critic policy network mapping (state, action) to Q-values
        :param state: state current description
        :param action: action current description
        :return: actions to be performed by agents controlling each tennis shovel
        """
        xs = F.relu(self.fcs1_layer(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2_layer(x))
        x = self.dropout(x)
        x = self.fc3_layer(x)

        return x
