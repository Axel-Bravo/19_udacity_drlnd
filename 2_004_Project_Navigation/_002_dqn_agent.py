import numpy as np
import random
from collections import namedtuple, deque

from _001_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
A = 0.66                 # experience replay sampling probability (0: Random -> 1: Strict probability)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, BE):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, BE)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, BE):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, priority, position) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, _, positions = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # TODO: modificar esto de por aqui tambine 
        # Compute priority = abs(TD error) + epsilon
        priority = abs(Q_targets - Q_expected) + 0.01

        # Update transition priority
        self.memory.update(priority.detach().numpy(), positions)

        # Importance sampling weight
        imp_sample_wht = pow((1/BATCH_SIZE)*(1/priority), BE)

        #TODO: adapt loss calculation
        # Compute loss
        loss = F.mse_loss(Q_expected*imp_sample_wht, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done",
                                                                "priority", "position"])
        self.seed = random.seed(seed)
        self.position_counter = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        try:
            initial_priority = max([e.priority for e in self.memory if e is not None])
        except ValueError:
            initial_priority = 0.01  # First initial case we allocate priority of 0.1

        e = self.experience(state, action, reward, next_state, done, initial_priority, self.position_counter)
        self.position_counter += 1
        self.memory.append(e)

    def update(self, priorities, positions):
        """Updates the priorities for the given position elements from ReplayBuffer."""
        init_deque_pos = self.memory[0].position
        deque_pos = positions - init_deque_pos

        for position in deque_pos:
            pos_value = int(position.item())
            if pos_value > 0:
                try:
                    state, action, reward, next_state, done, _, _ = self.memory[pos_value]
                    priority = priorities[pos_value]
                    self.memory[pos_value] = self.experience(state, action, reward, next_state, done, priority,
                                                             position)
                except IndexError:
                    pass

    def sample(self):
        """Priority based sampling a batch of experiences from memory."""
        sum_priority = pow(sum([e.priority for e in self.memory]), A)
        abs_priority = np.array([pow(e.priority, A) / sum_priority for e in self.memory])
        abs_priority = abs_priority/abs_priority.sum()

        experiences_index = np.random.choice(len(self.memory), size=self.batch_size, replace=False,
                                             p=abs_priority.reshape(-1))
        experiences = [self.memory[element] for element in experiences_index]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priority = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None]).astype(np.uint8))\
            .float().to(device)
        position = torch.from_numpy(np.vstack([e.position for e in experiences if e is not None]).astype(np.uint8))\
            .float().to(device)

        return states, actions, rewards, next_states, dones, priority, position

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
