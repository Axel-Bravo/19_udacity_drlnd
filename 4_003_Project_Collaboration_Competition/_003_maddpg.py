from _002_ddpg import DDPGAgent
import torch

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

class MADDPG(object):
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # Critical input = Full observation + Actions
        self.maddpg_agent = [DDPGAgent]
