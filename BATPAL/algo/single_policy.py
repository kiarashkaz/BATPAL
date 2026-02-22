
#from utils.buffer_recurrent import get_device, RecurrentReplayBuffer
import torch
import os
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, n_hidden, n_actions, rnn=True):
        super(RNNAgent, self).__init__()
        self.n_hidden = n_hidden
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.use_rnn = rnn

        self.fc1 = nn.Linear(input_shape, n_hidden)
        if self.use_rnn:
            self.rnn = nn.GRUCell(n_hidden, n_hidden)
        else:
            self.rnn = nn.Linear(n_hidden, n_hidden)    
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.n_hidden).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.n_hidden)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class PolicyAgent:
    def __init__(self, obs_size, hidden_size, action_size, soft_policy=True, rnn=True):
        
        self.obs_size = obs_size
        self.n_actions = action_size
        self.input_shape = self.obs_size
        self.model = RNNAgent(self.input_shape,  hidden_size, action_size, rnn=rnn)
        self.hidden = self.model.init_hidden()
        self.test_mode = True
        self.soft_policy = soft_policy


    def load_model(self, save_dir):
        self.model.load_state_dict(
            torch.load(save_dir, map_location=torch.device("cpu")))
        self.test_mode = True

    def save_model(self, save_dir):
        save_name = "adv_model.pth"
        torch.save(self.model.state_dict(), os.path.join(save_dir, save_name))

    def compute_action(self, obs, hidden_state, avail_actions=None):
        #print(avail_actions)
        if avail_actions is None:
            avail_actions = torch.ones(obs.shape[0], self.n_actions, dtype=np.int8)
        with torch.no_grad():
            obs = torch.from_numpy(obs) if type(obs) == np.ndarray else obs
            hidden_state = torch.from_numpy(hidden_state) if type(hidden_state) == np.ndarray else hidden_state
            avail_actions = torch.from_numpy(avail_actions) if type(avail_actions) == np.ndarray else avail_actions
            outs, hidden_state = self.model(obs, hidden_state)

            if self.soft_policy:
                outs[ avail_actions == 0]= -1e10
                p=torch.softmax(outs, dim=1)
                i = torch.multinomial(p, 1)
            else:
                outs[ avail_actions == 0]= -torch.inf
                m, i = torch.max(outs, dim=1)
                i = i.reshape(-1, 1)
            return i.tolist(), hidden_state


