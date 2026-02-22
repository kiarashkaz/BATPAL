import torch
import numpy as np
from BATPAL.util.util import _flatten, _sa_cast
from BATPAL.common.actor_buffer_advt_with_belief import ActorBufferAdvtBelief

class ActorBufferTypeBelief(ActorBufferAdvtBelief):
    """
    ActorBuffer contains data for on-policy actors.
    """
    def __init__(self, args, obs_space, act_space, num_agents, n_severity_types):
        super(ActorBufferTypeBelief, self).__init__(args, obs_space, act_space, num_agents)
        self.num_agents = num_agents
        self.n_severity_types = n_severity_types
        self.ground_truth_type = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents+self.n_severity_types), dtype=np.float32)
        self.belief_rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

    