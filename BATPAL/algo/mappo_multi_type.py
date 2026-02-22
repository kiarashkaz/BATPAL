import numpy as np
import torch
import torch.nn as nn
from BATPAL.util.util import get_grad_norm, check, softmax, update_linear_schedule
from BATPAL.algo.base import Base
from BATPAL.model.adv_actor import AdvActor


class MAPPOMultiType(Base):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu"), n_policies=None):
        """Initialize MAPPO algorithm."""
        super(MAPPOMultiType, self).__init__(args, obs_space, act_space, device)
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.adv_lr = args["adv_lr"]
        self.adv_epoch = args["adv_epoch"]
        self.adv_entropy_coef = args["adv_entropy_coef"]
        self.super_adversary = args["super_adversary"]  # whether the adversary has defenders' policies
        self.belief = args["belief"]
        self.num_agents = num_agents
        
        self.n_severity_types = args["n_severity_types"]
        self.n_policies = n_policies if n_policies is not None else self.n_severity_types 
        self.v_max = args["v_max"]
        self.v_min = args["v_min"]
        self.log_barrier_coef = args["log_barrier_coef"]

        # create actor networks
        self.adv_actors = [AdvActor(args, self.obs_space, self.act_space, self.num_agents, self.device) for _ in range(self.n_policies)]
        # create actor optimizer
        self.adv_actor_optimizers = [torch.optim.Adam(actor.parameters(),
                                                lr=self.adv_lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay) for actor in self.adv_actors]
        

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        super().lr_decay(episode, episodes)
        for optimizer in self.adv_actor_optimizers:
            update_linear_schedule(optimizer, episode, episodes, self.adv_lr)

    def get_adv_actions(self, obs, rnn_states_actor, masks, available_actions=None,
                        deterministic=False, agent_id=0, severity_ind=0):
        """Compute actions and value function predictions for the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, action_log_probs, rnn_states_actor = self.adv_actors[severity_ind](obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    available_actions,
                                                                    deterministic)
        return actions, action_log_probs, rnn_states_actor
    
    def get_adv_logits(self, obs, rnn_states_actor, masks, available_actions=None,
                        deterministic=False, agent_id=0, severity_ind=0):
        """Compute actions and value function predictions for the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        action_logits = self.adv_actors[severity_ind].get_logits(obs,
                                                rnn_states_actor,
                                                masks,
                                                available_actions,
                                                deterministic)
        return action_logits

    def act_adv(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0, severity_ind=0):
        """Compute actions using the given inputs.
        Args:
            obs (np.ndarray): local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.adv_actors[severity_ind](
            obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        (
            obs_batch,
            obs_next_batch,
            rnn_states_batch,
            adv_rnn_states_batch,
            actions_batch,
            adv_actions_batch,
            rewards_batch,
            masks_batch,
            active_masks_batch,
            adv_active_masks_batch,
            old_action_log_probs_batch,
            old_adv_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            adv_obs_batch,
        ) = sample        
               
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # update actor
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch).sum() 
            if active_masks_batch.sum() > 0:
                policy_action_loss /= active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()
        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
    

    def update_adv(self, sample, baseline_sample, baseline_value, severity_ind):
        """Update adv_actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        (
            obs_batch,
            obs_next_batch,
            rnn_states_batch,
            adv_rnn_states_batch,
            actions_batch,
            adv_actions_batch,
            rewards_batch,
            masks_batch,
            active_masks_batch,
            adv_active_masks_batch,
            old_action_log_probs_batch,
            old_adv_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            adv_obs_batch,
              ) = sample
        
        (
            baseline_obs_batch,
            baseline_obs_next_batch,
            baseline_rnn_states_batch,
            baseline_adv_rnn_states_batch,  
            baseline_actions_batch,
            baseline_adv_actions_batch,
            baseline_rewards_batch,     
            baseline_masks_batch,
            baseline_active_masks_batch,
            baseline_adv_active_masks_batch,
            baseline_old_action_log_probs_batch,
            baseline_old_adv_action_log_probs_batch,
            baseline_adv_targ,
            baseline_available_actions_batch,
            baseline_adv_obs_batch,
              ) = baseline_sample

        severity_level = severity_ind if self.n_severity_types > 1 else 0
        self.baseline_high_limit_normalized = 1 - (severity_level / self.n_severity_types)
        self.baseline_low_limit_normalized = 1 - (severity_level + 1) / self.n_severity_types
        self.baseline_high_limit = self.baseline_high_limit_normalized * (self.v_max - self.v_min) + self.v_min
        self.baseline_low_limit = self.baseline_low_limit_normalized * (self.v_max - self.v_min) + self.v_min        
        
        baseline_value = torch.tensor(baseline_value).to(**self.tpdv)
        baseline_value_normalized = (baseline_value - self.v_min) / (self.v_max - self.v_min)

        old_adv_action_log_probs_batch = check(old_adv_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        adv_active_masks_batch = check(adv_active_masks_batch).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        if self.super_adversary:
            obs_batch = adv_obs_batch

        adv_action_log_probs, adv_dist_entropy, _ = self.adv_actors[severity_ind].evaluate_actions(
            obs_batch,
            adv_rnn_states_batch,
            adv_actions_batch,
            masks_batch,
            available_actions_batch,
            adv_active_masks_batch,
        )
        # update adv_actor
        adv_imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(adv_action_log_probs - old_adv_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        adv_surr1 = adv_imp_weights * -adv_targ
        adv_surr2 = (
            torch.clamp(adv_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * -adv_targ
        )

        if self.use_policy_active_masks:
            adv_policy_action_loss = (
                -torch.sum(torch.min(adv_surr1, adv_surr2), dim=-1, keepdim=True)
                * adv_active_masks_batch).sum()
            if adv_active_masks_batch.sum() > 0:
                adv_policy_action_loss /= adv_active_masks_batch.sum()
        else:
            adv_policy_action_loss = -torch.sum(
                torch.min(adv_surr1, adv_surr2), dim=-1, keepdim=True
            ).mean()

        baseline_old_adv_action_log_probs_batch = check(baseline_old_adv_action_log_probs_batch).to(**self.tpdv)
        baseline_adv_targ = check(baseline_adv_targ).to(**self.tpdv)
        baseline_adv_active_masks_batch = check(baseline_adv_active_masks_batch).to(**self.tpdv)
        if self.super_adversary:
            baseline_obs_batch = baseline_adv_obs_batch

        baseline_adv_action_log_probs, _, _ = self.adv_actors[severity_ind].evaluate_actions(
            baseline_obs_batch,
            baseline_adv_rnn_states_batch,
            baseline_adv_actions_batch,
            baseline_masks_batch,
            baseline_available_actions_batch,
            baseline_adv_active_masks_batch,
        )
        #baseline_adv_imp_weights = getattr(torch, self.action_aggregation)(
        #    torch.exp(baseline_adv_action_log_probs - baseline_old_adv_action_log_probs_batch),
        #    dim=-1,
        #    keepdim=True,
        #)
        #baseline_adv_surr1 = baseline_adv_imp_weights * baseline_adv_targ
        #baseline_adv_surr2 = (
        #    torch.clamp(baseline_adv_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
        #    * baseline_adv_targ
        #)
        #if self.use_policy_active_masks:
        #    baseline_adv_policy_action_loss = (
        #        torch.sum(torch.min(baseline_adv_surr1, baseline_adv_surr2), dim=-1, keepdim=True)
        #        * baseline_adv_active_masks_batch).sum()
        #    if baseline_adv_active_masks_batch.sum() > 0:
        #        baseline_adv_policy_action_loss /= baseline_adv_active_masks_batch.sum()
        #else:
        #    baseline_adv_policy_action_loss = torch.sum(
        #        torch.min(baseline_adv_surr1, baseline_adv_surr2), dim=-1, keepdim=True
        #    ).mean()
        if self.use_policy_active_masks:
            baseline_adv_policy_action_loss = (
                torch.sum(baseline_adv_action_log_probs * baseline_adv_targ, dim=-1, keepdim=True)
                * baseline_adv_active_masks_batch).sum()
            if baseline_adv_active_masks_batch.sum() > 0:
                baseline_adv_policy_action_loss /= baseline_adv_active_masks_batch.sum()
        else:
            baseline_adv_policy_action_loss = torch.sum(
                baseline_adv_action_log_probs * baseline_adv_targ, dim=-1, keepdim=True
            ).mean()

        if baseline_value_normalized >= (self.baseline_low_limit_normalized + 1e-3 ) and baseline_value_normalized <= (self.baseline_high_limit_normalized - 1e-3):        
                high_const_loss = -baseline_adv_policy_action_loss/ (self.baseline_high_limit - baseline_value)
                low_const_loss = baseline_adv_policy_action_loss / (baseline_value - self.baseline_low_limit)
                adv_policy_loss = adv_policy_action_loss- adv_dist_entropy * self.adv_entropy_coef\
                      - self.log_barrier_coef * (high_const_loss + low_const_loss)
        else: 
             adv_policy_loss = torch.sign(baseline_value - 0.5 * (self.baseline_high_limit + self.baseline_low_limit)) * baseline_adv_policy_action_loss     
        

        self.adv_actor_optimizers[severity_ind].zero_grad()

        adv_policy_loss.backward()

        if self.use_max_grad_norm:
            adv_actor_grad_norm = nn.utils.clip_grad_norm_(
                self.adv_actors[severity_ind].parameters(), self.max_grad_norm
            )
        else:
            adv_actor_grad_norm = get_grad_norm(self.adv_actors[severity_ind].parameters())

        self.adv_actor_optimizers[severity_ind].step()

        return adv_policy_loss, adv_dist_entropy, adv_actor_grad_norm, adv_imp_weights


    def train(self, actor_buffer, advantages, state_type):
        raise NotImplementedError

    def train_adv(self, actor_buffer, advantages, state_type):
        raise NotImplementedError

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type, 
                          baseline_actor_buffer, baseline_advantages, baseline_value, severity_ind):
        """
        Perform a training update using minibatch GD.
        :param actor_buffer: (List[ActorBuffer]) buffer containing training data related to actor.
        :param advantages: (ndarray) advantages.
        :param num_agents: (int) number of agents.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info["adv_policy_loss"] = 0
        train_info["adv_dist_entropy"] = 0
        train_info["adv_actor_grad_norm"] = 0
        train_info["adv_ratio"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (std_advantages + 1e-5)
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":     
            advantages_list = []
            baseline_advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])
                baseline_advantages_list.append(baseline_advantages[:, :, agent_id])


        for _ in range(self.ppo_epoch):
            data_generators = []
            baseline_data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch, self.data_chunk_length)
                    basline_data_generator = baseline_actor_buffer[agent_id].recurrent_generator_actor(
                        baseline_advantages_list[agent_id], self.actor_num_mini_batch, self.data_chunk_length)
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch)
                    basline_data_generator = baseline_actor_buffer[agent_id].naive_recurrent_generator_actor(
                        baseline_advantages_list[agent_id], self.actor_num_mini_batch)
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch)
                    basline_data_generator = baseline_actor_buffer[agent_id].feed_forward_generator_actor(
                        baseline_advantages_list[agent_id], self.actor_num_mini_batch)
                data_generators.append(data_generator)
                baseline_data_generators.append(basline_data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(15)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(14):
                        batches[i].append(sample[i])
                for agent_id in range(num_agents):
                    def_act = np.concatenate([*batches[4][:agent_id], *batches[4][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([batches[0][agent_id], softmax(def_act)], axis=-1)
                    batches[14].append(adv_obs)
                for i in range(13):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[13][0] is None:
                    batches[13] = None
                else:
                    batches[13] = np.concatenate(batches[13], axis=0)
                batches[14] = np.concatenate(batches[14], axis=0)

                basline_batches = [[] for _ in range(15)]
                for baseline_generator in baseline_data_generators:
                    baseline_sample = next(baseline_generator)
                    for i in range(14):
                        basline_batches[i].append(baseline_sample[i])
                for agent_id in range(num_agents):
                    def_act = np.concatenate([*basline_batches[4][:agent_id], *basline_batches[4][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([basline_batches[0][agent_id], softmax(def_act)], axis=-1)
                    basline_batches[14].append(adv_obs)
                for i in range(13):
                    basline_batches[i] = np.concatenate(basline_batches[i], axis=0)
                if basline_batches[13][0] is None:
                    basline_batches[13] = None
                else:
                    basline_batches[13] = np.concatenate(basline_batches[13], axis=0)
                basline_batches[14] = np.concatenate(basline_batches[14], axis=0)        

                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(tuple(batches))

                for _ in range(self.adv_epoch):
                    adv_policy_loss, adv_dist_entropy, adv_actor_grad_norm, adv_imp_weights = self.update_adv(
                        tuple(batches), tuple(basline_batches), baseline_value, severity_ind)
                    train_info["adv_policy_loss"] += adv_policy_loss.item() / self.adv_epoch
                    train_info["adv_actor_grad_norm"] += adv_actor_grad_norm / self.adv_epoch
                    train_info["adv_dist_entropy"] += adv_dist_entropy.item() / self.adv_epoch
                    train_info["adv_ratio"] += adv_imp_weights.mean() / self.adv_epoch

                train_info["policy_loss"] += policy_loss.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["dist_entropy"] += dist_entropy.item() 
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info  
    
