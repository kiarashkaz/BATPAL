
import time
import numpy as np
import torch
from BATPAL.common.popart import PopArt
from BATPAL.common.actor_buffer_type_belief import ActorBufferTypeBelief
from BATPAL.common.critic_buffer_ep import CriticBufferEP
from BATPAL.common.critic_buffer_fp import CriticBufferFP
from BATPAL.algo import ALGO_REGISTRY
from BATPAL.common.v_critic import VCritic
from BATPAL.common.fgsm import FGSM
import time
import numpy as np
import torch
from BATPAL.model.adv_actor import AdvActor
from BATPAL.util.util import _t2n
import setproctitle
from BATPAL.util.util import make_eval_env, make_train_env, make_render_env, seed, init_device, init_dir, save_config, get_num_agents, softmax
from BATPAL.env import LOGGER_REGISTRY
from BATPAL.algo.single_policy import PolicyAgent
import copy
import os

#import ipdb



class OnPolicyMultiTypeRunner:
    """Runner for on-policy algorithms (adv training)."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyMARunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        # TODO: unify the type of args
        # args: argparse.Namespace -> dict
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        # get practical parameters
        for k, v in algo_args["train"].items():
            self.__dict__[k] = v
        for k, v in algo_args["eval"].items():
            self.__dict__[k] = v
        for k, v in algo_args["render"].items():
            self.__dict__[k] = v
        assert algo_args["algo"]["belief"] == True, "You are selecting to use belief for defense, yet not enabling belief"
        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_N = algo_args["model"]["recurrent_N"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.central_belief_option = algo_args["algo"].get("central_belief_option", 'mean')
        self.state_type = env_args.get("state_type", "FP")
        # TODO: don't use this default value
        self.share_param = algo_args["algo"].get("share_param", False)
        self.fixed_order = algo_args["algo"].get("fixed_order", False)
        # adv training
        self.adv_prob = algo_args["algo"].get("adv_prob", 0.5)  # probability of having adversary
        self.eval_critic_landscape = algo_args["algo"].get("eval_critic_landscape", False)  # probability of having adversary
        # adding adversary on observation
        self.obs_adversary = env_args.get("obs_agent_adversary", False)
        assert not self.obs_adversary, "The environment should not add additional dimensions itself"
        self.agent_adversary = algo_args["algo"].get("agent_adversary", 0)  # who is the adversary
        # use it if update adversary for multiple times
        self.victim_interval = algo_args["algo"].get("victim_interval", 1)
        # if self.agent_adversary<0, then we randomly assign adversary
        self.random_adversary = (self.agent_adversary < 0)
        self.episode_adversary = False  # which episode contains the adversary
        self.load_critic = algo_args["algo"].get("load_critic", False)
        self.load_adv_actor = algo_args["algo"].get("load_adv_actor", False)
        self.super_adversary = algo_args["algo"].get("super_adversary", False)  # whether the adversary has defenders' policies
        self.teacher_forcing = algo_args["algo"].get("teacher_forcing", False)
        self.adapt_adversary = algo_args["algo"].get("adapt_adversary", False)
        self.state_adversary = algo_args["algo"].get("state_adversary", False)
        self.render_mode = algo_args["render"].get("render_mode", None)
        self.save_checkpoint = False
        self.true_type_prob = 1  # linear decay from 1 to 0
        self.forced_belief = algo_args["algo"].get("forced_belief", False)  # whether to use forced belief
        if self.forced_belief:
            self.true_type_prob = 0  # if forced belief, then we always use belief
        self.forced_no_belief = algo_args["algo"].get("forced_no_belief", False)  # whether to use forced no belief

        self.baseline_policy = algo_args["algo"].get("baseline_policy", None)  # whether to use baseline policy
        self.n_severity_types = algo_args["algo"].get("n_severity_types", 1)


        self.algo_name = args.algo
        self.env_name = args.env
        # TODO: seed --> set_seed
        seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.use_render:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args.env, env_args, args.algo, args.exp_name, algo_args["seed"]["seed"])
            save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(str(args.algo) + "-" + str(args.env) + "-" + str(args.exp_name))

        # set the config of env
        if self.use_render:
            self.envs, self.manual_render, self.manual_expand_dims, self.manual_delay, self.env_num = make_render_env(
                args.env, algo_args["seed"]["seed"], env_args)
        else:
            if self.env_name == "toy":
                from BATPAL.env.toy_example.toy_example import ToyExample
                self.toy_env = ToyExample(env_args)
            self.envs = make_train_env(
                args.env, algo_args["seed"]["seed"], algo_args["train"]["n_rollout_threads"], env_args)
            if self.baseline_policy is not None:
                self.baseline_envs = make_train_env(
                    args.env, algo_args["seed"]["seed"], algo_args["train"]["n_rollout_threads"], env_args)
            self.eval_envs = make_eval_env(
                args.env, algo_args["seed"]["seed"], algo_args["eval"]["n_eval_rollout_threads"], env_args) if algo_args["eval"]["use_eval"] else None
        self.num_agents = get_num_agents(args.env, env_args, self.envs)
        self.ground_truth_type = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents+self.n_severity_types))  # self, belief of others

        self.adapt_adv_probs = np.zeros(self.n_severity_types)
        self.reward_max = 0
        self.belief_shape = self.num_agents + self.n_severity_types
        self.stack_share_obs = env_args.get("stack_share_obs", False)
        obs_shape = self._adjust_obs_shapes(self.envs.observation_space, self.belief_shape)
        share_obs_shape = self._adjust_obs_shapes(self.envs.share_observation_space, 2*self.belief_shape, stack=self.stack_share_obs)
        print("share_observation_space: {}".format(share_obs_shape))
        print("observation_space: {}".format(obs_shape)) # we add both the belief and the true type
        print("action_space: {}".format(self.envs.action_space))

        # actor
        if self.share_param:
            self.actor = []
            ac = ALGO_REGISTRY[args.algo](
                {**algo_args["model"], **algo_args["algo"]}, obs_shape, self.envs.action_space[0], self.num_agents, device=self.device)
            self.actor.append(ac)
            for agent_id in range(1, self.num_agents):
                assert self.envs.observation_space[agent_id] == self.envs.observation_space[
                    0], "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert self.envs.action_space[agent_id] == self.envs.action_space[
                    0], "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
            if self.baseline_policy is not None:
                self.baseline_actor = []
                baseline_obs_space = self._adjust_obs_shapes(self.envs.observation_space, self.num_agents)
                bac= ALGO_REGISTRY["mappo_advt_belief"](
                    {**algo_args["model"], **algo_args["algo"]}, baseline_obs_space, self.envs.action_space[0], self.num_agents, device=self.device)
                self.baseline_actor.append(bac)
                for agent_id in range(1, self.num_agents):
                    self.baseline_actor.append(self.baseline_actor[0])
            if self.adv_model_dir is not None:
                self.evaluate_external = algo_args["eval"].get("evaluate_external", False) # For unseen Dynamic advs
                if self.evaluate_external:
                    self.external_adv = PolicyAgent(45,128,6 , False, True) #2s3z(80, 64, 11), MMM(160,128,16), spread(18,128,5), LBF(45,128,6)
                    self.external_adv_hidden_size = 128
                    self.external_adv.load_model(self.adv_model_dir)
                else:
                    self.eval_adv_n_types = algo_args["eval"].get("eval_adv_n_types", 0)                 
                    eval_adv_obs_space = self._adjust_obs_shapes(self.envs.observation_space, self.num_agents + self.eval_adv_n_types)                
                    self.eval_adv_actor = AdvActor({**algo_args["model"], **algo_args["algo"]}, eval_adv_obs_space, self.envs.action_space[0], self.num_agents, self.device)
                    self.eval_adv_type_ind = algo_args["eval"].get("eval_adv_type_ind", 0)
                    state_dict = torch.load(self.adv_model_dir, map_location=torch.device('cpu'))
                    self.eval_adv_actor.load_state_dict(state_dict)                    
        else:
            print('Error: defense of non-shared algo with belief not implemented.')
            raise NotImplementedError

        if self.use_render is False:
            # Buffer for rendering
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = ActorBufferTypeBelief(
                    {**algo_args["train"], **algo_args["model"]}, obs_shape, self.envs.action_space[agent_id], self.num_agents, self.n_severity_types)
                self.actor_buffer.append(ac_bu)
            
            self.critic = VCritic(
                {**algo_args["model"], **algo_args["algo"]}, share_obs_shape, device=self.device)
            if self.state_type == "EP":  # note that we change it to be FP here, since we need multi critics
                self.critic_buffer = CriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]}, share_obs_shape)
            elif self.state_type == "FP":
                self.critic_buffer = CriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]}, share_obs_shape, self.num_agents)
            else:
                raise NotImplementedError           

            if self.use_popart is True:
                self.value_normalizer = PopArt(1, device=self.device)
            else:
                self.value_normalizer = None

            if self.baseline_policy is not None:
                self.baseline_critic = copy.deepcopy(self.critic)
                self.baseline_value_normalizer = copy.deepcopy(self.value_normalizer)
                self.baseline_critic_buffer = copy.deepcopy(self.critic_buffer)
                self.baseline_ground_truth_type = np.zeros((self.n_rollout_threads, self.num_agents, self.belief_shape))
                self.baseline_actor_buffer = copy.deepcopy(self.actor_buffer)
                            
            self.logger = LOGGER_REGISTRY[args.env](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir)

        if self.state_adversary or self.render_mode == "state":
            self.fgsm = FGSM(algo_args["algo"], self.obs_adversary, self.actor, device=self.device)

        if self.model_dir is not None:
            self.restore()
        if self.baseline_policy is not None:
            self.load_baseline()    

    def run(self):
        if self.use_render is True:
            self.render()
            return
        print("start running")
        self.warmup()  # reset
        if self.baseline_policy is not None:
            self.baseline_warmup()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        self.logger.init(episodes)

        self.logger.episode_init(0)

        if self.use_eval:
            self.prep_rollout()
            self.eval()
            if self.state_adversary:
                self.eval_adv_state()
                return
            else:               
                self.eval_adv()
                if self.eval_only:
                    if self.adv_model_dir is not None:
                        self.eval_adv_external()
                    return

        for episode in range(1, episodes + 1):

            if episode > episodes / 2:
                self.teacher_forcing = False
            else:
                self.save_checkpoint = True
                self.true_type_prob = 1 -  (episode - 1) / (episodes / 2)
            if self.forced_belief:
                self.true_type_prob = 0
            if self.forced_no_belief:
                self.true_type_prob = 1
                self.teacher_forcing = True

            self.ground_truth_type = np.zeros((self.n_rollout_threads, self.num_agents, self.belief_shape))
            if self.use_linear_lr_decay:
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(episode)

            self.prep_rollout()
            if self.random_adversary:                
                self.agent_adversary = np.random.choice(range(self.num_agents))
            if self.adapt_adversary:
                self.severity_ind = np.random.choice(range(self.n_severity_types), p=softmax(-1 * self.adapt_adv_probs / 1))
            else:
                self.severity_ind = np.random.choice(range(self.n_severity_types))
            if episode % self.victim_interval == 0:  # which means some episodes are not adversary
                self.episode_adversary = (np.random.rand(self.n_rollout_threads) < self.adv_prob)
            else:
                self.episode_adversary = (np.random.rand(self.n_rollout_threads) < 2)  # all True
            
            # kepts the same size as belief for convinence
            self.ground_truth_type[self.episode_adversary, :, self.agent_adversary] = 1
            self.ground_truth_type[self.episode_adversary, :, -self.n_severity_types + self.severity_ind] = 1

            if self.baseline_policy is not None:
                self.baseline_ground_truth_type.fill(0)
                self.baseline_ground_truth_type[:, :, self.agent_adversary] = 1
                self.baseline_ground_truth_type[:, :, -self.n_severity_types + self.severity_ind] = 1
            
            for step in range(self.episode_length):
                main_actions_and_stats = self._collect_main_step(step)
                main_input_actions = main_actions_and_stats[0]

                if self.baseline_policy is not None:
                    baseline_actions_and_stats = self._collect_baseline_step(step)
                    baseline_input_actions = baseline_actions_and_stats[0]

                    # Dispatch both env steps before waiting so subprocess workers overlap.
                    self.envs.step_async(main_input_actions)
                    self.baseline_envs.step_async(baseline_input_actions)
                    main_step_output = self.envs.step_wait()
                    baseline_step_output = self.baseline_envs.step_wait()

                    data = self._build_step_data(main_step_output, self.ground_truth_type, *main_actions_and_stats[1:])
                    baseline_data = self._build_step_data(baseline_step_output, self.baseline_ground_truth_type, *baseline_actions_and_stats[1:])

                    self.logger.per_step(data)
                    self.insert(data)
                    self.baseline_insert(baseline_data)
                else:
                    self.envs.step_async(main_input_actions)
                    main_step_output = self.envs.step_wait()
                    data = self._build_step_data(main_step_output, self.ground_truth_type, *main_actions_and_stats[1:])

                    self.logger.per_step(data)
                    self.insert(data)

            # compute return and update network
            self.compute()
            self.prep_training()

            actor_train_infos, critic_train_info = self.share_param_train()  # train adversary and victim
            # log information
            if episode % self.log_interval == 0:
                self.logger.episode_log(
                    actor_train_infos, critic_train_info, self.actor_buffer, self.critic_buffer)

            # eval
            if episode % self.eval_interval == 0:
                if self.use_eval:
                    self.prep_rollout()
                    self.eval()
                    if self.env_name not in ['pursuit']:
                        self.eval_adv()
                        if self.baseline_policy is not None:
                            self.baseline_eval_adv()
                else:
                    self.save()

            self.after_update()

    @torch.no_grad()
    def _collect_main_step(self, step):
        values, actions, adv_actions, action_log_probs, adv_action_log_probs, rnn_states, \
            adv_rnn_states, belief_rnn_states, rnn_states_critic = self.collect_adv(step)
        input_actions = actions.copy()
        input_actions[self.episode_adversary, self.agent_adversary] = adv_actions[self.episode_adversary, self.agent_adversary]
        return input_actions, values, actions, adv_actions, action_log_probs, adv_action_log_probs, \
            rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic

    @torch.no_grad()
    def _collect_baseline_step(self, step):
        values, actions, adv_actions, action_log_probs, adv_action_log_probs, rnn_states, \
            adv_rnn_states, belief_rnn_states, rnn_states_critic = self.baseline_collect_adv(step)
        input_actions = actions.copy()
        input_actions[:, self.agent_adversary] = adv_actions[:, self.agent_adversary]
        return input_actions, values, actions, adv_actions, action_log_probs, adv_action_log_probs, \
            rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic

    def _build_step_data(self, step_output, ground_truth_type, values, actions, adv_actions,
                         action_log_probs, adv_action_log_probs, rnn_states, adv_rnn_states,
                         belief_rnn_states, rnn_states_critic):
        obs, share_obs, rewards, dones, infos, available_actions = step_output

        obs = self._pad_obs(obs, self.belief_shape)
        if self.stack_share_obs:
            share_obs = self._stack_obs(share_obs)
        share_obs = self._pad_obs(share_obs, 2 * self.belief_shape)

        data = obs, share_obs, rewards, dones, infos, ground_truth_type, available_actions, \
            values, actions, adv_actions, action_log_probs, adv_action_log_probs, \
            rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic

        return data

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        obs = self._pad_obs(obs, self.belief_shape)
        if self.stack_share_obs:
            share_obs = self._stack_obs(share_obs)
        share_obs = self._pad_obs(share_obs, 2 * self.belief_shape) 
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

    def baseline_warmup(self):
        # reset env
        obs, share_obs, available_actions = self.baseline_envs.reset()
        obs = self._pad_obs(obs, self.belief_shape)
        if self.stack_share_obs:
            share_obs = self._stack_obs(share_obs)
        share_obs = self._pad_obs(share_obs, 2 * self.belief_shape) 
        # replay buffer
        for agent_id in range(self.num_agents):
            self.baseline_actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.baseline_actor_buffer[agent_id].available_actions is not None:
                self.baseline_actor_buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()
        if self.state_type == "EP":
            self.baseline_critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.baseline_critic_buffer.share_obs[0] = share_obs.copy()        
           

    @torch.no_grad()
    def baseline_collect_adv(self, step):
        belief_collector = []
        action_collector = []
        adv_action_collector = []
        action_log_prob_collector = []
        adv_action_log_prob_collector = []
        rnn_state_collector = []
        adv_rnn_state_collector = []
        belief_rnn_state_collector = []
        for agent_id in range(self.num_agents):
            belief_for_baseline= np.zeros((self.n_rollout_threads, self.num_agents))
            self.baseline_actor_buffer[agent_id].obs[step][:, -self.belief_shape: - self.n_severity_types] = _t2n(belief_for_baseline) #baseline network does not use belief

            action, action_log_prob, rnn_state = self.baseline_actor[agent_id].get_actions(self.baseline_actor_buffer[agent_id].obs[step][:, :- self.n_severity_types],
                                                                                  self.baseline_actor_buffer[agent_id].rnn_states[step],
                                                                                  self.baseline_actor_buffer[agent_id].masks[step],
                                                                                  self.baseline_actor_buffer[agent_id].available_actions[step] if self.baseline_actor_buffer[agent_id].available_actions is not None else None)

            belief = self.baseline_ground_truth_type[:, agent_id]
            belief_rnn_state= np.zeros((self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
            
            belief_collector.append(_t2n(belief))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            belief_rnn_state_collector.append(_t2n(belief_rnn_state))
        
        for agent_id in range(self.num_agents):
            adv_obs = self.baseline_actor_buffer[agent_id].obs[step].copy()
            adv_obs[:, -self.belief_shape: -self.n_severity_types] = np.eye(self.num_agents)[agent_id]
            adv_obs[:, -self.n_severity_types:] = np.eye(self.n_severity_types)[self.severity_ind]
            if self.super_adversary:
                def_act = np.concatenate([*action_collector[-self.num_agents:][:agent_id], 
                                          *action_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                adv_obs = np.concatenate([adv_obs, softmax(def_act)], axis=-1)
            
            adv_action, adv_action_log_prob, adv_rnn_state = self.actor[agent_id].get_adv_actions(adv_obs,
                                                                                                  self.baseline_actor_buffer[agent_id].adv_rnn_states[step],
                                                                                                  self.baseline_actor_buffer[agent_id].masks[step],
                                                                                                  self.baseline_actor_buffer[agent_id].available_actions[step] if self.baseline_actor_buffer[agent_id].available_actions is not None else None,
                                                                                                  severity_ind=self.severity_ind)
            adv_action_collector.append(_t2n(adv_action))
            adv_action_log_prob_collector.append(_t2n(adv_action_log_prob))
            adv_rnn_state_collector.append(_t2n(adv_rnn_state))
        # [self.envs, agents, dim]
        belief = np.array(belief_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        adv_actions = np.array(adv_action_collector).transpose(1, 0, 2)
        adv_action_log_probs = np.array(adv_action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        belief_rnn_states = np.array(belief_rnn_state_collector).transpose(1, 0, 2, 3)
        adv_rnn_states = np.array(adv_rnn_state_collector).transpose(1, 0, 2, 3)

        if self.central_belief_option == 'mean':
            belief_central = belief.mean(axis=1)
            belief_central = np.expand_dims(belief_central, axis=1).repeat(self.num_agents, axis=1)
        else:
            belief_central = belief

        if self.state_type == "EP":            
            self.baseline_critic_buffer.share_obs[step][:, -2*self.belief_shape: -self.belief_shape] = belief_central.mean(axis=1)
            self.baseline_critic_buffer.share_obs[step][:, -self.belief_shape:] = self.baseline_ground_truth_type[:, 0, :]


            # need to change to be compatible to our setting?
            value, rnn_state_critic = self.baseline_critic.get_values(np.concatenate(self.baseline_critic_buffer.share_obs[step]),
                                                             np.concatenate(self.baseline_critic_buffer.rnn_states_critic[step]),
                                                             np.concatenate(self.baseline_critic_buffer.masks[step]))
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        elif self.state_type == "FP":            
            self.baseline_critic_buffer.share_obs[step][:, :, -2*self.belief_shape: -self.belief_shape] = belief_central
            self.baseline_critic_buffer.share_obs[step][:, :, -self.belief_shape:] = self.baseline_ground_truth_type

            value, rnn_state_critic = self.baseline_critic.get_values(np.concatenate(self.baseline_critic_buffer.share_obs[step]),
                                                             np.concatenate(self.baseline_critic_buffer.rnn_states_critic[step]),
                                                             np.concatenate(self.baseline_critic_buffer.masks[step]))
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, adv_actions, action_log_probs, adv_action_log_probs, rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic


    @torch.no_grad()
    def collect_adv(self, step):
        belief_collector = []
        action_collector = []
        adv_action_collector = []
        action_log_prob_collector = []
        adv_action_log_prob_collector = []
        rnn_state_collector = []
        belief_rnn_state_collector = []
        adv_rnn_state_collector = []
        for agent_id in range(self.num_agents):
            belief, belief_rnn_state = self.actor[agent_id].get_belief(self.actor_buffer[agent_id].obs[step],
                                                                       self.actor_buffer[agent_id].belief_rnn_states[step],
                                                                       self.actor_buffer[agent_id].masks[step])
            if self.teacher_forcing and np.random.rand() < self.true_type_prob:
                belief = self.ground_truth_type[:, agent_id]
            self.actor_buffer[agent_id].obs[step][:, -self.belief_shape:] = _t2n(belief)

            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(self.actor_buffer[agent_id].obs[step],
                                                                                  self.actor_buffer[agent_id].rnn_states[step],
                                                                                  self.actor_buffer[agent_id].masks[step],
                                                                                  self.actor_buffer[agent_id].available_actions[step] if self.actor_buffer[agent_id].available_actions is not None else None)
            belief_collector.append(_t2n(belief))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            belief_rnn_state_collector.append(_t2n(belief_rnn_state))
        
        for agent_id in range(self.num_agents):
            adv_obs = self.actor_buffer[agent_id].obs[step].copy()
            adv_obs[:, -self.belief_shape: -self.n_severity_types] = np.eye(self.num_agents)[agent_id]
            adv_obs[:, -self.n_severity_types:] = np.eye(self.n_severity_types)[self.severity_ind]
            if self.super_adversary:
                def_act = np.concatenate([*action_collector[-self.num_agents:][:agent_id], 
                                          *action_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                adv_obs = np.concatenate([adv_obs, softmax(def_act)], axis=-1)

            adv_action, adv_action_log_prob, adv_rnn_state = self.actor[agent_id].get_adv_actions(adv_obs,
                                                                                                  self.actor_buffer[agent_id].adv_rnn_states[step],
                                                                                                  self.actor_buffer[agent_id].masks[step],
                                                                                                  self.actor_buffer[agent_id].available_actions[step] if self.actor_buffer[agent_id].available_actions is not None else None,
                                                                                                  severity_ind=self.severity_ind)
            adv_action_collector.append(_t2n(adv_action))
            adv_action_log_prob_collector.append(_t2n(adv_action_log_prob))
            adv_rnn_state_collector.append(_t2n(adv_rnn_state))
        # [self.envs, agents, dim]
        belief = np.array(belief_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        adv_actions = np.array(adv_action_collector).transpose(1, 0, 2)
        adv_action_log_probs = np.array(adv_action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        belief_rnn_states = np.array(belief_rnn_state_collector).transpose(1, 0, 2, 3)
        adv_rnn_states = np.array(adv_rnn_state_collector).transpose(1, 0, 2, 3)


        if self.central_belief_option == 'mean':
            belief_central = belief.mean(axis=1)
            belief_central = np.expand_dims(belief_central, axis=1).repeat(self.num_agents, axis=1)
        else:
            belief_central = belief

        if self.state_type == "EP":
            if self.teacher_forcing and np.random.rand() < self.true_type_prob:
                belief_central = self.ground_truth_type
            self.critic_buffer.share_obs[step][:, -2*self.belief_shape: -self.belief_shape] = belief_central.mean(axis=1)
            self.critic_buffer.share_obs[step][:, -self.belief_shape:] = self.ground_truth_type[:, 0, :]

            # need to change to be compatible to our setting?
            value, rnn_state_critic = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[step]),
                                                             np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                                                             np.concatenate(self.critic_buffer.masks[step]))
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        elif self.state_type == "FP":
            if self.teacher_forcing and np.random.rand() < self.true_type_prob:
                belief_central = self.ground_truth_type
            self.critic_buffer.share_obs[step][:, :, -2*self.belief_shape: -self.belief_shape] = belief_central
            self.critic_buffer.share_obs[step][:, :, -self.belief_shape:] = self.ground_truth_type

            value, rnn_state_critic = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[step]),
                                                             np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                                                             np.concatenate(self.critic_buffer.masks[step]))
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, adv_actions, action_log_probs, adv_action_log_probs, rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, ground_truth_type, available_actions, \
                    values, actions, adv_actions, action_log_probs, adv_action_log_probs, \
                    rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        adv_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        belief_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        # masks use 0 to mask out threads that just finish
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        adv_active_masks = active_masks.copy()
        active_masks[self.episode_adversary, self.agent_adversary] = 0
        adv_active_masks[~self.episode_adversary] = 0
        adv_active_masks[:, np.arange(self.num_agents)!=self.agent_adversary] = 0

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array([[0.0] if "bad_transition" in info[0].keys() and info[0]["bad_transition"] == True else [1.0] for info in infos])
        elif self.state_type == "FP":
            bad_masks = np.array([[[0.0] if "bad_transition" in info[agent_id].keys(
            ) and info[agent_id]['bad_transition'] == True else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(obs[:, agent_id], ground_truth_type[:, agent_id], rnn_states[:, agent_id], adv_rnn_states[:, agent_id], belief_rnn_states[:, agent_id], actions[:, agent_id], adv_actions[:, agent_id],
                                               action_log_probs[:, agent_id], adv_action_log_probs[:, agent_id], rewards[:, agent_id], masks[:, agent_id], active_masks[:, agent_id],
                                               adv_active_masks[:, agent_id], available_actions[:, agent_id] if available_actions[0] is not None else None)

        if self.state_type == "EP":
            self.critic_buffer.insert(share_obs, rnn_states_critic, values, rewards, masks, bad_masks)
        elif self.state_type == "FP":
            self.critic_buffer.insert(share_obs, rnn_states_critic, values, rewards, masks, bad_masks)

    def baseline_insert(self, data):
        obs, share_obs, rewards, dones, infos, ground_truth_type, available_actions, \
                    values, actions, adv_actions, action_log_probs, adv_action_log_probs, \
                    rnn_states, adv_rnn_states, belief_rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)


        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        adv_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        belief_rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        # masks use 0 to mask out threads that just finish
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        adv_active_masks = active_masks.copy()
        active_masks[:, self.agent_adversary] = 0
        adv_active_masks[:, np.arange(self.num_agents)!=self.agent_adversary] = 0

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array([[0.0] if "bad_transition" in info[0].keys() and info[0]["bad_transition"] == True else [1.0] for info in infos])
        elif self.state_type == "FP":
            bad_masks = np.array([[[0.0] if "bad_transition" in info[agent_id].keys(
            ) and info[agent_id]['bad_transition'] == True else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        for agent_id in range(self.num_agents):
            self.baseline_actor_buffer[agent_id].insert(obs[:, agent_id], ground_truth_type[:, agent_id], rnn_states[:, agent_id], adv_rnn_states[:, agent_id], belief_rnn_states[:, agent_id], actions[:, agent_id], adv_actions[:, agent_id],
                                               action_log_probs[:, agent_id], adv_action_log_probs[:, agent_id], rewards[:, agent_id], masks[:, agent_id], active_masks[:, agent_id],
                                               adv_active_masks[:, agent_id], available_actions[:, agent_id] if available_actions[0] is not None else None)

        if self.state_type == "EP":
            self.baseline_critic_buffer.insert(share_obs, rnn_states_critic, values, rewards, masks, bad_masks)
        elif self.state_type == "FP":
            self.baseline_critic_buffer.insert(share_obs, rnn_states_critic, values, rewards, masks, bad_masks)            

    @torch.no_grad()    
    def compute(self):
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[-1]),
                                                   np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                                                   np.concatenate(self.critic_buffer.masks[-1]))
            next_value = np.array(
                np.split(_t2n(next_value), self.n_rollout_threads))
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(np.concatenate(self.critic_buffer.share_obs[-1]),
                                                   np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                                                   np.concatenate(self.critic_buffer.masks[-1]))
            next_value = np.array(
                np.split(_t2n(next_value), self.n_rollout_threads))
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)
        if self.baseline_policy is not None:
            if self.state_type == "EP":
                next_value, _ = self.baseline_critic.get_values(np.concatenate(self.baseline_critic_buffer.share_obs[-1]),
                                                                np.concatenate(self.baseline_critic_buffer.rnn_states_critic[-1]),
                                                                np.concatenate(self.baseline_critic_buffer.masks[-1]))
                next_value = np.array(
                    np.split(_t2n(next_value), self.n_rollout_threads))
            elif self.state_type == "FP":
                next_value, _ = self.baseline_critic.get_values(np.concatenate(self.baseline_critic_buffer.share_obs[-1]),
                                                                np.concatenate(self.baseline_critic_buffer.rnn_states_critic[-1]),
                                                                np.concatenate(self.baseline_critic_buffer.masks[-1]))
                next_value = np.array(
                    np.split(_t2n(next_value), self.n_rollout_threads))
            self.baseline_critic_buffer.compute_returns(next_value, self.baseline_value_normalizer)
            self.baseline_mean_value = self.baseline_critic_buffer.get_mean_episodic_value()



    def share_param_train(self):
        """
        Training procedure for parameter-sharing MAPPO.
        """

        actor_train_infos = []

        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - \
                self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        if self.state_type == "FP":
            active_masks_collector = [self.actor_buffer[i].active_masks for i in range(self.num_agents)]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        if self.eval_critic_landscape:
            self.critic.visualize_critic_value_landscape(self.critic_buffer, self.value_normalizer)

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        

        if self.baseline_policy is not None:
            if self.value_normalizer is not None:
                baseline_advantages = self.baseline_critic_buffer.returns[:-1] - \
                    self.baseline_value_normalizer.denormalize(self.baseline_critic_buffer.value_preds[:-1])
            else:
                baseline_advantages = self.baseline_critic_buffer.returns[:-1] - self.baseline_critic_buffer.value_preds[:-1]

            if self.state_type == "FP":
                baseline_active_masks_collector = [self.baseline_actor_buffer[i].active_masks for i in range(self.num_agents)]
                baseline_active_masks_array = np.stack(baseline_active_masks_collector, axis=2)
                baseline_advantages_copy = baseline_advantages.copy()
                baseline_advantages_copy[baseline_active_masks_array[:-1] == 0.0] = np.nan
                mean_baseline_advantages = np.nanmean(baseline_advantages_copy)
                std_baseline_advantages = np.nanstd(baseline_advantages_copy)
                baseline_advantages = (baseline_advantages - mean_baseline_advantages) / (std_baseline_advantages + 1e-5)

            if self.eval_critic_landscape:
                self.baseline_critic.visualize_critic_value_landscape(self.baseline_critic_buffer, self.baseline_value_normalizer)

            critic_train_info_baseline = self.baseline_critic.train(self.baseline_critic_buffer, self.baseline_value_normalizer)

            actor_train_info = self.actor[0].share_param_train(self.actor_buffer, advantages.copy(), self.num_agents, self.state_type, 
                                                               self.baseline_actor_buffer, baseline_advantages.copy(), self.baseline_mean_value, self.severity_ind)

        else:
            actor_train_info = self.actor[0].share_param_train(self.actor_buffer, advantages.copy(), self.num_agents, self.state_type)

        
        if not self.forced_no_belief:
            actor_train_info_belief = self.actor[0].share_param_train_belief(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type)
            actor_train_info.update(actor_train_info_belief)    
                   
 
        for agent_id in torch.randperm(self.num_agents):
            actor_train_infos.append(actor_train_info)

        return actor_train_infos, critic_train_info



    def after_update(self):
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()
        if self.baseline_policy is not None:
            for agent_id in range(self.num_agents):
                self.baseline_actor_buffer[agent_id].after_update()
            self.baseline_critic_buffer.after_update()
            

    @torch.no_grad()
    def eval(self):
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = self._pad_obs(eval_obs, self.belief_shape)
        if self.stack_share_obs:
            eval_share_obs = self._stack_obs(eval_share_obs)
        eval_share_obs = self._pad_obs(eval_share_obs, 2 * self.belief_shape)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        ground_truth_type = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.belief_shape))

        eval_rnn_states_critic = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        while True:
            eval_actions_collector = []
            eval_belief_collector = []
            
            for agent_id in range(self.num_agents):
                eval_belief, temp_rnn_state_belief = \
                    self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                             eval_rnn_states_belief[:, agent_id],
                                             eval_masks[:, agent_id])
                eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                eval_belief_collector.append(_t2n(eval_belief))

                if self.teacher_forcing and np.random.rand() < self.true_type_prob:
                    eval_belief = ground_truth_type[:, agent_id]
                eval_obs[:, agent_id, -self.belief_shape:] = _t2n(eval_belief)
                eval_share_obs[:, agent_id, -2*self.belief_shape:-self.belief_shape] = ground_truth_type[:, agent_id]
                eval_share_obs[:, agent_id, -self.belief_shape:] = ground_truth_type[:, agent_id]

                eval_actions, temp_rnn_state = \
                    self.actor[agent_id].act(eval_obs[:, agent_id],
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                             deterministic=False)
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            if self.use_critic_for_value:
                value, eval_rnn_states_critic = self.critic.get_values(np.concatenate(eval_share_obs),
                                                                       np.concatenate(eval_rnn_states_critic),
                                                                       np.concatenate(eval_masks))
                eval_values = np.array(np.split(_t2n(value), self.n_eval_rollout_threads))
                if self.value_normalizer is not None:
                    eval_values = self.value_normalizer.denormalize(eval_values)
                eval_value = np.mean(eval_values, axis=1)
                eval_rnn_states_critic = np.array(np.split(_t2n(eval_rnn_states_critic), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            eval_obs = self._pad_obs(eval_obs, self.belief_shape)
            if self.stack_share_obs:
                eval_share_obs = self._stack_obs(eval_share_obs)
            eval_share_obs = self._pad_obs(eval_share_obs, 2 * self.belief_shape)

            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
            eval_rnn_states_belief[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
            eval_rnn_states_critic[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)            

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    next_value = eval_value[eval_i].item() if self.use_critic_for_value else None
                    self.logger.eval_thread_done(eval_i, next_value)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                save_model = self.logger.eval_log(eval_episode)
                if save_model and self.env_name != "toy":
                    self.save()
                break

    @torch.no_grad()
    def eval_adv(self):
        for ind in range(self.n_severity_types):
            self._eval_adv(ind)

    @torch.no_grad()
    def _eval_adv(self, severity_ind):
        if self.random_adversary:
            adv_id = np.random.randint(0, self.num_agents, size=self.n_eval_rollout_threads)
        else:
            adv_id = np.full(self.n_eval_rollout_threads, self.agent_adversary)
        
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = self._pad_obs(eval_obs, self.belief_shape)
        if self.stack_share_obs:
            eval_share_obs = self._stack_obs(eval_share_obs)
        eval_share_obs = self._pad_obs(eval_share_obs, 2 * self.belief_shape)

        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)

        eval_rnn_states_critic = np.zeros((eval_obs.shape[0], self.num_agents,
                                            self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        while True:
            ground_truth_type = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.belief_shape))
            ground_truth_type[np.arange(self.n_eval_rollout_threads)[:, None], np.arange(self.num_agents), adv_id[:, None]] = 1
            ground_truth_type[:, :, -self.n_severity_types + severity_ind] = 1

            eval_actions_collector = []
            eval_adv_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_belief, temp_rnn_state_belief = \
                    self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                
                if self.teacher_forcing and np.random.rand() < self.true_type_prob:
                    eval_belief = ground_truth_type[:, agent_id]
                eval_obs[:, agent_id, -self.belief_shape:] = _t2n(eval_belief)
                eval_share_obs[:, agent_id, -2*self.belief_shape:-self.belief_shape] = _t2n(eval_belief)
                eval_share_obs[:, agent_id, -self.belief_shape:] = ground_truth_type[:, agent_id]

                eval_actions, temp_rnn_state = \
                    self.actor[agent_id].act(eval_obs[:, agent_id],
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                             deterministic=False)
                    
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            for agent_id in range(self.num_agents):
                # adv_obs = eval_obs[:, agent_id].copy()
                adv_obs = eval_obs[:, agent_id].copy()
                adv_obs[:, -self.belief_shape: -self.n_severity_types] = np.eye(self.num_agents)[agent_id]
                adv_obs[:, -self.n_severity_types:] = np.eye(self.n_severity_types)[severity_ind]

                # adv_obs[:, -self.num_agents:] = np.eye(self.num_agents)[agent_id]
                if self.super_adversary:
                    def_act = np.concatenate([*eval_actions_collector[-self.num_agents:][:agent_id], 
                                              *eval_actions_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([adv_obs, softmax(def_act)], axis=-1)
                eval_adv_actions, temp_adv_rnn_state = \
                    self.actor[agent_id].act_adv(adv_obs,
                                                 eval_adv_rnn_states[:, agent_id],
                                                 eval_masks[:, agent_id],
                                                 eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                 deterministic=False,
                                                 severity_ind=severity_ind)
                eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                eval_adv_actions_collector.append(_t2n(eval_adv_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)

            eval_actions[np.arange(eval_actions.shape[0]), adv_id] = eval_adv_actions[np.arange(eval_actions.shape[0]), adv_id]

            if self.use_critic_for_value:
                value, eval_rnn_states_critic = self.critic.get_values(np.concatenate(eval_share_obs),
                                                                       np.concatenate(eval_rnn_states_critic),
                                                                       np.concatenate(eval_masks))
                eval_values = np.array(np.split(_t2n(value), self.n_eval_rollout_threads))
                if self.value_normalizer is not None:
                    eval_values = self.value_normalizer.denormalize(eval_values)
                eval_value = np.mean(eval_values, axis=1)
                eval_rnn_states_critic = np.array(np.split(_t2n(eval_rnn_states_critic), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            eval_obs = self._pad_obs(eval_obs, self.belief_shape)
            if self.stack_share_obs:
                eval_share_obs = self._stack_obs(eval_share_obs)
            eval_share_obs = self._pad_obs(eval_share_obs, 2 * self.belief_shape)


            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
            eval_rnn_states_belief[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)            
            eval_rnn_states_critic[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    next_value = eval_value[eval_i].item() if self.use_critic_for_value else None
                    self.logger.eval_thread_done(eval_i, next_value)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                ret_mean = self.logger.eval_log_severity(eval_episode, severity_ind)
                break

        self.adapt_adv_probs[severity_ind] = np.mean(self.logger.eval_episode_rewards)
   

    @torch.no_grad()
    def render(self):
        print("start rendering")
        obs_traj = []
        action_traj = []
        action_prob_traj = []
        belief_traj = []
        done_traj = []

        features = []
        hooks = []
        def hook_fn(module, input, output):
            features.append(output.detach().cpu().numpy())

        if "ddpg" in self.args.algo:
            for ii in range(self.num_agents):
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[1].register_forward_hook(hook_fn))
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[3].register_forward_hook(hook_fn))
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[5].register_forward_hook(hook_fn))
        else:
            hooks.append(self.actor[0].actor.base.mlp.fc[2].register_forward_hook(hook_fn))
            hooks.append(self.actor[0].actor.base.mlp.fc[5].register_forward_hook(hook_fn))
            hooks.append(self.actor[0].actor.base.mlp.fc[8].register_forward_hook(hook_fn))

        if self.manual_expand_dims:
            for _ in range(self.render_episodes):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = np.expand_dims(np.array(
                    eval_available_actions), axis=0) if eval_available_actions is not None else None
                eval_rnn_states = np.zeros((self.env_num, self.num_agents,
                                            self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_rnn_states_belief = np.zeros((self.env_num, self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)
                rewards = 0
                while True:
                    eval_actions_collector = []
                    eval_action_probs_collector = []
                    eval_belief_collector = []
                    for agent_id in range(self.num_agents):
                        eval_belief, temp_rnn_state_belief = \
                                            self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                        eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                        eval_belief_collector.append(_t2n(eval_belief))

                        eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)
                        eval_actions, eval_action_probs, temp_rnn_state = \
                            self.actor[agent_id].act_with_probs(eval_obs[:, agent_id],
                                                                eval_rnn_states[:, agent_id],
                                                                eval_masks[:, agent_id],
                                                                eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                                deterministic=False)
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                        eval_action_probs_collector.append(_t2n(eval_action_probs))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    eval_action_probs = np.array(eval_action_probs_collector).transpose(1, 0, 2)
                    eval_belief = np.array(eval_belief_collector).transpose(1, 0, 2)

                    obs_traj.append(eval_obs)
                    action_traj.append(eval_actions)
                    action_prob_traj.append(eval_action_probs)
                    belief_traj.append(eval_belief)


                    # Obser reward and next obs
                    eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions = self.envs.step(
                        eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = np.expand_dims(np.array(
                        eval_available_actions), axis=0) if eval_available_actions is not None else None
                    if self.manual_render:
                        if "smac" not in self.args.env:  # replay for smac, no rendering
                            self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        done_traj.append(True)
                        print(f"total reward of this episode: {rewards}")
                        break
                    else:
                        done_traj.append(False)
            if "smac" in self.args.env:
                if 'v2' in self.args.env:
                    self.envs.env.save_replay()
                else:
                    self.envs.save_replay()

        else:
            raise NotImplementedError

        # np.save("traj/obs.npy", np.array(obs_traj))
        # np.save("traj/action.npy", np.array(action_traj))
        # np.save("traj/action_prob.npy", np.array(action_prob_traj))
        # np.save("traj/belief.npy", np.array(belief_traj))
        # np.save("traj/done.npy", np.array(done_traj))

        for hook in hooks:
            hook.remove()
        features_cat = []
        for i in range(len(features)//3):
            features_cat.append(np.concatenate([features[3*i], features[3*i+1], features[3*i+2]], axis=1))
        features_cat = np.concatenate(features_cat)

        np.save(f"traj/{self.args.exp_name}_features.npy", features_cat)

    @torch.no_grad()
    def render_adv(self):
        if self.random_adversary:
            for i in range(self.num_agents):
                self._render_adv(i)
        else:
            self._render_adv(self.agent_adversary)

    @torch.no_grad()
    def _render_adv(self, adv_id):
        print("start adv rendering")
        obs_traj = []
        action_traj = []
        action_prob_traj = []
        belief_traj = []
        done_traj = []

        features = []
        hooks = []
        def hook_fn(module, input, output):
            features.append(output.detach().cpu().numpy())

        if "ddpg" in self.args.algo:
            for ii in range(self.num_agents):
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[1].register_forward_hook(hook_fn))
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[3].register_forward_hook(hook_fn))
                hooks.append(self.actor[0].maddpg[ii].actor.pi.mlp[5].register_forward_hook(hook_fn))
        else:
            hooks.append(self.actor[0].actor.base.mlp.fc[2].register_forward_hook(hook_fn))
            hooks.append(self.actor[0].actor.base.mlp.fc[5].register_forward_hook(hook_fn))
            hooks.append(self.actor[0].actor.base.mlp.fc[8].register_forward_hook(hook_fn))

        if self.manual_expand_dims:
            for _ in range(self.render_episodes):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)

                eval_available_actions = np.expand_dims(np.array(
                    eval_available_actions), axis=0) if eval_available_actions is not None else None
                eval_rnn_states = np.zeros((self.env_num, self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_rnn_states_belief = np.zeros((self.env_num, self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_adv_rnn_states = np.zeros((self.env_num, self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)
                rewards = 0
                while True:
                    eval_actions_collector = []
                    eval_adv_actions_collector = []
                    eval_action_probs_collector = []
                    eval_belief_collector = []
                    for agent_id in range(self.num_agents):
                        eval_belief, temp_rnn_state_belief = \
                                            self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                            eval_rnn_states_belief[:, agent_id],
                                            eval_masks[:, agent_id])
                        eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)
                        eval_belief_collector.append(_t2n(eval_belief))

                        eval_obs[:, agent_id, -self.num_agents:] = _t2n(eval_belief)
                        eval_actions, eval_action_probs, temp_rnn_state = \
                            self.actor[agent_id].act_with_probs(eval_obs[:, agent_id],
                                                                eval_rnn_states[:, agent_id],
                                                                eval_masks[:, agent_id],
                                                                eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                                deterministic=False)
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                        eval_action_probs_collector.append(_t2n(eval_action_probs))

                    for agent_id in range(self.num_agents):
                        if self.super_adversary:
                            def_act = np.concatenate([*eval_actions_collector[-self.num_agents:][:agent_id], 
                                                    *eval_actions_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                            adv_obs = np.concatenate([eval_obs[:, agent_id], softmax(def_act)], axis=-1)
                        else:
                            adv_obs = eval_obs[:, agent_id]
                        eval_adv_actions, temp_adv_rnn_state = \
                            self.actor[agent_id].act_adv(adv_obs,
                                                    eval_adv_rnn_states[:, agent_id],
                                                    eval_masks[:, agent_id],
                                                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                    deterministic=False)
                        eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                        eval_adv_actions_collector.append(_t2n(eval_adv_actions))

                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)
                    eval_action_probs = np.array(eval_action_probs_collector).transpose(1, 0, 2)
                    eval_belief = np.array(eval_belief_collector).transpose(1, 0, 2)

                    eval_actions[:, adv_id] = eval_adv_actions[:, adv_id]

                    obs_traj.append(eval_obs)
                    action_traj.append(eval_actions)
                    action_prob_traj.append(eval_action_probs)
                    belief_traj.append(eval_belief)
                    # Obser reward and next obs
                    eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions = self.envs.step(
                        eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)

                    eval_available_actions = np.expand_dims(np.array(
                    eval_available_actions), axis=0) if eval_available_actions is not None else None
                    if self.manual_render:
                        if "smac" not in self.args.env:  # replay for smac, no rendering
                            self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        done_traj.append(True)
                        print(f"total reward of this episode: {rewards}")
                        break
                    else:
                        done_traj.append(False)
            if "smac" in self.args.env:
                if 'v2' in self.args.env:
                    self.envs.env.save_replay()
                else:
                    self.envs.save_replay()
        else:
            raise NotImplementedError
        
        # np.save("traj/obs.npy", np.array(obs_traj))
        # np.save("traj/action.npy", np.array(action_traj))
        # np.save("traj/action_prob.npy", np.array(action_prob_traj))
        # np.save("traj/belief.npy", np.array(belief_traj))
        # np.save("traj/done.npy", np.array(done_traj))

        for hook in hooks:
            hook.remove()
        features_cat = []
        for i in range(len(features)//3):
            features_cat.append(np.concatenate([features[3*i], features[3*i+1], features[3*i+2]], axis=1))
        features_cat = np.concatenate(features_cat)

        np.save(f"traj/{self.args.exp_name}_adv{adv_id}_features.npy", features_cat)


    def prep_rollout(self):
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()
        if self.baseline_policy is not None:
            for agent_id in range(self.num_agents):
                self.baseline_actor[agent_id].prep_rollout()
            self.baseline_critic.prep_rollout()    

    def prep_training(self):
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()
        if self.baseline_policy is not None:            
            self.baseline_critic.prep_training()

    def save(self):
        step_num = self.logger.total_num_steps
        checkpoint = (step_num // self.save_interval) * self.save_interval
        save_dir = str(self.save_dir) + "/" + str(checkpoint)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        policy_actor = self.actor[0].actor
        torch.save(policy_actor.state_dict(), str(
            save_dir) + "/actor_agent" + ".pt")

        policy_belief = self.actor[0].belief
        torch.save(policy_belief.state_dict(), str(
            save_dir) + "/actor_belief" + ".pt")
        for ind in range(self.n_severity_types):
            adv_policy_actor = self.actor[0].adv_actors[ind]
            torch.save(adv_policy_actor.state_dict(), str(
                save_dir) + "/adv_actor_agent_type" + str(ind) + ".pt")
        policy_critic = self.critic.critic
        torch.save(policy_critic.state_dict(), str(
            save_dir) + "/critic_agent" + ".pt")
        if self.value_normalizer is not None:
            torch.save(self.value_normalizer.state_dict(), str(
                save_dir) + "/value_normalizer" + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + '/actor_agent' + '.pt', map_location=torch.device('cpu'))
            self.actor[agent_id].actor.load_state_dict(
                policy_actor_state_dict)

            policy_belief_state_dict = torch.load(
                str(self.model_dir) + '/actor_belief' + '.pt', map_location=torch.device('cpu'))
            self.actor[agent_id].belief.load_state_dict(
                policy_belief_state_dict)
            if self.load_adv_actor:
                for ind in range(self.n_severity_types):
                    adv_policy_actor_state_dict = torch.load(
                        str(self.model_dir) + '/adv_actor_agent_type' + str(ind) + '.pt', map_location=torch.device('cpu'))
                    self.actor[agent_id].adv_actors[ind].load_state_dict(
                        adv_policy_actor_state_dict)

        if not self.use_render and self.load_critic:
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + '/critic_agent' + '.pt', map_location=torch.device('cpu'))
            self.critic.critic.load_state_dict(
                policy_critic_state_dict)
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(str(
                    self.model_dir) + "/value_normalizer" + ".pt", map_location=torch.device('cpu'))
                self.value_normalizer.load_state_dict(
                    value_normalizer_state_dict)
                
    def load_baseline(self):            
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.baseline_policy) + '/actor_agent' + str(agent_id) + '.pt', map_location=torch.device('cpu'))
            self.baseline_actor[agent_id].actor.load_state_dict(
                policy_actor_state_dict)

            policy_belief_state_dict = torch.load(
                str(self.baseline_policy) + '/actor_belief' + str(agent_id) + '.pt', map_location=torch.device('cpu'))
            self.baseline_actor[agent_id].belief.load_state_dict(
                policy_belief_state_dict)
            
    def close(self):
        # post process
        if self.use_render:
            self.envs.close()
        else:
            self.envs.close()
            if self.baseline_policy is not None:
                self.baseline_envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(
                str(self.log_dir + '/summary.json'))
            self.writter.close()
            self.logger.close()

    @torch.no_grad()
    def eval_adv_external(self):
        if self.random_adversary:
            for i in range(self.num_agents):
                self._eval_adv_external(i)
        else:
            self._eval_adv_external(self.agent_adversary)

    @torch.no_grad()
    def _eval_adv_external(self, adv_id):
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = self._pad_obs(eval_obs, self.belief_shape)
        if self.stack_share_obs:   
            eval_share_obs = self._stack_obs(eval_share_obs)
        eval_share_obs = self._pad_obs(eval_share_obs, 2 * self.belief_shape)

        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                    self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((eval_obs.shape[0], self.num_agents,
                                           self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)

        if self.evaluate_external:
            external_adv_hidden = torch.zeros((eval_obs.shape[0], self.external_adv_hidden_size), dtype=torch.float32)


        while True:
            eval_actions_collector = []
            eval_adv_actions_collector = []
            external_adv_obs = torch.tensor(eval_obs[:, adv_id, :-self.belief_shape], dtype=torch.float32) #-self.num_agents for SMAC
            #print(eval_obs.shape)
            if self.evaluate_external:
                external_adv_act, external_adv_hidden = self.external_adv.compute_action(external_adv_obs, external_adv_hidden, eval_available_actions[:, adv_id])
            for agent_id in range(self.num_agents):
                eval_belief, temp_rnn_state_belief = \
                    self.actor[agent_id].get_belief(eval_obs[:, agent_id],
                                                    eval_rnn_states_belief[:, agent_id],
                                                    eval_masks[:, agent_id])
                eval_rnn_states_belief[:, agent_id] = _t2n(temp_rnn_state_belief)

                eval_obs[:, agent_id, -self.belief_shape:] = _t2n(eval_belief)

                eval_actions, temp_rnn_state = \
                    self.actor[agent_id].act(eval_obs[:, agent_id],
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[
                                                                                        0] is not None else None,
                                             deterministic=False)

                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            for agent_id in range(self.num_agents):
                adv_obs = eval_obs[:, agent_id, :-self.n_severity_types].copy()
                adv_obs[:, -self.num_agents:] = np.eye(self.num_agents)[agent_id]
                if self.eval_adv_n_types > 0:
                    adv_type= np.zeros((self.n_eval_rollout_threads, self.eval_adv_n_types))
                    adv_type[:, self.eval_adv_type_ind] = 1
                    adv_obs = np.concatenate([adv_obs, adv_type], axis=-1)                
                if self.super_adversary:
                    def_act = np.concatenate([*eval_actions_collector[-self.num_agents:][:agent_id],
                                              *eval_actions_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([adv_obs, softmax(def_act)], axis=-1)
                if not self.evaluate_external:
                    eval_adv_actions, _, temp_adv_rnn_state = \
                        self.eval_adv_actor(adv_obs,
                                            eval_adv_rnn_states[:, agent_id],
                                            eval_masks[:, agent_id],
                                            eval_available_actions[:, agent_id] if eval_available_actions[
                                                                                        0] is not None else None,
                                            deterministic=False)
                    eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                    eval_adv_actions_collector.append(_t2n(eval_adv_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            if self.evaluate_external:
                eval_actions[:, adv_id] = external_adv_act
            else:
                eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)
                eval_actions[:, adv_id] = eval_adv_actions[:, adv_id]

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            eval_obs = self._pad_obs(eval_obs, self.belief_shape)
            if self.stack_share_obs:
                eval_share_obs = self._stack_obs(eval_share_obs)
            eval_share_obs = self._pad_obs(eval_share_obs, 2 * self.belief_shape)

            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

            if self.evaluate_external:
                external_adv_hidden[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(
                ), self.external_adv_hidden_size), dtype=torch.float32)

            eval_masks = np.ones(
                (eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                ret_mean = self.logger.eval_log_adv(eval_episode, adv_id)
                break



    @torch.no_grad()
    def baseline_eval_adv(self):
        for ind in range(self.n_severity_types):
            self._baseline_eval_adv(ind)

    @torch.no_grad()
    def _baseline_eval_adv(self, severity_ind):
        if self.random_adversary:
            adv_id = np.random.randint(0, self.num_agents, size=self.n_eval_rollout_threads)
        else:
            adv_id = np.full(self.n_eval_rollout_threads, self.agent_adversary)
        
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = self._pad_obs(eval_obs, self.belief_shape)
        if self.stack_share_obs:   
            eval_share_obs = self._stack_obs(eval_share_obs)
        eval_share_obs = self._pad_obs(eval_share_obs, 2 * self.belief_shape)

        eval_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((eval_obs.shape[0], self.num_agents,
                                        self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_rnn_states_belief = np.zeros((eval_obs.shape[0], self.num_agents,
                                   self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)

        eval_rnn_states_critic = np.zeros((eval_obs.shape[0], self.num_agents,
                                            self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

        while True:
            ground_truth_type = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.belief_shape))
            ground_truth_type[np.arange(self.n_eval_rollout_threads)[:, None], np.arange(self.num_agents), adv_id[:, None]] = 1
            ground_truth_type[:, :, -self.n_severity_types + severity_ind] = 1

            eval_actions_collector = []
            eval_adv_actions_collector = []
            for agent_id in range(self.num_agents):
                baseline_eval_belief = np.zeros((self.n_eval_rollout_threads, self.num_agents))

                eval_obs[:, agent_id, -self.belief_shape:-self.n_severity_types] = _t2n(baseline_eval_belief)
                eval_share_obs[:, agent_id, -2*self.belief_shape:-self.belief_shape] = ground_truth_type[:, agent_id]
                eval_share_obs[:, agent_id, -self.belief_shape:] = ground_truth_type[:, agent_id]

                eval_actions, temp_rnn_state = \
                    self.baseline_actor[agent_id].act(eval_obs[:, agent_id, :-self.n_severity_types],
                                             eval_rnn_states[:, agent_id],
                                             eval_masks[:, agent_id],
                                             eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                             deterministic=False)
                    
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            for agent_id in range(self.num_agents):
                # adv_obs = eval_obs[:, agent_id].copy()
                adv_obs = eval_obs[:, agent_id].copy()
                adv_obs[:, -self.belief_shape: -self.n_severity_types] = np.eye(self.num_agents)[agent_id]
                adv_obs[:, -self.n_severity_types:] = np.eye(self.n_severity_types)[severity_ind]

                # adv_obs[:, -self.num_agents:] = np.eye(self.num_agents)[agent_id]
                if self.super_adversary:
                    def_act = np.concatenate([*eval_actions_collector[-self.num_agents:][:agent_id], 
                                              *eval_actions_collector[-self.num_agents:][agent_id + 1:]], axis=-1)
                    adv_obs = np.concatenate([adv_obs, softmax(def_act)], axis=-1)
                eval_adv_actions, temp_adv_rnn_state = \
                    self.actor[agent_id].act_adv(adv_obs,
                                                 eval_adv_rnn_states[:, agent_id],
                                                 eval_masks[:, agent_id],
                                                 eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                                                 deterministic=False,
                                                 severity_ind=severity_ind)
                eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                eval_adv_actions_collector.append(_t2n(eval_adv_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)

            eval_actions[np.arange(eval_actions.shape[0]), adv_id] = eval_adv_actions[np.arange(eval_actions.shape[0]), adv_id]

            if self.use_critic_for_value:
                value, eval_rnn_states_critic = self.baseline_critic.get_values(np.concatenate(eval_share_obs),
                                                                               np.concatenate(eval_rnn_states_critic),
                                                                               np.concatenate(eval_masks))
                eval_values = np.array(np.split(_t2n(value), self.n_eval_rollout_threads))
                if self.baseline_value_normalizer is not None:
                    eval_values = self.baseline_value_normalizer.denormalize(eval_values)
                eval_value = np.mean(eval_values, axis=1)
                eval_rnn_states_critic = np.array(np.split(_t2n(eval_rnn_states_critic), self.n_eval_rollout_threads)) 

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            eval_obs = self._pad_obs(eval_obs, self.belief_shape)
            if self.stack_share_obs:
                eval_share_obs = self._stack_obs(eval_share_obs)
            eval_share_obs = self._pad_obs(eval_share_obs, 2 * self.belief_shape)


            eval_data = eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
            eval_rnn_states_belief[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)            
            eval_rnn_states_critic[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (eval_obs.shape[0], self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    next_value = eval_value[eval_i].item() if self.use_critic_for_value else None
                    self.logger.eval_thread_done(eval_i, next_value)

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                print("baseline evaluation")
                ret_mean = self.logger.eval_log_severity(eval_episode, severity_ind)
                break

    def _pad_obs(self, obs, pad_size):
        pad= np.zeros((obs.shape[0], obs.shape[1], pad_size), dtype=np.float32)
        return np.concatenate([obs, pad], axis=-1)
    
    def _stack_obs(self, obs):
        concat_obs = np.concatenate([obs[:,i,:] for i in range(obs.shape[1])], axis=-1)
        repeated_obs = np.repeat(concat_obs[:, np.newaxis, :], obs.shape[1], axis=1)
        return repeated_obs

    def _adjust_obs_shapes(self, space, offset=0 ,stack=False):
        obs_space = copy.deepcopy(space[0])
        if obs_space.__class__.__name__ == 'Box':
            from gym.spaces import Box
            if stack:
                new_shape = (obs_space.shape[0]*self.num_agents + offset,)
            else:
                new_shape = (obs_space.shape[0] + offset,)
            obs_space = Box(low=obs_space.low.min(), high=obs_space.high.max(), shape=new_shape, dtype=obs_space.dtype)
        elif obs_space.__class__.__name__ == 'list':
            if stack:
                obs_space[0] = obs_space[0]*self.num_agents
            obs_space[0] = obs_space[0] + offset
        return obs_space            
           
