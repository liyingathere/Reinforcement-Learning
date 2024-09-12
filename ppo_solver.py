import io

import gym
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from functools import reduce

from .ppo_model import generate_ppo_model
from ..algorithm import Algorithm
from ..util import ensure_tensor


class DataLoader:
    def __init__(self, batch, k_epochs, minibatch_size):
        self.n = 0
        self.data = batch
        self.k_epochs = k_epochs
        self.minibatch_size = minibatch_size

    def __iter__(self):
        return self

    def __next__(self):
        # when self.n>self.k_epochs, it will be reset(self.n = 0) and begin next iteration
        if self.n > self.k_epochs:
            raise StopIteration

        begin_index = self.minibatch_size * self.n
        end_index = min(len(self.data[2]), self.minibatch_size * (self.n + 1))
        states = self.data[0][begin_index:end_index]
        actions = self.data[1][begin_index:end_index]
        rewards = self.data[2][begin_index:end_index]
        logprobs = self.data[3][begin_index:end_index]

        self.n += 1
        return states, actions, rewards, logprobs


class PPOSolver(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        models=None,
        # sgd_minibatch_size=200,
        # train_batch_size=800,
        k_epochs = 5,
        eps_clip=0.2,
        vf_loss_coeff=1.0,
        entropy_coeff=0.01,
        learning_rate=0.01,
        hidden_size=64,
        device=None,
    ):

        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gym.spaces.Box
        )
        if isinstance(action_space, gym.spaces.Discrete):
            self.num_actions = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            self.num_actions = reduce(lambda x, y: x * y, action_space.shape)
            self.action_shape = action_space.shape
        super().__init__(device)
        if models is None:
            models = generate_ppo_model(
                observation_space,
                action_space,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
            )

        assert models["actor_network"]
        assert models["value_network"]
        assert models["actor_optimizer"]
        assert models["value_optimizer"]
        self.actor_network = models["actor_network"]
        self.value_network = models["value_network"]
        self.actor_optimizer = models["actor_optimizer"]
        self.value_optimizer = models["value_optimizer"]

        self.actor_network.to(self.device, non_blocking=True)
        self.value_network.to(self.device, non_blocking=True)

        # self.sgd_minibatch_size = sgd_minibatch_size
        self.k_epochs = k_epochs#math.ceil(train_batch_size // sgd_minibatch_size)
        self.eps_clip = eps_clip
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.action_space = action_space
        self.state_shape = reduce(lambda x, y: x * y, observation_space.shape)

    def _evaluate_policy(self, state, action):
        state = state.view(-1, self.state_shape)
        action_probs = self.actor_network(state)
        dist = (
            Categorical(action_probs)
            if isinstance(self.action_space, gym.spaces.Discrete)
            else action_probs
        )
        action_logprobs = dist.log_prob(action)
        dist_entropy = (
            dist.entropy()
            if isinstance(self.action_space, gym.spaces.Discrete)
            else None
        )
        state_value = self.value_network(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
    def cal_discount(self, rewards,s_next):
        # s_next = torch.FloatTensor(s_next)
        s_next = s_next.view(-1, self.state_shape)
        discr = self.value_network(s_next).detach()                 
        discr_list = []
        rewards = rewards.detach().cpu().numpy()
        for r in rewards[::-1]:
            discr = r + 0.9 * discr
            discr_list.append(discr)
        discr_list.reverse()
        discr_list = torch.cat(discr_list)                   
        return discr_list


    def update(self, batch, weights=None):
        states, actions, rewards, logprobs, s_next = batch

        states = ensure_tensor(states, torch.float, self.device)
        s_next = ensure_tensor(s_next, torch.float, self.device)
        actions = ensure_tensor(actions, torch.long, self.device)
        rewards = ensure_tensor(rewards, torch.float, self.device)
        rewards = self.cal_discount(rewards,s_next[-1])
        if isinstance(self.action_space, gym.spaces.Box):
            rewards = rewards.view(-1, 1)
            rewards = rewards.expand(-1, self.num_actions)
        logprobs = ensure_tensor(logprobs, torch.float, self.device)

        # states, actions, rewards, logprobs = batch

        # states = ensure_tensor(states, torch.float, self.device)
        # actions = ensure_tensor(actions, torch.long, self.device)
        # rewards = ensure_tensor(rewards, torch.float, self.device)
        # if isinstance(self.action_space, gym.spaces.Box):
        #     rewards = rewards.view(-1, 1)
        #     rewards = rewards.expand(-1, self.num_actions)
        # logprobs = ensure_tensor(logprobs, torch.float, self.device)
        

        for _ in range(self.k_epochs):

            # Evaluating old actions and values :
            new_logprobs, state_values, dist_entropy = self._evaluate_policy(
                states, actions
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            # logprobs: [batch, action_shape], new_logprobs:[batch, nums_actions]
            logprobs = logprobs.view(new_logprobs.shape)

            ratios = torch.exp(new_logprobs - logprobs.detach())

            # Finding Surrogate Loss:
            # rewards:[batch,num_actions], state_values:[batch],  make sure that rewards and state_values has the same shape
            if isinstance(self.action_space, gym.spaces.Box):
                state_values = state_values.view(-1, 1)
                state_values = state_values.expand(-1, self.num_actions)
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            if isinstance(self.action_space, gym.spaces.Discrete):
                loss_actor = (
                    -torch.min(surr1, surr2) - self.entropy_coeff * dist_entropy
                )
            elif isinstance(self.action_space, gym.spaces.Box):
                loss_actor = -torch.min(surr1, surr2)

            loss_critic = self.vf_loss_coeff * F.mse_loss(state_values, rewards)
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss_actor.mean().backward()
            loss_critic.backward()
            self.actor_optimizer.step()
            self.value_optimizer.step()

        # loss = dict(actor = loss_actor.detach().cpu().abs(), critic = loss_critic.detach().cpu().abs())
        # return loss
        return loss_actor.detach().cpu().abs() + loss_critic.detach().cpu().abs()

    def act(self, state):
        state = ensure_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        state = state.reshape(-1, 1).squeeze(1)
        action_probs = self.actor_network(state)
        dist = (
            Categorical(action_probs)
            if isinstance(self.action_space, gym.spaces.Discrete)
            else action_probs
        )
        action = dist.sample()
        #action = torch.clip(action,-2,2)
        logprobs = dist.log_prob(action)
        if isinstance(self.action_space, gym.spaces.Discrete):
            return action.item(), logprobs.item()
        elif isinstance(self.action_space, gym.spaces.Box):
            # action: [batch,num_actions] -> [batch, action_shape]
            return (
                action.squeeze().view(self.action_shape).cpu().numpy(),
                logprobs.squeeze().view(self.action_shape).cpu().detach().numpy(),
            )

    def sync_weights(self, src_ppo):
        assert isinstance(src_ppo, PPOSolver)
        self.actor_network.load_state_dict(src_ppo.actor_network.state_dict())
        # self.value_network.load_state_dict(src_ppo.value_network.state_dict())

    def load_weights(self, stream):
        states = torch.load(stream, map_location=self.device)
        self.actor_network.load_state_dict(states)
        self.actor_network.to(self.device, non_blocking=True)

    def save_weights(self, stream=None):
        if stream is None:
            stream = io.BytesIO()
        torch.save(self.actor_network.state_dict(), stream)
        return stream
