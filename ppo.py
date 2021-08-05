"""
A simple demo of Proximal Policy Optimization (PPO).
paper: [https://arxiv.org/abs/1707.06347]

Dependencies:
pytorch 1.8.0
gym 0.17.3
env: Pendulum-v0   https://www.jianshu.com/p/af3a7853268f
data: 2021/07/14
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import argparse


class ActorNet(nn.Module):
    """
    actor network to interact with env or to learn policy
    """

    def __init__(self, n_states, n_hidden, n_actions, bound):
        super(ActorNet, self).__init__()
        self.n_states = n_states
        self.bound = bound  # action: [-2,2],  bound = 2.0

        self.layer = nn.Sequential(nn.Linear(self.n_states, n_hidden), nn.ReLU())
        self.mu_out = nn.Linear(n_hidden, n_actions)
        self.sigma_out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = F.relu(self.layer(x))
        mu = self.bound * torch.tanh(self.mu_out(x))
        sigma = F.softplus(self.sigma_out(x))
        return mu, sigma


class CriticNet(nn.Module):
    """
    critic network to calculate advantage
    """

    def __init__(self, n_states, n_hidden, n_actions):
        super(CriticNet, self).__init__()
        self.n_states = n_states

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        value = self.layer(x)
        return value
        ###


class NormalizedEnv(gym.RewardWrapper):
    """Wrap reward"""

    def __init__(self, env, normalize_scale):
        super().__init__(env)
        self.normlize_scale = normalize_scale

    def reward(self, reward):
        reward = (reward + self.normlize_scale) / self.normlize_scale
        return reward


class PPO(nn.Module):
    """
    PPO algorithm :https://blog.csdn.net/ksvtsipert/article/details/118615472
    """

    def __init__(self, n_states, n_actions, bound, args):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.actor_update_steps = args.actor_update_steps
        self.critic_update_steps = args.critic_update_steps
        self.method = args.method
        if self.method == "kl_penalty":
            self.beta = args.beta
            self.kl_target = args.kl_target
        self._build_net()

    def _build_net(self):
        self.actor = ActorNet(n_states, args.n_hidden, n_actions, bound)
        self.actor_old = ActorNet(n_states, args.n_hidden, n_actions, bound)
        self.critic = CriticNet(n_states, args.n_hidden, n_actions)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, state):
        s = torch.FloatTensor(state)
        mu, sigma = self.actor(state)
        dist_norm = torch.distributions.Normal(mu, sigma)
        action = dist_norm.sample()
        return torch.clip(action, -self.bound, self.bound).numpy()

    def cal_advantage(self, states, discount_reward):
        ## states [32,3]
        ## discount_reward[32]
        ## value [32,1]
        states = torch.FloatTensor(states)
        value = self.critic(states)
        advantage = discount_reward - value.reshape(1, -1).squeeze(0)
        return advantage.detach()

    def discount_reward(self, rewards, state):
        state = torch.FloatTensor(state)
        discr = self.critic(state).detach()
        discr_list = []
        for r in rewards[::-1]:
            discr = r + self.gamma * discr
            discr_list.append(discr)
        discr_list.reverse()
        discr_list = torch.cat(discr_list)
        return discr_list

    # def get_actor_loss
    def actor_learn(self, states, actions, advantage):
        ## states:[32,3]  actions[32,1]
        states = torch.FloatTensor(states)
        # actions = torch.FloatTensor(actions).reshape(args.batch,n_actions)
        actions = torch.FloatTensor(actions)
        # mu, sigma [32,1]
        mu, sigma = self.actor(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)
        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))

        # surr = ratio * advantage.reshape(args.batch,n_actions)
        surr = ratio * advantage.reshape(actions.shape)
        if self.method == "clip":
            # loss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(args.batch,n_actions)))
            loss = -torch.mean(
                torch.min(
                    surr,
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * advantage.reshape(actions.shape),
                )
            )
        else:
            beta = torch.tensor(self.beta)
            actor_dist = torch.distributions.normal.Normal(mu, sigma)
            actor_old_dist = torch.distributions.normal.Normal(old_mu, old_sigma)
            kl = torch.distributions.kl_divergence(actor_old_dist, actor_dist)
            kl_mean = torch.mean(kl)
            if kl_mean > 4 * self.kl_target:
                return
            if kl_mean < self.kl_target:
                beta /= 2
            elif kl_mean > self.kl_target:
                beta *= 2
            self.beta = torch.clip(beta, 1e-4, 10)

            loss = -torch.mean(surr - beta * kl)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, discount_reward):
        states = torch.FloatTensor(states)
        value = self.critic(states).reshape(1, -1).squeeze(0)

        loss_func = nn.MSELoss()
        loss = loss_func(value, discount_reward)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def update(self, states, actions, discout_reward):
        self.actor_old.load_state_dict(self.actor.state_dict())
        advantage = self.cal_advantage(states, discout_reward)

        for i in range(self.actor_update_steps):
            self.actor_learn(states, actions, advantage)

        for i in range(self.critic_update_steps):
            self.critic_learn(states, discout_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--len_episode", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--actor_update_steps", type=int, default=10)
    parser.add_argument("--critic_update_steps", type=int, default=10)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--method", type=str, default="kl_penalty")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--kl_target", type=float, default=0.01)
    args = parser.parse_args()
    env = gym.make("Pendulum-v0")
    # wrap environment
    # env = NormalizedEnv(env,8)

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    bound = env.action_space.high[0]

    ppo = PPO(n_states, n_actions, bound, args)

    all_ep_r = []
    for episode in range(args.n_episodes):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(args.len_episode):
            # env.render()
            a = ppo.choose_action(s)
            s_next, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)
            s = s_next
            ep_r += r

            if (t + 1) % args.batch == 0 or t == args.len_episode - 1:
                buffer_s = np.array(buffer_s)
                buffer_a = np.array(buffer_a)
                buffer_r = np.array(buffer_r)

                discount_reward = ppo.discount_reward(buffer_r, s_next)
                ppo.update(buffer_s, buffer_a, discount_reward)
                buffer_s, buffer_a, buffer_r = [], [], []

        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print("Episode: %i" % episode, "|Ep_reward: %i" % ep_r)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel("Episode")
    plt.ylabel("Moving averaged episode reward")
    plt.show()
