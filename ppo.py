"""
A simple demo of Proximal Policy Optimization (PPO).
paper: [https://arxiv.org/abs/1707.06347]

Dependencies:
torch 1.8.1
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
    actor network用于与环境交互
    生成均值mu和方差sigma, 在交互过程中通过从正态分布中采样，得到action
    """
    def __init__(self, n_states, bound):
        super(ActorNet, self).__init__()
        self.n_states = n_states
        self.bound = bound     #action: [-2,2],  bound = 2.0  
        
        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU()
        )
        self.mu_out = nn.Linear(128, 1)
        self.sigma_out = nn.Linear(128, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x,dtype = torch.float)
        x = F.relu(self.layer(x))
        mu = self.bound * torch.tanh(self.mu_out(x))
        sigma = F.softplus(self.sigma_out(x))
        return mu, sigma
    
    
class CriticNet(nn.Module):
    '''
    critic network 用于评估优势函数
    '''
    def __init__(self, n_states):
        super(CriticNet, self).__init__()
        self.n_states = n_states

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x,dtype = torch.float)
        value = self.layer(x)
        return value


class PPO(nn.Module):
    '''
    PPO算法部分
    详解见https://blog.csdn.net/ksvtsipert/article/details/118615472 
    '''
    def __init__(self, n_states, n_actions, bound, args):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.update_steps = args.update_steps
        self._build_net()

    def _build_net(self):
        self.actor = ActorNet(n_states, bound)
        self.actor_old = ActorNet(n_states, bound)
        self.critic = CriticNet(n_states)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, state):
        s = torch.FloatTensor(state)
        mu, sigma = self.actor(state)
        dist_norm = torch.distributions.Normal(mu, sigma)
        action = dist_norm.sample()
        return torch.clip(action, -self.bound, self.bound).numpy()

    def cal_advantage(self, states, discout_reward):
        states = torch.FloatTensor(states)
        value = self.critic(states)                            
        advantage = discout_reward - value.reshape(1, -1).squeeze(0)
        return advantage.detach()                                 

    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_)
        discr = self.critic(s_).detach()                 
        discr_list = []
        for r in rewards[::-1]:
            discr = r + self.gamma * discr
            discr_list.append(discr)
        discr_list.reverse()
        discr_list = torch.cat(discr_list)                   
        return discr_list

    def update_actor(self):
        self.actor_old.load_state_dict(self.actor.state_dict())      #更新旧actor

    def actor_learn(self, states, actions, advantage):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).reshape(-1, 1)

        mu, sigma = self.actor(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)
        
        # ratio = pi.log_prob(actions)/(old_pi.log_prob(actions)+1e-5) #####按照除法做，几乎没有明显学习效果
        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))
            
        surr = ratio * advantage.reshape(-1, 1)                 
        loss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, discount_reward):
        states = torch.FloatTensor(states)
        value = self.critic(states).reshape(1, -1).squeeze(0)

        loss_func = nn.MSELoss()
        loss = loss_func(value, discout_reward)
        # loss = torch.mean(torch.square(discount_reward-value))
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def update(self, states, actions, discout_reward):
        self.update_actor()
        advantage = self.cal_advantage(states, discout_reward)

        for i in range(self.update_steps):                      
            self.actor_learn(states, actions, advantage)                     
            self.critic_learn(states, discout_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--len_episode', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--update_steps', type=int, default=10)
    args = parser.parse_args()
    env = gym.make('Pendulum-v0')
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
            env.render()
            a = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)   
            s = s_
            ep_r += r

            if (t + 1) % args.batch == 0 or t == args.len_episode - 1:   
                buffer_s = np.array(buffer_s)
                buffer_a = np.array(buffer_a)
                buffer_r = np.array(buffer_r)

                discout_reward = ppo.discount_reward(buffer_r, s_)          
                ppo.update(buffer_s, buffer_a, discout_reward)           
                buffer_s, buffer_a, buffer_r = [], [], []

        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1) 
        print('Episode: %i' % episode,"|Ep_reward: %i" % ep_r)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()       


    
