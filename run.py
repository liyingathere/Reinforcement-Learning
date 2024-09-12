import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as path
import yaml
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

from reth.algorithm.util import calculate_discount_rewards_with_dones
from reth.buffer import DynamicSizeBuffer
from reth.presets.config import get_solver, get_worker, get_trainer

torch.manual_seed(10)
np.random.seed(10)

if __name__ == "__main__":
    config_path = path.join(path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    max_ts = config["common"]["max_ts"]
    gamma = config["common"]["gamma"]
    update_interval = config["common"]["batch_size"]
    env = gym.make('Pendulum-v0')
    env.seed(10)

    solver = get_solver(config_path, env = env)
    act_solver = get_solver(config_path, env = env)
    # shared solver
    
    worker = get_worker(config_path, env = env, solver=act_solver)
    trainer = get_trainer(config_path, env = env, solver=solver)
    
    all_ep_r = []
    act_solver.sync_weights(solver)
    for episode in range(600):
        worker.s0 = worker.env.reset()
        buffer_s, buffer_a, buffer_r,buffer_logprob, buffer_snext= [], [], [], [], []
        ep_r = 0
        for t in range(update_interval):
            s = worker.s0
            a, logprob = act_solver.act(s)
            s_, r, done, _ = env.step(a)
            # s0, a, r, s1, done = worker.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r +10) /10) 
            buffer_logprob.append(logprob)
            buffer_snext.append(s_)   
            ep_r += r

            if (t + 1) % 32 == 0 or t == 200 - 1:   
                buffer_s = np.array(buffer_s)
                buffer_a = np.array(buffer_a)
                buffer_r = np.array(buffer_r)
                buffer_logprob = np.array(buffer_logprob)
                # print(len(buffer_s))             
                loss = trainer.step([buffer_s, buffer_a, buffer_r, buffer_logprob,buffer_snext])
                act_solver.sync_weights(solver)          
                buffer_s, buffer_a, buffer_r, buffer_logprob,buffer_snext = [], [], [], [],[]

        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1) 
        print('Episode: %i' % episode,"|Ep_reward: %i" % ep_r)
    
    print('r /10' )
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()   



#old reth
    # episode_buffer = DynamicSizeBuffer(64)
    # data_buffer = DynamicSizeBuffer(64)
    # act_solver.sync_weights(solver)
    # for _ in range(max_ts):
    #     worker.s0 = worker.env.reset()
    #     for t in range(update_interval):
    #         a, logprob = act_solver.act(worker.s0)
    #         s0, a, r, s1, done = worker.step(a)
    #         # r = (r+10)/10
    #         # episode_buffer.append((s0, a, r, logprob, done))
    #         # if done or t == update_interval - 1:
    #         #     s0, a, r, logprob, done = episode_buffer.data
    #         #     r = calculate_discount_rewards_with_dones(r, done, gamma)
    #         #     data_buffer.append_batch((s0, a, r, logprob))
    #         #     episode_buffer.clear()

    #         episode_buffer.append((s0, a, r, logprob, s1))
    #         if (t+1) % 32 == 0 or t == update_interval - 1:
    #             s0, a, r, logprob, s1 = episode_buffer.data
                
    #             # r = calculate_discount_rewards_with_dones(r, done, gamma)
    #             data_buffer.append_batch((s0, a, r, logprob, s1))
    #             episode_buffer.clear()

    #             loss = trainer.step(data_buffer.data)
    #             act_solver.sync_weights(solver)
    #             data_buffer.clear()
    #     # loss = trainer.step(data_buffer.data)
    #     # act_solver.sync_weights(solver)
    #     # data_buffer.clear()







# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import gym
# import argparse

# class ActorNet(nn.Module):
#     """
#     actor network用于与环境交互
#     生成均值mu和方差sigma, 在交互过程中通过从正态分布中采样，得到action
#     """
#     def __init__(self, n_states, bound):
#         super(ActorNet, self).__init__()
#         self.n_states = n_states
#         self.bound = bound     #action: [-2,2],  bound = 2.0  
        
#         self.layer = nn.Sequential(
#             nn.Linear(self.n_states, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#         )
#         self.mu_out = nn.Linear(64, 1)
#         self.sigma_out = nn.Linear(64, 1)


#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x,dtype = torch.float)
#         x = self.layer(x)
#         mu = 2.0*torch.tanh(self.mu_out(x))
#         sigma = torch.nn.functional.softplus(self.sigma_out(x))  # sigma>0
#         return mu, sigma
    
    
# class CriticNet(nn.Module):
#     '''
#     critic network 用于评估优势函数
#     '''
#     def __init__(self, n_states):
#         super(CriticNet, self).__init__()
#         self.n_states = n_states

#         self.layer = nn.Sequential(
#             nn.Linear(self.n_states, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#         )

#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x,dtype = torch.float)
#         value = self.layer(x)
#         return value



# class PPO(nn.Module):
#     '''
#     PPO算法部分
#     详解见https://blog.csdn.net/ksvtsipert/article/details/118615472 
#     '''
#     def __init__(self, n_states, n_actions, bound, args):
#         super().__init__()
#         self.n_states = n_states
#         self.n_actions = n_actions
#         self.bound = bound
#         self.lr = args.lr
#         self.gamma = args.gamma
#         self.epsilon = args.epsilon
#         self.update_steps = args.update_steps
#         self._build_net()

#     def _build_net(self):
#         self.actor = ActorNet(n_states, bound)
#         self.actor_old = ActorNet(n_states, bound)
#         self.critic = CriticNet(n_states)
#         self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

#     def choose_action(self, state):
#         s = torch.FloatTensor(state)
#         mu, sigma = self.actor(state)
#         dist_norm = torch.distributions.Normal(mu, sigma)
#         action = dist_norm.sample()
#         return torch.clip(action, -self.bound, self.bound).numpy()

#     def cal_advantage(self, states, discout_reward):
#         states = torch.FloatTensor(states)
#         value = self.critic(states)                            
#         advantage = discout_reward - value.reshape(1, -1).squeeze(0)
#         return advantage.detach()                                 

#     def discount_reward(self, rewards, s_):
#         s_ = torch.FloatTensor(s_)
#         discr = self.critic(s_).detach()        
#         # discr = torch.zeros(temp.shape)         
#         discr_list = []
#         for r in rewards[::-1]:
#             discr = r + self.gamma * discr
#             discr_list.append(discr)
#         discr_list.reverse()
#         discr_list = torch.cat(discr_list)                   
#         return discr_list

#     def update_actor(self):
#         self.actor_old.load_state_dict(self.actor.state_dict())      #更新旧actor

#     def actor_learn(self, states, actions, advantage):
#         states = torch.FloatTensor(states)
#         actions = torch.FloatTensor(actions).reshape(-1, 1)

#         mu, sigma = self.actor(states)
#         pi = torch.distributions.Normal(mu, sigma)

#         old_mu, old_sigma = self.actor_old(states)
#         old_pi = torch.distributions.Normal(old_mu, old_sigma)
        
#         ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))
            
#         surr = ratio * advantage.reshape(-1, 1)                 
#         loss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))

#         self.actor_optim.zero_grad()
#         loss.backward()
#         self.actor_optim.step()

#     def critic_learn(self, states, discount_reward):
#         states = torch.FloatTensor(states)
#         value = self.critic(states).reshape(1, -1).squeeze(0)

#         loss_func = nn.MSELoss()
#         loss = loss_func(value, discout_reward)
#         # loss = torch.mean(torch.square(discount_reward-value))
#         self.critic_optim.zero_grad()
#         loss.backward()
#         self.critic_optim.step()

#     def update(self, states, actions, discout_reward):
#         self.update_actor()
#         advantage = self.cal_advantage(states, discout_reward)

#         for i in range(self.update_steps):                      
#             self.actor_learn(states, actions, advantage)                     
#             self.critic_learn(states, discout_reward)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_episodes', type=int, default=1500)
#     parser.add_argument('--len_episode', type=int, default=200)
#     parser.add_argument('--lr', type=float, default=0.0001)
#     parser.add_argument('--batch', type=int, default=32)
#     parser.add_argument('--gamma', type=float, default=0.9)
#     parser.add_argument('--seed', type=int, default=10)
#     parser.add_argument('--epsilon', type=float, default=0.2)
#     parser.add_argument('--update_steps', type=int, default=10)
#     args = parser.parse_args()
#     env = gym.make('Pendulum-v0')
#     env.seed(args.seed)
#     torch.manual_seed(args.seed)

    
#     def calculate_discount_rewards_with_dones(r, done, gamma=0.99):
#         discounted_r = np.zeros_like(r, dtype=np.float32)
#         running_add = 0
#         for t in reversed(range(len(r))):
#             if done[t]:
#                 # import pdb;pdb.set_trace()
#                 discounted_r[t] = 0
#             else:
#                 running_add = running_add * gamma + r[t]
#                 discounted_r[t] = running_add

#         # discounted_r -= discounted_r.mean()
#         # discounted_r /= discounted_r.std() + 1e-7
#         return torch.tensor(discounted_r)


#     n_states = env.observation_space.shape[0]
#     n_actions = env.action_space.shape[0]
#     bound = env.action_space.high[0]

#     ppo = PPO(n_states, n_actions, bound, args)

#     all_ep_r = []
#     for episode in range(args.n_episodes):
#         s = env.reset()
#         buffer_s, buffer_a, buffer_r , buffer_done= [], [], [], []
#         ep_r = 0
#         for t in range(args.len_episode):
#             # env.render()
#             a = ppo.choose_action(s)
#             s_, r, done, _ = env.step(a)
#             # print(r)
#             buffer_s.append(s)
#             buffer_a.append(a)
#             buffer_r.append((r +10) /10) 
#             # buffer_r.append(r)
#             buffer_done.append(done)   
#             s = s_
#             ep_r += r

#             if (t + 1) % args.batch == 0 or t == args.len_episode - 1:   
#                 buffer_s = np.array(buffer_s)
#                 buffer_a = np.array(buffer_a)
#                 buffer_r = np.array(buffer_r)
#                 buffer_done = np.array(buffer_done)
#                 discout_reward = calculate_discount_rewards_with_dones(buffer_r,buffer_done)     
#                 # discout_reward = ppo.discount_reward(buffer_r, s_)          
#                 ppo.update(buffer_s, buffer_a, discout_reward)           
#                 buffer_s, buffer_a, buffer_r, buffer_done = [], [], [], []

#         if episode == 0:
#             all_ep_r.append(ep_r)
#         else:
#             all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1) 
#         print('Episode: %i' % episode,"|Ep_reward: %i" % ep_r)
    
#     print('r /10' )
#     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
#     plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()       



#---------------------------------------------
# GAE TEST
# class ActorNet(nn.Module):
#     """
#     actor network用于与环境交互
#     生成均值mu和方差sigma, 在交互过程中通过从正态分布中采样，得到action
#     """
#     def __init__(self, n_states, bound):
#         super(ActorNet, self).__init__()
#         self.n_states = n_states
#         self.bound = bound     #action: [-2,2],  bound = 2.0  
        
#         self.layer = nn.Sequential(
#             nn.Linear(self.n_states, 128),
#             nn.ReLU()
#         )
#         self.mu_out = nn.Linear(128, 1)
#         self.sigma_out = nn.Linear(128, 1)

#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x,dtype = torch.float)
#         x = F.relu(self.layer(x))
#         mu = self.bound * torch.tanh(self.mu_out(x))
#         sigma = F.softplus(self.sigma_out(x))
#         return mu, sigma
    
    
# class CriticNet(nn.Module):
#     '''
#     critic network 用于评估优势函数
#     '''
#     def __init__(self, n_states):
#         super(CriticNet, self).__init__()
#         self.n_states = n_states

#         self.layer = nn.Sequential(
#             nn.Linear(self.n_states, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x,dtype = torch.float)
#         value = self.layer(x)
#         return value

# class PPO(nn.Module):
#     '''
#     PPO算法部分
#     详解见https://blog.csdn.net/ksvtsipert/article/details/118615472 
#     '''
#     def __init__(self, n_states, n_actions, bound, args):
#         super().__init__()
#         self.n_states = n_states
#         self.n_actions = n_actions
#         self.bound = bound
#         self.lr = args.lr
#         self.gamma = args.gamma
#         self.epsilon = args.epsilon
#         self.update_steps = args.update_steps
#         self._build_net()

#     def _build_net(self):
#         self.actor = ActorNet(n_states, bound)
#         self.actor_old = ActorNet(n_states, bound)
#         self.critic = CriticNet(n_states)
#         self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

#     def choose_action(self, state):
#         s = torch.FloatTensor(state)
#         mu, sigma = self.actor(state)
#         dist_norm = torch.distributions.Normal(mu, sigma)
#         action = dist_norm.sample()
#         return torch.clip(action, -self.bound, self.bound).numpy()

#     def cal_advantage(self, states, discout_reward):
#         states = torch.FloatTensor(states)
#         value = self.critic(states)                            
#         advantage = discout_reward - value.reshape(1, -1).squeeze(0)
#         return advantage.detach()                                 

#     def discount_reward(self, rewards, s_):
#         s_ = torch.FloatTensor(s_)
#         discr = self.critic(s_).detach()        
#         # discr = torch.zeros(temp.shape)         
#         discr_list = []
#         for r in rewards[::-1]:
#             discr = r + self.gamma * discr
#             discr_list.append(discr)
#         discr_list.reverse()
#         discr_list = torch.cat(discr_list)                   
#         return discr_list

#     def update_actor(self):
#         self.actor_old.load_state_dict(self.actor.state_dict())      #更新旧actor

#     def actor_learn(self, states, actions, advantage):
#         states = torch.FloatTensor(states)
#         actions = torch.FloatTensor(actions).reshape(-1, 1)

#         mu, sigma = self.actor(states)
#         pi = torch.distributions.Normal(mu, sigma)

#         old_mu, old_sigma = self.actor_old(states)
#         old_pi = torch.distributions.Normal(old_mu, old_sigma)
        
#         # ratio = pi.log_prob(actions)/(old_pi.log_prob(actions)+1e-5) #####按照除法做，几乎没有明显学习效果
#         ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))
            
#         surr = ratio * advantage.reshape(-1, 1)                 
#         loss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))

#         self.actor_optim.zero_grad()
#         loss.backward()
#         self.actor_optim.step()

#     def critic_learn(self, states, returns, advants, values):
#         states = torch.FloatTensor(states)
#         new_values = self.critic(states).reshape(1, -1).squeeze(0)
#         clipped_values = values + torch.clamp(new_values - values,-10,10)
#         loss_func = nn.MSELoss()
#         critic_loss1 = loss_func(clipped_values, returns)
#         critic_loss2 = loss_func(new_values, returns)
#         loss = torch.max(critic_loss1, critic_loss2).mean()

#         self.critic_optim.zero_grad()
#         loss.backward()
#         self.critic_optim.step()

#     def update(self, states, actions, returns, advants, values):
#         self.update_actor()
#         # advantage = self.cal_advantage(states, discout_reward)

#         for i in range(self.update_steps):                      
#             self.actor_learn(states, actions, advants)                     
#             self.critic_learn(states, returns, advants, values)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_episodes', type=int, default=1500)
#     parser.add_argument('--len_episode', type=int, default=200)
#     parser.add_argument('--lr', type=float, default=0.0001)
#     parser.add_argument('--batch', type=int, default=32)
#     parser.add_argument('--gamma', type=float, default=0.9)
#     parser.add_argument('--seed', type=int, default=10)
#     parser.add_argument('--epsilon', type=float, default=0.2)
#     parser.add_argument('--update_steps', type=int, default=10)
#     args = parser.parse_args()
#     env = gym.make('Pendulum-v0')
#     env.seed(args.seed)
#     torch.manual_seed(args.seed)

    
#     def calculate_discount_rewards_with_dones(r, done, gamma=0.99):
#         discounted_r = np.zeros_like(r, dtype=np.float32)
#         running_add = 0
#         for t in reversed(range(len(r))):
#             if done[t]:
#                 # import pdb;pdb.set_trace()
#                 discounted_r[t] = 0
#             else:
#                 running_add = running_add * gamma + r[t]
#                 discounted_r[t] = running_add

#         # discounted_r -= discounted_r.mean()
#         # discounted_r /= discounted_r.std() + 1e-7
#         return torch.tensor(discounted_r)

#     def get_gae(rewards, dones, values, gamma, lamda):
#         if isinstance(rewards, np.ndarray):
#             rewards = torch.tensor(rewards,dtype = torch.float)
#         returns = torch.zeros_like(rewards)
#         advants = torch.zeros_like(rewards)

#         running_returns = 0
#         previous_value = 0
#         running_advants = 0

#         for t in reversed(range(0, len(rewards))):
#             running_returns = rewards[t] + gamma * running_returns * dones[t]
#             running_tderror = rewards[t] + gamma * previous_value * dones[t] - values.data[t]
#             running_advants = running_tderror + gamma * lamda * running_advants * dones[t]

#             returns[t] = running_returns
#             previous_value = values.data[t]
#             advants[t] = running_advants

#         advants = (advants - advants.mean()) / advants.std()
#         return returns, advants

#     n_states = env.observation_space.shape[0]
#     n_actions = env.action_space.shape[0]
#     bound = env.action_space.high[0]

#     ppo = PPO(n_states, n_actions, bound, args)

#     all_ep_r = []
#     for episode in range(args.n_episodes):
#         s = env.reset()
#         buffer_s, buffer_a, buffer_r , buffer_done, buffer_v= [], [], [], [],[]
#         ep_r = 0
#         for t in range(args.len_episode):
#             # env.render()
#             a = ppo.choose_action(s)
#             s_, r, done, _ = env.step(a)
#             # print(r)
#             buffer_s.append(s)
#             buffer_a.append(a)
#             # buffer_r.append((r +10) /10) 
#             buffer_r.append(r)
#             buffer_done.append(done)
#             # buffer_v.append(self.critic(s).detach())   
#             s = s_
#             ep_r += r

#             if (t + 1) % args.batch == 0 or t == args.len_episode - 1:   
#                 buffer_s = np.array(buffer_s)
#                 buffer_a = np.array(buffer_a)
#                 buffer_r = np.array(buffer_r)
#                 buffer_done = np.array(buffer_done)
#                 values = ppo.critic(buffer_s).detach()
#                 # import pdb;pdb.set_trace()
#                 returns, advantages = get_gae(buffer_r, buffer_done, values, 0.99, 0.05)

#                 # discout_reward = calculate_discount_rewards_with_dones(buffer_r,buffer_done)     
#                 # discout_reward = ppo.discount_reward(buffer_r, s_)          
#                 ppo.update(buffer_s, buffer_a, returns, advantages, values)        
#                 buffer_s, buffer_a, buffer_r, buffer_done = [], [], [], []

#         if episode == 0:
#             all_ep_r.append(ep_r)
#         else:
#             all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1) 
#         print('Episode: %i' % episode,"|Ep_reward: %i" % ep_r)
    
#     print('r /10' )
#     plt.plot(np.arange(len(all_ep_r)), all_ep_r)
#     plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()  