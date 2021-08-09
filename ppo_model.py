import gym
import torch
import torch.nn as nn
from functools import reduce
# torch.manual_seed(10)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.constant_(m.weight,0.25)
        nn.init.constant_(m.bias, 0.0)




class ActorNetwork(nn.Module):
    def __init__(self, observation_shape, action_space, n_latent_var):
        super().__init__()
        self.action_space = action_space
        if isinstance(action_space, gym.spaces.Discrete):
            output_shape = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            output_shape = reduce(lambda x, y: x * y, action_space.shape)
        input_shape = reduce(lambda x, y: x * y, observation_shape)
        # discrete action
        if isinstance(action_space, gym.spaces.Discrete):
            self.discrete_action_layer = nn.Sequential(
                nn.Linear(input_shape, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, output_shape),
                nn.Softmax(dim=-1),
            )
        # continuous action
        elif isinstance(action_space, gym.spaces.Box):
            self.continuous_action_layer = nn.Sequential(
                nn.Linear(input_shape, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
            )
            self.mu_out = nn.Linear(n_latent_var, output_shape)
            self.sigma_out = nn.Linear(n_latent_var, output_shape)

    def forward(self, x):
        # discrete action
        if isinstance(self.action_space, gym.spaces.Discrete):
            out = self.discrete_action_layer(x)
            return out
        # continuous action
        elif isinstance(self.action_space, gym.spaces.Box):
            x = self.continuous_action_layer(x)
            mu = 2.0*torch.tanh(self.mu_out(x))
            sigma = torch.nn.functional.softplus(self.sigma_out(x))  # sigma>0
            # import pdb;pdb.set_trace()
            return torch.distributions.Normal(mu, sigma)

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
#         return torch.distributions.Normal(mu, sigma)




class ValueNetwork(nn.Module):
    def __init__(self, observation_shape, output_shape, n_latent_var):
        super().__init__()
        input_shape = reduce(lambda x, y: x * y, observation_shape)
        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(input_shape, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, output_shape),
        )

    def forward(self, x):
        out = self.value_layer(x)
        return out
    
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


def generate_ppo_model(
    observation_space, action_space, hidden_size=64, learning_rate=0.01
):

    assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
        action_space, gym.spaces.Box
    )
    assert isinstance(observation_space, gym.spaces.Box)

    value_network = ValueNetwork(observation_space.shape, 1, hidden_size)
    # value_network = CriticNet(reduce(lambda x, y: x * y, observation_space.shape))
    # value_network.apply(weights_init)
    value_network.load_state_dict(torch.load('critic_params.pth'))
    value_network_optimizer = torch.optim.Adam(
        value_network.parameters(), lr=learning_rate
    )

    # init actor network
    actor_network = ActorNetwork(observation_space.shape, action_space, hidden_size)
    # actor_network = ActorNet(reduce(lambda x, y: x * y, observation_space.shape),2.0)
    # actor_network.apply(weights_init)
    actor_network.load_state_dict(torch.load('actor_params.pth'))
    # import pdb;pdb.set_trace
    actor_network_optimizer = torch.optim.Adam(
        actor_network.parameters(), lr=learning_rate
    )
    # import pdb;pdb.set_trace()

    return {
        "value_network": value_network,
        "value_optimizer": value_network_optimizer,
        "actor_network": actor_network,
        "actor_optimizer": actor_network_optimizer,
    }
