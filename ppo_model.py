import gym
import torch
import torch.nn as nn
from functools import reduce


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
            return torch.distributions.Normal(mu, sigma)


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


def generate_ppo_model(
    observation_space, action_space, hidden_size=64, learning_rate=0.01
):

    assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
        action_space, gym.spaces.Box
    )
    assert isinstance(observation_space, gym.spaces.Box)

    value_network = ValueNetwork(observation_space.shape, 1, hidden_size)
    value_network_optimizer = torch.optim.Adam(
        value_network.parameters(), lr=learning_rate
    )

    # init actor network
    actor_network = ActorNetwork(observation_space.shape, action_space, hidden_size)
    actor_network_optimizer = torch.optim.Adam(
        actor_network.parameters(), lr=learning_rate
    )

    return {
        "value_network": value_network,
        "value_optimizer": value_network_optimizer,
        "actor_network": actor_network,
        "actor_optimizer": actor_network_optimizer,
    }
