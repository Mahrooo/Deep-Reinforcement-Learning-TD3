import torch
import torch.nn as nn
import torch.nn.functional as F


# build actor neural networks
class Actor(nn.Module):
    # dim of input of NN, dim of output, to clip the action in certaion range
    def __init__(self, state_dim, action_dim,max_action):
        super(Actor, self).__init__()  # activating the inheritage with super function.
        self.h1 = nn.Linear(state_dim, 400)
        self.h2 = nn.Linear(400, 300)
        self.h3 = nn.Linear(300, action_dim)
        self.max_action = max_action  # for cliping the actions

    def forward(self, x):  # forward propagation x:the input state.
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.max_action * torch.tanh(self.h3(x))  # value between (-1,1), to go back to original action value.
        return x


# build two neural networks for the two Critic models and two neural networks for the two Critic targets
class Critic(nn.Module):

    def __init__(self, state_dim,
                 action_dim):  # dim of input of NN, dim of output, to clip the action in certaion range
        super(Critic, self).__init__()
        # first critic neuron:
        self.h1 = nn.Linear(state_dim + action_dim, 400)
        self.h2 = nn.Linear(400, 300)
        self.h3 = nn.Linear(300, 1)
        # second critic neuron:
        self.h4 = nn.Linear(state_dim + action_dim, 400)
        self.h5 = nn.Linear(400, 300)
        self.h6 = nn.Linear(300, 1)

    def forward(self, x, u):  # forward propagation x:the input state. concatinate of action and state
        xu = torch.cat([x, u], 1)  # vertically
        x1 = F.relu(self.h1(xu))
        x1 = F.relu(self.h2(x1))
        x1 = self.h3(x1)

        x2 = F.relu(self.h4(xu))
        x2 = F.relu(self.h5(x2))
        x2 = self.h6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.h1(xu))
        x1 = F.relu(self.h2(x1))
        x1 = self.h3(x1)
        return x1
