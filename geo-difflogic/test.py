from difflogic import LogicLayer, GroupSum
import torch

model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LogicLayer(36, 64000),
            torch.nn.Tanh(),
            LogicLayer(64000, 64000),
            torch.nn.Tanh(),
            LogicLayer(64000, 64000),
            torch.nn.Tanh(),
            LogicLayer(64000, 64000),
            torch.nn.Tanh(),
            # LogicLayer(64_000, 64_000),
            # LogicLayer(64_000, 64_000),
            # LogicLayer(64_000, 64_000),
            # LogicLayer(64_000, 64_000),
            # LogicLayer(64_000, 32_000),
            # LogicLayer(32_000, 16_000),
            GroupSum(k=4, tau=10),
            # torch.nn.Softmax(dim=1),
        )
model = torch.load('/media/juno/research/blackbox research/difflogic_test/experiments/gdqn-tiny-softmax/model.pth')
num_actions = 4
state_shape = [36]
mlp_layers = [64,64]
import numpy as np
import torch.nn as nn
# build the Q network
# layer_dims = [np.prod(state_shape)] + mlp_layers
# fc = [nn.Flatten()]
# fc.append(nn.BatchNorm1d(layer_dims[0]))
# for i in range(len(layer_dims)-1):
#     fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
#     fc.append(nn.Tanh())
# fc.append(nn.Linear(layer_dims[-1], num_actions, bias=True))
# fc_layers = nn.Sequential(*fc)
# fc_layers.load_state_dict(torch.load('/media/juno/research/blackbox research/difflogic_test/experiments/dqn-2/model.pth'))
model2=torch.load('/media/juno/research/blackbox research/difflogic_test/experiments/dqn-2/model.pth')
print('CUDA?',torch.cuda.is_available())

import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make('leduc-holdem')
# env.set_agents([RandomAgent(num_actions=env.num_actions),RandomAgent(num_actions=env.num_actions)])
model3 = RandomAgent(num_actions=env.num_actions)
env.set_agents([model2, model])

print(env.num_actions) # 2
print(env.num_players) # 1
print(env.state_shape) # [[2]]
print(env.action_shape) # [None]
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)
models = [
    'gated',
    'non-gated',
    'random',
]
rewards = tournament(env, 1000)
for position, reward in enumerate(rewards):
    print(position, models[position], reward)