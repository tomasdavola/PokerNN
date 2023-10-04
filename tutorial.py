import tensorflow as tf
import numpy as np
import rlcard
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.utils.logger import Logger
from rlcard.agents import (
    CFRAgent,
    RandomAgent,
)

from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

# Set the global random seed for reproducibility
# Initialize RLCard environment for No-Limit Texas Hold'em
env = rlcard.make('no-limit-holdem')
eval_env = rlcard.make('no-limit-holdem')

# Initialize the DQN agent
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64,64]
)
agent2 = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64,64]
)
agent3 = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64,64]
)
env.set_agents([agent,agent2,agent3])


with Logger("experiments/leduc_holdem_dqn_result/") as logger:
    for episode in range(1000):

        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        # print(f"HIIIII {len(trajectories[0][0]['obs'])}")
        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)
        print(trajectories)


        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agent.feed(ts)

        # Evaluate the performance.
        if episode % 50 == 0:
            logger.log_performance(
                env.timestep,
                tournament(
                    env,
                    10000,
                )[0]
            )
#
#     # Get the paths
    csv_path, fig_path = logger.csv_path, logger.fig_path


# with Logger("experiments/leduc_holdem_cfr_result") as logger:
#     for episode in range(1000):
#         agent.train()
#         print('\rIteration {}'.format(episode), end='')
#         # Evaluate the performance. Play with Random agents.
#         if episode % 50 == 0:
#             logger.log_performance(
#                 env.timestep,
#                 tournament(
#                     eval_env,
#                     10000,
#                 )[0]
#             )

    # Get the paths
    # csv_path, fig_path = logger.csv_path, logger.fig_path
plot_curve(csv_path, fig_path, "DQN")