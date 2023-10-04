import tensorflow as tf
import numpy as np
import rlcard
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.utils.logger import Logger

# Set the global random seed for reproducibility
np.random.seed(0)

# Initialize RLCard environment for No-Limit Texas Hold'em
env = rlcard.make('no-limit-holdem')

# Define the neural network model using TensorFlow
def build_model():
    input_shape = (env.state_shape[0],)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(env.action_num, activation='linear')
    ])
    return model

# Define hyperparameters for DQN
total_episodes = 100
memory_size = 2000
batch_size = 128
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 5000
learning_rate = 0.001
update_target_freq = 100

# Initialize the DQN agent
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64,64],
)

# Initialize logger
logger = Logger('./experiments/dqn_no_limit_holdem')

# Training loop
for episode in range(total_episodes):
    trajectories, _ = env.run(is_training=True)
    for trajectory in trajectories:
        agent.feed(trajectory)

    # Train the agent
    loss = agent.train()
    logger.log_performance(episode, loss)

    # Print episode info
    if episode % 100 == 0:
        print(f'Episode {episode}/{total_episodes}, Loss: {loss:.4f}')

# Save the trained model
agent.save('./models/dqn_no_limit_holdem')

# Evaluate the trained agent
evaluate_num = 1000
eval_env = rlcard.make('no-limit-holdem', config={'seed': 0})
eval_env.set_agents([agent, random_agent])
reward = 0
for _ in range(evaluate_num):
    _, payoffs = eval_env.run(is_training=False)
    reward += payoffs[0]
average_reward = reward / evaluate_num
print(f'Average reward: {average_reward:.2f}')
