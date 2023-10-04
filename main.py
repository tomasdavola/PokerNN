from rlcard.games.nolimitholdem.round import Action
import matplotlib.pyplot as plt
from rlcard.games.nolimitholdem.game import NolimitholdemGame
from toBinary import state_to_binary
env = NolimitholdemGame(allow_step_back=False, num_players=2)
import tensorflow as tf
import numpy as np
actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
env = NolimitholdemGame(allow_step_back=False, num_players=2)

# Define the neural network model
class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(NeuralNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Define the environment (replace with your specific RL environment)
# For this example, we'll create a random environment
num_states = 147
num_actions = 5

# Create the neural network model
model = NeuralNetwork(num_actions)

# Define the optimizer (e.g., Adam)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop (for demonstration purposes)
num_episodes = 1000
for episode in range(num_episodes):
    state = env.init_game()
    state = state_to_binary(state)
    state = np.array(state)

    with tf.GradientTape() as tape:
        for _ in range(10000):  # Adjust the maximum time steps as needed
            # Replace this with your policy for selecting actions
            action_probs = model(tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32))
            action = np.argmax(action_probs)
            action = actions[action]

            # Simulate the environment and get the next state and reward
            try:
                next_state=env.step(action)
                next_state=np.array(state_to_binary(next_state))
            except:
                done=True
                reward=-1

            # Replace with your environment's next state
            reward = np.random.randn()

            # Calculate the TD error and loss
            target = reward + model(tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32))
            current = model(tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32))
            loss = tf.reduce_mean(tf.square(target - current))

            state = next_state
            done=True
            if done:
                break

    # Update the model's weights
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if episode % 10 == 0:
        plt.plot(episode,reward)
        print(f"Episode: {episode}, Total Reward: {reward}")
plt.show()
# After training, you can use the trained model for decision making.
