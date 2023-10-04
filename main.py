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


num_states = 147 #Length of Binary Inputs
num_actions = 5 #Output

# Create the neural network model
model = NeuralNetwork(num_actions)

# Define the optimizer (e.g., Adam)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


num_episodes = 1000
for episode in range(num_episodes):
    state = env.init_game()
    #Converts state to a binary list
    state = state_to_binary(state)
    #Converts state to a numpy array
    state = np.array(state)

    with tf.GradientTape() as tape:
        for _ in range(10000):
            action_probs = model(tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32))
            action = np.argmax(action_probs)
            #Returns a number through 0-5
            action = actions[action]
            #Turns number to an action object

            try:
                next_state=env.step(action)
                next_state=np.array(state_to_binary(next_state))
            except:
                #If this is exception is caught the NN has played an illegal move
                done=True
                reward=-1



            #Below is not fully finished/tested
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
