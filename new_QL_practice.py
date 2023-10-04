import numpy as np
import gym
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
import rlcard
from rlcard.games.nolimitholdem.round import Action
from rlcard.games.nolimitholdem.game import NolimitholdemGame
from toBinary import encode

env =NolimitholdemGame(allow_step_back=False, num_players=2)
discount_factor = 0.95
eps = 0.5
eps_decay_factor = 0.999
learning_rate = 0.8
num_episodes = 500

model = Sequential()
model.add(InputLayer(batch_input_shape=(2, 155)))
model.add(Dense(200, activation='relu'))
model.add(Dense(5, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]

q_table = np.zeros([155, 5])
list=[]
for i in range(num_episodes):

    eps *= eps_decay_factor
    done = False
    state = env.init_game()
    state = encode(state)
    if np.random.random() < eps-1000:
        action = np.random.randint(0, 4)
        action = actions[action]
    else:
        print(model.predict(state).shape)
        action=np.argmax(model.predict(state))
        print(action)
        list.append(action)
        # action=actions[action]
print(max(list), min(list))