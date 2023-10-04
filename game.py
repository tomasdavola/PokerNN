from rlcard.games.nolimitholdem.round import Action
from rlcard.games.nolimitholdem.game import NolimitholdemGame
from rlcard.games.nolimitholdem.judger import NolimitholdemJudger
env = NolimitholdemGame(allow_step_back=False, num_players=2)
import tensorflow as tf
import numpy as np
actions = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF_POT, Action.RAISE_POT, Action.ALL_IN]
env = NolimitholdemGame(allow_step_back=False, num_players=2)

state=env.init_game()
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
print(env.step(Action.CHECK_CALL))
judge=NolimitholdemJudger(2)
print(judge.judge_game([0,1],[{'hand': ['CA', 'CJ']}]))