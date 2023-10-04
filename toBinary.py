from rlcard.games.nolimitholdem.game import Stage
from rlcard.games.nolimitholdem.game import Action
import numpy as np
def int_to_binary(pot):
    if pot>=0:
        pos=0
    else:
        pos=1
    pot=abs(pot)
    num = list(bin(pot)[2:])
    num = [eval(i) for i in num]
    while len(num)!=7:
        num.insert(0,0)
    num.insert(0,pos)
    return(num)

def stage_to_binary(stage):
    stages = [0, 0 , 0 , 0 , 0, 0]
    if stage == Stage.PREFLOP:
        stages[0]=1
        return stages
    elif stage ==  Stage.FLOP:
        stages[1]=1
        return stages
    elif stage ==  Stage.TURN:
        stages[2]=1
        return stages
    elif stage ==  Stage.RIVER:
        stages[3]=1
        return stages
    elif stage ==  Stage.END_HIDDEN:
        stages[4]=1
        return stages
    elif stage ==  Stage.SHOWDOWN:
        stages[5]=1
        return stages

def action_to_binary(legal_actions):
    actions = [0, 0 , 0 , 0 , 0]
    for action in actions:
        if action==Action.FOLD:
            actions[0]=1
        elif action==Action.CHECK_CALL:
            actions[1]=1
        elif action==Action.RAISE_HALF_POT:
            actions[2]=1
        elif action==Action.RAISE_POT:
            actions[3]=1
        elif action == Action.RAISE_FULL_POT:
            actions[3] = 1
        elif action==Action.ALL_IN:
            actions[4]=1
    return actions

def cards_to_binary(cards):
    binary_list = [0] * 52
    for card in cards:
        value=0
        suit=0
        if card[1] == "K":
            value+=12
        elif card[1] == "Q":
            value+=11
        elif card[1] == "J":
            value+=10
        elif card[1] == "T":
            value+=9
        elif card[1] == "A":
            pass
        else:
            value+=int(card[1])-1
        if card[0] == "S":
            suit+=1
        elif card[0] == "C":
            suit+=2
        elif card[0] == "D":
            suit+=3
        binary_list[suit*13+value]+=1
    return binary_list

def state_to_binary(trajectories):
    items = []
    # Hand
    hand = trajectories[0]['hand']  # 52
    items.append(cards_to_binary(hand))
    # Public card
    public_cards = trajectories[0]['public_cards']  # 52
    items.append(cards_to_binary(public_cards))
    # Legal actioms
    legal_actions = trajectories[0]["legal_actions"]  # 5
    items.append(action_to_binary(legal_actions))
    # Stage
    stage = trajectories[0]["stage"]  # 6
    items.append(stage_to_binary(stage))
    # Ints
    pot = trajectories[0]["pot"]  # 8
    items.append(int_to_binary(pot))
    my_chips = trajectories[0]["my_chips"]  # 8
    items.append(int_to_binary(my_chips))
    all_chips = trajectories[0]["all_chips"]  # 16
    items.append(int_to_binary(all_chips[0]))
    items.append(int_to_binary(all_chips[0]))
    items = [item for sublist in items for item in sublist]
    return items

def encode(state):
    state=state_to_binary(state)
    encoded_array = np.zeros((len(state), 2), dtype=int)
    encoded_array[np.arange(len(state)), state] = 1
    return encoded_array