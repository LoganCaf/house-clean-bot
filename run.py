## run.py
## Author: Logan Caffey
# This file runs the AI 1 time

from map import Map
from DataReader import svg_to_color_grid, colorGridTo3dGrid
from DQAgent import DQAgent
import random
from collections import deque


MAXSIZE = 300
MAXBANDS = 3
GRID_SIZE = [100, 100]

static_rgb = svg_to_color_grid('train-00/0000-0003.svg', grid_size=(MAXSIZE, MAXSIZE))    # may want to adapt this into an array of the maps eventually
def reset_svg():
    mask = static_rgb   # change this to the function calling a random map once it works
    m = Map(mask.shape[0], mask.shape[1], 11, MAXSIZE, MAXBANDS)
    m.grid[:,:,0] = (colorGridTo3dGrid(mask).copy()[:,:,[0,4,5,6]].sum(axis=2)>0) # sets walls
    m.add_agent(50, 200)
    return m

agent = DQAgent((MAXSIZE, MAXSIZE, MAXBANDS), 16)
agent.loadModel("models/Model-30-04-2025-00-34.weights.h5")
agent.epsilon = 0
agent.learningRate = 0



GOAL_REWARD      = 300
STEP_PENALTY     = -0.00001
NEW_CELL_REWARD  = +.1

# map set up
m = reset_svg()
mapsize = m.getMovableCount()
allRewards = 0
lastPos = deque(maxlen=30)
for i in range(1000): # run for 1000 steps in 1 game

    # agemnt action
    startLen = m.grid[:,:,2].sum()
    lastPos.append(m.agent)
    m.move_direction(agent.act(m.getGrid3D()))
    options = list(range(16))
    random.shuffle(options)
    agent.epsilon = max(0.01, agent.epsilon - 0.1)
    while m.agent in lastPos: # if stuck move the agent randomly
        m.move_direction(options.pop(0))
        agent.epsilon = 1
        if len(options) == 0:
            print("No more options")
            break
    afterLen = m.grid[:,:,2].sum()

    m.displayBase()

    #rewards
    if afterLen >= (mapsize):
        print("-------------------------------------Visited all cells")
        reward = GOAL_REWARD
    if m.agent == lastPos[-1]:
        reward = STEP_PENALTY*10
    elif afterLen > startLen:
        reward = NEW_CELL_REWARD * (afterLen - startLen) * ((afterLen/mapsize))
    else:
        reward = STEP_PENALTY
    
    agent.remember(m.getGrid3D(), reward, reward == GOAL_REWARD)
    allRewards += reward
    if reward == GOAL_REWARD:
        break
print("Total rewards:", allRewards)
m.close()