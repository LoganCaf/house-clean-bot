## main.py
## Author: Logan Caffey
# This file contains the main function that initializes the environment and the DQAgent, and runs the training loop.

from map import Map
from DataReader import svg_to_color_grid, svg_to_binary_grid, colorGridTo3dGrid
from DQAgent import DQAgent
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os

# helper Incrementally update a moving-average list
def update_ma(window, running_sum, new_val, ma_store):
    # drop the value that just fell out of the window (if any)
    if len(window) == window.maxlen:
        running_sum -= window[0]
    # push the new value
    window.append(new_val)
    running_sum += new_val
    # append current average
    ma_store.append(running_sum / len(window))
    return running_sum


MAXSIZE = 300
MAXBANDS = 3
GRID_SIZE = [100, 100]

def reset():
    m = Map(100, 100, 11, MAXSIZE, MAXBANDS)
    m.add_wall(0, 0, 100, 10)
    m.add_wall(0, 0, 10, 100)
    m.add_wall(90, 0, 100, 100)
    m.add_wall(0, 90, 100, 100)
    m.add_agent(random.randint(20, 70), random.randint(20, 70))
    return m

static_rgb = svg_to_color_grid('train-00/0000-0003.svg', grid_size=(MAXSIZE, MAXSIZE))



def reset_svg():
    mask = static_rgb
    m = Map(mask.shape[0], mask.shape[1], 11, MAXSIZE, MAXBANDS)
    m.grid[:,:,0] = (colorGridTo3dGrid(mask).copy()[:,:,[0,4,5,6]].sum(axis=2)>0) # sets walls
    
    m.add_agent(50, 200)
    return m


agent = DQAgent((MAXSIZE, MAXSIZE, MAXBANDS), 16)
#agent.loadModel("models/Model-latest.weights.h5")
# agent.epsilon = 0.75


rew_w100, rew_w10 = deque(maxlen=100), deque(maxlen=10)
vis_w100, vis_w10 = deque(maxlen=100), deque(maxlen=10)

sum_rew100 = sum_rew10 = 0.0
sum_vis100 = sum_vis10 = 0.0

MA_rew100, MA_rew10 = [], []        # what will plot
MA_vis100, MA_vis10 = [], []


# game log setup
roundNum = 0
GOAL_REWARD      = 300
STEP_PENALTY     = -0.00001     # every time step
NEW_CELL_REWARD  = +.1
currGoal = 1
goalCount = deque(maxlen=20)
while roundNum < 10000:
    # round setup
    roundNum += 1
    m = reset_svg()
    mapsize = m.getMovableCount()
    allRewards = 0
    print("Round:", roundNum, "Epsilon:", agent.epsilon)
    if roundNum % 100 == 0:
        oldEps = agent.epsilon
    for i in range(1000):
        # agent action
        startLen = m.grid[:,:,2].sum()
        startPos = m.agent
        m.move_direction(agent.act(m.getGrid3D()))
        afterLen = m.grid[:,:,2].sum()

        if roundNum % 10 == 0: # show every 10 rounds
            m.displayBase()
        
        # rewards
        if afterLen >= (mapsize*currGoal):
            goalCount.append(1)
            print("-------------------------------------Visited all cells",currGoal, "Goal count:", sum(goalCount))
            reward = GOAL_REWARD
        if m.agent == startPos:
            reward = STEP_PENALTY*10
        elif afterLen > startLen:
            reward = NEW_CELL_REWARD * (afterLen - startLen) * ((afterLen/mapsize))
        else:
            reward = STEP_PENALTY
        agent.remember(m.getGrid3D(), reward, reward == GOAL_REWARD)
        allRewards += reward
        if reward == GOAL_REWARD:
            break
    if afterLen < (mapsize*currGoal):
        goalCount.append(0)
    print("Total rewards:", allRewards)
    if roundNum % 100 == 0:
        agent.epsilon = oldEps
    
    # update logs and graphs
    visited_pct = (m.grid[:, :, 2].sum() / mapsize) * 100.0   # % of cells cleaned this episode

    # update running moving averages
    sum_rew100 = update_ma(rew_w100, sum_rew100, allRewards, MA_rew100)
    sum_rew10  = update_ma(rew_w10 , sum_rew10 , allRewards, MA_rew10)

    sum_vis100 = update_ma(vis_w100, sum_vis100, visited_pct, MA_vis100)
    sum_vis10  = update_ma(vis_w10 , sum_vis10 , visited_pct, MA_vis10)

    # plots
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # rewards plots
    ax1.plot(MA_rew100, label="100-pt MA")
    ax1.plot(MA_rew10 , label="10-pt MA")
    ax1.set_ylabel("Total reward")
    ax1.legend()
    ax1.grid(True, linestyle=":")

    # % of spaces visited
    ax2.plot(MA_vis100, label="% visited (100-pt MA)")
    ax2.plot(MA_vis10 , label="% visited (10-pt MA)")
    ax2.set_ylabel("Visited %")
    ax2.set_xlabel("Episode")
    ax2.legend()
    ax2.grid(True, linestyle=":")

    fig.tight_layout()
    fig.savefig("rewards.png")
    plt.close(fig)

m.displayMove()
m.close()