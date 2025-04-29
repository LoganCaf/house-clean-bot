### main.py
# This file contains the main function that initializes the environment and the DQAgent, and runs the training loop.

from map import Map
from DataReader import svg_to_color_grid, svg_to_binary_grid
from DQAgent import DQAgent
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

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

static_rgb = svg_to_color_grid('train-00/0000-0003.svg', grid_size=(MAXSIZE, MAXSIZE))    # may want to adapt this into an array of the maps eventually
map_files = [
    'train-00/0000-0002.svg',
    'train-00/0000-0003.svg',
    'train-00/0000-0005.svg'
]
static_maps = [
    svg_to_color_grid(path, grid_size=(MAXSIZE, MAXSIZE))
    for path in map_files
]


# This function may require more implementation to incorporate color
def reset_svg():
    mask = static_rgb   # change this to the function calling a random map once it works
    m = Map(GRID_SIZE[0], GRID_SIZE[1])
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            if mask[i, j]:
                m.grid[i][j] = '0'
    
    # Start agent on a free cell
    free_cells = list(zip(*np.where(mask == 0)))
    start = random.choice(free_cells)
    m.add_agent(*start)


# Starting with binary maps
def reset_svg_binary():
    mask = svg_to_binary_grid(static_rgb)
    m = Map(GRID_SIZE[0], GRID_SIZE[1])
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            if mask[i, j]:
                m.grid[i][j] = '0'      # check this value...
    
    # Start agent on a free cell
    free_cells = list(zip(*np.where(mask == 0)))
    start = random.choice(free_cells)
    m.add_agent(*start)
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


roundNum = 0
m = reset()
mapsize = m.getMovableCount()
print("Map size:", mapsize)
GOAL_REWARD      = 300
STEP_PENALTY     = -0.00001     # every time step
NEW_CELL_REWARD  = +.1
currGoal = 1
goalCount = deque(maxlen=20)
while roundNum < 10000:
    roundNum += 1
    m.close()
    m = reset()     # this may not be the portion to comment out but I think it will help establish the map as the training data? -Z
    allRewards = 0
    print("Round:", roundNum, "Epsilon:", agent.epsilon)
    if roundNum % 100 == 0:
        oldEps = agent.epsilon
    for i in range(300):
        startLen = m.grid[:,:,2].sum()
        m.move_direction(agent.act(m.getGrid3D()))
        afterLen = m.grid[:,:,2].sum()

        if roundNum % 10 == 0:
            m.displayBase()
        if afterLen >= (mapsize*currGoal):
            goalCount.append(1)
            if sum(goalCount) >= 15:
                currGoal += .01
                if currGoal > 1:
                    currGoal = 1
                    break
                agent.epsilon = max(.2,agent.epsilon*1.2)
                goalCount.clear()
                agent.saveModel(currGoal)
                print("Goal reached, making harder",currGoal)
            print("-------------------------------------Visited all cells",currGoal, "Goal count:", sum(goalCount))
            reward = GOAL_REWARD
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
    

    visited_pct = (m.grid[:, :, 2].sum() / mapsize) * 100.0   # % of cells cleaned this episode

    # update running moving averages
    sum_rew100 = update_ma(rew_w100, sum_rew100, allRewards, MA_rew100)
    sum_rew10  = update_ma(rew_w10 , sum_rew10 , allRewards, MA_rew10)

    sum_vis100 = update_ma(vis_w100, sum_vis100, visited_pct, MA_vis100)
    sum_vis10  = update_ma(vis_w10 , sum_vis10 , visited_pct, MA_vis10)

    # --- plotting ---
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # rewards (existing plot)
    ax1.plot(MA_rew100, label="100-pt MA")
    ax1.plot(MA_rew10 , label="10-pt MA")
    ax1.set_ylabel("Total reward")
    ax1.legend()
    ax1.grid(True, linestyle=":")

    # NEW: % of spaces visited
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