### main.py
# This file contains the main function that initializes the environment and the DQAgent, and runs the training loop.

from map import Map
from DQAgent import DQAgent
import random
import matplotlib.pyplot as plt

print(1111)

MAXSIZE = 100
MAXBANDS = 3

def reset():
    m = Map(100, 100, 10, MAXSIZE, MAXBANDS)
    m.add_wall(0, 0, 100, 10)
    m.add_wall(0, 0, 10, 100)
    m.add_wall(90, 0, 100, 100)
    m.add_wall(0, 90, 100, 100)
    m.add_agent(random.randint(20, 70), random.randint(20, 70))
    return m


agent = DQAgent((MAXSIZE, MAXSIZE, MAXBANDS), 16)
#agent.loadModel("models/Model-latest.weights.h5")

STEP_PENALTY     = -1.0     # every time step
NEW_CELL_REWARD  = +5.0

roundNum = 0
EveryReward = []
m = reset()
mapsize = m.getMovableCount()
print("Map size:", mapsize)
GOAL_REWARD      = mapsize * 2
while roundNum < 10000:
    roundNum += 1
    m = reset()
    allRewards = 0
    visited = set()
    visited.add(m.agent)
    print("Round:", roundNum, "Epsilon:", agent.epsilon)
    if roundNum % 100 == 0:
        oldEps = agent.epsilon
    for i in range(200):
        m.move_direction(agent.act(m.getGrid3D()))
        startLen = len(visited)
        visited.add(m.agent)
        if roundNum % 10 == 0:
            m.displayBase()
        if len(visited) >= mapsize:
            print("-------------------------------------Visited all cells")
            reward = GOAL_REWARD
        elif len(visited) > startLen:
            reward = NEW_CELL_REWARD * (len(visited) - startLen)
        else:
            reward = STEP_PENALTY
        agent.remember(m.getGrid3D(), reward, reward == GOAL_REWARD)
        allRewards += reward
        if reward == GOAL_REWARD:
            break
    print("Total rewards:", allRewards)
    if roundNum % 100 == 0:
        agent.epsilon = oldEps
    EveryReward.append(allRewards)
    plt.plot([sum(EveryReward[max(0,i-100):i+1])/min(100,i+1) for i in range(len(EveryReward))])
    plt.plot([sum(EveryReward[max(0,i-10):i+1])/min(10,i+1) for i in range(len(EveryReward))])
    plt.savefig("rewards.png")

m.displayMove()
m.close()