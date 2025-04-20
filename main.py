### main.py
# This file contains the main function that initializes the environment and the DQAgent, and runs the training loop.

from map import Map
from DQAgent import DQAgent
import random
import matplotlib.pyplot as plt

print(1111)


def reset():
    m = Map(10, 10)
    m.add_wall(0, 0, 10, 1)
    m.add_wall(0, 0, 1, 10)
    m.add_wall(9, 0, 10, 10)
    m.add_wall(0, 9, 10, 10)
    m.add_agent(random.randint(2, 7), random.randint(2, 7))
    return m


agent = DQAgent((10, 10, 4), 4)
agent.loadModel("models/Model-latest.weights.h5")

STEP_PENALTY     = -1     # every time step
NEW_CELL_REWARD  = +1.0

roundNum = 0
EveryReward = []
m = reset()
mapsize = m.getMovableCount()
GOAL_REWARD      = mapsize * 2
while roundNum < 10000:
    roundNum += 1
    m = reset()
    allRewards = 0
    visited = set()
    print("Round:", roundNum, "Epsilon:", agent.epsilon)
    if roundNum % 100 == 0:
        oldEps = agent.epsilon
    for i in range(200):
        m.move_direction(agent.act(m.getGrid3D()))
        if roundNum % 10 == 0:
            m.displayBase()
        startLen = len(visited)
        visited.add(m.agent)
        if len(visited) >= mapsize:
            print("-------------------------------------Visited all cells")
            reward = GOAL_REWARD
        elif len(visited) > startLen:
            reward = NEW_CELL_REWARD
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