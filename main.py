from map import Map
from DQAgent import DQAgent
import random

print(1111)


def reset():
    m = Map(10, 10)
    m.add_wall(0, 0, 10, 1)
    m.add_wall(0, 0, 1, 10)
    m.add_wall(9, 0, 10, 10)
    m.add_wall(0, 9, 10, 10)
    m.add_agent(random.randint(2, 7), random.randint(2, 7))
    return m


agent = DQAgent((10, 10, 1), 4)

roundNum = 0
while roundNum < 10000:
    roundNum += 1
    m = reset()
    allRewards = 0
    visited = set()
    print("Round:", roundNum, "Epsilon:", agent.epsilon)
    for i in range(200):
        m.move_direction(agent.act(m.getGrid()))
        if roundNum % 1 == 0:
            m.displayBase()
            startLen = len(visited)
        visited.add(m.agent)
        if len(visited) >= 64:
            print("Visited all cells")
            reward = 1000
        elif len(visited) > startLen:
            reward = 1
        else:
            reward = len(visited)-64
        agent.remember(m.getGrid(), reward, reward == 1000)
        allRewards += reward
        if reward == 1000:
            break
    print("Total rewards:", allRewards)

m.displayMove()
m.close()