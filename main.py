from map import Map
from DQAgent import DQAgent
import random

print(1111)
m = Map(10, 10)
m.add_wall(0, 0, 10, 1)
m.add_wall(0, 0, 1, 10)
m.add_wall(9, 0, 10, 10)
m.add_wall(0, 9, 10, 10)
m.add_agent(5, 5)

agent = DQAgent((10, 10, 1), 4)
visited = set()

roundNum = 0
while roundNum < 10000:
    roundNum += 1
    m.move_agent(random.randint(2, 7), random.randint(2, 7))
    print("Round:", roundNum)
    for i in range(1000):
        m.move_direction(agent.act(m.getGrid()))
        if roundNum % 100 == 0:
            m.displayBase()
        visited.add(m.agent)
        if len(visited) >= 64:
            print("Visited all cells")
            reward = 1000
        else:
            reward = len(visited)-64
        agent.remember(m.getGrid(), reward, reward == 1000)
        if reward == 1000:
            break

m.displayMove()
m.close()