### File: map.py
# This file contains the Map class, which represents a grid-based environment for an agent to navigate.

import cv2 as cv
import numpy as np
from PIL import Image

class Map:
    def __init__(self, length, width,agentSize=11,MAXSIZE=400,MAXBANDS=15):
        self.length = length
        self.width = width
        self.agentSize = agentSize
        self.agent = None
        self.grid = np.zeros((MAXSIZE, MAXSIZE, MAXBANDS)).astype(np.float16)
        # grind bands:
        # if all 0 then assumed nothing is there
        # 0 - wall
        # 1 - agent
        # 2 - cleaned


    def add_wall(self, x1, y1, x2, y2):
        self.grid[x1:x2, y1:y2, 0] = 1 # set wall

    def add_agent(self, x, y):
        self.agent = (x, y)
        self.grid[self.agent[0]-(self.agentSize//2):self.agent[0]+(self.agentSize//2)+1,self.agent[1]-(self.agentSize//2):self.agent[1]+(self.agentSize//2)+1,1] = 1 # remove agent
        self.grid[self.agent[0]-(self.agentSize//2):self.agent[0]+(self.agentSize//2)+1,self.agent[1]-(self.agentSize//2):self.agent[1]+(self.agentSize//2)+1,2] = 1 # set clean
    
    def remove_agent(self):
        self.grid[self.agent[0]-(self.agentSize//2):self.agent[0]+(self.agentSize//2)+1,self.agent[1]-(self.agentSize//2):self.agent[1]+(self.agentSize//2)+1,1] = 0 # remove agent
        self.agent = None
    
    def checkCollision(self, x, y):
        return x-(self.agentSize//2) < 0 or x+(self.agentSize//2) >= self.length or y-(self.agentSize//2) < 0 or y+(self.agentSize//2) >= self.width or self.grid[x-(self.agentSize//2):x+(self.agentSize//2)+1,y-(self.agentSize//2):y+(self.agentSize//2)+1,0].sum() != 0

    def displayMove(self):
        while True:
            img = self.displayBase(False)
            key = cv.waitKey(1) & 0xFF
            if key == ord('w'):
                self.move_agent(self.agent[0] - 1, self.agent[1])
            elif key == ord('a'):
                self.move_agent(self.agent[0], self.agent[1] - 1)
            elif key == ord('s'):
                self.move_agent(self.agent[0] + 1, self.agent[1])
            elif key == ord('d'):
                self.move_agent(self.agent[0], self.agent[1] + 1)
            if key == ord('q'):
                break
            cv.imshow("Environment", img)
    
    def displayBase(self,show=True):
        cell_size = 5
        img = np.zeros((self.length * cell_size, self.width * cell_size, 3), np.uint8)
        for x in range(self.length):
            for y in range(self.width):
                color = [0, 0, 0]
                if self.grid[x,y,0] == 1:
                    color = [255, 255, 255]
                elif self.grid[x,y,1] == 1:
                    color = [0, 255, 0]
                elif self.grid[x,y,2] == 1:
                    color = [50, 50, 50]
                img[x * cell_size:(x + 1) * cell_size, y * cell_size:(y + 1) * cell_size] = color

        if show:
            cv.imshow("Environment", img)
            cv.waitKey(1)
        return img
    
    def close(self):
        cv.destroyAllWindows()

    def move_agent(self, x, y):
        if self.checkCollision(x, y):
            return False
        else:
            self.remove_agent()
            self.add_agent(x, y)
            return True

    
    def move_direction(self, direction):
        mult = (((direction // 4)+1)*self.agentSize)//4 # movement speed
        direction = direction%4 #movement direction
        while True:
            if mult == 0:
                break
            match direction:
                case 0:
                    if self.move_agent(self.agent[0] - mult, self.agent[1]):
                        break
                case 1:
                    if self.move_agent(self.agent[0] + mult, self.agent[1]):
                        break
                case 2:
                    if self.move_agent(self.agent[0], self.agent[1] - mult):
                        break
                case 3:
                    if self.move_agent(self.agent[0], self.agent[1] + mult):
                        break
            mult -= 1
                
    

    def getGrid3D(self):
        return self.grid

    def getMovableCount(self):
        count = 0
        for x in range(self.length):
            for y in range(self.width):
                if self.grid[x,y,0] != 1:
                    count += 1
        return count
    
    def getCleaned(self):
        arr = np.zeros((self.length, self.width))
        for x in range(self.length):
            for y in range(self.width):
                if self.grid[x][y] == 'c':  # or however this is set to read if a spot is clean / has been visited
                    arr[x, y] = 1.0
        return arr
    
    def getAgent(self):
        arr = np.zeros((self.length, self.width))
        ax, ay = self.agent
        arr[ax, ay] = 1.0
        return arr


        

if __name__ == "__main__":
    m = Map(10, 10)
    m.add_wall(0, 0, 10, 1)
    m.add_wall(0, 0, 1, 10)
    m.add_wall(9, 0, 10, 10)
    m.add_wall(0, 9, 10, 10)
    m.add_agent(5, 5)
    m.displayMove()
    m.close()

