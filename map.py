## File: map.py
## Author: Logan Caffey
# This file contains the Map class, which represents a grid-based environment for an agent to navigate.

import cv2 as cv
import numpy as np

class Map:
    def __init__(self, length, width,agentSize,MAXSIZE,MAXBANDS):
        self.length = length
        self.width = width
        self.agentSize = agentSize
        self.agent = None
        self.grid = np.zeros((MAXSIZE, MAXSIZE, MAXBANDS), dtype=np.uint8)
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

    # displays map and allows the user to move the agent
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
    
    # displays the map
    def displayBase(self, show=True):
        cell_size = max(1, 1000 // max(self.length, self.width, 1)) # Avoid division by zero
        img_height = self.length * cell_size
        img_width = self.width * cell_size
        img = np.zeros((img_height, img_width, 3), np.uint8) # Base black

        # Walls (white)
        wall_coords = np.argwhere(self.grid[:, :, 0] == 1)
        for x, y in wall_coords:
            cv.rectangle(img, (y * cell_size, x * cell_size),
                        ((y + 1) * cell_size -1 , (x + 1) * cell_size - 1),
                        (255, 255, 255), -1) # Use OpenCV drawing

        # Cleaned (gray)
        cleaned_coords = np.argwhere(self.grid[:, :, 2] == 1)
        for x, y in cleaned_coords:
            cv.rectangle(img, (y * cell_size, x * cell_size),
                        ((y + 1) * cell_size-1, (x + 1) * cell_size-1),
                        (50, 50, 50), -1)

        # Agent (green)
        agent_coords = np.argwhere(self.grid[:, :, 1] == 1)
        for x, y in agent_coords:
            cv.rectangle(img, (y * cell_size, x * cell_size),
                        ((y + 1) * cell_size-1, (x + 1) * cell_size-1),
                        (0, 255, 0), -1)

        if show:
            cv.imshow("Environment", img)
            cv.waitKey(1)
        return img
    
    def close(self):
        cv.destroyAllWindows()

    # moves the agen to a position if there is no collision
    def move_agent(self, x, y):
        if self.checkCollision(x, y):
            return False
        else:
            self.remove_agent()
            self.add_agent(x, y)
            return True

    # movces the agent in a direction at a given speed
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
                
    
    # returns the grid
    def getGrid3D(self):
        return self.grid

    # returns the count of non wall spaces
    def getMovableCount(self):
        return self.length * self.width - self.grid[:,:,0].sum()

