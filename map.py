import cv2 as cv
import numpy as np
from PIL import Image

class Map:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.grid = [['e' for _ in range(width)] for _ in range(length)]
        self.agent = None

    def add_wall(self, x1, y1, x2, y2):
        for x in range(x1, x2):
            for y in range(y1, y2):
                self.grid[x][y] = 'w'

    def add_agent(self, x, y):
        self.grid[x][y] = 'a'
        self.agent = (x, y)

    def display(self):
        cell_size = 50
        img = np.zeros((self.length * cell_size, self.width * cell_size, 3), np.uint8)
        while True:
            for x in range(self.length):
                for y in range(self.width):
                    color = [0, 0, 0]
                    if self.grid[x][y] == 'w':
                        color = [255, 255, 255]
                    elif self.grid[x][y] == 'a':
                        color = [0, 255, 0]
                    img[x * cell_size:(x + 1) * cell_size, y * cell_size:(y + 1) * cell_size] = color
                    
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
        
        cv.destroyAllWindows()

    def move_agent(self, x, y):
        if self.grid[x][y] != 'w':
            self.grid[self.agent[0]][self.agent[1]] = 'e'
            self.grid[x][y] = 'a'
            self.agent = (x, y)

m = Map(10, 10)
m.add_wall(0, 0, 10, 1)
m.add_wall(0, 0, 1, 10)
m.add_wall(9, 0, 10, 10)
m.add_wall(0, 9, 10, 10)
m.add_agent(5, 5)
m.display()
