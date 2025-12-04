import random

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 10

    def move(self, grid):
        dx, dy = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
        new_x = max(0, min(self.x + dx, len(grid[0]) - 1))
        new_y = max(0, min(self.y + dy, len(grid) - 1))

        if grid[new_y][new_x] == 1:
            self.energy += 5
            grid[new_y][new_x] = 0
        else:
            self.energy -= 1
        
        self.x = new_x
        self.y = new_y

        