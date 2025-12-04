from config import *

class Agent:
    def __init__(self, x, y, grid):
        self.x = x
        self.y = y
        self.energy = INITIAL_ENERGY
        self.water = INITIAL_WATER
        self.grid = grid

    def is_alive(self):
        return self.energy > 0 and self.water > 0

    def get_perception_indices(self):
        directions = [(-2, 0), (-1, 0), (1, 0), (2, 0),
                      (0, -2), (0, -1), (0, 1), (0, 2),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        indices = []
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                indices.append((nx, ny))
        return indices
    
    def perceive(self):
        return [(nx, ny in self.grid[nx][ny].type) for nx, ny in self.get_perception_indices()]

    def step(self, action):
        dxdy = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        if action in range(5):
            dx, dy = dxdy[action]
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                self.x, self.y = nx, ny

        elif action == 5:
            if self.grid[self.y][self.x].consume_food():
                self.energy = min(self.energy + ENERGY_GAIN, MAX_ENERGY)

        elif action == 6:
            if self.grid[self.y][self.x].is_water:
                self.water = min(self.water + WATER_GAIN, MAX_WATER)

        self.water -= WATER_DECAY
        self.energy -= ENERGY_DECAY

        