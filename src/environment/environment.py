import numpy as np

import config.config as cfg
from environment.tiles import *

class grid_env():
    def __init__(self, width, height):
        
        self.width = width
        self.height = height
        
        self.grid = np.zeros((width, height), dtype=tuple)
        self.tiles = np.empty((width, height), dtype=object)
        
        self.tile_types = [grass, forest, vegetation, water]
        
    def generate(self):
        for x in range(self.width):
            for y in range(self.height):
                tile_type = np.random.choice(self.tile_types)
                self.tiles[x][y] = tile_type(x, y)
                self.grid[x][y] = self.tiles[x][y].color
                