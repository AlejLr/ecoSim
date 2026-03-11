import numpy as np
from PIL import Image

import config.config as cfg
from environment.tiles import *

class grid_env():
    def __init__(self, width, height):
        
        self.width = width
        self.height = height
        
        self.grid = np.zeros((width, height), dtype=tuple)
        self.tiles = np.empty((width, height), dtype=object)
        self.agents = np.zeros((width, height), dtype=int)
        
        self.tile_class = {
            "grass": GrassTile,
            "forest": ForestTile,
            "vegetation": VegetationTile,
            "water": WaterTile,
            "ground": GroundTile
        }
        
        
    def generate(self):
        for x in range(self.width):
            for y in range(self.height):
                tile_type = np.random.choice(list(self.tile_class.values()))
                self.tiles[x][y] = tile_type(x, y)
                self.grid[x][y] = self.tiles[x][y].color
                
    def use_test(self, path):
        image_grid = self._import_color_grid(path)
        
        if image_grid.shape[:2] != (self.height, self.width):
            raise ValueError("Imported map size does not match environment dimensions.")

        self.grid = np.zeros((self.width, self.height), dtype=object)
        self.tiles = np.empty((self.width, self.height), dtype=object)
        
        for x in range(self.width):
            for y in range(self.height):
                color = tuple(image_grid[y, x])
                tile_type = self._map_color_to_tile(color)
                
                tile_class = self.tile_class[tile_type]
                self.tiles[x][y] = tile_class(x, y)
                self.grid[x][y] = color
        
    def _import_color_grid(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height))
        return np.array(img)
                    
    def _map_color_to_tile(self, color, threshold=50):
        closest_tile = None
        min_dist = float('inf')

        for tile_type, valid_colors in colors.items():
            for ref_color in valid_colors:
                dist = np.linalg.norm(np.array(color) - np.array(ref_color))
                if dist < min_dist and dist <= threshold:
                    min_dist = dist
                    closest_tile = tile_type

        if closest_tile is not None:
            return closest_tile

        raise ValueError(f"Color {color} does not match any known tile type (min_dist={min_dist:.2f})")