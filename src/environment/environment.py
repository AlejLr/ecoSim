import numpy as np
from PIL import Image
from collections import defaultdict

import config.config as cfg
from environment.tiles import *

class grid_env():
    def __init__(self, width, height):
        
        self.width = width
        self.height = height
        
        self.grid = np.zeros((width, height), dtype=tuple)
        self.tiles = np.empty((width, height), dtype=object)
        self.agent_grid = np.zeros((width, height), dtype=int)
        self.agents = []
        self.agents_by_position = defaultdict(list)
        
        self.tile_class = {
            "grass": GrassTile,
            "water": WaterTile,
            "empty": EmptyTile
        }
        
    def generate(self):
        """Randomly generate the environment grid with different tile types."""
        from config.config import TILE_DISTRIBUTION
        
        tile_types = list(TILE_DISTRIBUTION.keys())
        tile_probs = list(TILE_DISTRIBUTION.values())
        
        for x in range(self.width):
            for y in range(self.height):
                tile_type = np.random.choice(tile_types, p=tile_probs)
                tile_class = self.tile_class[tile_type]
                self.tiles[x][y] = tile_class(x, y)
                self.grid[x][y] = self.tiles[x][y].color
                
    def use_test(self, path):
        """Load a test map from an image file."""
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
        """Imports an image and converts it to a grid of RGB tuples."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height))
        return np.array(img)
                    
    def _map_color_to_tile(self, color, threshold=50):
        """Maps an RGB color to a tile type based on the closest match"""
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
    
    def update_agent_position(self, agent, old_pos, new_pos):
        """Update agent's position in the agent_grid."""
        if not (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height):
            return False
        self.agent_grid[old_pos] -= 1
        self.agent_grid[new_pos] += 1
        self.agents_by_position[old_pos].remove(agent)
        self.agents_by_position[new_pos].append(agent)
        return True
    
    def get_agents_at(self, pos):
        """Get count of agents at a specific position."""
        if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
            return self.agent_grid[pos]
        return 0
    
    def get_agents_nearby(self, pos, radius):
        """Get all agents within a certain radius of a position."""
        agents_nearby = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_pos = (pos[0] + dx, pos[1] + dy)
                if check_pos in self.agents_by_position:
                    agents_nearby.extend(self.agents_by_position[check_pos])
        return agents_nearby

    def get_tiles_nearby(self, pos, radius) :
        """Get all tiles within radius."""
        tiles_nearby = []
        x, y = pos
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_x = x + dx
                check_y = y + dy
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    tiles_nearby.append(self.tiles[check_x][check_y])
        return tiles_nearby