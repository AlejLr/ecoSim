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
        self.agent_grid = np.zeros((width, height), dtype=int)
        self.agents = []
        
        self.tile_class = {
            "grass": GrassTile,
            "forest": ForestTile,
            "vegetation": VegetationTile,
            "water": WaterTile,
            "ground": GroundTile
        }
        
    def generate(self):
        """Randomly generate the environment grid with different tile types."""
        for x in range(self.width):
            for y in range(self.height):
                tile_type = np.random.choice(list(self.tile_class.values()))
                self.tiles[x][y] = tile_type(x, y)
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
        return True
    
    def get_agents_at(self, pos):
        """Get count of agents at a specific position."""
        if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
            return self.agent_grid[pos]
        return 0
    
    def get_agents_nearby(self, pos, radius):
        """Get all agents within radius using spatial partitioning via agent_grid."""
        agents_nearby = []
        x, y = pos
        # Check only nearby grid positions instead of all agents
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_x = x + dx
                check_y = y + dy
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    # If there are agents at this position, find them
                    if self.agent_grid[check_x, check_y] > 0:
                        for agent in self.agents:
                            if agent.position == (check_x, check_y):
                                agents_nearby.append(agent)
        return agents_nearby
    
    def remove_agent_from_grid(self, agent):
        """Remove agent from the agent_grid."""
        pos = agent.position
        if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
            self.agent_grid[pos] -= 1
            
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