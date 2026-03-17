from random import random

from config.config import *
class tile():
    def __init__(self, x, y, tile_type):
        self.x = x
        self.y = y
        self.tile_type = tile_type
        self.has_energy = False
        self.color = colors[tile_type][int(random() * len(colors[tile_type]))]
        
    def eat(self):
        """If not overwritten, this tile provides no energy"""
        return 0
    
    def drink(self):
        """If not overwritten, this tile provides no water"""
        return 0
    
    def grow(self):
        """Growth logic for renewable tiles. Override in subclasses."""
        pass

class RenewableTile(tile):
    """Base class for tiles that regenerate energy (grass, forest, vegetation)."""
    def __init__(self, x, y, tile_type):
        super().__init__(x, y, tile_type)
        self.growth_rate = GROWTH_RATE[tile_type]
        self.energy_production = ENERGY_PRODUCTION[tile_type]
        self.energy = MAX_ENERGY[tile_type]//2
        self.has_energy = True
        
    def eat(self):
        """Consume this tile's energy if available."""
        gain = self.energy
        self.energy = 0
        return gain
    
    def grow(self):
        """Regenerate energy over time based on growth_rate."""
        if self.energy < MAX_ENERGY[self.tile_type]:
            self.energy += self.energy_production * self.growth_rate
            if self.energy >= MAX_ENERGY[self.tile_type]:
                self.energy = MAX_ENERGY[self.tile_type]

class GrassTile(RenewableTile):
    def __init__(self, x, y):
        super().__init__(x, y, "grass")
        
class ForestTile(RenewableTile):
    def __init__(self, x, y):
        super().__init__(x, y, "forest")
                
class VegetationTile(RenewableTile):
    def __init__(self, x, y):
        super().__init__(x, y, "vegetation")

class WaterTile(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "water")
        self.water_gain = HYDRATION_GAIN_WATER
        
    def drink(self):
        return self.water_gain
        
class GroundTile(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "ground")