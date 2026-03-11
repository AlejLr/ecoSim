from random import random

from config.config import *
class tile():
    def __init__(self, x, y, tile_type):
        self.x = x
        self.y = y
        self.tile_type = tile_type
        self.color = colors[tile_type][int(random() * len(colors[tile_type]))]

class GrassTile(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "grass")
        self.growth_rate = GROWTH_RATE["grass"]
        self.energy_production = ENERGY_PRODUCTION["grass"]
        
        self.energy = self.energy_production
        self.has_energy = True
        
        self.counter = 0
        
    def eat(self):
        if self.has_energy:
            self.energy = 0
            self.has_energy = False
            return self.energy_production
        else:
            return 0
    
    def grow(self):
        if not self.has_energy:
            self.counter += 1
            if self.counter >= (1 / self.growth_rate):
                self.energy = self.energy_production
                self.has_energy = True
                self.counter = 0
        
class ForestTile(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "forest")
        self.growth_rate = GROWTH_RATE["forest"]
        self.energy_production = ENERGY_PRODUCTION["forest"]
        
        self.energy = self.energy_production
        self.has_energy = True
        
        self.counter = 0
    def eat(self):
        if self.has_energy:
            self.energy = 0
            self.has_energy = False
            return self.energy_production
        else:
            return 0
        
    def grow(self):
        
        if not self.has_energy:
            self.counter += 1
            if self.counter >= (1 / self.growth_rate):
                self.energy = self.energy_production
                self.has_energy = True
                self.counter = 0
                
class VegetationTile(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "vegetation")
        self.growth_rate = GROWTH_RATE["vegetation"]
        self.energy_production = ENERGY_PRODUCTION["vegetation"]
        
    def eat(self):
        if self.has_energy:
            self.energy = 0
            self.has_energy = False
            return self.energy_production
        else:
            return 0

    def grow(self):
        if not self.has_energy:
            self.counter += 1
            if self.counter >= (1 / self.growth_rate):
                self.energy = self.energy_production
                self.has_energy = True
                self.counter = 0

class WaterTile(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "water")
        self.water_gain = HYDRATION_GAIN_WATER
        
    def drink(self):
        return self.water_gain
        
class GroundTile(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "ground")