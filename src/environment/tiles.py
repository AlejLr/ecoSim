from config.config import *

class Tile():
    """Base tile class"""
    def __init__(self, x, y, tile_type):
        self.x = x
        self.y = y
        self.tile_type = tile_type
        self.has_energy = False
        self.color = colors.get(tile_type, (200, 200, 200))
        
    def eat(self):
        """Returns energy if this tile can be eaten, 0 otherwise"""
        return 0
    
    def drink(self):
        """Returns hydration if this tile provides water, 0 otherwise"""
        return 0


class GrassTile(Tile):
    """Grass tile provides constant energy when eaten"""
    def __init__(self, x, y):
        super().__init__(x, y, "grass")
        self.has_energy = True
        
    def eat(self):
        """Each grass tile provides constant energy"""
        return GRASS_ENERGY


class WaterTile(Tile):
    """Water tile provides constant hydration when drunk"""
    def __init__(self, x, y):
        super().__init__(x, y, "water")
        self.has_energy = False
        
    def drink(self):
        """Each water tile provides constant hydration"""
        return WATER_HYDRATION


class EmptyTile(Tile):
    """Empty/ground tile with no resources"""
    def __init__(self, x, y):
        super().__init__(x, y, "empty")
        self.has_energy = False
