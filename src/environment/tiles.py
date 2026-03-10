from config.config import *
class tile():
    def __init__(self, x, y, tile_type):
        self.x = x
        self.y = y
        self.tile_type = tile_type
        self.color = colors[tile_type]

class grass(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "grass")
        self.growth_rate = GROWTH_RATE["grass"]
        self.energy_production = ENERGY_PRODUCTION["grass"]
        
class forest(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "forest")
        self.growth_rate = GROWTH_RATE["forest"]
        self.energy_production = ENERGY_PRODUCTION["forest"]
        
class vegetation(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "vegetation")
        self.growth_rate = GROWTH_RATE["vegetation"]
        self.energy_production = ENERGY_PRODUCTION["vegetation"]
        
class water(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "water")
        self.water_gain = HYDRATION_GAIN_WATER
        
