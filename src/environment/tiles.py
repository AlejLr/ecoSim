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
        
class forest(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "forest")
        
class vegetation(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "vegetation")
        
class water(tile):
    def __init__(self, x, y):
        super().__init__(x, y, "water")
        
