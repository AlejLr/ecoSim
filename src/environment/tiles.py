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

    def update(self):
        """Advance tile state by one simulation step."""
        return None


class GrassTile(Tile):
    """Grass tile that depletes when eaten and regrows after a delay"""
    def __init__(self, x, y):
        super().__init__(x, y, "grass")
        self.has_energy = True
        self.regrowth_timer = 0
        
    def eat(self):
        """Return energy if grass is available, otherwise 0."""
        if not self.has_energy:
            return 0

        self.has_energy = False
        self.regrowth_timer = GRASS_REGROWTH_STEPS
        return GRASS_ENERGY

    def update(self):
        """Count down to regrowth and make the tile available again."""
        if not self.has_energy and self.regrowth_timer > 0:
            self.regrowth_timer -= 1
            if self.regrowth_timer <= 0:
                self.has_energy = True
                self.regrowth_timer = 0


class EmptyTile(Tile):
    """Empty/ground tile with no resources"""
    def __init__(self, x, y):
        super().__init__(x, y, "empty")
        self.has_energy = False
