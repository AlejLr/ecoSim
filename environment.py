import pygame
import random
from config import *

COLOR_EMPTY = (220, 220, 220)
COLOR_VEGETATION = (34, 139, 34)
COLOR_VISIBLE = (180, 180, 180)
COLOR_AGENT = (255, 0, 0)
COLOR_WATER = (0, 191, 255)

class Tile:
    def __init__(self):
        self.agent = None
        self.has_food = False
        self.is_water = False

class FoodTile(Tile):
    def __init__(self):
        super().__init__()
        self.regrowth = FOOD_REGROWTH_TIME
        self.has_food = True

    def update(self):
        if not self.has_food:
            self.regrowth -= 1
            if self.regrowth <= 0:
                self.has_food = True
                self.regrowth = FOOD_REGROWTH_TIME

    def consume(self):
        if self.has_food:
            self.has_food = False
            self.regrowth = FOOD_REGROWTH_TIME
            return True
        return False
    
class WaterTile(Tile):
    def __init__(self):
        super().__init__()
        self.is_water = True

    def consume(self):
        return True

def generate_water_patch(grid, patch_size):
    patch_w = random.randint(1, patch_size)
    patch_h = random.randint(1, patch_size)
    x = random.randint(0, GRID_SIZE - patch_w)
    y = random.randint(0, GRID_SIZE - patch_h)

    for i in range(patch_h):
        for j in range(patch_w):
            grid[y + i][x + j] = WaterTile()
    
def generate_environement():
    grid = [[Tile() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    for _ in range(round(GRID_SIZE * GRID_SIZE * 0.25)):
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        grid[y][x] = FoodTile()

    for _ in range(WATER_PATCHES):
        generate_water_patch(grid, WATER_PATCH_MAX_SIZE)

    return grid

def update_grid(grid):
    for row in grid:
        for tile in row:
            if isinstance(tile, FoodTile):
                tile.update()

def draw(screen, grid, agent):
    screen.fill((0, 0, 0))
    
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

            tile = grid[y][x]
            if isinstance(tile, WaterTile):
                color = COLOR_WATER
            elif isinstance(tile, FoodTile):
                color = COLOR_VEGETATION
            else:
                color = COLOR_EMPTY

            if not agent.can_see(x, y):
                color = tuple(max(0, c - 80) for c in color)

            pygame.draw.rect(screen, color, rect)

    pygame.draw.rect(screen, COLOR_AGENT, pygame.Rect(agent.x * TILE_SIZE, agent.y * TILE_SIZE, TILE_SIZE, TILE_SIZE))