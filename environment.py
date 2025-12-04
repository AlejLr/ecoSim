import pygame
import random
from agent import Agent
import config

class Ecosystem:
    def __init__(self, grid_size, tile_size):
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.agent = Agent(grid_size//2, grid_size//2)
        self._generate_energy_tiles()

    def _generate_energy_tiles(self):
        for _ in range(self.grid_size * 2):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.grid[y][x] = 1

    def update(self):
        self.agent.move(self.grid)

    def render(self, screen):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = config.COLOR_EMPTY
                if self.grid[y][x] == 1:
                    color = config.COLOR_ENERGY
                rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
                pygame.draw.rect(screen, color, rect)
        
        ax, ay = self.agent.x, self.agent.y
        agent_rect = pygame.Rect(ax * self.tile_size, ay * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(screen, config.COLOR_AGENT, agent_rect)