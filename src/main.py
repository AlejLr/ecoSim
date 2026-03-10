import pygame
import sys
import random

from config.config import *
from environment.environment import *


def main():
    pygame.init()
    
    width, height = 150, 150
    scale_factor = 3
    env = grid_env(width, height)
    env.generate()
    
    surface = pygame.Surface((width, height))
    for x in range(width):
        for y in range(height):
            surface.set_at((x, y), env.grid[x][y])
    
    scaled_surface = pygame.transform.scale(surface, (width * scale_factor, height * scale_factor))
    screen = pygame.display.set_mode((width * scale_factor, height * scale_factor))
    pygame.display.set_caption("EcoSim")
    screen.blit(scaled_surface, (0, 0))
    

    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()