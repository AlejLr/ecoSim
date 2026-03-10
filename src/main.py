import pygame
import sys
import random

from config.config import *
from environment.environment import *


def main():
    pygame.init()
    
    env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])
    env.generate()
    
    surface = pygame.Surface((GRID_SUBENV[0], GRID_SUBENV[1]))
    for x in range(GRID_SUBENV[0]):
        for y in range(GRID_SUBENV[1]):
            surface.set_at((x, y), env.grid[x][y])
    
    scaled_surface = pygame.transform.scale(surface, SUB_GRID_SIZE)
    screen = pygame.display.set_mode(SUB_GRID_SIZE)
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