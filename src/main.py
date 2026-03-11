import pygame
import sys
import random

from config.config import *
from environment.environment import *
from agents.agent import *


def main():
    pygame.init()
    
    env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])
    env.use_test("maps/test_map.png")
    
    agent = Agent(position=(75, 75), agent_type="prey")
    env.agents[agent.position[0]][agent.position[1]] = 1
    
    screen = pygame.display.set_mode(SUB_GRID_SIZE)
    pygame.display.set_caption("EcoSim")
    
    clock = pygame.time.Clock()
    tick_counter = 0
    
    while tick_counter < 100:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        agent.test(env)
        
        surface = pygame.Surface((GRID_SUBENV[0], GRID_SUBENV[1]))
        for x in range(GRID_SUBENV[0]):
            for y in range(GRID_SUBENV[1]):
                surface.set_at((x, y), env.grid[x][y])
                
        scaled_surface = pygame.transform.scale(surface, SUB_GRID_SIZE)

        ax, ay = agent.position
        scaled_x = int(ax * SUB_TILE_SIZE)
        scaled_y = int(ay * SUB_TILE_SIZE)
        pygame.draw.circle(scaled_surface, (255, 255, 255), (scaled_x, scaled_y), 5)

        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        clock.tick(MAX_FPS)
        tick_counter += 1
        
    pygame.quit()
        
if __name__ == "__main__":
    main()