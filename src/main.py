import pygame
import sys
from random import random, randint

from config.config import *
from environment.environment import *
from agents.agent import *


def main():
    pygame.init()
    
    env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])
    env.use_test("maps/test_map.png")

    for _ in range(INITIAL_PREY_NUMBER):
        x = randint(0, env.width - 1)
        y = randint(0, env.height - 1)
        env.agents.append(Prey(position=(x, y)))
        env.agent_grid[x, y] += 1
    
    for _ in range(INITIAL_PREDATOR_NUMBER):
        x = randint(0, env.width - 1)
        y = randint(0, env.height - 1)
        env.agents.append(Predator(position=(x, y)))
        env.agent_grid[x, y] += 1
    
    screen = pygame.display.set_mode(SUB_GRID_SIZE)
    pygame.display.set_caption("EcoSim")
    
    clock = pygame.time.Clock()
    tick_counter = 0
    
    while tick_counter < 100:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Update all tiles
        for x in range(GRID_SUBENV[0]):
            for y in range(GRID_SUBENV[1]):
                env.tiles[x][y].grow()
        
        # Agents take their actions
        for agent in env.agents:        
            agent.test(env)
        
        # Remove dead agents
        dead_agents = [agent for agent in env.agents if not agent.is_alive()]
        for agent in dead_agents:
            env.remove_agent_from_grid(agent)
        env.agents = [agent for agent in env.agents if agent.is_alive()]

        # Render environment
        surface = pygame.Surface((GRID_SUBENV[0], GRID_SUBENV[1]))
        for x in range(GRID_SUBENV[0]):
            for y in range(GRID_SUBENV[1]):
                surface.set_at((x, y), env.grid[x][y])
                
        scaled_surface = pygame.transform.scale(surface, SUB_GRID_SIZE)

        # Render all agents
        for agent in env.agents:
            ax, ay = agent.position
            scaled_x = int(ax * SUB_TILE_SIZE)
            scaled_y = int(ay * SUB_TILE_SIZE)
            color = (255, 255, 255) if agent.agent_type == "PREY" else (255, 0, 0)
            pygame.draw.circle(scaled_surface, color, (scaled_x, scaled_y), 5)

        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        clock.tick(MAX_FPS)
        tick_counter += 1
        
    pygame.quit()
        
if __name__ == "__main__":
    main()