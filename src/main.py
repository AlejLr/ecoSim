import pygame
import sys
from random import randint

from config.config import *
from environment.environment import *
from agents.agent import *


def main():
    """Simple Pygame visualization for ecosystem simulation"""
    pygame.init()
    
    # Create environment
    env = grid_env(GRID_SUBENV[0], GRID_SUBENV[1])
    env.generate()  # Generate random tiles

    # Create initial agents
    initial_predator_number = max(1, NUM_AGENTS // 5)
    initial_prey_number = max(1, NUM_AGENTS - initial_predator_number)

    agent_id = 0
    for i in range(initial_prey_number):
        x = randint(0, env.width - 1)
        y = randint(0, env.height - 1)
        prey = Prey(agent_id, (x, y))
        env.agents.append(prey)
        env.agent_grid[x, y] += 1
        env.agents_by_position[(x, y)].append(prey)
        agent_id += 1
    
    for i in range(initial_predator_number):
        x = randint(0, env.width - 1)
        y = randint(0, env.height - 1)
        predator = Predator(agent_id, (x, y))
        env.agents.append(predator)
        env.agent_grid[x, y] += 1
        env.agents_by_position[(x, y)].append(predator)
        agent_id += 1
    
    # Setup display
    screen = pygame.display.set_mode(SUB_GRID_SIZE)
    pygame.display.set_caption("EcoSim - Ecosystem Simulation")
    
    clock = pygame.time.Clock()
    tick_counter = 0
    
    print(f"Starting simulation with {len(env.agents)} agents...")
    
    while tick_counter < 1000:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        # Agents take their actions
        for agent in list(env.agents):  # Use list() to avoid modification during iteration
            if agent.is_alive():
                agent.test(env)  # Random movement for visualization
        
        # Remove dead agents
        dead_agents = [agent for agent in env.agents if not agent.is_alive()]
        for agent in dead_agents:
            agent.die(env)

        # Render environment tiles
        surface = pygame.Surface((GRID_SUBENV[0], GRID_SUBENV[1]))
        for x in range(GRID_SUBENV[0]):
            for y in range(GRID_SUBENV[1]):
                tile_color = env.tiles[x][y].color if hasattr(env.tiles[x][y], 'color') else (100, 100, 100)
                surface.set_at((x, y), tile_color)
                
        # Scale up for better visibility
        scaled_surface = pygame.transform.scale(surface, SUB_GRID_SIZE)
        sub_tile_scale_x = SUB_GRID_SIZE[0] / GRID_SUBENV[0]
        sub_tile_scale_y = SUB_GRID_SIZE[1] / GRID_SUBENV[1]

        # Render all agents as circles
        for agent in env.agents:
            if agent.is_alive():
                ax, ay = agent.position
                scaled_x = int(ax * sub_tile_scale_x)
                scaled_y = int(ay * sub_tile_scale_y)
                
                # Color: white for prey, red for predators
                color = (255, 255, 255) if agent.agent_type == "PREY" else (255, 0, 0)
                radius = 3 + int(agent.energy / MAX_AGENT_ENERGY * 5)  # Size based on energy
                pygame.draw.circle(scaled_surface, color, (scaled_x, scaled_y), radius)

        screen.blit(scaled_surface, (0, 0))
        
        # Display info
        font = pygame.font.Font(None, 24)
        prey_count = len([a for a in env.agents if a.is_alive() and a.agent_type == "PREY"])
        predator_count = len([a for a in env.agents if a.is_alive() and a.agent_type == "PREDATOR"])
        text = font.render(f"Tick: {tick_counter} | Prey: {prey_count} | Predators: {predator_count}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(MAX_FPS)
        tick_counter += 1
        
        if tick_counter % 100 == 0:
            print(f"Tick {tick_counter}: {prey_count} prey, {predator_count} predators")
        
    pygame.quit()
    print("Simulation ended.")
        
if __name__ == "__main__":
    main()