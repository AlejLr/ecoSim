import pygame
import sys
import random
from config import *
from environment import generate_environement, draw, update_grid
from agent import Agent


def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE * TILE_SIZE, GRID_SIZE * TILE_SIZE))
    pygame.display.set_caption("Ecosim: Agent Environment Simulation")
    clock = pygame.time.Clock()

    num_episodes = 300

    for episode in range(num_episodes):
        grid = generate_environement()
        agent = Agent(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1), grid)

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 500:
            pass

        print(f"Episode {episode + 1}/{num_episodes} - Steps: {steps}, Total Reward: {total_reward}")

    pygame.quit()

if __name__ == "__main__":
    main()