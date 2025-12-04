import pygame
import sys
import random
from config import *
from environment import generate_environement, draw, update_grid
from agent import Agent
from models.q_learning import QLearningAgent


def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE * TILE_SIZE, GRID_SIZE * TILE_SIZE))
    pygame.display.set_caption("Ecosim: Agent Environment Simulation")
    clock = pygame.time.Clock()

    q_agent = QLearningAgent()

    num_episodes = 300

    for episode in range(num_episodes):
        grid = generate_environement()
        agent = Agent(random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1), grid)

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 500:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state = q_agent.get_state(agent, grid)
            action = q_agent.select_action(state)

            prev_tile = grid[agent.y][agent.x]
            reward = 0

            alive = agent.step(action)
            new_tile = grid[agent.y][agent.x]

            done = not alive

            next_state = q_agent.get_state(agent, grid)
            reward = q_agent.get_reward(agent, prev_tile, new_tile, done)

            q_agent.update(state, action, reward, next_state)

            update_grid(grid)
            total_reward += reward
            steps += 1

            if episode == num_episodes - 1:
                draw(screen, grid, agent)
                pygame.display.flip()
                clock.tick(60)

        print(f"Episode {episode + 1}/{num_episodes} - Steps: {steps}, Total Reward: {total_reward}")

    pygame.quit()

if __name__ == "__main__":
    main()