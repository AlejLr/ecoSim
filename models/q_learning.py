import random
import numpy as np
from collections import defaultdict
from config import *

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(7))

    def get_state(self, agent, grid):
        radius_tiles = []
        directions = [(-2, 0), (-1, 0), (1, 0), (2, 0),
                      (0, -2), (0, -1), (0, 1), (0, 2),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dx, dy in directions:
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):
                tile = grid[ny][nx]
                tile_info = (
                    int(tile.has_food),
                    int(tile.is_water)
                )
            else:
                tile_info = (0, 0)
            radius_tiles.append(tile_info)

        energy_level = agent.energy // 10
        water_level = agent.water // 10

        return tuple(radius_tiles + [energy_level, water_level])
                

    def select_action(self, state):
        if random.random() < EPSILON:
            return random.randint(0, 6)   
        return int(np.argmax(self.q_table[state]))
    
    def update(self, state, action, reward, next_state):
        
        max_next = np.max(self.q_table[next_state])
        old_value = self.q_table[state][action]
        self.q_table[state][action] += ALPHA * (reward + GAMMA * max_next - old_value)

    def get_reward(self, agent, prev_tile, new_tile, done):
        reward = 0

        if done:
            reward -= 500

        reward += 1

        if new_tile.has_food and agent.energy < MAX_ENERGY:
            reward += 25

        if new_tile.is_water and agent.water < MAX_WATER:
            reward += 25

        if agent.energy < 10:
            reward -= 10
        if agent.water < 10:
            reward -= 10

        return reward