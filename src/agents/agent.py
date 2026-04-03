from random import choice

from config.config import *


class Action():
    """Simple class to represent an action"""
    def __init__(self, key, value):
        self.key = key
        self.value = value


class Agent():
    """Simplified agent: no type distinction, standardized energy"""
    def __init__(self, agent_id, position):
        self.agent_id = agent_id
        self.position = position
        self.energy = MAX_AGENT_ENERGY
        self.thirst = MAX_THIRST
        self.vision_radius = VISION_RADIUS
        self.episode_reward = 0
        
    def test(self, environment):
        """Random movement for testing"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        self.decay_resources()
        self.move(choice(directions + diagonals), environment)
    
    def action(self, action_idx, environment):
        """Execute action and return immediate reward"""
        immediate_reward = 0
        
        # Energy and thirst decay each step
        self.decay_resources()
        
        # Map action indices to actions
        if action_idx < 8:
            # Actions 0-7: 8 directions
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]
            self.move(directions[action_idx], environment)
        elif action_idx == 8:
            # Action 8: eat
            immediate_reward = self.eat(environment)
        elif action_idx == 9:
            # Action 9: drink
            immediate_reward = self.drink(environment)
        elif action_idx == 10:
            # Action 10: idle
            pass
        
        # Add base reward
        immediate_reward += STEP_PENALTY
        
        return immediate_reward

    def decay_resources(self):
        """Decay energy and thirst each step"""
        self.energy -= ENERGY_DECAY_PER_STEP
        self.thirst -= THIRST_DECAY_PER_STEP

    def move(self, direction, environment):
        """Move the agent in the given direction if within bounds"""
        new_x = self.position[0] + direction[0]
        new_y = self.position[1] + direction[1]
        
        if 0 <= new_x < environment.width and 0 <= new_y < environment.height:
            environment.update_agent_position(self, self.position, (new_x, new_y))
            self.position = (new_x, new_y)

    def eat(self, environment):
        """Eat from any food tile within ACTION_RADIUS"""
        for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
            for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                tile_x = self.position[0] + dx
                tile_y = self.position[1] + dy
                if 0 <= tile_x < environment.width and 0 <= tile_y < environment.height:
                    tile = environment.tiles[tile_x][tile_y]
                    gain = tile.eat()
                    if gain > 0:
                        self.energy = min(MAX_AGENT_ENERGY, self.energy + gain)
                        # Return energy gained as reward
                        return gain * ENERGY_REWARD_SCALE
        return 0

    def drink(self, environment):
        """Drink from any water tile within ACTION_RADIUS"""
        for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
            for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                tile_x = self.position[0] + dx
                tile_y = self.position[1] + dy
                if 0 <= tile_x < environment.width and 0 <= tile_y < environment.height:
                    tile = environment.tiles[tile_x][tile_y]
                    gain = tile.drink()
                    if gain > 0:
                        self.thirst = min(MAX_THIRST, self.thirst + gain)
                        return 0  # No explicit reward for drinking (thirst is implicit)
        return 0

    def get_nearby_agents(self, environment):
        """Get all agents nearby within vision radius"""
        return environment.get_agents_nearby(self.position, self.vision_radius)
    
    def get_observation(self, environment):
        """Build observation state for Q-learning: normalized [0,1] values
        
        Returns dict with keys:
        - energy: normalized energy level
        - thirst: normalized thirst level  
        - food_nearby: count of food items in vision
        - water_nearby: count of water items in vision
        - other_agents_nearby: count of other agents in vision
        """
        obs = {}
        
        # Agent's own state (normalized)
        obs["energy"] = self.energy / MAX_AGENT_ENERGY
        obs["thirst"] = self.thirst / MAX_THIRST
        
        # Count resources in vision
        nearby_tiles = environment.get_tiles_nearby(self.position, self.vision_radius)
        food_count = sum(1 for t in nearby_tiles if t.has_energy)
        water_count = sum(1 for t in nearby_tiles if t.tile_type == "water")
        
        obs["food_nearby"] = food_count / (len(nearby_tiles) + 1)  # Normalize by total tiles
        obs["water_nearby"] = water_count / (len(nearby_tiles) + 1)
        
        # Count other agents
        nearby_agents = self.get_nearby_agents(environment)
        obs["other_agents_nearby"] = len(nearby_agents) / max(1, len(environment.agents))
        
        return obs
        
    def is_alive(self):
        """Agent dies when energy or thirst reaches 0"""
        return self.energy > 0 and self.thirst > 0
    
    def die(self, environment):
        """Handle agent death and cleanup"""
        environment.agents.remove(self)
        environment.agent_grid[self.position] -= 1
        environment.agents_by_position[self.position].remove(self)
