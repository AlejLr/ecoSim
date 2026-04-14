from random import choice

from config.config import *

class Agent():
    """Base agent class"""
    def __init__(self, agent_id, position, agent_type):
        self.agent_id = agent_id
        self.position = position
        self.agent_type = agent_type.upper()  # "PREY" or "PREDATOR"
        self.energy = MAX_AGENT_ENERGY  # Standardized for all
        self.thirst = MAX_THIRST
        self.vision_radius = VISION_RADIUS  # Same for all
        self.episode_reward = 0
        self.q_learning = None  # Optional Q-learning agent (set by environment)
        
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
        
        # Add base penalty per step
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
        """Eat from tiles/agents within ACTION_RADIUS. Override in subclasses."""
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
        
        Returns numpy array with 5 floats:
        [energy, thirst, food_nearby, water_nearby, other_agents_nearby]
        """
        import numpy as np
        
        # Agent's own state (normalized)
        energy_norm = self.energy / MAX_AGENT_ENERGY
        thirst_norm = self.thirst / MAX_THIRST
        
        # Count resources in vision
        nearby_tiles = environment.get_tiles_nearby(self.position, self.vision_radius)
        food_count = sum(1 for t in nearby_tiles if t.has_energy)
        water_count = sum(1 for t in nearby_tiles if t.tile_type == "water")
        
        food_norm = food_count / (len(nearby_tiles) + 1)  # Normalize by total tiles
        water_norm = water_count / (len(nearby_tiles) + 1)
        
        # Count other agents
        nearby_agents = self.get_nearby_agents(environment)
        agents_norm = len(nearby_agents) / max(1, len(environment.agents))
        
        return np.array([energy_norm, thirst_norm, food_norm, water_norm, agents_norm], dtype=np.float32)
        
    def is_alive(self):
        """Agent dies when energy or thirst reaches 0"""
        return self.energy > 0 and self.thirst > 0
    
    def die(self, environment):
        """Handle agent death and cleanup"""
        if self in environment.agents:
            environment.agents.remove(self)
        
        x, y = self.position
        if self in environment.agents_by_position[self.position]:
            environment.agents_by_position[self.position].remove(self)
        
        environment.agent_grid[x, y] = max(0, environment.agent_grid[x, y] - 1)


class Prey(Agent):
    """Prey agent: eats grass only"""
    def __init__(self, agent_id, position):
        super().__init__(agent_id, position, "PREY")
    
    def eat(self, environment):
        """Eat from grass tiles within ACTION_RADIUS"""
        for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
            for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                tile_x = self.position[0] + dx
                tile_y = self.position[1] + dy
                if 0 <= tile_x < environment.width and 0 <= tile_y < environment.height:
                    tile = environment.tiles[tile_x][tile_y]
                    if tile.tile_type == "grass":
                        gain = tile.eat()
                        if gain > 0:
                            self.energy = min(MAX_AGENT_ENERGY, self.energy + gain)
                            # Return energy gained as reward
                            return gain * ENERGY_REWARD_SCALE
        return 0


class Predator(Agent):
    """Predator agent: hunts and eats prey"""
    def __init__(self, agent_id, position):
        super().__init__(agent_id, position, "PREDATOR")
    
    def eat(self, environment, target_position=None):
        """Hunt and eat prey within ACTION_RADIUS
        
        If target_position is given, eat only that prey.
        Otherwise, eat first prey found within ACTION_RADIUS.
        """
        if target_position is None:
            # Find any prey within ACTION_RADIUS
            for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                    check_x = self.position[0] + dx
                    check_y = self.position[1] + dy
                    if 0 <= check_x < environment.width and 0 <= check_y < environment.height:
                        # Check if any prey at this position
                        prey_list = [a for a in environment.agents_by_position[(check_x, check_y)]
                                    if a.agent_type == "PREY" and a is not self]
                        if prey_list:
                            prey = prey_list[0]
                            target_position = prey.position
                            break
        
        if target_position is None:
            return 0  # No prey found
        
        # Check if target is within ACTION_RADIUS
        if abs(target_position[0] - self.position[0]) > ACTION_RADIUS or \
           abs(target_position[1] - self.position[1]) > ACTION_RADIUS:
            return 0
        
        # Find and eat prey at target position
        prey_list = [a for a in environment.agents_by_position[target_position]
                    if a.agent_type == "PREY" and a is not self]
        
        if not prey_list:
            return 0
        
        prey = prey_list[0]
        energy_gained = prey.energy
        prey.energy = 0  # Kill the prey (will be cleaned up later)
        
        # Predator gains energy
        self.energy = min(MAX_AGENT_ENERGY, self.energy + energy_gained)
        
        # Return energy gained as reward
        return energy_gained * ENERGY_REWARD_SCALE
