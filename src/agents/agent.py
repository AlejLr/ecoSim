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
        
        # Small detection bonuses/penalties (configurable)
        if self.agent_type == "PREDATOR":
            obs = self.get_observation(environment)
            if obs[5] > 0:  # prey_detected
                immediate_reward += PREDATOR_DETECTION_BONUS
        elif self.agent_type == "PREY":
            obs = self.get_observation(environment)
            if obs[5] > 0:  # pred_detected
                immediate_reward += PREY_DETECTION_PENALTY
        
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
            self.energy = max(0, self.energy - MOVEMENT_ENERGY_COST)

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
        
        For PREY (6 dims): [energy, thirst, pred_distance, pred_dir_x, pred_dir_y, pred_detected]
                          Focus on survival - track nearest predator
                          (Food is abundant everywhere, not critical)
        
        For PREDATOR (6 dims): [energy, thirst, prey_distance, prey_dir_x, prey_dir_y, prey_detected]
                              Track and hunt prey
        
        Returns numpy array with 6 floats.
        """
        import numpy as np
        
        # Agent's own state (normalized)
        energy_norm = self.energy / MAX_AGENT_ENERGY
        thirst_norm = self.thirst / MAX_THIRST
        
        if self.agent_type == "PREY":
            # PREY observation: focus on avoiding nearest predator
            # Food is abundant (70% grass), so survival takes priority
            pred_distance_norm = 1.0
            pred_dir_x = 0.5
            pred_dir_y = 0.5
            pred_detected = 0
            
            # Find nearest predator
            nearby_agents = self.get_nearby_agents(environment)
            predators = [a for a in nearby_agents if a.agent_type == "PREDATOR"]
            
            if predators:
                # Find closest predator
                closest_pred = min(predators, key=lambda a:
                    abs(a.position[0] - self.position[0]) + abs(a.position[1] - self.position[1]))
                
                dx = closest_pred.position[0] - self.position[0]
                dy = closest_pred.position[1] - self.position[1]
                distance = (abs(dx) + abs(dy)) / (2 * self.vision_radius)
                
                pred_distance_norm = min(1.0, distance)
                pred_dir_x = np.clip((dx / self.vision_radius + 1) / 2, 0, 1)
                pred_dir_y = np.clip((dy / self.vision_radius + 1) / 2, 0, 1)
                pred_detected = 1 if distance < 0.5 else 0
            
            return np.array([energy_norm, thirst_norm, pred_distance_norm, 
                            pred_dir_x, pred_dir_y, pred_detected], dtype=np.float32)
            
        else:  # PREDATOR
            # PREDATOR observation: find nearest prey
            target_distance_norm = 1.0
            direction_x_norm = 0.5
            direction_y_norm = 0.5
            target_detected = 0
            
            nearby_agents = self.get_nearby_agents(environment)
            prey_list = [a for a in nearby_agents if a.agent_type == "PREY"]
            
            if prey_list:
                # Find closest prey
                closest = min(prey_list, key=lambda a:
                    abs(a.position[0] - self.position[0]) + abs(a.position[1] - self.position[1]))
                
                dx = closest.position[0] - self.position[0]
                dy = closest.position[1] - self.position[1]
                distance = (abs(dx) + abs(dy)) / (2 * self.vision_radius)
                
                target_distance_norm = min(1.0, distance)
                direction_x_norm = np.clip((dx / self.vision_radius + 1) / 2, 0, 1)
                direction_y_norm = np.clip((dy / self.vision_radius + 1) / 2, 0, 1)
                target_detected = 1 if distance < 0.5 else 0
            
            return np.array([energy_norm, thirst_norm, target_distance_norm, 
                            direction_x_norm, direction_y_norm, target_detected], dtype=np.float32)
        
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

    def _find_nearby_mate(self, all_agents, search_radius=3):
        """Find nearby alive mate of same species for sexual reproduction.

        Returns mate agent or None.
        """
        for other in all_agents:
            if (other.is_alive() and
                other.agent_type == self.agent_type and
                other.agent_id != self.agent_id and
                getattr(other, 'reproduction_cooldown', 0) == 0):
                dx = abs(other.position[0] - self.position[0])
                dy = abs(other.position[1] - self.position[1])
                if dx <= search_radius and dy <= search_radius:
                    return other
        return None


class Prey(Agent):
    """Prey agent: eats grass only"""
    def __init__(self, agent_id, position):
        super().__init__(agent_id, position, "PREY")
        self.reproduction_cooldown = 0

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
                            return gain * ENERGY_REWARD_SCALE
        return 0

    def reproduce(self, environment, new_agent_id, current_prey_count=None, carrying_capacity=None, all_agents=None):
        """Prey reproduces if energy is sufficient and a mate is nearby (SEXUAL)."""
        from config.config import (
            REPRODUCTION_ENABLED,
            PREY_REPRODUCTION_THRESHOLD,
            PREY_REPRODUCTION_ENERGY_COST,
            PREY_OFFSPRING_ENERGY,
            PREY_REPRODUCTION_SEARCH_RADIUS,
            PREY_REPRODUCTION_COOLDOWN,
        )

        if not REPRODUCTION_ENABLED or self.reproduction_cooldown > 0:
            return None

        if carrying_capacity is not None and current_prey_count is not None:
            if current_prey_count >= carrying_capacity:
                return None

        if all_agents is None:
            all_agents = []

        mate = self._find_nearby_mate(all_agents, search_radius=PREY_REPRODUCTION_SEARCH_RADIUS)
        if mate is None:
            return None

        if self.energy < PREY_REPRODUCTION_THRESHOLD or mate.energy < PREY_REPRODUCTION_THRESHOLD:
            return None

        free_positions = []
        for dx in range(-PREY_REPRODUCTION_SEARCH_RADIUS, PREY_REPRODUCTION_SEARCH_RADIUS + 1):
            for dy in range(-PREY_REPRODUCTION_SEARCH_RADIUS, PREY_REPRODUCTION_SEARCH_RADIUS + 1):
                new_x = self.position[0] + dx
                new_y = self.position[1] + dy
                if not (0 <= new_x < environment.width and 0 <= new_y < environment.height):
                    continue
                if environment.is_position_free((new_x, new_y)):
                    free_positions.append((new_x, new_y))

        if not free_positions:
            return None

        import random
        new_pos = random.choice(free_positions)

        # Both parents pay cost and enter cooldown
        self.energy -= PREY_REPRODUCTION_ENERGY_COST
        mate.energy -= PREY_REPRODUCTION_ENERGY_COST
        self.reproduction_cooldown = PREY_REPRODUCTION_COOLDOWN
        mate.reproduction_cooldown = PREY_REPRODUCTION_COOLDOWN

        offspring = Prey(new_agent_id, new_pos)
        offspring.energy = PREY_OFFSPRING_ENERGY
        return offspring

    def decay_resources(self):
        """Decay energy and thirst, and cooldown"""
        super().decay_resources()
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1


class Predator(Agent):
    """Predator agent: hunts and eats prey"""
    def __init__(self, agent_id, position):
        super().__init__(agent_id, position, "PREDATOR")
        self.reproduction_cooldown = 0
    
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
        predation_energy_transfer = energy_gained * 0.85
        self.energy = min(MAX_AGENT_ENERGY, self.energy + predation_energy_transfer)

        current_prey_count = len([
            agent for agent in environment.agents
            if agent.is_alive() and agent.agent_type == "PREY"
        ])
        reward_scale = 1.0
        if PREY_PREDATION_SUSTAINABILITY_THRESHOLD > 0:
            reward_scale = min(1.0, current_prey_count / PREY_PREDATION_SUSTAINABILITY_THRESHOLD)
        
        # Return energy gained as reward + small configurable hunting bonus
        bonus = (HUNTING_SUCCESS_BONUS if energy_gained > 0 else 0) * reward_scale
        return energy_gained * ENERGY_REWARD_SCALE + bonus

    def reproduce(self, environment, new_agent_id, current_prey_count=None, carrying_capacity=None, all_agents=None):
        """Predator reproduces when energy is sufficient, a mate is nearby, and free nearby space (SEXUAL reproduction)."""
        from config.config import (
            PREDATOR_REPRODUCTION_ENABLED,
            PREDATOR_REPRODUCTION_THRESHOLD,
            PREDATOR_REPRODUCTION_ENERGY_COST,
            PREDATOR_OFFSPRING_ENERGY,
            PREDATOR_REPRODUCTION_SEARCH_RADIUS,
            PREDATOR_REPRODUCTION_COOLDOWN,
        )

        if not PREDATOR_REPRODUCTION_ENABLED or self.reproduction_cooldown > 0:
            return None

        if current_prey_count is not None and current_prey_count <= 0:
            return None

        # SEXUAL REPRODUCTION: Require a nearby mate
        if all_agents is None:
            all_agents = []
        mate = self._find_nearby_mate(all_agents, search_radius=PREDATOR_REPRODUCTION_SEARCH_RADIUS)
        if mate is None:
            return None

        if self.energy < PREDATOR_REPRODUCTION_THRESHOLD or mate.energy < PREDATOR_REPRODUCTION_THRESHOLD:
            return None

        free_positions = []
        for dx in range(-PREDATOR_REPRODUCTION_SEARCH_RADIUS, PREDATOR_REPRODUCTION_SEARCH_RADIUS + 1):
            for dy in range(-PREDATOR_REPRODUCTION_SEARCH_RADIUS, PREDATOR_REPRODUCTION_SEARCH_RADIUS + 1):
                new_x = self.position[0] + dx
                new_y = self.position[1] + dy
                if not (0 <= new_x < environment.width and 0 <= new_y < environment.height):
                    continue
                if environment.is_position_free((new_x, new_y)):
                    free_positions.append((new_x, new_y))

        if not free_positions:
            return None

        import random
        new_pos = random.choice(free_positions)

        # Both parents pay energy cost
        self.energy -= PREDATOR_REPRODUCTION_ENERGY_COST
        mate.energy -= PREDATOR_REPRODUCTION_ENERGY_COST
        self.reproduction_cooldown = PREDATOR_REPRODUCTION_COOLDOWN
        mate.reproduction_cooldown = PREDATOR_REPRODUCTION_COOLDOWN

        offspring = Predator(new_agent_id, new_pos)
        offspring.energy = PREDATOR_OFFSPRING_ENERGY
        return offspring

    def decay_resources(self):
        """Decay energy/thirst and reproduction cooldown."""
        super().decay_resources()
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
