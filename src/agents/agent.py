from random import choice

from config.config import *

class Action():
    def __init__(self, key, value):
        self.key = key
        self.value = value

class Agent():
    def __init__(self, position, agent_type):
        self.position = position
        self.agent_type = agent_type.upper()
        self.energy = PREY_MAX_ENERGY if self.agent_type == "PREY" else PREDATOR_MAX_ENERGY
        self.thirst = MAX_THIRST
        self.vision_radius = PREY_VISION_RADIUS if self.agent_type == "PREY" else PREDATOR_VISION_RADIUS
        
    def test(self, environment):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        self.energy -= PREY_ENERGY_DECAY if self.agent_type == "PREY" else PREDATOR_ENERGY_DECAY
        self.thirst -= THIRST_DECAY
        
        self.move(choice(directions + diagonals), environment)
    
    def decide_action(self, environment):
        pass
    
    def action(self, action, environment):
        
        # Action will be a dictionary with the action as key and the position as value
        
        self.energy -= PREY_ENERGY_DECAY if self.agent_type == "PREY" else PREDATOR_ENERGY_DECAY
        self.thirst -= THIRST_DECAY
        
        match action.key:
            case "move":
                self.move(action.value, environment)
            case "eat":
                self.eat(environment)
            case "drink":
                self.drink(environment)
            case "idle":
                pass

    def move(self, direction, environment):
        
        new_x = self.position[0] + direction[0]
        new_y = self.position[1] + direction[1]
        
        if 0 <= new_x < environment.width and 0 <= new_y < environment.height:
            environment.update_agent_position(self, self.position, (new_x, new_y))
            self.position = (new_x, new_y)
    
    def eat(self, environment):
        """Eat from tiles within ACTION_RADIUS. Override in subclasses."""
        pass
    
    def drink(self, environment):
        """Drink from water tiles within ACTION_RADIUS."""
        for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
            for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                tile_x = self.position[0] + dx
                tile_y = self.position[1] + dy
                if 0 <= tile_x < environment.width and 0 <= tile_y < environment.height:
                    gain = environment.tiles[tile_x][tile_y].drink()
                    if gain > 0:
                        self.thirst = min(MAX_THIRST, self.thirst + gain)
                        return
    
    def get_nearby_agents(self, environment):
        """Get all agents nearby within vision radius."""
        return environment.get_agents_nearby(self.position, self.vision_radius)
    
    def reproduce(self, environment):
        """Reproduce if energy exceeds threshold. Returns new agent or None."""
        if self.energy >= REPRODUCTION_ENERGY_THRESHOLD:
            self.energy *= 0.5
            offspring = self.__class__(self.position)
            offspring.energy = self.energy
            return offspring
        return None
    
    def get_observation(self, environment):
        """Get observation for Q-learning: nearby food, enemies, own state."""

        food_count = 0
        water_count = 0
        enemy_count = 0
        
        for dx in range(-self.vision_radius, self.vision_radius + 1):
            for dy in range(-self.vision_radius, self.vision_radius + 1):
                tile_x = self.position[0] + dx
                tile_y = self.position[1] + dy
                if 0 <= tile_x < environment.width and 0 <= tile_y < environment.height:
                    tile = environment.tiles[tile_x][tile_y]
                    if tile.tile_type in ["grass", "forest", "vegetation"] and tile.has_energy:
                        food_count += 1
                    elif tile.tile_type == "water":
                        water_count += 1
        
        nearby_agents = self.get_nearby_agents(environment)
        for agent in nearby_agents:
            if agent is not self:
                if agent.agent_type != self.agent_type:
                    enemy_count += 1
        
        return {
            'energy': self.energy,
            'thirst': self.thirst,
            'food_nearby': food_count,
            'water_nearby': water_count,
            'enemies_nearby': enemy_count,
            'position': self.position
        }
        
    def is_alive(self):
        return self.energy > 0 and self.thirst > 0
        

class Prey(Agent):
    def __init__(self, position):
        super().__init__(position, "prey")
        
    def eat(self, environment):
        """Eat from plants within ACTION_RADIUS."""
        for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
            for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                tile_x = self.position[0] + dx
                tile_y = self.position[1] + dy
                if 0 <= tile_x < environment.width and 0 <= tile_y < environment.height:
                    gain = environment.tiles[tile_x][tile_y].eat()
                    if gain > 0:
                        self.energy = min(PREY_MAX_ENERGY, self.energy + gain * ENERGY_TRANSFER_EFFICIENCY)
                        return
        
class Predator(Agent):
    def __init__(self, position):
        super().__init__(position, "predator")
        
    def eat(self, environment):
        """Hunt nearby prey within ACTION_RADIUS and gain their energy."""

        prey_to_hunt = [agent for agent in self.get_nearby_agents(environment)
                        if agent.agent_type == "PREY" and agent is not self 
                        and abs(agent.position[0] - self.position[0]) <= ACTION_RADIUS
                        and abs(agent.position[1] - self.position[1]) <= ACTION_RADIUS]
        
        if prey_to_hunt:

            for prey in prey_to_hunt:
                energy_gained = prey.energy
                prey.energy = 0
                self.energy = min(PREDATOR_MAX_ENERGY, 
                                self.energy + energy_gained * ENERGY_TRANSFER_EFFICIENCY)
                environment.remove_agent_from_grid(prey)