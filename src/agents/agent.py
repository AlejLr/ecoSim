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
    
    def action(self, action, position, environment):
        
        # Action will be a dictionary with the action as key and the position as value
        
        self.energy -= PREY_ENERGY_DECAY if self.agent_type == "PREY" else PREDATOR_ENERGY_DECAY
        self.thirst -= THIRST_DECAY
        
        match action.key:
            case "move":
                self.move(action.value, environment)
            case "eat":
                self.eat(position, environment)
            case "drink":
                self.drink(position, environment)
            case "reproduce":
                self.reproduce(environment)
            case "idle":
                pass

    def move(self, direction, environment):
        
        new_x = self.position[0] + direction[0]
        new_y = self.position[1] + direction[1]
        
        if 0 <= new_x < environment.width and 0 <= new_y < environment.height:
            environment.update_agent_position(self, self.position, (new_x, new_y))
            self.position = (new_x, new_y)
    
    def eat(self, position, environment):
        """Eat from tiles within ACTION_RADIUS. Override in subclasses."""
        pass
    
    def drink(self, position, environment):
        """Drink from water tiles within ACTION_RADIUS."""
        for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
            for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                tile_x = position[0] + dx
                tile_y = position[1] + dy
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
        """Build a local observation dict for Q-learning.

        All values normalized to [0,1]. Spatial info is relative to the agent's
        current position so the agent learns position-independent policies.
        The k nearest items per category give a fixed-size state vector.
        """
        max_own_energy = PREY_MAX_ENERGY if self.agent_type == "PREY" else PREDATOR_MAX_ENERGY

        state = {}
        state["energy"] = self.energy / max_own_energy
        state["thirst"] = self.thirst / MAX_THIRST

        food_tiles = []
        water_tiles = []

        nearby_tiles = environment.get_tiles_nearby(self.position, self.vision_radius)
        for t in nearby_tiles:
            dx = t.x - self.position[0]
            dy = t.y - self.position[1]
            dist = abs(dx) + abs(dy)
            if dist == 0:
                continue
            if t.has_energy:
                energy_ratio = t.energy / MAX_ENERGY[t.tile_type]
                food_tiles.append((dist, dx, dy, energy_ratio))
            elif t.tile_type == "water":
                water_tiles.append((dist, dx, dy))

        food_tiles.sort(key=lambda t: t[0])
        water_tiles.sort(key=lambda t: t[0])
        top_food  = [(dx, dy, e) for _, dx, dy, e in food_tiles[:3]]
        top_water = [(dx, dy) for _, dx, dy in water_tiles[:2]]
        while len(top_food) < 3: top_food.append((0, 0, 0.0))
        while len(top_water) < 2: top_water.append((0, 0))

        state["food_tiles"]  = top_food
        state["water_tiles"] = top_water

        prey_list = []
        predator_list = []

        for agent in self.get_nearby_agents(environment):
            if agent is self:
                continue
            dx = agent.position[0] - self.position[0]
            dy = agent.position[1] - self.position[1]
            dist = abs(dx) + abs(dy)
            agent_max_energy = PREY_MAX_ENERGY if agent.agent_type == "PREY" else PREDATOR_MAX_ENERGY
            energy_ratio = agent.energy / agent_max_energy
            if agent.agent_type == "PREY":
                prey_list.append((dist, dx, dy, energy_ratio))
            else:
                predator_list.append((dist, dx, dy, energy_ratio))

        prey_list.sort(key=lambda a: a[0])
        predator_list.sort(key=lambda a: a[0])
        top_prey = [(dx, dy, e) for _, dx, dy, e in prey_list[:3]]
        top_predators = [(dx, dy, e) for _, dx, dy, e in predator_list[:3]]
        while len(top_prey) < 3: top_prey.append((0, 0, 0.0))
        while len(top_predators) < 3: top_predators.append((0, 0, 0.0))

        state["prey_nearby"] = top_prey
        state["predators_nearby"] = top_predators

        state["position"] = (self.position[0] / environment.width,
                             self.position[1] / environment.height)

        return state
        
    def is_alive(self):
        return self.energy > 0 and self.thirst > 0
        

class Prey(Agent):
    def __init__(self, position):
        super().__init__(position, "prey")
        
    def eat(self, position, environment):
        """Eat from plants within ACTION_RADIUS."""
        for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
            for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                tile_x = position[0] + dx
                tile_y = position[1] + dy
                if 0 <= tile_x < environment.width and 0 <= tile_y < environment.height:
                    gain = environment.tiles[tile_x][tile_y].eat()
                    if gain > 0:
                        self.energy = min(PREY_MAX_ENERGY, self.energy + gain * ENERGY_TRANSFER_EFFICIENCY)
                        return
        
class Predator(Agent):
    def __init__(self, position):
        super().__init__(position, "predator")
        
    def eat(self, environment, target_position):
        """Hunt prey at the specified target position"""
        
        if abs(target_position[0] - self.position[0]) > ACTION_RADIUS or \
           abs(target_position[1] - self.position[1]) > ACTION_RADIUS:
            return
        
        prey_at_position = [agent for agent in environment.agents
                           if agent.agent_type == "PREY" 
                           and agent.position == target_position]
        
        if prey_at_position:
            prey = prey_at_position[0]
            energy_gained = prey.energy
            prey.energy = 0
            
            self.energy = min(PREDATOR_MAX_ENERGY, 
                            self.energy + energy_gained * ENERGY_TRANSFER_EFFICIENCY)
            
            environment.remove_agent_from_grid(prey)