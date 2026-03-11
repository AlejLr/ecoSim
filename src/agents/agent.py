from random import choice

from config.config import *


class Agent():
    def __init__(self, position, agent_type):
        self.position = position
        self.agent_type = agent_type
        self.energy = MAX_ENERGY
        self.thirst = MAX_THIRST
        
    def test(self, environment):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        self.move(choice(directions + diagonals), environment)
        
    def action(self, action, environment):
        
        # Action will be a dictionary with the action as key and the position as value
        
        match action.key:
            case "move":
                self.move(action.value, environment)
            case "eat":
                self.eat(action.value, environment)
            case "drink":
                self.drink(action.value, environment)
            case "idle":
                pass

    def move(self, direction, environment):
        
        new_x = self.position[0] + direction[0]
        new_y = self.position[1] + direction[1]
        if 0 <= new_x < environment.width and 0 <= new_y < environment.height:
            environment.agents[self.position[0]][self.position[1]] = 0
            environment.agents[new_x][new_y] = 1
            self.position = (new_x, new_y)
    
    def eat(self, position, environment):
        gain = environment.tiles[position[0]][position[1]].eat()
        self.energy = min(MAX_ENERGY, self.energy + gain)
    
    def drink(self, position, environment):
        gain = environment.tiles[position[0]][position[1]].drink()
        self.thirst = min(MAX_THIRST, self.thirst + gain)
        
        