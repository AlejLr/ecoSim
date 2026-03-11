
from config.config import *


def agent():
    def __init__(self, position, agent_type):
        self.position = position
        self.agent_type = agent_type
        self.energy = MAX_ENERGY
        self.thirst = MAX_THIRST
        
    def action(self, action, environment):
        
        # Action is a dictionary with the action as key and the position as value
        
        match action.key:
            case "move":
                self.move(action.value, environment)
            case "eat":
                self.eat(action.value, environment)
            case "drink":
                self.drink(action.value, environment)

    def move(self, direction, environment):
        
        environment.agents[self.position[0]][self.position[1]] = 0
        environment.agents[direction[0]][direction[1]] = 1
        self.position = direction
    
    def eat(self, position, environment):
        pass
    
    def drink(self, position, environment):
        pass

        