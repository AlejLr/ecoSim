import numpy as np
from collections import defaultdict

from config.config import *
        
class QLearningAgent():
    def __init__(self, agent_type):
        self.agent_type = agent_type
        self.q_table = defaultdict(lambda: np.zeros(len(11)))
        self.epsilon = EPSILON_START
        
    def discretize_state(self, state):
        """Convert continuous state into a discrete representation for Q-table."""
        pass
    
    def select_action(self, state):
        """Select an action using epsilon-greedy strategy."""
        pass
    
    def update_q(self, state, action, reward, next_state):
        """Update Q-values based on the observed transition."""
        pass
        