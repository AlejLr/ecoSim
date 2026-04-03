# EcoSim - Simplified Configuration

# Pygame
GRID_SUBENV = (150, 150)
SUB_TILE_SIZE = 4
SUB_GRID_SIZE = (GRID_SUBENV[0] * SUB_TILE_SIZE, GRID_SUBENV[1] * SUB_TILE_SIZE)

MAX_SIMULATION_STEPS = 1000
MAX_FPS = 5

RENDER = True
VERBOSE = False
SAVE_STATS = True

# ENVIRONMENT
GRASS_ENERGY = 50
WATER_HYDRATION = 20

TILE_DISTRIBUTION = {
    "grass": 0.7,
    "water": 0.2,
    "empty": 0.1
}

# AGENTS
NUM_AGENTS = 10                 # Total agents in environment
MAX_AGENT_ENERGY = 100          # All agents have same max energy
ENERGY_DECAY_PER_STEP = 1       # Agents lose 1 energy per step
MAX_THIRST = 100
THIRST_DECAY_PER_STEP = 1       # Agents lose 1 thirst per step
VISION_RADIUS = 4               # All agents can see 4 tiles away
ACTION_RADIUS = 2               # All agents can act within 2 tiles

# REINFORCEMENT LEARNING
LEARNING_RATE = 0.1             # Q-learning alpha
DISCOUNT_FACTOR = 0.95          # Q-learning gamma
EPSILON_START = 1.0             # Exploration factor (start at 100% random)
EPSILON_DECAY = 0.995           # Decay exploration each episode
EPSILON_MIN = 0.01              # Don't go below 1% random
NUM_EPISODES = 1000             # Total training episodes
STEPS_PER_EPISODE = 500         # Max steps per episode


# REWARDS
ENERGY_REWARD_SCALE = 1.0       # Reward = energy_gained * scale
DEATH_PENALTY = -10             # Penalty for dying
STEP_PENALTY = -0.01            # Small penalty per step (encourage efficiency)
SURVIVE_BONUS = 0.1             # Small bonus for staying alive

# COLORS
colors = {
    "grass": (88, 143, 61),
    "water": (122, 197, 255),
    "empty": (200, 200, 200),
}

# Agent rendering colors
AGENT_COLOR_ALIVE = (255, 255, 255)
AGENT_COLOR_SELECTED = (255, 0, 0)
