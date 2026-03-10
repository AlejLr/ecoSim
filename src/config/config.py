# Configuration file for EcoSim

# Pygame
GRID_SUBENV = (150, 150)
SUB_TILE_SIZE = 4
SUB_GRID_SIZE = (GRID_SUBENV[0] * SUB_TILE_SIZE, GRID_SUBENV[1] * SUB_TILE_SIZE)

GRID_SIZE = (250, 250)
TILE_SIZE = 3
GRID_SIZE = (GRID_SIZE[0] * TILE_SIZE, GRID_SIZE[1] * TILE_SIZE)

MAX_SIMULATION_STEPS = 1000
MAX_FPS = 5

RENDER = True
VERBOSE = False
SAVE_STATS = True


# Colors
colors = {
    "grass": (0, 255, 0), #light green
    "forest": (0, 128, 0), #dark green
    "vegetation": (128, 128, 0), #yellow-green
    "water": (0, 0, 255)  #blue
}

# Tiles
GROWTH_RATE = {
    "grass": 0.1,
    "forest": 0.05,
    "vegetation": 0.08
}

ENERGY_PRODUCTION = {
    "grass": 1,
    "forest": 2,
    "vegetation": 1.5
}

# Biological parameters

ENERGY_TRANSFER_EFFICIENCY = 0.1

# Agents

INITIAL_PREY_NUMBER = 10
INITIAL_PREDATOR_NUMBER = 5

MAX_THIRST = 100
THIRST_DECAY = 1
HYDRATION_GAIN_WATER = 20
DEHYDRATION_PENALTY = 0.2


MAX_ENERGY = 100
ENERGY_DECAY = 1
ENERGY_GAIN_FOOD = 20

REPRODUCTION_ENERGY_THRESHOLD = 80
ACTION_RADIUS = 2
VISION_RADIUS = 5

# Reinforcement Learning

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1
NUM_EPISODES = 500
CTDE = True
