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

PREY_MAX_ENERGY = 100
PREDATOR_MAX_ENERGY = 150
PREY_ENERGY_DECAY = 1
PREDATOR_ENERGY_DECAY = 1.5
PREY_VISION_RADIUS = 5
PREDATOR_VISION_RADIUS = 10

REPRODUCTION_ENERGY_THRESHOLD = 80
ACTION_RADIUS = 2

# Reinforcement Learning

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1
NUM_EPISODES = 500
CTDE = True

# Reward

ALPHA = 0.5
BETA = 0.5
GAMMA = 0.5
DELTA = 0.5
EPSILON = 0.1

# Idea is to reward individual agents for their own survival
# reward = (ALPHA * (energy_gain_this_step) - BETA * (energy_decay) + GAMMA * (alive_bonus))

# And the population for surviving and not dying
# global_reward = (DELTA * (sum(agent_energies) / num_agents) - EPSILON * (num_dead_agents / total_agents))

colors = {
    "grass": [
        (88, 143, 61),
        (54, 99, 61),
        (170, 191, 64),
        (127, 182, 50),
    ],
    "forest": [
        (111, 242, 174),
        (23, 166, 104),
        (2, 115, 46),
        (0, 71, 38),
    ],
    "vegetation": [
        (122, 136, 94),
        (82, 87, 71),
        (143, 159, 145),
        (176, 192, 154),
    ],
    "water": [
        (122, 197, 255),
        (33, 150, 243),
        (23, 95, 143),
        (9, 64, 105),
    ],
    "ground": [
        (237, 152, 124),
        (209, 123, 115),
        (191, 95, 95),
        (168, 74, 94),
    ]
}