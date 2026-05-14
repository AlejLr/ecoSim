# Pygame
GRID_SUBENV = (150, 150)
SUB_TILE_SIZE = 4
SUB_GRID_SIZE = (GRID_SUBENV[0] * SUB_TILE_SIZE, GRID_SUBENV[1] * SUB_TILE_SIZE)

MAX_SIMULATION_STEPS = 1000
MAX_FPS = 5

RENDER = True
VERBOSE = False
SAVE_STATS = True

# RANDOM SEED FOR REPRODUCIBILITY
SEED = 42  # Global seed for numpy, random, gym - set to None for non-deterministic runs

# ENVIRONMENT
GRASS_ENERGY = 40
WATER_HYDRATION = 20
GRASS_REGROWTH_STEPS = 8      # Steps before a depleted grass tile becomes available again

TILE_DISTRIBUTION = {
    "grass": 0.7,
    "water": 0.2,
    "empty": 0.1
}

# AGENTS
NUM_AGENTS = 10                 # Total agents in environment
MAX_AGENT_ENERGY = 100          # All agents have same max energy
ENERGY_DECAY_PER_STEP = 0.3     # Agents lose 0.3 energy per step (rebalanced for sustainability)
MOVEMENT_ENERGY_COST = 0.15     # Extra energy cost when a move action is taken
MAX_THIRST = 100
THIRST_DECAY_PER_STEP = 0.5     # Agents lose 0.5 thirst per step (reduced for sustainability)
VISION_RADIUS = 4               # All agents can see 4 tiles away
ACTION_RADIUS = 3               # All agents can act within 3 tiles (increased for predator hunting)

# AGENT STARTING ENERGY (below max to require learning)
START_ENERGY_PREY = 70          # Prey start below max (prevents trivial early reproduction)
START_ENERGY_PREDATOR = 85      # Predators start with slight margin

# REINFORCEMENT LEARNING
LEARNING_RATE = 0.1             # Q-learning alpha
DISCOUNT_FACTOR = 0.95          # Q-learning gamma
EPSILON_START = 1.0             # Exploration factor (start at 100% random)
EPSILON_DECAY = 0.995           # Decay exploration each episode
EPSILON_MIN = 0.01              # Don't go below 1% random
NUM_EPISODES = 1000             # Total training episodes
STEPS_PER_EPISODE = 250         # Max steps per episode


# REWARDS
ENERGY_REWARD_SCALE = 1.0       # Reward = energy_gained * scale
DRINKING_REWARD = 0.5           # Reward for successfully drinking
REPRODUCTION_REWARD = 0.5        # Reward for successful reproduction
THIRST_PENALTY_THRESHOLD = 20   # If thirst drops below this, additional penalty
THIRST_CRITICAL_PENALTY = -2.0  # Penalty when thirst is critically low
DEATH_PENALTY = -10             # Penalty for dying
STEP_PENALTY = -0.01            # Small penalty per step
SURVIVE_BONUS = 0.1             # Small bonus for staying alive

# Hunting and detection reward tuning
HUNTING_SUCCESS_BONUS = 0.25    # Small bonus for successful hunt (added to energy-based reward)
PREDATOR_DETECTION_BONUS = 0.0  # Small bonus when predator detects prey (kept 0 to avoid bias)
PREY_DETECTION_PENALTY = 0.0    # Small penalty when prey detects predator (avoid large penalties)
PREY_PREDATION_SUSTAINABILITY_THRESHOLD = 75  # Below this prey count, hunting reward is reduced

# REPRODUCTION
REPRODUCTION_ENABLED = True     # Enable prey reproduction
PREY_REPRODUCTION_THRESHOLD = 60    # Min energy needed to reproduce (was 70, now delayed)
PREY_REPRODUCTION_ENERGY_COST = 30  # Energy cost for prey to reproduce
PREY_OFFSPRING_ENERGY = 40      # Offspring start with this energy
PREY_REPRODUCTION_SEARCH_RADIUS = 2  # Nearby tiles to search for a free birth location
PREY_CARRYING_CAPACITY_RATIO = 0.01  # Fraction of grid cells that can be occupied by prey
PREY_CARRYING_CAPACITY = int(GRID_SUBENV[0] * GRID_SUBENV[1] * PREY_CARRYING_CAPACITY_RATIO)
PREY_REPRODUCTION_COOLDOWN = 5
PREY_REPRODUCTION_PROB_SCALE = 0.8  # Reproduction probability = (energy_surplus / max_surplus) * scale
PREDATOR_REPRODUCTION_ENABLED = True
PREDATOR_REPRODUCTION_THRESHOLD = 80  # Min energy (was 85, now more achievable)
PREDATOR_REPRODUCTION_ENERGY_COST = 35
PREDATOR_OFFSPRING_ENERGY = 45
PREDATOR_REPRODUCTION_SEARCH_RADIUS = 2
PREDATOR_REPRODUCTION_COOLDOWN = 8
PREDATOR_REPRODUCTION_PROB_SCALE = 0.7  # Higher chance than prey (encourage pack formation)
PREDATOR_PREY_RATIO_FOR_REPRODUCTION = 0.1  # Max predators = prey_count * this ratio

# COLORS
colors = {
    "grass": (88, 143, 61),
    "water": (122, 197, 255),
    "empty": (200, 200, 200),
}

# Agent rendering colors
AGENT_COLOR_ALIVE = (255, 255, 255)
AGENT_COLOR_SELECTED = (255, 0, 0)
