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
GRASS_REGROWTH_STEPS = 8      # Steps before a depleted grass tile becomes available again

TILE_DISTRIBUTION = {
    "grass": 0.8,
    "empty": 0.2
}

# AGENTS
NUM_AGENTS = 10                 # Total agents in environment
MAX_AGENT_ENERGY = 100          # All agents have same max energy
ENERGY_DECAY_PER_STEP = 0.12    # Agents lose 0.12 energy per step (reduced from 0.3 for sustainability)
MOVEMENT_ENERGY_COST = 0.15     # Extra energy cost when a move action is taken
VISION_RADIUS = 6               # All agents can see 6 tiles away
ACTION_RADIUS = 3               # All agents can act within 3 tiles (increased for predator hunting)
PREDATOR_VISION_RADIUS = 8      # Predators see farther to compensate for prey's extra information

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
STEPS_PER_EPISODE = 500         # Max steps per episode
ONE_V_ONE_STEPS = 350           # Step limit for 1v1 protocol (prey survival threshold)


# REWARDS
ENERGY_REWARD_SCALE = 1.0       # Reward = energy_gained * scale
REPRODUCTION_REWARD = 15.0       # Reward for successful reproduction
DEATH_PENALTY = -10             # Penalty for dying
STEP_PENALTY = -0.01            # Small penalty per step

# Hunting and detection reward tuning
HUNTING_SUCCESS_BONUS = 2.0     # Flat reward for a successful hunt
PREDATOR_DETECTION_BONUS = 0.0  # Small bonus when predator detects prey (kept 0 to avoid bias)
PREY_DETECTION_PENALTY = -0.5   # Penalty when prey detects predator (teaches evasion)

# Coexistence scoring: used for evaluation metrics only, not for reward shaping.
# Coexistence should emerge from population dynamics, not be hardcoded into rewards.
COEXISTENCE_PREY_FLOOR = 4            # Healthy prey floor for the score curve
COEXISTENCE_PREDATOR_FLOOR = 2        # Healthy predator floor for the score curve
COEXISTENCE_TARGET_PREY_TO_PREDATOR_RATIO = 2.0  # Rough target balance

# REPRODUCTION
REPRODUCTION_ENABLED = True     # Enable prey reproduction
PREY_REPRODUCTION_THRESHOLD = 55    # Min energy needed to reproduce (lowered to enable pop growth)
PREY_REPRODUCTION_ENERGY_COST = 25  # Energy cost for prey to reproduce (reduced from 30)
PREY_OFFSPRING_ENERGY = 50      # Offspring start with this energy (increased from 40 for viability)
PREY_REPRODUCTION_SEARCH_RADIUS = 6  # Nearby tiles to search for a free birth location
PREY_CARRYING_CAPACITY_RATIO = 0.015  # Fraction of grid cells that can be occupied by prey (increased from 0.01)
PREY_CARRYING_CAPACITY = int(GRID_SUBENV[0] * GRID_SUBENV[1] * PREY_CARRYING_CAPACITY_RATIO)
PREY_REPRODUCTION_COOLDOWN = 8
PREY_REPRODUCTION_PROB_SCALE = 0.8  # Reproduction probability = (energy_surplus / max_surplus) * scale
PREDATOR_REPRODUCTION_ENABLED = True
PREDATOR_REPRODUCTION_THRESHOLD = 70  # Min energy (lowered for more reproduction opportunities)
PREDATOR_REPRODUCTION_ENERGY_COST = 35
PREDATOR_OFFSPRING_ENERGY = 45
PREDATOR_REPRODUCTION_SEARCH_RADIUS = 8
PREDATOR_REPRODUCTION_COOLDOWN = 8
PREDATOR_REPRODUCTION_PROB_SCALE = 0.7  # Higher chance than prey (encourage pack formation)
PREDATOR_PREY_RATIO_FOR_REPRODUCTION = 0.5  # Max predators = prey_count * this ratio

# COLORS
colors = {
    "grass": (88, 143, 61),
    "water": (122, 197, 255),
    "empty": (200, 200, 200),
}

# Agent rendering colors
AGENT_COLOR_ALIVE = (255, 255, 255)
AGENT_COLOR_SELECTED = (255, 0, 0)
