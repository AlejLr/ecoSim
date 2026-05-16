"""Environment dynamics logger - documents all ecosystem rules and step ordering.

Saves comprehensive documentation of environment behavior including:
- Step ordering within each timestep
- Resource spawning and consumption rules
- Agent removal and kill rules
- Reproduction dynamics and mate-finding logic
"""

from pathlib import Path
from config.config import (
    ENERGY_DECAY_PER_STEP,
    MOVEMENT_ENERGY_COST,
    GRASS_ENERGY,
    MAX_AGENT_ENERGY,
    THIRST_PENALTY_THRESHOLD,
    THIRST_CRITICAL_PENALTY,
    DRINKING_REWARD,
    STEP_PENALTY,
    DEATH_PENALTY,
    REPRODUCTION_REWARD,
    START_ENERGY_PREY,
    START_ENERGY_PREDATOR,
    PREY_REPRODUCTION_THRESHOLD,
    PREY_REPRODUCTION_ENERGY_COST,
    PREY_OFFSPRING_ENERGY,
    PREY_REPRODUCTION_COOLDOWN,
    PREY_REPRODUCTION_PROB_SCALE,
    PREDATOR_REPRODUCTION_THRESHOLD,
    PREDATOR_REPRODUCTION_ENERGY_COST,
    PREDATOR_OFFSPRING_ENERGY,
    PREDATOR_REPRODUCTION_COOLDOWN,
    PREDATOR_REPRODUCTION_PROB_SCALE,
    PREDATOR_PREY_RATIO_FOR_REPRODUCTION,
    PREY_CARRYING_CAPACITY_RATIO,
)

# Define hunting efficiency constant (same as in agent.py line 436)
HUNTING_EFFICIENCY = 0.25


def generate_environment_documentation(run_number):
    """Generate comprehensive documentation of environment dynamics.
    
    Args:
        run_number: Current run number for logging
        
    Returns:
        str: Formatted documentation string
    """
    doc = f"""# Environment Dynamics Documentation - Run #{run_number}

## 1. STEP ORDERING (Per Timestep)

Each timestep follows this exact sequence:

1. **Resource Spawning**
   - Grass grows on empty tiles at configured rate
   - Water tiles always available (static 20% of grid)
   
2. **Agent Action Phase (Randomized Order)**
   - All agents execute their action in random order
   - Actions include: move, eat, drink, idle, reproduce
   - Each agent receives immediate reward signal
   
3. **Energy Decay & Cleanup**
   - Apply per-step energy decay to all agents: -{ENERGY_DECAY_PER_STEP} energy
   - Check for agent starvation (energy <= 0): mark as dead
   - Check for critical thirst (thirst < {THIRST_PENALTY_THRESHOLD}): apply penalty {THIRST_CRITICAL_PENALTY}
   
4. **Dead Agent Removal**
   - Remove all marked-dead agents from environment
   - Prevent dead agents from affecting next step
   - Apply death penalty to their Q-table
   
5. **Reproduction Phase**
   - Check all living agents for reproduction eligibility
   - Execute pairwise mate-finding and breeding
   - Place offspring with appropriate starting energy

---

## 2. RESOURCE RULES

### Grass Spawning
- **Location**: Empty tiles (not water, not agent)
- **Energy Value**: {GRASS_ENERGY} energy per grass
- **Consumption**: Prey eat grass, gain {GRASS_ENERGY} energy
- **Regrowth**: Continuous regeneration at environment rate

### Water Access
- **Location**: Static 20% of grid (150×150 = ~4,500 water tiles)
- **Consumption**: Agents drink to reduce thirst
- **Thirst Reduction**: Drinking reduces thirst value
- **Observation**: Agents observe if water is nearby (binary: yes/no)

### Energy System
- **Maximum**: {MAX_AGENT_ENERGY} energy per agent
- **Decay**: -{ENERGY_DECAY_PER_STEP} per step (base metabolic cost)
- **Movement Cost**: -{MOVEMENT_ENERGY_COST} per move action
- **Starting Energy - Prey**: {START_ENERGY_PREY}
- **Starting Energy - Predator**: {START_ENERGY_PREDATOR}

---

## 3. KILL/REMOVAL RULES

### Starvation
- **Trigger**: Agent energy <= 0
- **Result**: Agent marked dead and removed
- **Penalty**: DEATH_PENALTY = {DEATH_PENALTY}
- **Timing**: Applied during cleanup phase after decay

### Predation
- **Trigger**: Predator executes EAT action on prey
- **Consequence**: 
  - Prey dies immediately (is_alive() = False)
  - Prey cannot act in future timesteps
  - Prevents double-kills (multiple predators eating same prey)
- **Energy Transfer**: Predator gains (prey_energy × {HUNTING_EFFICIENCY})
  - Example: Prey with 50 energy → Predator gains 5 energy (10% efficiency)
- **Reward**: Predator gets energy_transferred + {REPRODUCTION_REWARD} bonus (hunting success reward)

### Critical Thirst
- **Trigger**: Thirst < {THIRST_PENALTY_THRESHOLD}
- **Penalty**: -{THIRST_CRITICAL_PENALTY} reward immediately
- **Does NOT Kill**: Agent survives but receives negative reinforcement to learn drinking

---

## 4. REPRODUCTION RULES

### PREY Reproduction

**Eligibility Criteria:**
- Energy >= {PREY_REPRODUCTION_THRESHOLD}
- reproduction_cooldown == 0 (not in cooldown from recent reproduction)
- Population < global carrying capacity ({PREY_CARRYING_CAPACITY_RATIO} × grid area)
- Partner found: another prey with energy >= {PREY_REPRODUCTION_THRESHOLD} and cooldown == 0

**Probability:**
- P(reproduce) = (energy_surplus / max_surplus) × {PREY_REPRODUCTION_PROB_SCALE}
  - energy_surplus = energy - {PREY_REPRODUCTION_THRESHOLD}
  - max_surplus = {MAX_AGENT_ENERGY} - {PREY_REPRODUCTION_THRESHOLD}
  - Example: Prey with 80 energy → (80-60)/(100-60) × 0.8 = 0.4 = 40% chance
  
**Energy Cost:**
- Parent 1 loses {PREY_REPRODUCTION_ENERGY_COST} energy
- Parent 2 loses {PREY_REPRODUCTION_ENERGY_COST} energy
- Offspring born with {PREY_OFFSPRING_ENERGY} energy

**Cooldown & Mate-Locking:**
- Both parents get reproduction_cooldown = {PREY_REPRODUCTION_COOLDOWN}
- Prevents same agents from reproducing twice in quick succession
- Mate-locking: Check mate.reproduction_cooldown > 0 to prevent double-mating

**Reward:**
- Both parents receive REPRODUCTION_REWARD = {REPRODUCTION_REWARD}
- Encourages Q-learning policy to pursue reproduction

### PREDATOR Reproduction

**Dynamic Capacity:**
- Maximum predators allowed = prey_count × {PREDATOR_PREY_RATIO_FOR_REPRODUCTION}
- Prevents predator overpopulation
- If current_predators >= capacity, reproduction blocked

**Eligibility Criteria:**
- Energy >= {PREDATOR_REPRODUCTION_THRESHOLD}
- reproduction_cooldown == 0
- Below dynamic capacity
- Partner found: another predator with same criteria

**Probability:**
- P(reproduce) = (energy_surplus / max_surplus) × {PREDATOR_REPRODUCTION_PROB_SCALE}
  - energy_surplus = energy - {PREDATOR_REPRODUCTION_THRESHOLD}
  - max_surplus = {MAX_AGENT_ENERGY} - {PREDATOR_REPRODUCTION_THRESHOLD}
  - Example: Predator with 90 energy → (90-80)/(100-80) × 0.7 ≈ 0.35 = 35% chance
  
**Energy Cost:**
- Parent 1 loses {PREDATOR_REPRODUCTION_ENERGY_COST} energy
- Parent 2 loses {PREDATOR_REPRODUCTION_ENERGY_COST} energy
- Offspring born with {PREDATOR_OFFSPRING_ENERGY} energy

**Cooldown & Mate-Locking:**
- Both parents get reproduction_cooldown = {PREDATOR_REPRODUCTION_COOLDOWN}
- Mate-locking: Check mate.reproduction_cooldown > 0

**Reward:**
- Both parents receive REPRODUCTION_REWARD = {REPRODUCTION_REWARD}

---

## 5. REWARD SYSTEM

### Per-Action Rewards

| Action | Prey | Predator | Trigger |
|--------|------|----------|---------|
| Eating Grass | energy_gained × 1.0 | N/A | Grass consumed |
| Hunting | N/A | transfer × 1.0 + 0.25 | Prey killed |
| Drinking | 0.5 | 0.5 | Water consumed |
| Reproduction | 0.5 | 0.5 | Offspring born |
| Thirst Penalty | -2.0 | -2.0 | Thirst < {THIRST_PENALTY_THRESHOLD} |
| Step Penalty | -0.01 | -0.01 | Every step |
| Death Penalty | -10 | -10 | Agent dies |

---

## 6. OBSERVATION SPACE (Agent View)

Each agent observes 8 normalized dimensions [0, 1]:

1. **Energy**: current_energy / {MAX_AGENT_ENERGY}
2. **Thirst**: current_thirst / MAX_THIRST
3. **Predator Distance** (Prey only): normalized distance to nearest predator
4. **Predator Direction X** (Prey only): -1 (left) to +1 (right)
5. **Predator Direction Y** (Prey only): -1 (up) to +1 (down)
6. **Predator Detected** (Prey only): 1 if nearby, 0 otherwise
7. **Can Reproduce**: 1 if eligible (energy ≥ threshold, cooldown=0, mate found), 0 otherwise
8. **Water Nearby**: 1 if water within 1 tile, 0 otherwise

State discretization: 5,400 unique states (5⁸ combinations)

---

## 7. ACTION SPACE

Each agent can perform 11 actions:

| Index | Action | Effect |
|-------|--------|--------|
| 0-7 | Move (N, NE, E, SE, S, SW, W, NW) | Move to adjacent tile, costs {MOVEMENT_ENERGY_COST} energy |
| 8 | Eat | Consume grass (prey) or hunt (predator) |
| 9 | Drink | Reduce thirst if water nearby |
| 10 | Idle | Do nothing |

---

## 8. GRID CONFIGURATION

- **Dimensions**: 150 × 150 = 22,500 tiles
- **Grass Coverage**: ~70% of available space (empty tiles)
- **Water Coverage**: ~20% (static)
- **Empty Tiles**: ~10%

---

## 9. INITIAL POPULATION

- **Prey**: 6 agents starting at {START_ENERGY_PREY} energy
- **Predators**: 2 agents starting at {START_ENERGY_PREDATOR} energy
- **Placement**: Random on grid

---

## Summary

This environment implements a biologically realistic ecosystem simulation with:
✓ Proper trophic efficiency (10% predator energy transfer)
✓ Sexual reproduction with mate-finding and cooldown mechanics
✓ Dynamic predator carrying capacity
✓ Probabilistic breeding based on energy surplus
✓ Energy costs coupled to realistic metabolic rates
✓ Clear reward signals for learnable behaviors (eating, drinking, reproducing)
✓ Randomized action order to prevent agent coordination artifacts
✓ Deterministic mate-locking to prevent double-mating
✓ Comprehensive logging and reproducibility (seed control)
"""
    return doc


def save_environment_log(run_number):
    """Save environment documentation to log file.
    
    Args:
        run_number: Current run number
        
    Returns:
        Path: Path to saved log file
    """
    results_dir = Path(__file__).parent.parent / "models" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = results_dir / f"log_run_{run_number}.txt"
    
    doc = generate_environment_documentation(run_number)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    print(f"✓ Environment documentation saved: {log_path}")
    return log_path


def save_simulation_metrics(run_number, metrics_list, agent_type="PREY"):
    """Save per-episode simulation metrics to CSV for analysis.
    
    Args:
        run_number: Current run number
        metrics_list: List of dicts with episode data
        agent_type: "PREY", "PREDATOR", or "MARL"
        
    Example metrics_list entry:
        {
            'episode': 1,
            'prey_population': 6,
            'predator_population': 2,
            'prey_avg_energy': 85.3,
            'predator_avg_energy': 76.2,
            'total_kills': 5,
            'prey_births': 2,
            'predator_births': 1,
            'starvation_deaths': 0,
            'prey_avg_reward': 120.5,
            'predator_avg_reward': 45.2,
        }
    """
    import csv
    
    if not metrics_list:
        return None
        
    results_dir = Path(__file__).parent.parent / "models" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get fieldnames from first entry
    fieldnames = metrics_list[0].keys()
    
    # Create filename based on agent type and run number
    csv_path = results_dir / f"simulation_metrics_{agent_type.lower()}_run_{run_number}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    
    print(f"✓ Simulation metrics saved: {csv_path}")
    return csv_path


def log_episode_snapshot(environment, episode, agent_type="PREY"):
    """Extract current snapshot of simulation metrics from environment.
    
    Called after each episode to capture population dynamics, energy stats, etc.
    
    Args:
        environment: The environment object with agents
        episode: Episode number
        agent_type: Type of agent being trained
        
    Returns:
        dict: Metrics snapshot for this episode
    """
    import numpy as np
    
    # Count agents by type and status
    prey_agents = [a for a in environment.agents if a.agent_type == "PREY" and a.is_alive()]
    predator_agents = [a for a in environment.agents if a.agent_type == "PREDATOR" and a.is_alive()]
    
    # Calculate energy statistics
    prey_energies = [a.energy for a in prey_agents] if prey_agents else [0]
    predator_energies = [a.energy for a in predator_agents] if predator_agents else [0]
    
    metrics = {
        'episode': episode,
        'prey_population': len(prey_agents),
        'predator_population': len(predator_agents),
        'prey_avg_energy': float(np.mean(prey_energies)) if prey_agents else 0,
        'prey_max_energy': float(np.max(prey_energies)) if prey_agents else 0,
        'prey_min_energy': float(np.min(prey_energies)) if prey_agents else 0,
        'predator_avg_energy': float(np.mean(predator_energies)) if predator_agents else 0,
        'predator_max_energy': float(np.max(predator_energies)) if predator_agents else 0,
        'predator_min_energy': float(np.min(predator_energies)) if predator_agents else 0,
    }
    
    # Add tracked events if environment has them (requires env to track during episode)
    if hasattr(environment, 'episode_kills'):
        metrics['total_kills'] = environment.episode_kills
    if hasattr(environment, 'prey_births'):
        metrics['prey_births'] = environment.prey_births
    if hasattr(environment, 'predator_births'):
        metrics['predator_births'] = environment.predator_births
    if hasattr(environment, 'starvation_deaths'):
        metrics['starvation_deaths'] = environment.starvation_deaths
    
    return metrics
