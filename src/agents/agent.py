from random import choice

from config.config import *


def reward_for_agent(
    agent_type,
    *,
    event="step",
    energy_gained=0.0,
    detected=False,
):
    """Explicit reward equation for PREY and PREDATOR agents.

    event can be one of: step, eat, reproduce.
    STEP_PENALTY is only applied for event="step" to avoid double-counting.
    """
    reward = 0.0
    agent_kind = agent_type.upper()

    if event == "step":
        reward += STEP_PENALTY

    if agent_kind == "PREY":
        if event == "eat":
            reward += energy_gained * ENERGY_REWARD_SCALE
        elif event == "reproduce":
            reward += REPRODUCTION_REWARD

        if detected:
            reward += PREY_DETECTION_PENALTY

    elif agent_kind == "PREDATOR":
        if event == "eat":
            reward += (energy_gained * 0.25) * ENERGY_REWARD_SCALE
            if energy_gained > 0:
                reward += HUNTING_SUCCESS_BONUS
        elif event == "reproduce":
            reward += REPRODUCTION_REWARD

        if detected:
            reward += PREDATOR_DETECTION_BONUS

    return reward

class Agent():
    """Base agent class"""
    def __init__(self, agent_id, position, agent_type):
        self.agent_id = agent_id
        self.position = position
        self.agent_type = agent_type.upper()  # "PREY" or "PREDATOR"
        
        # Set starting energy based on agent type (not at max, requires learning)
        if self.agent_type == "PREY":
            self.energy = START_ENERGY_PREY
        else:
            self.energy = START_ENERGY_PREDATOR
        

        self.vision_radius = VISION_RADIUS  # Same for all
        self.speed = 1
        self.episode_reward = 0
        self.q_learning = None  # Optional Q-learning agent (set by environment)
        
    def test(self, environment):
        """Random movement for testing"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        self.decay_resources()
        self.move(choice(directions + diagonals), environment)
    
    def action(self, action_idx, environment):
        """Execute action and return immediate reward"""
        immediate_reward = 0
        
        # Energy and thirst decay each step
        self.decay_resources()
        
        # Map action indices to actions
        if action_idx < 8:
            # Actions 0-7: 8 directions
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]
            self.move(directions[action_idx], environment)
        elif action_idx == 8:
            # Action 8: eat
            immediate_reward = self.eat(environment)
        elif action_idx == 9:
            # Action 9: idle
            pass

        obs = self.get_observation(environment)
        # PREY: obs[5] = predator_detected (conditional encoding — switches obs[1-3] semantics).
        # PREDATOR: obs[4] = prey_detected.
        if self.agent_type == "PREY":
            detected = obs[5] > 0
        else:
            detected = obs[4] > 0
        immediate_reward += reward_for_agent(
            self.agent_type,
            event="step",
            detected=detected,
        )
        
        return immediate_reward

    def decay_resources(self):
        """Decay energy each step"""
        self.energy -= ENERGY_DECAY_PER_STEP

    def move(self, direction, environment):
        """Move the agent in the given direction if within bounds"""
        # Apply speed multiplier so faster agents can move multiple tiles per action
        new_x = self.position[0] + direction[0] * self.speed
        new_y = self.position[1] + direction[1] * self.speed

        # Ensure integer positions
        new_x = int(new_x)
        new_y = int(new_y)

        if 0 <= new_x < environment.width and 0 <= new_y < environment.height:
            environment.update_agent_position(self, self.position, (new_x, new_y))
            self.position = (new_x, new_y)
            # Movement energy cost scales with speed (more tiles moved costs more energy)
            self.energy = max(0, self.energy - MOVEMENT_ENERGY_COST * self.speed)

    def eat(self, environment):
        """Eat from tiles/agents within ACTION_RADIUS. Override in subclasses."""
        return 0

    def get_nearby_agents(self, environment):
        """Get all agents nearby within vision radius"""
        return environment.get_agents_nearby(self.position, self.vision_radius)
    
    def get_observation(self, environment):
        """Build 6-dim observation for Q-learning (normalized [0,1]).

        PREY (6 dims):
          [energy, primary_dist, primary_dir_x, primary_dir_y, food_detected, predator_detected]
          Conditional encoding: when predator_detected=1, obs[1-3] point toward the nearest
          predator so prey can learn to flee; otherwise they point toward nearest food.
          obs[4] is always food-based for eat-action masking.

        PREDATOR (6 dims):
          [energy, prey_dist, prey_dir_x, prey_dir_y, prey_detected, prey_density]
          prey_density=1 when 2+ prey are within vision radius.

        State space: 5×3×3×3×2×2 = 540 states.
        """
        import numpy as np

        energy_norm = self.energy / MAX_AGENT_ENERGY

        if self.agent_type == "PREY":
            # --- Food scan (always needed for obs[4] and as default obs[1-3]) ---
            food_distance_norm = 1.0
            food_dir_x = 0.5
            food_dir_y = 0.5
            food_detected = 0
            closest_grass = None
            closest_food_dist = float('inf')

            for dx in range(-VISION_RADIUS, VISION_RADIUS + 1):
                for dy in range(-VISION_RADIUS, VISION_RADIUS + 1):
                    check_x = self.position[0] + dx
                    check_y = self.position[1] + dy
                    if 0 <= check_x < environment.width and 0 <= check_y < environment.height:
                        tile = environment.tiles[check_x][check_y]
                        if tile.tile_type == "grass" and tile.has_energy:
                            dist = abs(dx) + abs(dy)
                            if dist < closest_food_dist:
                                closest_food_dist = dist
                                closest_grass = (dx, dy)

            if closest_grass is not None:
                dx, dy = closest_grass
                food_distance_norm = min(1.0, (abs(dx) + abs(dy)) / (2 * VISION_RADIUS))
                food_dir_x = np.clip((dx / VISION_RADIUS + 1) / 2, 0, 1)
                food_dir_y = np.clip((dy / VISION_RADIUS + 1) / 2, 0, 1)
                food_detected = 1 if closest_food_dist <= ACTION_RADIUS else 0

            # --- Predator scan ---
            nearby_agents = self.get_nearby_agents(environment)
            predators = [a for a in nearby_agents if a.agent_type == "PREDATOR"]
            predator_detected = 0
            primary_dist = food_distance_norm
            primary_dir_x = food_dir_x
            primary_dir_y = food_dir_y

            if predators:
                closest_pred = min(
                    predators,
                    key=lambda a: abs(a.position[0] - self.position[0]) + abs(a.position[1] - self.position[1]),
                )
                pdx = closest_pred.position[0] - self.position[0]
                pdy = closest_pred.position[1] - self.position[1]
                predator_detected = 1
                # Switch obs[1-3] to point toward predator so prey can learn to flee
                primary_dist = min(1.0, (abs(pdx) + abs(pdy)) / (2 * self.vision_radius))
                primary_dir_x = np.clip((pdx / self.vision_radius + 1) / 2, 0, 1)
                primary_dir_y = np.clip((pdy / self.vision_radius + 1) / 2, 0, 1)

            return np.array(
                [energy_norm, primary_dist, primary_dir_x, primary_dir_y, food_detected, predator_detected],
                dtype=np.float32,
            )

        else:  # PREDATOR
            target_distance_norm = 1.0
            direction_x_norm = 0.5
            direction_y_norm = 0.5
            target_detected = 0

            nearby_agents = self.get_nearby_agents(environment)
            prey_list = [a for a in nearby_agents if a.agent_type == "PREY"]

            if prey_list:
                closest = min(
                    prey_list,
                    key=lambda a: abs(a.position[0] - self.position[0]) + abs(a.position[1] - self.position[1]),
                )
                dx = closest.position[0] - self.position[0]
                dy = closest.position[1] - self.position[1]
                distance = (abs(dx) + abs(dy)) / (2 * self.vision_radius)
                target_distance_norm = min(1.0, distance)
                direction_x_norm = np.clip((dx / self.vision_radius + 1) / 2, 0, 1)
                direction_y_norm = np.clip((dy / self.vision_radius + 1) / 2, 0, 1)
                target_detected = 1 if (abs(dx) + abs(dy)) <= ACTION_RADIUS else 0

            # Binary prey density: 1 if 2+ prey visible, encourages staying in rich areas
            prey_density = 1 if len(prey_list) >= 2 else 0
            return np.array(
                [energy_norm, target_distance_norm, direction_x_norm, direction_y_norm, target_detected, prey_density],
                dtype=np.float32,
            )
        
    def is_alive(self):
        """Agent dies when energy reaches 0"""
        return self.energy > 0
    
    def die(self, environment):
        """Handle agent death and cleanup."""
        # Mark agent as dead (prevents zombie agents from is_alive() checks)
        self.energy = 0

        if self in environment.agents:
            environment.agents.remove(self)
        
        x, y = self.position
        if self in environment.agents_by_position[self.position]:
            environment.agents_by_position[self.position].remove(self)
        
        environment.agent_grid[x, y] = max(0, environment.agent_grid[x, y] - 1)

    def _find_nearby_mate(self, all_agents, search_radius=3):
        """Find nearby alive mate of same species for sexual reproduction.

        Returns mate agent or None.
        """
        for other in all_agents:
            if (other.is_alive() and
                other.agent_type == self.agent_type and
                other.agent_id != self.agent_id and
                getattr(other, 'reproduction_cooldown', 0) == 0):
                dx = abs(other.position[0] - self.position[0])
                dy = abs(other.position[1] - self.position[1])
                if dx <= search_radius and dy <= search_radius:
                    return other
        return None


class Prey(Agent):
    """Prey agent: eats grass only"""
    def __init__(self, agent_id, position):
        super().__init__(agent_id, position, "PREY")
        self.reproduction_cooldown = 0

    def eat(self, environment):
        """Eat from grass tiles within ACTION_RADIUS"""
        for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
            for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                tile_x = self.position[0] + dx
                tile_y = self.position[1] + dy
                if 0 <= tile_x < environment.width and 0 <= tile_y < environment.height:
                    tile = environment.tiles[tile_x][tile_y]
                    if tile.tile_type == "grass":
                        gain = tile.eat()
                        if gain > 0:
                            self.energy = min(MAX_AGENT_ENERGY, self.energy + gain)
                            return reward_for_agent(self.agent_type, event="eat", energy_gained=gain)
        return 0

    def reproduce(self, environment, new_agent_id, current_prey_count=None, carrying_capacity=None, all_agents=None):
        """Prey reproduces if energy is sufficient and a mate is nearby (SEXUAL).
        
        Returns offspring agent on success, None on failure.
        Reproduction reward is applied by the caller based on return value.
        """
        import random
        import numpy as np
        from config.config import (
            REPRODUCTION_ENABLED,
            PREY_REPRODUCTION_THRESHOLD,
            PREY_REPRODUCTION_ENERGY_COST,
            PREY_OFFSPRING_ENERGY,
            PREY_REPRODUCTION_SEARCH_RADIUS,
            PREY_REPRODUCTION_COOLDOWN,
            PREY_REPRODUCTION_PROB_SCALE,
            MAX_AGENT_ENERGY,
        )

        if not REPRODUCTION_ENABLED or self.reproduction_cooldown > 0:
            return None

        # Soft logistic cap: probability decreases smoothly as population approaches capacity.
        # Hard block only triggers at or above capacity to prevent over-shooting.
        density_factor = 1.0
        if carrying_capacity is not None and current_prey_count is not None:
            density = current_prey_count / carrying_capacity
            if density >= 1.0:
                return None
            density_factor = 1.0 - density

        if all_agents is None:
            all_agents = []

        # Find mate and check mate is not on cooldown (partner locking)
        mate = self._find_nearby_mate(all_agents, search_radius=PREY_REPRODUCTION_SEARCH_RADIUS)
        if mate is None or mate.reproduction_cooldown > 0:
            return None

        if self.energy < PREY_REPRODUCTION_THRESHOLD or mate.energy < PREY_REPRODUCTION_THRESHOLD:
            return None

        # Probabilistic reproduction: energy surplus × density suppression
        energy_surplus = self.energy - PREY_REPRODUCTION_THRESHOLD
        max_surplus = MAX_AGENT_ENERGY - PREY_REPRODUCTION_THRESHOLD
        if max_surplus > 0:
            reproduction_prob = min(1.0, (energy_surplus / max_surplus) * PREY_REPRODUCTION_PROB_SCALE * density_factor)
            if random.random() > reproduction_prob:
                return None

        free_positions = []
        for dx in range(-PREY_REPRODUCTION_SEARCH_RADIUS, PREY_REPRODUCTION_SEARCH_RADIUS + 1):
            for dy in range(-PREY_REPRODUCTION_SEARCH_RADIUS, PREY_REPRODUCTION_SEARCH_RADIUS + 1):
                new_x = self.position[0] + dx
                new_y = self.position[1] + dy
                if not (0 <= new_x < environment.width and 0 <= new_y < environment.height):
                    continue
                if environment.is_position_free((new_x, new_y)):
                    free_positions.append((new_x, new_y))

        if not free_positions:
            return None

        new_pos = random.choice(free_positions)

        # Both parents share the cost equally and enter cooldown (partner locking)
        half_cost = PREY_REPRODUCTION_ENERGY_COST / 2
        self.energy -= half_cost
        mate.energy -= half_cost
        self.reproduction_cooldown = PREY_REPRODUCTION_COOLDOWN
        mate.reproduction_cooldown = PREY_REPRODUCTION_COOLDOWN

        offspring = Prey(new_agent_id, new_pos)
        offspring.energy = PREY_OFFSPRING_ENERGY
        return offspring

    def decay_resources(self):
        """Decay energy and thirst, and cooldown"""
        super().decay_resources()
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1


class Predator(Agent):
    """Predator agent: hunts and eats prey"""
    def __init__(self, agent_id, position):
        super().__init__(agent_id, position, "PREDATOR")
        self.reproduction_cooldown = 0
        self.vision_radius = PREDATOR_VISION_RADIUS
        self.speed = 2
    
    def eat(self, environment, target_position=None):
        """Hunt and eat prey within ACTION_RADIUS
        
        If target_position is given, eat only that prey.
        Otherwise, eat first prey found within ACTION_RADIUS.
        Prevents double-kills by checking is_alive() and removing prey immediately.
        """
        # Record that an eat() was invoked (attempt)
        try:
            environment.predation_attempts += 1
        except Exception:
            pass

        if VERBOSE:
            print(f"[PREDATION ATTEMPT] Predator {self.agent_id} at {self.position} called eat(); initial target={target_position}")

        if target_position is None:
            # Find any prey within ACTION_RADIUS
            for dx in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                for dy in range(-ACTION_RADIUS, ACTION_RADIUS + 1):
                    check_x = self.position[0] + dx
                    check_y = self.position[1] + dy
                    if 0 <= check_x < environment.width and 0 <= check_y < environment.height:
                        # Check if any alive prey at this position
                        prey_list = [a for a in environment.agents_by_position[(check_x, check_y)]
                                    if a.agent_type == "PREY" and a is not self and a.is_alive()]
                        if prey_list:
                            prey = prey_list[0]
                            target_position = prey.position
                            break
                if target_position is not None:
                    break
        
        if target_position is None:
            # No prey found within ACTION_RADIUS
            try:
                environment.predation_failures += 1
            except Exception:
                pass
            if VERBOSE:
                print(f"[PREDATION FAILURE] Predator {self.agent_id} found no prey to eat from {self.position}")
            return 0  # No prey found
        
        # Check if target is within ACTION_RADIUS
        if abs(target_position[0] - self.position[0]) > ACTION_RADIUS or \
           abs(target_position[1] - self.position[1]) > ACTION_RADIUS:
            try:
                environment.predation_failures += 1
            except Exception:
                pass
            if VERBOSE:
                print(f"[PREDATION FAILURE] Predator {self.agent_id} target {target_position} out of ACTION_RADIUS from {self.position}")
            return 0
        
        # Find and eat alive prey at target position
        prey_list = [a for a in environment.agents_by_position[target_position]
                    if a.agent_type == "PREY" and a is not self and a.is_alive()]
        
        if not prey_list:
            try:
                environment.predation_failures += 1
            except Exception:
                pass
            if VERBOSE:
                print(f"[PREDATION FAILURE] Predator {self.agent_id} no alive prey found at target {target_position}")
            return 0

        prey = prey_list[0]
        energy_gained = prey.energy

        # Count prey before removal for logging
        pre_count = len([agent for agent in environment.agents if agent.is_alive() and agent.agent_type == "PREY"])

        from config.config import DEATH_PENALTY
        if VERBOSE:
            print(f"[PREDATION SUCCESS -> EXECUTING] Predator {self.agent_id} will eat Prey {prey.agent_id} at {prey.position} (energy={energy_gained}); prey_count_before={pre_count}")

        prey.die(environment)

        # Predator gains energy with 25% efficiency (realistic energy transfer)
        predation_energy_transfer = energy_gained * 0.25
        self.energy = min(MAX_AGENT_ENERGY, self.energy + predation_energy_transfer)

        post_count = len([agent for agent in environment.agents if agent.is_alive() and agent.agent_type == "PREY"])
        try:
            environment.predation_successes += 1
        except Exception:
            pass

        # Record predation telemetry by discrete state (if QLearning discretizer available)
        try:
            from models.Q_learning import QLearningAgent
            obs = self.get_observation(environment)
            state = QLearningAgent().discretize_state(obs)
            environment.predation_by_state[state] += 1
        except Exception:
            # Best-effort only; don't fail the action if telemetry fails
            pass

        if VERBOSE:
            print(f"[PREDATION DONE] Predator {self.agent_id} ate Prey {prey.agent_id}; energy_gained={energy_gained} -> predator_energy={self.energy}; prey_count_after={post_count}")

        return reward_for_agent(
            self.agent_type,
            event="eat",
            energy_gained=energy_gained,
        )

    def reproduce(self, environment, new_agent_id, current_prey_count=None, current_predator_count=None, all_agents=None):
        """Predator reproduces when energy sufficient and mate nearby (enforces population ratio)."""
        import random
        import numpy as np
        from config.config import (
            PREDATOR_REPRODUCTION_ENABLED,
            PREDATOR_REPRODUCTION_THRESHOLD,
            PREDATOR_REPRODUCTION_ENERGY_COST,
            PREDATOR_OFFSPRING_ENERGY,
            PREDATOR_REPRODUCTION_SEARCH_RADIUS,
            PREDATOR_REPRODUCTION_COOLDOWN,
            PREDATOR_REPRODUCTION_PROB_SCALE,
            PREDATOR_PREY_RATIO_FOR_REPRODUCTION,
            MAX_AGENT_ENERGY,
        )

        if not PREDATOR_REPRODUCTION_ENABLED or self.reproduction_cooldown > 0:
            return None

        if current_prey_count is not None and current_prey_count <= 0:
            return None

        # Enforce predator population ratio: max_predators = prey_count * ratio
        if current_prey_count is not None and current_predator_count is not None:
            max_predators_allowed = current_prey_count * PREDATOR_PREY_RATIO_FOR_REPRODUCTION
            if current_predator_count >= max_predators_allowed:
                return None  # Population cap reached

        if all_agents is None:
            all_agents = []
        
        # Find mate and check mate is not on cooldown (partner locking)
        mate = self._find_nearby_mate(all_agents, search_radius=PREDATOR_REPRODUCTION_SEARCH_RADIUS)
        if mate is None or mate.reproduction_cooldown > 0:
            return None

        if self.energy < PREDATOR_REPRODUCTION_THRESHOLD or mate.energy < PREDATOR_REPRODUCTION_THRESHOLD:
            return None

        # Probabilistic reproduction: higher chance with more energy surplus
        energy_surplus = self.energy - PREDATOR_REPRODUCTION_THRESHOLD
        max_surplus = MAX_AGENT_ENERGY - PREDATOR_REPRODUCTION_THRESHOLD
        if max_surplus > 0:
            reproduction_prob = min(1.0, (energy_surplus / max_surplus) * PREDATOR_REPRODUCTION_PROB_SCALE)
            if random.random() > reproduction_prob:
                return None

        free_positions = []
        for dx in range(-PREDATOR_REPRODUCTION_SEARCH_RADIUS, PREDATOR_REPRODUCTION_SEARCH_RADIUS + 1):
            for dy in range(-PREDATOR_REPRODUCTION_SEARCH_RADIUS, PREDATOR_REPRODUCTION_SEARCH_RADIUS + 1):
                new_x = self.position[0] + dx
                new_y = self.position[1] + dy
                if not (0 <= new_x < environment.width and 0 <= new_y < environment.height):
                    continue
                if environment.is_position_free((new_x, new_y)):
                    free_positions.append((new_x, new_y))

        if not free_positions:
            return None

        new_pos = random.choice(free_positions)

        # Both parents share the cost equally and enter cooldown (partner locking)
        half_cost = PREDATOR_REPRODUCTION_ENERGY_COST / 2
        self.energy -= half_cost
        mate.energy -= half_cost
        self.reproduction_cooldown = PREDATOR_REPRODUCTION_COOLDOWN
        mate.reproduction_cooldown = PREDATOR_REPRODUCTION_COOLDOWN

        offspring = Predator(new_agent_id, new_pos)
        offspring.energy = PREDATOR_OFFSPRING_ENERGY
        return offspring

    def decay_resources(self):
        """Decay energy/thirst and reproduction cooldown."""
        super().decay_resources()
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
