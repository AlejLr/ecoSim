"""Training protocols for stable Multi-Agent Reinforcement Learning.

Issue 14 & 15: Address MARL Instability

MARL (Multi-Agent Reinforcement Learning) presents unique challenges:

1. **Non-Stationarity**: Other agents' policies change during training,
   violating standard RL convergence assumptions (stationary environment).
   
2. **Independent Learners Problem**: Each agent independently optimizes its
   policy, but changing one agent's policy invalidates all other agents'
   value function estimates (Q-table becomes outdated).

3. **Convergence Guarantees**: Standard Q-learning convergence proofs assume
   a stationary environment. Multi-agent environments are non-stationary
   by definition.

This module implements three protocols with increasing complexity:

Protocol 1: FIXED OPPONENTS
- Train one species while others use fixed policy (random or pre-trained)
- Ensures stationary environment for learning agent
- Baseline approach, high stability, limited interaction
- Phases: (1) Train PREY vs random PREDATORS
           (2) Train PREDATOR vs fixed PREY

Protocol 2: ALTERNATING TRAINING  
- Take turns: Train PREY for N episodes, freeze, train PREDATOR, freeze, repeat
- Reduces non-stationarity by limiting how often policies change
- Moderate stability, more realistic species interactions
- Cycles multiple times to allow adaptation

Protocol 3: CO-LEARNING (Both Learn Simultaneously)
- Both species learn at the same time
- Most realistic, most unstable
- Only use after proving stability with protocols 1-2
- Requires explicit non-stationarity awareness

References:
- Tan, M. (1993). Multi-Agent Reinforcement Learning: Independent vs Cooperative Agents
- Busoniu et al. (2008). A Comprehensive Survey of MARL
- Palmer et al. (2018). Lenient Multi-Agent Deep RL
"""

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config.config import *
from config.utils import set_global_seed, get_next_run_number, get_latest_model_path
from environment.gym_env import EcoSimEnv
from environment.logger import save_environment_log
from agents.agent import Predator
from models.Q_learning import QLearningAgent


class TrainingProtocol:
    """Base class for training protocols."""
    
    def __init__(self, agent_type: str, num_episodes: int, run_number: int):
        self.agent_type = agent_type
        self.num_episodes = num_episodes
        self.run_number = run_number
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'eval_rewards': [],
            'populations': [],  # Store population snapshots
            'outcomes': [],
        }
    
    def _collect_episode_snapshot(self, env: EcoSimEnv):
        """Collect population and energy metrics from environment."""
        all_agents = [env.agent] + list(env.other_agents)
        alive_agents = [agent for agent in all_agents if agent.is_alive()]
        alive_prey = [agent for agent in alive_agents if agent.agent_type == "PREY"]
        alive_predators = [agent for agent in alive_agents if agent.agent_type == "PREDATOR"]
        
        return {
            'prey_population': len(alive_prey),
            'predator_population': len(alive_predators),
            'avg_prey_energy': float(np.mean([a.energy for a in alive_prey] or [0])),
            'avg_predator_energy': float(np.mean([a.energy for a in alive_predators] or [0])),
        }

    @contextmanager
    def _track_predation_events(self):
        """Count successful predator hunts during a block of environment steps."""
        original_predator_eat = Predator.eat
        tracker = {'count': 0}

        def wrapped_predator_eat(predator_self, *args, **kwargs):
            reward = original_predator_eat(predator_self, *args, **kwargs)
            if reward > 0:
                tracker['count'] += 1
            return reward

        Predator.eat = wrapped_predator_eat
        try:
            yield tracker
        finally:
            Predator.eat = original_predator_eat

    def _episode_seed(self, cycle: int, phase: int, episode: int) -> int:
        """Build a deterministic but distinct seed for a cycle/phase/episode."""
        return SEED + (self.run_number * 100000) + (cycle * 1000) + (phase * 100) + episode

    def _clone_agent_policy(self, source_agent: QLearningAgent) -> QLearningAgent:
        """Create a frozen snapshot of an agent policy (Q-table + exploration params)."""
        cloned_agent = QLearningAgent(
            agent_id=source_agent.agent_id,
            num_actions=source_agent.num_actions,
            num_states=source_agent.num_states,
        )
        cloned_agent.epsilon = source_agent.epsilon
        cloned_agent.learning_rate = source_agent.learning_rate
        cloned_agent.discount_factor = source_agent.discount_factor

        for state, q_values in source_agent.q_table.items():
            cloned_agent.q_table[state] = np.array(q_values, copy=True)

        return cloned_agent

    def _load_latest_saved_agent(self, agent_type: str) -> QLearningAgent | None:
        """Load the most recently saved model for an agent type, if available."""
        latest_model_path = get_latest_model_path(agent_type)
        if latest_model_path is None:
            return None
        return QLearningAgent.load_model_from_file(str(latest_model_path))
    
    def train(self, resume_prey: str = None, resume_predator: str = None) -> Dict:
        """Train agent according to protocol. Return metrics."""
        raise NotImplementedError("Subclasses must implement train()")
    
    def _log_progress(self, episode: int, reward: float, epsilon: float, phase: str = ""):
        """Log training progress."""
        if (episode + 1) % max(10, self.num_episodes // 10) == 0:
            phase_str = f" [{phase}]" if phase else ""
            print(f"  Episode {episode+1:4d}/{self.num_episodes} | "
                  f"Reward: {reward:8.2f} | Epsilon: {epsilon:.3f}{phase_str}")
    
    def _evaluate_greedy(
        self,
        env: EcoSimEnv,
        agent: QLearningAgent,
        num_evals: int = 10,
        seed_base: int | None = None,
        track_predation: bool = False,
    ):
        """Evaluate agent with greedy policy (epsilon=0)."""
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        eval_rewards = []
        eval_predation_events = []
        for eval_index in range(num_evals):
            reset_seed = None if seed_base is None else seed_base + eval_index
            obs = env.reset(seed=reset_seed)
            state = agent.discretize_state(obs)
            episode_reward = 0
            done = False
            
            with self._track_predation_events() as predation_tracker:
                while not done:
                    action = agent.select_action(state, training=False)
                    next_obs, reward, done, info = env.step(action)
                    next_state = agent.discretize_state(next_obs)
                    state = next_state
                    episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_predation_events.append(predation_tracker['count'])
        
        agent.epsilon = original_epsilon
        if track_predation:
            return {
                'avg_reward': float(np.mean(eval_rewards)),
                'avg_predation_events': float(np.mean(eval_predation_events)) if eval_predation_events else 0.0,
                'total_predation_events': int(sum(eval_predation_events)),
                'eval_rewards': eval_rewards,
                'predation_events': eval_predation_events,
            }
        return float(np.mean(eval_rewards))
    
    def _save_results(self, protocol_name: str, additional_info: str = ""):
        """Save training results to CSV with population metrics."""
        csv_path = self.results_dir / f"protocol_{protocol_name}_{self.agent_type.lower()}_{self.run_number}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Include population metrics if available
            if self.metrics['populations']:
                fieldnames = ['episode', 'reward', 'steps', 'eval_reward', 'outcome',
                             'prey_population', 'predator_population', 
                             'avg_prey_energy', 'avg_predator_energy']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for ep, (reward, steps, eval_r, pop) in enumerate(zip(
                    self.metrics['rewards'],
                    self.metrics['steps'],
                    self.metrics['eval_rewards'],
                    self.metrics['populations']
                )):
                    writer.writerow({
                        'episode': ep + 1,
                        'reward': reward,
                        'steps': steps,
                        'eval_reward': eval_r,
                        'outcome': self.metrics['outcomes'][ep] if ep < len(self.metrics['outcomes']) else '',
                        'prey_population': pop.get('prey_population', 0),
                        'predator_population': pop.get('predator_population', 0),
                        'avg_prey_energy': pop.get('avg_prey_energy', 0),
                        'avg_predator_energy': pop.get('avg_predator_energy', 0),
                    })
            else:
                # Fallback to basic CSV without population data
                writer = csv.writer(f)
                writer.writerow(['episode', 'reward', 'steps', 'eval_reward', 'outcome'])
                for ep, (reward, steps, eval_r) in enumerate(zip(
                    self.metrics['rewards'],
                    self.metrics['steps'],
                    self.metrics['eval_rewards']
                )):
                    outcome = self.metrics['outcomes'][ep] if ep < len(self.metrics['outcomes']) else ''
                    writer.writerow([ep + 1, reward, steps, eval_r, outcome])
        
        print(f"  ✓ Results saved: {csv_path.name}")
        if additional_info:
            print(f"    {additional_info}")
        
        return csv_path


class FixedOpponentProtocol(TrainingProtocol):
    """Protocol 1: Train one agent while other uses fixed (random) policy.
    
    Guarantees stationary environment for learning agent.
    Most stable, suitable for initial training.
    """
    
    def train(self, resume_prey: str = None, resume_predator: str = None) -> Dict:
        """Protocol 1 redefined: 1v1 training (both prey and predator learn).

        Episode ends when either:
          - Prey survives `ONE_V_ONE_STEPS` (prey wins), or
          - Predator successfully eats the prey (predator wins).

        Both agents are Q-learning agents that train simultaneously on the same episodes.
        The protocol otherwise follows the same environment and reward rules
        as the full MARL ecosystem (resource dynamics, reproduction disabled by
        default in this small setting).
        """

        print(f"\n{'='*70}")
        print(f"PROTOCOL 1: 1v1 TRAINING (Prey vs Predator) - DUAL AGENT LEARNING")
        print(f"{'='*70}")
        print(f"Mode: Both PREY and PREDATOR learn simultaneously")
        print(f"Episodes: {self.num_episodes}")
        print(f"Environment: 1 prey, 1 predator | Episode ends on predation or {ONE_V_ONE_STEPS} steps")
        print(f"{'='*70}\n")

        # Initialize two learning agents (allow resuming from checkpoints)
        if resume_prey:
            try:
                prey_agent = QLearningAgent.load_model_from_file(str(resume_prey))
                print(f"✓ Loaded PREY model from {resume_prey}")
            except Exception:
                print(f"! Failed to load PREY model from {resume_prey}; starting from scratch")
                prey_agent = QLearningAgent(agent_id=0, num_actions=10, num_states=2160)
        else:
            prey_agent = QLearningAgent(agent_id=0, num_actions=10, num_states=2160)

        if resume_predator:
            try:
                predator_agent = QLearningAgent.load_model_from_file(str(resume_predator))
                print(f"✓ Loaded PREDATOR model from {resume_predator}")
            except Exception:
                print(f"! Failed to load PREDATOR model from {resume_predator}; starting from scratch")
                predator_agent = QLearningAgent(agent_id=1, num_actions=10, num_states=2160)
        else:
            predator_agent = QLearningAgent(agent_id=1, num_actions=10, num_states=2160)

        # Training loop (1v1 episodes)
        prey_rewards_per_ep = []
        predator_rewards_per_ep = []
        outcomes_per_ep = []
        eval_rewards_per_ep = []

        for episode in range(self.num_episodes):
            # Create fresh 1v1 environment for this episode
            env = EcoSimEnv(
                agent_id=0,
                num_prey=1,
                num_predators=1,
                agent_type="PREY",
                map_path=None,
                memory=False,
            )
            np.random.seed(self._episode_seed(0, 0, episode))
            import random
            random.seed(self._episode_seed(0, 0, episode))
            obs_prey = env.reset(seed=self._episode_seed(0, 0, episode))

            # Extract prey and predator agents from environment
            prey_agent_in_env = env.agent  # The main agent (always PREY in setup)
            predator_agent_in_env = env.other_agents[1] if len(env.other_agents) > 1 else None

            # Link the Q-learning agents to the physical agents
            prey_agent_in_env.q_learning = prey_agent
            if predator_agent_in_env:
                predator_agent_in_env.q_learning = predator_agent

            # Initialize states
            prey_state = prey_agent.discretize_state(obs_prey)
            predator_obs = predator_agent_in_env.get_observation(env.env) if predator_agent_in_env else obs_prey
            predator_state = predator_agent.discretize_state(predator_obs)

            prey_episode_reward = 0
            predator_episode_reward = 0
            episode_step_count = 0
            done = False
            outcome = ""
            predation_occurred = False

            # Dual-agent 1v1 episode loop
            while not done and episode_step_count < ONE_V_ONE_STEPS:
                # Prey action
                prey_action = prey_agent.select_action(prey_state, training=True)
                prey_reward_step = prey_agent_in_env.action(prey_action, env.env)

                # Predator action
                if predator_agent_in_env and predator_agent_in_env.is_alive():
                    predator_action = predator_agent.select_action(predator_state, training=True)
                    predator_reward_step = predator_agent_in_env.action(predator_action, env.env)
                else:
                    predator_reward_step = 0

                # Advance environment
                env.env.update_resources()

                # Check for deaths and clean up
                dead_agents = [a for a in env.env.agents if not a.is_alive()]
                for dead_agent in dead_agents:
                    dead_agent.die(env.env)

                # Check if predation occurred (predator ate prey)
                if not prey_agent_in_env.is_alive():
                    predation_occurred = True
                    outcome = "PREDATOR_WIN"
                    done = True
                    # Predator gets full reward, prey gets death penalty
                    predator_reward_step += HUNTING_SUCCESS_BONUS * 0.5
                    prey_reward_step = DEATH_PENALTY

                # Get next observations
                next_prey_obs = prey_agent_in_env.get_observation(env.env)
                next_prey_state = prey_agent.discretize_state(next_prey_obs)

                if predator_agent_in_env and predator_agent_in_env.is_alive():
                    next_predator_obs = predator_agent_in_env.get_observation(env.env)
                    next_predator_state = predator_agent.discretize_state(next_predator_obs)
                else:
                    next_predator_state = predator_state

                # Update Q-tables for both agents
                prey_agent.update(prey_state, prey_action, prey_reward_step, next_prey_state, done)
                if predator_agent_in_env and predator_agent_in_env.is_alive():
                    predator_agent.update(predator_state, predator_action, predator_reward_step, next_predator_state, done)

                # Update states and rewards
                prey_state = next_prey_state
                predator_state = next_predator_state
                prey_episode_reward += prey_reward_step
                predator_episode_reward += predator_reward_step
                episode_step_count += 1

            # If loop ends due to step limit, prey survived
            if not predation_occurred and episode_step_count >= ONE_V_ONE_STEPS:
                outcome = "PREY_WIN"
            elif not outcome:
                outcome = "PREY_WIN" if episode_step_count >= ONE_V_ONE_STEPS else "PREDATOR_WIN"

            # Decay epsilon for both agents
            prey_agent.decay_epsilon()
            predator_agent.decay_epsilon()

            # Save metrics (use predator's perspective by default, but track both)
            prey_rewards_per_ep.append(prey_episode_reward)
            predator_rewards_per_ep.append(predator_episode_reward)
            outcomes_per_ep.append(outcome)

            self.metrics['episodes'].append(episode + 1)
            self.metrics['rewards'].append(predator_episode_reward)
            self.metrics['steps'].append(episode_step_count)
            self.metrics['eval_rewards'].append(0.0)  # Placeholder; final eval computed at end
            self.metrics['populations'].append(self._collect_episode_snapshot(env))
            self.metrics['outcomes'].append(outcome)

            # Log progress for both agents
            if (episode + 1) % max(10, self.num_episodes // 10) == 0:
                print(f"  Episode {episode+1:4d}/{self.num_episodes} | "
                      f"Prey: {prey_episode_reward:8.2f} | Predator: {predator_episode_reward:8.2f} | "
                      f"Outcome: {outcome} | Eps: {prey_agent.epsilon:.3f}")

        # Evaluate both agents greedily
        original_prey_epsilon = prey_agent.epsilon
        original_predator_epsilon = predator_agent.epsilon
        prey_agent.epsilon = 0.0
        predator_agent.epsilon = 0.0

        eval_rewards = []
        num_evals = min(10, max(1, self.num_episodes // 10))

        for ei in range(num_evals):
            env = EcoSimEnv(
                agent_id=0,
                num_prey=1,
                num_predators=1,
                agent_type="PREY",
                map_path=None,
                memory=False,
            )
            obs = env.reset(seed=self._episode_seed(0, 9, ei))
            prey_agent_in_env = env.agent
            predator_agent_in_env = env.other_agents[1] if len(env.other_agents) > 1 else None

            prey_state = prey_agent.discretize_state(obs)
            predator_obs = predator_agent_in_env.get_observation(env.env) if predator_agent_in_env else obs
            predator_state = predator_agent.discretize_state(predator_obs)

            eval_pred_reward = 0
            steps = 0
            done = False

            while steps < ONE_V_ONE_STEPS and not done:
                prey_action = prey_agent.select_action(prey_state, training=False)
                prey_reward_step = prey_agent_in_env.action(prey_action, env.env)

                if predator_agent_in_env and predator_agent_in_env.is_alive():
                    predator_action = predator_agent.select_action(predator_state, training=False)
                    predator_reward_step = predator_agent_in_env.action(predator_action, env.env)
                else:
                    predator_reward_step = 0

                env.env.update_resources()

                dead_agents = [a for a in env.env.agents if not a.is_alive()]
                for dead_agent in dead_agents:
                    dead_agent.die(env.env)

                if not prey_agent_in_env.is_alive():
                    done = True
                    predator_reward_step += HUNTING_SUCCESS_BONUS * 0.5

                prey_state = prey_agent.discretize_state(prey_agent_in_env.get_observation(env.env))
                if predator_agent_in_env and predator_agent_in_env.is_alive():
                    predator_state = predator_agent.discretize_state(predator_agent_in_env.get_observation(env.env))

                eval_pred_reward += predator_reward_step
                steps += 1

            eval_rewards.append(eval_pred_reward)

        prey_agent.epsilon = original_prey_epsilon
        predator_agent.epsilon = original_predator_epsilon

        avg_eval_reward = float(np.mean(eval_rewards)) if eval_rewards else 0.0
        predator_wins = sum(1 for o in outcomes_per_ep if o == "PREDATOR_WIN")
        prey_wins = sum(1 for o in outcomes_per_ep if o == "PREY_WIN")

        print(f"\n1v1 Training Summary:")
        print(f"  Predator Wins: {predator_wins}/{self.num_episodes}")
        print(f"  Prey Wins: {prey_wins}/{self.num_episodes}")
        print(f"  Eval Predator Reward (greedy): {avg_eval_reward:.2f}")

        # Save both models
        prey_model_path = self.results_dir.parent / f"trained_prey_{self.run_number}_protocol1_1v1.pkl"
        predator_model_path = self.results_dir.parent / f"trained_predator_{self.run_number}_protocol1_1v1.pkl"
        prey_agent.save_model(str(prey_model_path))
        predator_agent.save_model(str(predator_model_path))

        self._save_results("1v1", f"Model: prey+predator models saved")

        return {
            'protocol': '1v1_dual_agent',
            'agents': ['PREY', 'PREDATOR'],
            'episodes': self.num_episodes,
            'final_eval': avg_eval_reward,
            'predator_rewards': predator_rewards_per_ep,
            'prey_rewards': prey_rewards_per_ep,
            'outcomes': outcomes_per_ep,
            'predator_wins': predator_wins,
            'prey_wins': prey_wins,
            'prey_model_path': prey_model_path,
            'predator_model_path': predator_model_path,
        }


class AlternatingTrainingProtocol(TrainingProtocol):
    """Protocol 2: Alternating training of prey and predator.
    
    Train PREY for N episodes, freeze, train PREDATOR for N episodes, freeze.
    Reduces non-stationarity by limiting policy change frequency.
    Moderate stability, more realistic interactions.
    """
    
    def train(self, num_cycles: int = 2, start_from_latest: bool = True) -> Dict:
        """Train with alternating phases and carry agents forward across cycles."""

        print(f"\n{'='*70}")
        print(f"PROTOCOL 2: ALTERNATING TRAINING (STATEFUL)")
        print(f"{'='*70}")
        print(f"Num Cycles: {num_cycles}")
        print(f"Episodes per Agent: {self.num_episodes}")
        if start_from_latest:
            print(f"Cycle 1: resume from latest saved prey/predator models when available")
        else:
            print(f"Cycle 1: train both species from scratch")
        print(f"Cycle 2+: continue from previous cycle's learned agents")
        print(f"Each episode gets a distinct seed so the map changes over time")
        print(f"{'='*70}\n")

        all_results = {
            'cycles': [],
            'prey_agents': [],
            'predator_agents': [],
        }

        prey_agent = self._load_latest_saved_agent('PREY') if start_from_latest else None
        predator_agent = self._load_latest_saved_agent('PREDATOR') if start_from_latest else None

        if prey_agent is not None:
            print("✓ Resumed PREY from latest saved model")
        if predator_agent is not None:
            print("✓ Resumed PREDATOR from latest saved model")
        if start_from_latest and prey_agent is None and predator_agent is None:
            print("(No saved models found; starting from scratch)")

        for cycle in range(num_cycles):
            print(f"\n--- CYCLE {cycle + 1}/{num_cycles} ---\n")

            if prey_agent is None:
                prey_agent = QLearningAgent(agent_id=0, num_actions=10, num_states=2160)
            else:
                prey_agent.epsilon = EPSILON_START

            if predator_agent is None:
                predator_agent = QLearningAgent(agent_id=0, num_actions=10, num_states=2160)
            else:
                predator_agent.epsilon = EPSILON_START

            print(f"PHASE A: Training PREY (cycle {cycle + 1})...")
            prey_opponent = predator_agent
            if prey_opponent is None:
                print(f"  (PREDATOR: random)")
            elif cycle == 0:
                print(f"  (PREDATOR: loaded checkpoint)")
            else:
                print(f"  (PREDATOR: carried over from previous cycle)")

            prey_result = self._train_species_phase(
                agent=prey_agent,
                agent_type="PREY",
                cycle=cycle + 1,
                phase_name="A_prey",
                phase_seed_offset=0,
                opponent_agent=prey_opponent,
                opponent_type="PREDATOR",
            )
            prey_agent = prey_result['agent']

            print(f"\nPHASE B: Training PREDATOR (cycle {cycle + 1})...")
            print(f"  (PREY policy frozen from phase A)")
            predator_result = self._train_species_phase(
                agent=predator_agent,
                agent_type="PREDATOR",
                cycle=cycle + 1,
                phase_name="B",
                phase_seed_offset=1,
                opponent_agent=prey_agent,
                opponent_type="PREY",
            )
            predator_agent = predator_result['agent']

            all_results['prey_agents'].append(prey_agent)
            all_results['predator_agents'].append(predator_agent)
            all_results['cycles'].append({
                'cycle': cycle + 1,
                'prey_result': prey_result,
                'predator_result': predator_result,
            })

            print(f"✓ PREDATOR eval reward: {predator_result['eval']:.2f}")

        self._save_alternating_results(all_results, num_cycles)
        return all_results

    def _train_species_phase(
        self,
        agent: QLearningAgent,
        agent_type: str,
        cycle: int,
        phase_name: str,
        phase_seed_offset: int,
        opponent_agent=None,
        opponent_type=None,
    ) -> Dict:
        """Train one species for one phase, optionally against a carried opponent."""

        frozen_same_species_agent = self._clone_agent_policy(agent)

        env = EcoSimEnv(
            agent_id=0,
            num_prey=6,
            num_predators=2,
            agent_type=agent_type,
            map_path=None,
            memory=False,
            opponent_agent=opponent_agent,
            opponent_type=opponent_type,
            same_species_agent=frozen_same_species_agent,
            same_species_type=agent_type,
            frozen_policy_epsilon=0.01,
        )

        if opponent_agent is not None and opponent_type is not None:
            print(f"✓ Using carried {opponent_type} opponent")

        rewards = []
        steps = []
        populations = []
        predation_events = []

        for episode in range(self.num_episodes):
            obs = env.reset(seed=self._episode_seed(cycle, phase_seed_offset, episode))
            state = agent.discretize_state(obs)
            episode_reward = 0
            episode_step_count = 0
            done = False

            with self._track_predation_events() as predation_tracker:
                while not done:
                    action = agent.select_action(state, training=True)
                    next_obs, reward, done, info = env.step(action)
                    next_state = agent.discretize_state(next_obs)

                    agent.update(state, action, reward, next_state, done)

                    state = next_state
                    episode_reward += reward
                    episode_step_count += 1

            agent.decay_epsilon()

            rewards.append(episode_reward)
            steps.append(episode_step_count)
            populations.append(self._collect_episode_snapshot(env))
            predation_events.append(predation_tracker['count'])

            self._log_progress(episode, episode_reward, agent.epsilon, phase=phase_name)
            if (episode + 1) % max(10, self.num_episodes // 10) == 0:
                print(f"    Predation events (last episode): {predation_tracker['count']}")

        avg_train_predation = float(np.mean(predation_events)) if predation_events else 0.0
        eval_result = self._evaluate_greedy(
            env,
            agent,
            seed_base=self._episode_seed(cycle, phase_seed_offset, 1000),
            track_predation=True,
        )
        eval_reward = eval_result['avg_reward']

        model_path = self.results_dir.parent / f"trained_{agent_type.lower()}_{self.run_number}_protocol2_cycle{cycle}.pkl"
        agent.save_model(str(model_path))

        print(f"  ✓ Training predation: {avg_train_predation:.2f} per episode")
        print(f"  ✓ Eval: {eval_reward:.2f} | Predation: {eval_result['avg_predation_events']:.2f} per eval episode")

        return {
            'agent': agent,
            'rewards': rewards,
            'steps': steps,
            'populations': populations,
            'predation_events': predation_events,
            'avg_train_predation': avg_train_predation,
            'eval': eval_reward,
            'eval_predation_events': eval_result['avg_predation_events'],
            'eval_predation_total': eval_result['total_predation_events'],
            'model_path': model_path,
        }

    def _save_alternating_results(self, all_results: Dict, num_cycles: int):
        """Save alternating training results with population metrics."""

        csv_path = self.results_dir / f"protocol_alternating_{self.run_number}.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'cycle', 'phase', 'agent_type', 'episode', 'reward', 'steps', 'eval_reward',
                'prey_population', 'predator_population', 'avg_prey_energy', 'avg_predator_energy',
                'predation_events', 'eval_predation_events',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for cycle_data in all_results['cycles']:
                cycle_num = cycle_data['cycle']

                prey_data = cycle_data['prey_result']
                for ep, reward in enumerate(prey_data['rewards']):
                    pop = prey_data['populations'][ep] if ep < len(prey_data['populations']) else {}
                    writer.writerow({
                        'cycle': cycle_num,
                        'phase': 'A',
                        'agent_type': 'PREY',
                        'episode': ep + 1,
                        'reward': reward,
                        'steps': prey_data['steps'][ep] if ep < len(prey_data['steps']) else 0,
                        'eval_reward': prey_data['eval'],
                        'prey_population': pop.get('prey_population', 0),
                        'predator_population': pop.get('predator_population', 0),
                        'avg_prey_energy': pop.get('avg_prey_energy', 0),
                        'avg_predator_energy': pop.get('avg_predator_energy', 0),
                        'predation_events': prey_data['predation_events'][ep] if ep < len(prey_data['predation_events']) else 0,
                        'eval_predation_events': prey_data['eval_predation_events'],
                    })

                pred_data = cycle_data['predator_result']
                for ep, reward in enumerate(pred_data['rewards']):
                    pop = pred_data['populations'][ep] if ep < len(pred_data['populations']) else {}
                    writer.writerow({
                        'cycle': cycle_num,
                        'phase': 'B',
                        'agent_type': 'PREDATOR',
                        'episode': ep + 1,
                        'reward': reward,
                        'steps': pred_data['steps'][ep] if ep < len(pred_data['steps']) else 0,
                        'eval_reward': pred_data['eval'],
                        'prey_population': pop.get('prey_population', 0),
                        'predator_population': pop.get('predator_population', 0),
                        'avg_prey_energy': pop.get('avg_prey_energy', 0),
                        'avg_predator_energy': pop.get('avg_predator_energy', 0),
                        'predation_events': pred_data['predation_events'][ep] if ep < len(pred_data['predation_events']) else 0,
                        'eval_predation_events': pred_data['eval_predation_events'],
                    })

        print(f"  ✓ Alternating results saved: {csv_path.name}")


class CoLearningProtocol(TrainingProtocol):
    """Protocol 3: Co-learning (both agents learn simultaneously).
    
    Most realistic but most unstable.
    Use only after demonstrating stability with protocols 1-2.
    Explicitly acknowledges non-stationarity.
    """
    
    def train(self) -> Dict:
        """Train with co-learning."""
        
        print(f"\n{'='*70}")
        print(f"PROTOCOL 3: CO-LEARNING (BOTH AGENTS LEARN)")
        print(f"{'='*70}")
        print(f"Episodes: {self.num_episodes}")
        print(f"PREY & PREDATOR: Both training simultaneously")
        print(f"Environment: NON-STATIONARY ✗ (policies change every step)")
        print(f"Convergence: Poor (violates RL convergence assumptions)")
        print(f"Use Only After: Verifying stability with Protocols 1-2")
        print(f"Acknowledgment: Results may be unstable and less reproducible")
        print(f"{'='*70}\n")
        
        from environment.multi_agent_gym_env import MultiAgentEcoSimEnv
        
        env = MultiAgentEcoSimEnv(num_prey=6, num_predators=2)
        
        # Would need to implement full multi-agent learning
        # For now, document the approach
        
        print("⚠ Co-learning protocol requires multi-agent gym environment")
        print("  Documented but not yet implemented (requires careful instability handling)")
        
        return {
            'protocol': 'co_learning',
            'status': 'documented_not_implemented',
            'note': 'Requires explicit non-stationarity awareness and stabilization techniques'
        }


def compare_protocols():
    """Generate comparison table of training protocols."""
    
    comparison = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                   MARL TRAINING PROTOCOL COMPARISON TABLE                      ║
╠═══════════════════╦═════════════════╦════════════════════╦════════════════════╣
║ Aspect            ║ Fixed Opponents ║ Alternating Train  ║ Co-Learning        ║
╠═══════════════════╬═════════════════╬════════════════════╬════════════════════╣
║ Environment       ║ Stationary ✓    ║ Semi-Stationary    ║ Non-Stationary ✗   ║
║ Stability         ║ High            ║ Moderate           ║ Low                ║
║ Realism           ║ Low             ║ Moderate           ║ High               ║
║ Convergence       ║ Guaranteed*     ║ Partial            ║ None               ║
║ Interaction       ║ Limited         ║ Moderate           ║ Full               ║
║ Learning Speed    ║ Fast            ║ Moderate           ║ Slow/Unstable      ║
║ Reproducibility   ║ High            ║ High               ║ Low                ║
║ Comm. Complexity  ║ Simple          ║ Simple             ║ Complex            ║
╠═══════════════════╬═════════════════╬════════════════════╬════════════════════╣
║ Use When          ║ Starting Out    ║ Exploring Stability║ After 1&2 Proven   ║
║ Training Phase    ║ 1: PREY→random  ║ Cycle: PREY/PRED   ║ Both Simultaneous  ║
║                   ║ 2: PRED→random  ║ Both use latest    ║                    ║
╠═══════════════════╬═════════════════╬════════════════════╬════════════════════╣
║ Model Outputs     ║ trained_*_p1.pkl║ trained_*_p2_c*.pkl║ trained_*_p3.pkl   ║
║ CSV Output        ║ protocol_fixed_* ║ protocol_altern_* ║ protocol_co_*      ║
╚═══════════════════╩═════════════════╩════════════════════╩════════════════════╝

* Standard Q-learning convergence proofs apply

RECOMMENDATION:
1. Start with Protocol 1 (Fixed Opponents) - Baseline
2. Move to Protocol 2 (Alternating) - Explore interactions
3. Only use Protocol 3 (Co-Learning) after 1&2 demonstrate stability
   and document instability explicitly in thesis

This progression acknowledges the MARL non-stationarity problem and proves
that the ecosystem simulation is sound before introducing learning complexity.
"""
    
    return comparison
