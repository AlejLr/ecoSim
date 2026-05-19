"""Run training using different MARL stability protocols.

Usage:
    python -m src.models.run_training_protocol 1 PREY 100
    # Protocol 1 (Fixed Opponents): Train PREY for 100 episodes
    
    python -m src.models.run_training_protocol 2 3 100
    # Protocol 2 (Alternating): 3 cycles, 100 episodes each
    
    python -m src.models.run_training_protocol compare
    # Show comparison table of protocols
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.utils import set_global_seed, get_next_run_number
from environment.logger import save_environment_log
from models.training_protocols import (
    FixedOpponentProtocol,
    AlternatingTrainingProtocol,
    CoLearningProtocol,
    compare_protocols
)


def main():
    """Main protocol runner."""
    
    # Set global seed
    set_global_seed()
    
    # Get run number
    run_number = get_next_run_number()
    
    # Save environment documentation
    save_environment_log(run_number)
    
    print(f"\n{'='*70}")
    print(f"MARL STABILITY PROTOCOLS - Run #{run_number}")
    print(f"{'='*70}")
    print(f"Addressing Issue 14 & 15: MARL Non-Stationarity\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.models.run_training_protocol 1 PREY 100")
        print("    Protocol 1: Fixed Opponents, train PREY, 100 episodes")
        print()
        print("  python -m src.models.run_training_protocol 2 3 100")
        print("    Protocol 2: Alternating training, 3 cycles, 100 episodes each")
        print()
        print("  python -m src.models.run_training_protocol compare")
        print("    Show comparison table of all protocols")
        print()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Show comparison
    if command == "compare":
        print(compare_protocols())
        return
    
    # Protocol 1: Fixed Opponents
    if command == "1":
        if len(sys.argv) < 4:
            print("Usage: python -m src.models.run_training_protocol 1 [PREY|PREDATOR] [episodes]")
            sys.exit(1)
        
        agent_type = sys.argv[2].upper()
        num_episodes = int(sys.argv[3])

        # Optional resume argument: --resume <prey_model_path> <predator_model_path>
        resume_prey = None
        resume_predator = None
        if '--resume' in sys.argv:
            idx = sys.argv.index('--resume')
            # Expect two file paths after --resume
            if len(sys.argv) > idx + 2:
                resume_prey = sys.argv[idx + 1]
                resume_predator = sys.argv[idx + 2]
            else:
                print("Usage: --resume <prey_model_path> <predator_model_path>")
                sys.exit(1)

        protocol = FixedOpponentProtocol(agent_type, num_episodes, run_number)
        result = protocol.train(resume_prey=resume_prey, resume_predator=resume_predator)
        
        print(f"\n{'='*70}")
        print(f"PROTOCOL 1 COMPLETE")
        print(f"{'='*70}")
        print(f"Final Evaluation: {result['final_eval']:.2f}")
        print(f"Predator Wins: {result.get('predator_wins', 0)}/{result['episodes']}")
        print(f"Prey Wins: {result.get('prey_wins', 0)}/{result['episodes']}")
        if 'prey_model_path' in result:
            print(f"Prey Model: {result['prey_model_path'].name}")
        if 'predator_model_path' in result:
            print(f"Predator Model: {result['predator_model_path'].name}")
        elif 'model_path' in result:
            print(f"Model saved: {result['model_path'].name}")
        
    # Protocol 2: Alternating Training
    elif command == "2":
        if len(sys.argv) < 4:
            print("Usage: python -m src.models.run_training_protocol 2 [cycles] [episodes_per_cycle] [new]")
            sys.exit(1)
        
        num_cycles = int(sys.argv[2])
        num_episodes = int(sys.argv[3])
        start_from_latest = True
        if len(sys.argv) >= 5 and sys.argv[4].strip().lower() == "new":
            start_from_latest = False
        
        protocol = AlternatingTrainingProtocol("", num_episodes, run_number)
        result = protocol.train(num_cycles=num_cycles, start_from_latest=start_from_latest)
        
        print(f"\n{'='*70}")
        print(f"PROTOCOL 2 COMPLETE")
        print(f"{'='*70}")
        print(f"Completed {num_cycles} cycles")
        print(f"Prey agents trained: {len(result['prey_agents'])}")
        print(f"Predator agents trained: {len(result['predator_agents'])}")
        if start_from_latest:
            print("Resume mode: latest saved models were used when available")
        else:
            print("Fresh mode: training started from scratch")
        if result.get('cycles'):
            prey_train_predation = [cycle_data['prey_result'].get('avg_train_predation', 0.0) for cycle_data in result['cycles']]
            predator_train_predation = [cycle_data['predator_result'].get('avg_train_predation', 0.0) for cycle_data in result['cycles']]
            prey_eval_predation = [cycle_data['prey_result'].get('eval_predation_events', 0.0) for cycle_data in result['cycles']]
            predator_eval_predation = [cycle_data['predator_result'].get('eval_predation_events', 0.0) for cycle_data in result['cycles']]

            def mean(values):
                return sum(values) / len(values) if values else 0.0

            print(f"Training predation / episode - PREY: {mean(prey_train_predation):.2f} | PREDATOR: {mean(predator_train_predation):.2f}")
            print(f"Eval predation / episode - PREY: {mean(prey_eval_predation):.2f} | PREDATOR: {mean(predator_eval_predation):.2f}")
        
    # Protocol 3: Co-Learning
    elif command == "3":
        protocol = CoLearningProtocol("", 0, run_number)
        result = protocol.train()
        
        print(f"\n{'='*70}")
        print(f"PROTOCOL 3 STATUS")
        print(f"{'='*70}")
        print(f"Status: {result['status']}")
        print(f"Note: {result['note']}")
    
    else:
        print(f"Unknown protocol: {command}")
        print("Use: 1 (Fixed), 2 (Alternating), 3 (Co-Learning), or compare")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Run #{run_number} Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
