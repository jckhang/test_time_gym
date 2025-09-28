"""Advanced experiment runner with A/B testing framework."""

import sys
sys.path.append('..')

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.agents.dummy_agent import DummyAgent, RandomAgent
from test_time_gym.agents.learning_agent import LearningAgent
from test_time_gym.evaluation.metrics import ExperimentRunner, MetricsCalculator
from test_time_gym.utils.logger import TrajectoryLogger


class TestTimeGymExperiment:
    """Complete experiment framework for the Test-Time Gym."""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.runner = ExperimentRunner(str(self.output_dir))
        self.calculator = MetricsCalculator()
    
    def run_ablation_study(self, num_episodes: int = 500, num_seeds: int = 3) -> Dict[str, Any]:
        """
        Run comprehensive ablation study as described in the evaluation framework.
        
        Tests the following conditions:
        1. No-Memory: Pure planning without experience
        2. Memory: Full learning agent with skills and bandit
        3. Shuffled-Memory: Skills but random selection
        4. Skills-Off: Memory but no skill extraction
        5. Bandit-Off: Skills but no Thompson Sampling
        6. Random: Completely random baseline
        """
        print("Starting Ablation Study...")
        print(f"Episodes per condition: {num_episodes}")
        print(f"Random seeds: {num_seeds}")
        
        # Define experimental conditions
        conditions = {
            "no_memory": {
                "agent": {"enable_skills": False, "enable_bandit": False, "enable_shield": True},
                "env": {"enable_3ds": True, "enable_payment_failures": True}
            },
            "memory": {
                "agent": {"enable_skills": True, "enable_bandit": True, "enable_shield": True},
                "env": {"enable_3ds": True, "enable_payment_failures": True}
            },
            "shuffled_memory": {
                "agent": {"enable_skills": True, "enable_bandit": False, "enable_shield": True},
                "env": {"enable_3ds": True, "enable_payment_failures": True}
            },
            "skills_off": {
                "agent": {"enable_skills": False, "enable_bandit": True, "enable_shield": True},
                "env": {"enable_3ds": True, "enable_payment_failures": True}
            },
            "bandit_off": {
                "agent": {"enable_skills": True, "enable_bandit": False, "enable_shield": True},
                "env": {"enable_3ds": True, "enable_payment_failures": True}
            }
        }
        
        def agent_factory(**kwargs):
            if kwargs.get("enable_skills", False) or kwargs.get("enable_bandit", False):
                return LearningAgent(**kwargs)
            else:
                return DummyAgent(verbose=False)
        
        def env_factory(**kwargs):
            return FlightBookingEnv(**kwargs)
        
        # Run experiment
        results = self.runner.run_ablation_study(
            agent_factory=agent_factory,
            env_factory=env_factory,
            conditions=conditions,
            num_episodes=num_episodes,
            num_seeds=num_seeds
        )
        
        # Generate report
        report = self.runner.generate_report(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"ablation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        report_file = self.output_dir / f"ablation_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Report saved to: {report_file}")
        
        return results
    
    def run_ood_robustness_test(self, num_episodes: int = 200) -> Dict[str, Any]:
        """Test robustness under out-of-distribution conditions."""
        print("Running OOD Robustness Test...")
        
        # Test different levels of distribution shift
        ood_levels = {
            "baseline": {"price_noise_std": 0.1, "enable_3ds": True, "enable_payment_failures": True},
            "light_shift": {"price_noise_std": 0.3, "enable_3ds": True, "enable_payment_failures": True},
            "medium_shift": {"price_noise_std": 0.5, "enable_3ds": True, "enable_payment_failures": True},
            "heavy_shift": {"price_noise_std": 0.8, "enable_3ds": True, "enable_payment_failures": True}
        }
        
        results = {}
        
        for level_name, env_config in ood_levels.items():
            print(f"Testing {level_name} distribution shift...")
            
            episodes = []
            for seed in range(3):
                env = FlightBookingEnv(seed=seed, **env_config)
                agent = LearningAgent(seed=seed, enable_skills=True, enable_bandit=True)
                
                for ep_idx in range(num_episodes // 3):
                    obs, info = env.reset()
                    done = False
                    truncated = False
                    
                    while not (done or truncated):
                        action = agent.select_action(obs, info)
                        obs, reward, done, truncated, info = env.step(action)
                    
                    if hasattr(env, 'episode'):
                        episodes.append(env.episode)
            
            metrics = self.calculator.calculate_metrics(episodes)
            results[level_name] = metrics.dict()
        
        return results
    
    def run_learning_curve_analysis(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """Analyze learning curves over extended training."""
        print("Running Learning Curve Analysis...")
        
        # Create learning agent
        env = FlightBookingEnv(seed=42)
        agent = LearningAgent(seed=42, enable_skills=True, enable_bandit=True, verbose=True)
        
        episodes = []
        rewards = []
        
        for episode_idx in range(num_episodes):
            obs, info = env.reset(seed=42 + episode_idx)
            episode_reward = 0.0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.select_action(obs, info)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            
            # Notify agent of episode end
            success = info.get("success", False) if done else False
            agent.end_episode(episode_reward, success, env.episode if hasattr(env, 'episode') else None)
            
            if hasattr(env, 'episode'):
                episodes.append(env.episode)
            
            # Progress reporting
            if (episode_idx + 1) % 100 == 0:
                recent_success_rate = sum(1 for r in rewards[-100:] if r > 0.5) / 100
                print(f"Episode {episode_idx + 1}: Recent success rate = {recent_success_rate:.3f}")
                
                agent_stats = agent.get_agent_stats()
                print(f"  Skills learned: {agent_stats['total_skills']}")
        
        # Analyze results
        metrics = self.calculator.calculate_metrics(episodes)
        learning_curve = self.calculator.calculate_learning_curve(episodes)
        
        results = {
            "final_metrics": metrics.dict(),
            "learning_curve": {
                "episode_numbers": learning_curve[0],
                "success_rates": learning_curve[1]
            },
            "reward_history": rewards,
            "total_episodes": len(episodes)
        }
        
        return results


def main():
    """Run comprehensive experiments."""
    print("Test-Time Gym Comprehensive Experiment Suite")
    print("=" * 50)
    
    experiment = TestTimeGymExperiment()
    
    # 1. Basic usage demo
    print("\n1. Basic Usage Demo")
    run_dummy_agent_example()
    
    # 2. Quick ablation study (reduced episodes for demo)
    print("\n2. Quick Ablation Study")
    ablation_results = experiment.run_ablation_study(num_episodes=50, num_seeds=2)
    
    print("\nAblation Study Summary:")
    for condition, data in ablation_results.items():
        if condition != "comparison" and "metrics" in data:
            metrics = data["metrics"]
            print(f"  {condition}: Success={metrics['success_rate']:.3f}, Steps={metrics['avg_steps_to_success']:.1f}")
    
    # 3. OOD robustness test
    print("\n3. OOD Robustness Test")
    ood_results = experiment.run_ood_robustness_test(num_episodes=60)
    
    print("\nOOD Robustness Summary:")
    for level, metrics in ood_results.items():
        print(f"  {level}: Success={metrics['success_rate']:.3f}, Violations={metrics['constraint_violation_rate']:.3f}")
    
    # 4. Learning curve analysis (reduced for demo)
    print("\n4. Learning Curve Analysis")
    learning_results = experiment.run_learning_curve_analysis(num_episodes=200)
    
    print(f"\nLearning Analysis Summary:")
    print(f"  Final Success Rate: {learning_results['final_metrics']['success_rate']:.3f}")
    print(f"  Final Avg Steps: {learning_results['final_metrics']['avg_steps_to_success']:.1f}")
    print(f"  Skill Reuse Rate: {learning_results['final_metrics']['skill_reuse_rate']:.3f}")
    
    print(f"\n🎉 All experiments completed!")
    print(f"📊 Results saved to: {experiment.output_dir}")


if __name__ == "__main__":
    main()