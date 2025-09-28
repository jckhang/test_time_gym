"""Evaluation metrics and analysis tools."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils.models import Episode, EvaluationMetrics, Skill
from ..utils.logger import TrajectoryLogger
from datetime import datetime


class MetricsCalculator:
    """Calculate various evaluation metrics from episode data."""
    
    def __init__(self):
        self.logger = TrajectoryLogger()
    
    def calculate_metrics(self, episodes: List[Episode]) -> EvaluationMetrics:
        """Calculate comprehensive metrics from episodes."""
        if not episodes:
            return EvaluationMetrics()
        
        total_episodes = len(episodes)
        successful_episodes = [ep for ep in episodes if ep.success]
        
        # Basic success metrics
        success_rate = len(successful_episodes) / total_episodes
        
        # Steps to success (only for successful episodes)
        if successful_episodes:
            avg_steps_to_success = np.mean([len(ep.steps) for ep in successful_episodes])
        else:
            avg_steps_to_success = 0.0
        
        # Constraint violations
        total_violations = sum(ep.constraint_violations for ep in episodes)
        violation_rate = total_violations / total_episodes
        
        # Regret
        avg_regret = np.mean([ep.final_regret for ep in episodes if ep.final_regret > 0])
        if np.isnan(avg_regret):
            avg_regret = 0.0
        
        # Exploration ratio
        exploration_ratios = [ep.exploration_ratio for ep in episodes if ep.exploration_ratio > 0]
        avg_exploration_ratio = np.mean(exploration_ratios) if exploration_ratios else 0.0
        
        # Skill reuse rate
        skill_usage_count = 0
        total_actions = 0
        for episode in episodes:
            for step in episode.steps:
                total_actions += 1
                if step.skill_used:
                    skill_usage_count += 1
        
        skill_reuse_rate = skill_usage_count / total_actions if total_actions > 0 else 0.0
        
        # Invalid action rate (actions that resulted in negative immediate reward < -0.04)
        invalid_actions = 0
        total_steps = 0
        for episode in episodes:
            for step in episode.steps:
                total_steps += 1
                if step.reward < -0.04:  # Threshold for invalid actions
                    invalid_actions += 1
        
        invalid_action_rate = invalid_actions / total_steps if total_steps > 0 else 0.0
        
        return EvaluationMetrics(
            success_rate=success_rate,
            avg_steps_to_success=avg_steps_to_success,
            constraint_violation_rate=violation_rate,
            avg_regret=avg_regret,
            exploration_ratio=avg_exploration_ratio,
            skill_reuse_rate=skill_reuse_rate,
            invalid_action_rate=invalid_action_rate,
            episodes_evaluated=total_episodes
        )
    
    def calculate_learning_curve(
        self, 
        episodes: List[Episode], 
        window_size: int = 100
    ) -> Tuple[List[int], List[float]]:
        """Calculate learning curve (success rate over time)."""
        if len(episodes) < window_size:
            window_size = len(episodes)
        
        episode_numbers = []
        success_rates = []
        
        for i in range(window_size, len(episodes) + 1, window_size // 2):
            window_episodes = episodes[max(0, i - window_size):i]
            success_rate = sum(1 for ep in window_episodes if ep.success) / len(window_episodes)
            
            episode_numbers.append(i)
            success_rates.append(success_rate)
        
        return episode_numbers, success_rates
    
    def analyze_skill_performance(self, skills: List[Skill]) -> Dict[str, Any]:
        """Analyze performance of extracted skills."""
        if not skills:
            return {"total_skills": 0}
        
        skill_stats = []
        for skill in skills:
            stats_dict = {
                "id": skill.id,
                "name": skill.name,
                "success_rate": skill.confidence,
                "attempts": skill.attempt_count,
                "successes": skill.success_count,
                "confidence_interval": self._beta_confidence_interval(skill.alpha, skill.beta),
                "last_used": skill.last_used.isoformat() if skill.last_used else None
            }
            skill_stats.append(stats_dict)
        
        # Sort by success rate
        skill_stats.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return {
            "total_skills": len(skills),
            "avg_success_rate": np.mean([s["success_rate"] for s in skill_stats]),
            "skills": skill_stats
        }
    
    def _beta_confidence_interval(self, alpha: float, beta: float, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for Beta distribution."""
        lower = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
        upper = stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)
        return (lower, upper)
    
    def compare_experimental_conditions(
        self, 
        control_episodes: List[Episode], 
        treatment_episodes: List[Episode]
    ) -> Dict[str, Any]:
        """Compare two experimental conditions."""
        control_metrics = self.calculate_metrics(control_episodes)
        treatment_metrics = self.calculate_metrics(treatment_episodes)
        
        # Statistical tests
        control_successes = [1 if ep.success else 0 for ep in control_episodes]
        treatment_successes = [1 if ep.success else 0 for ep in treatment_episodes]
        
        # Two-sample t-test for success rates
        success_stat, success_pval = stats.ttest_ind(control_successes, treatment_successes)
        
        # Mann-Whitney U test for steps to success
        control_steps = [len(ep.steps) for ep in control_episodes if ep.success]
        treatment_steps = [len(ep.steps) for ep in treatment_episodes if ep.success]
        
        if control_steps and treatment_steps:
            steps_stat, steps_pval = stats.mannwhitneyu(control_steps, treatment_steps, alternative='two-sided')
        else:
            steps_stat, steps_pval = 0, 1.0
        
        return {
            "control": control_metrics.dict(),
            "treatment": treatment_metrics.dict(),
            "statistical_tests": {
                "success_rate": {"statistic": success_stat, "p_value": success_pval},
                "steps_to_success": {"statistic": steps_stat, "p_value": steps_pval}
            },
            "effect_sizes": {
                "success_rate_diff": treatment_metrics.success_rate - control_metrics.success_rate,
                "steps_reduction": control_metrics.avg_steps_to_success - treatment_metrics.avg_steps_to_success
            }
        }


class ExperimentRunner:
    """Run controlled experiments with different agent configurations."""
    
    def __init__(self, log_dir: str = "experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.calculator = MetricsCalculator()
    
    def run_ablation_study(
        self,
        agent_factory,
        env_factory,
        conditions: Dict[str, Dict[str, Any]],
        num_episodes: int = 1000,
        num_seeds: int = 3
    ) -> Dict[str, Any]:
        """
        Run ablation study with different experimental conditions.
        
        Args:
            agent_factory: Function that creates agent with given config
            env_factory: Function that creates environment with given config
            conditions: Dict mapping condition names to agent/env configs
            num_episodes: Number of episodes per condition per seed
            num_seeds: Number of random seeds to test
            
        Returns:
            Comprehensive results dictionary
        """
        results = {}
        
        for condition_name, config in conditions.items():
            print(f"Running condition: {condition_name}")
            condition_episodes = []
            
            for seed in range(num_seeds):
                print(f"  Seed {seed + 1}/{num_seeds}")
                
                # Create environment and agent
                env = env_factory(seed=seed, **config.get("env", {}))
                agent = agent_factory(**config.get("agent", {}))
                
                # Run episodes
                seed_episodes = []
                for episode_idx in range(num_episodes):
                    obs, info = env.reset()
                    episode_reward = 0
                    
                    done = False
                    truncated = False
                    
                    while not (done or truncated):
                        action = agent.select_action(obs, info)
                        obs, reward, done, truncated, info = env.step(action)
                        episode_reward += reward
                    
                    # Get episode data from environment
                    if hasattr(env, 'episode'):
                        seed_episodes.append(env.episode)
                
                condition_episodes.extend(seed_episodes)
            
            # Calculate metrics for this condition
            metrics = self.calculator.calculate_metrics(condition_episodes)
            results[condition_name] = {
                "metrics": metrics.dict(),
                "episodes": len(condition_episodes),
                "learning_curve": self.calculator.calculate_learning_curve(condition_episodes)
            }
        
        # Statistical comparisons
        if len(conditions) == 2:
            condition_names = list(conditions.keys())
            comparison = self.calculator.compare_experimental_conditions(
                results[condition_names[0]]["episodes"],
                results[condition_names[1]]["episodes"]
            )
            results["comparison"] = comparison
        
        # Save results
        experiment_file = self.log_dir / f"ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(experiment_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate a human-readable report from experiment results."""
        report_lines = [
            "# Test-Time Gym Experiment Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary"
        ]
        
        for condition_name, condition_data in results.items():
            if condition_name == "comparison":
                continue
            
            metrics = condition_data["metrics"]
            report_lines.extend([
                f"### {condition_name}",
                f"- Success Rate: {metrics['success_rate']:.3f}",
                f"- Avg Steps to Success: {metrics['avg_steps_to_success']:.1f}",
                f"- Constraint Violation Rate: {metrics['constraint_violation_rate']:.3f}",
                f"- Avg Regret: ${metrics['avg_regret']:.2f}",
                f"- Skill Reuse Rate: {metrics['skill_reuse_rate']:.3f}",
                f"- Episodes: {metrics['episodes_evaluated']}",
                ""
            ])
        
        if "comparison" in results:
            comp = results["comparison"]
            report_lines.extend([
                "## Statistical Comparison",
                f"- Success Rate Difference: {comp['effect_sizes']['success_rate_diff']:.3f}",
                f"- Steps Reduction: {comp['effect_sizes']['steps_reduction']:.1f}",
                f"- Success Rate p-value: {comp['statistical_tests']['success_rate']['p_value']:.6f}",
                f"- Steps p-value: {comp['statistical_tests']['steps_to_success']['p_value']:.6f}",
                ""
            ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report