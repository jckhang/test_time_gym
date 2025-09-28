"""Logging and trajectory recording utilities."""

import json
import jsonlines
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from .models import Episode, TrajectoryStep, Skill, EvaluationMetrics


class TrajectoryLogger:
    """Handles logging of agent trajectories and episodes."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / "episodes").mkdir(exist_ok=True)
        (self.log_dir / "skills").mkdir(exist_ok=True)
        (self.log_dir / "metrics").mkdir(exist_ok=True)
        
        self.session_id = str(uuid.uuid4())
        self.episodes_file = self.log_dir / f"episodes_{self.session_id}.jsonl"
        self.metrics_file = self.log_dir / "metrics.jsonl"
    
    def log_episode(self, episode: Episode) -> None:
        """Log a complete episode to file."""
        with jsonlines.open(self.episodes_file, mode='a') as writer:
            writer.write(episode.dict())
    
    def log_step(self, step: TrajectoryStep) -> None:
        """Log a single trajectory step."""
        step_file = self.log_dir / f"steps_{self.session_id}.jsonl"
        with jsonlines.open(step_file, mode='a') as writer:
            writer.write(step.dict())
    
    def log_metrics(self, metrics: EvaluationMetrics, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log evaluation metrics."""
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "metrics": metrics.dict(),
            "metadata": metadata or {}
        }
        
        with jsonlines.open(self.metrics_file, mode='a') as writer:
            writer.write(metric_entry)
    
    def save_skill(self, skill: Skill) -> None:
        """Save a skill to the skill database."""
        skill_file = self.log_dir / "skills" / f"{skill.id}.json"
        with open(skill_file, 'w') as f:
            json.dump(skill.dict(), f, indent=2, default=str)
    
    def load_skills(self) -> List[Skill]:
        """Load all skills from the skill database."""
        skills = []
        skills_dir = self.log_dir / "skills"
        
        if skills_dir.exists():
            for skill_file in skills_dir.glob("*.json"):
                try:
                    with open(skill_file, 'r') as f:
                        skill_data = json.load(f)
                        skills.append(Skill(**skill_data))
                except Exception as e:
                    print(f"Error loading skill {skill_file}: {e}")
        
        return skills
    
    def load_episodes(self, limit: Optional[int] = None) -> List[Episode]:
        """Load episodes from log files."""
        episodes = []
        
        for episodes_file in self.log_dir.glob("episodes_*.jsonl"):
            try:
                with jsonlines.open(episodes_file) as reader:
                    for episode_data in reader:
                        episodes.append(Episode(**episode_data))
                        if limit and len(episodes) >= limit:
                            break
            except Exception as e:
                print(f"Error loading episodes from {episodes_file}: {e}")
        
        return episodes[:limit] if limit else episodes
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        episodes = self.load_episodes()
        
        if not episodes:
            return {"episodes": 0}
        
        total_episodes = len(episodes)
        successful_episodes = sum(1 for ep in episodes if ep.success)
        total_steps = sum(len(ep.steps) for ep in episodes)
        total_violations = sum(ep.constraint_violations for ep in episodes)
        
        return {
            "session_id": self.session_id,
            "episodes": total_episodes,
            "success_rate": successful_episodes / total_episodes if total_episodes > 0 else 0,
            "avg_steps": total_steps / total_episodes if total_episodes > 0 else 0,
            "violation_rate": total_violations / total_episodes if total_episodes > 0 else 0,
            "avg_reward": sum(ep.total_reward for ep in episodes) / total_episodes if total_episodes > 0 else 0
        }


class SkillExtractor:
    """Extracts reusable skills from successful trajectories."""
    
    def __init__(self, min_success_rate: float = 0.6, min_occurrences: int = 3):
        self.min_success_rate = min_success_rate
        self.min_occurrences = min_occurrences
    
    def extract_skills_from_episodes(self, episodes: List[Episode]) -> List[Skill]:
        """Extract skills from a collection of episodes."""
        # Find successful episodes
        successful_episodes = [ep for ep in episodes if ep.success]
        
        if len(successful_episodes) < self.min_occurrences:
            return []
        
        # Extract common action sequences
        action_sequences = []
        for episode in successful_episodes:
            sequence = []
            for step in episode.steps:
                if not step.done:  # Exclude final step
                    sequence.append((step.action.verb, step.observation.view))
            action_sequences.append(sequence)
        
        # Find frequent subsequences (simplified pattern mining)
        skills = []
        for seq_length in range(2, 6):  # Look for sequences of length 2-5
            subsequences = {}
            
            for sequence in action_sequences:
                for i in range(len(sequence) - seq_length + 1):
                    subseq = tuple(sequence[i:i + seq_length])
                    if subseq not in subsequences:
                        subsequences[subseq] = 0
                    subsequences[subseq] += 1
            
            # Create skills from frequent subsequences
            for subseq, count in subsequences.items():
                if count >= self.min_occurrences:
                    skill = self._create_skill_from_sequence(subseq, count, len(successful_episodes))
                    if skill:
                        skills.append(skill)
        
        return skills
    
    def _create_skill_from_sequence(self, sequence: tuple, count: int, total_episodes: int) -> Optional[Skill]:
        """Create a skill from an action sequence."""
        if len(sequence) < 2:
            return None
        
        skill_id = f"skill_{hash(sequence) & 0x7fffffff}"
        
        # Convert sequence to skill steps
        steps = []
        for i, (verb, expected_view) in enumerate(sequence):
            step = SkillStep(
                action=Action(verb=verb, payload={}),
                expected_view=expected_view if i < len(sequence) - 1 else None
            )
            steps.append(step)
        
        # Calculate confidence based on frequency
        success_rate = count / total_episodes
        
        skill = Skill(
            id=skill_id,
            name=f"Sequence: {' -> '.join([verb for verb, _ in sequence])}",
            description=f"Common action sequence extracted from {count} successful episodes",
            steps=steps,
            success_count=count,
            attempt_count=count,  # Assume all extractions were successful
            confidence=success_rate,
            alpha=count + 1,  # Beta prior with pseudocounts
            beta=max(1, total_episodes - count + 1)
        )
        
        return skill