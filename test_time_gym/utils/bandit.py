"""Thompson Sampling and bandit algorithms for skill selection."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import random

from .models import Skill, ObservationState
from datetime import datetime


class ThompsonSamplingBandit:
    """Thompson Sampling bandit for skill selection."""
    
    def __init__(self, exploration_bonus: float = 0.1, min_confidence: float = 0.3):
        self.exploration_bonus = exploration_bonus
        self.min_confidence = min_confidence
        self.rng = np.random.RandomState()
    
    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)
    
    def select_skill(
        self, 
        skills: List[Skill], 
        observation: ObservationState,
        exclude_recent: bool = True
    ) -> Optional[Skill]:
        """
        Select a skill using Thompson Sampling.
        
        Args:
            skills: Available skills
            observation: Current observation state
            exclude_recent: Whether to exclude recently used skills
            
        Returns:
            Selected skill or None if no suitable skill found
        """
        if not skills:
            return None
        
        # Filter applicable skills based on preconditions
        applicable_skills = []
        for skill in skills:
            if self._is_skill_applicable(skill, observation):
                applicable_skills.append(skill)
        
        if not applicable_skills:
            return None
        
        # Sample from Beta posterior for each skill
        skill_samples = []
        for skill in applicable_skills:
            # Thompson sampling: sample from Beta(alpha, beta)
            sampled_success_rate = self.rng.beta(skill.alpha, skill.beta)
            
            # Add exploration bonus for less-tried skills
            total_attempts = skill.alpha + skill.beta - 2  # Subtract prior pseudocounts
            exploration_boost = self.exploration_bonus / (1 + total_attempts)
            
            final_score = sampled_success_rate + exploration_boost
            skill_samples.append((skill, final_score))
        
        # Select skill with highest sampled value
        skill_samples.sort(key=lambda x: x[1], reverse=True)
        selected_skill = skill_samples[0][0]
        
        # Only return if confidence is above threshold
        if selected_skill.confidence >= self.min_confidence:
            return selected_skill
        
        return None
    
    def update_skill(self, skill: Skill, success: bool) -> None:
        """Update skill statistics based on outcome."""
        skill.attempt_count += 1
        
        if success:
            skill.success_count += 1
            skill.alpha += 1
        else:
            skill.beta += 1
        
        # Update confidence (posterior mean)
        skill.confidence = skill.alpha / (skill.alpha + skill.beta)
        skill.last_used = datetime.now()
    
    def _is_skill_applicable(self, skill: Skill, observation: ObservationState) -> bool:
        """Check if a skill is applicable given the current observation."""
        # Basic applicability check based on current view
        if not skill.steps:
            return False
        
        first_action = skill.steps[0].action.verb
        current_view = observation.view
        
        # Define valid transitions
        valid_transitions = {
            "search_form": ["search_flights"],
            "search_results": ["filter_results", "select_flight", "add_to_cart"],
            "cart": ["proceed_to_payment", "apply_coupon"],
            "payment": ["enter_card", "confirm_payment"],
            "error": ["restart", "search_flights"]
        }
        
        valid_actions = valid_transitions.get(current_view, [])
        
        # Check if first action of skill is valid for current view
        if first_action not in valid_actions:
            return False
        
        # Check preconditions if any
        if skill.preconditions:
            # Simple precondition checking (can be extended)
            if "min_flights" in skill.preconditions:
                if len(observation.flights) < skill.preconditions["min_flights"]:
                    return False
            
            if "cart_not_empty" in skill.preconditions:
                if skill.preconditions["cart_not_empty"] and not observation.cart.items:
                    return False
        
        return True
    
    def get_exploration_ratio(self, recent_actions: List[str], skill_actions: List[str]) -> float:
        """Calculate exploration vs exploitation ratio."""
        if not recent_actions:
            return 0.0
        
        exploration_count = sum(1 for action in recent_actions if action not in skill_actions)
        return exploration_count / len(recent_actions)


class ContextualBandit:
    """Contextual bandit that considers observation context for skill selection."""
    
    def __init__(self, context_dim: int = 10, alpha: float = 1.0):
        self.context_dim = context_dim
        self.alpha = alpha
        self.skill_contexts = {}  # skill_id -> list of context vectors
        self.skill_rewards = {}   # skill_id -> list of rewards
    
    def extract_context(self, observation: ObservationState) -> np.ndarray:
        """Extract context features from observation."""
        context = np.zeros(self.context_dim)
        
        # View encoding (one-hot)
        view_mapping = {
            "search_form": 0, "search_results": 1, "cart": 2, 
            "payment": 3, "receipt": 4, "error": 5
        }
        if observation.view in view_mapping:
            context[view_mapping[observation.view]] = 1.0
        
        # Budget ratio
        if observation.cart.total > 0 and observation.constraints.budget > 0:
            context[6] = min(1.0, observation.cart.total / observation.constraints.budget)
        
        # Number of flights (normalized)
        context[7] = min(1.0, len(observation.flights) / 20.0)
        
        # Cart status
        context[8] = 1.0 if observation.cart.items else 0.0
        
        # Step progress
        context[9] = min(1.0, observation.step_count / 50.0)
        
        return context
    
    def select_skill_contextual(
        self, 
        skills: List[Skill], 
        observation: ObservationState
    ) -> Optional[Skill]:
        """Select skill using contextual information."""
        if not skills:
            return None
        
        context = self.extract_context(observation)
        
        best_skill = None
        best_score = -float('inf')
        
        for skill in skills:
            if skill.id not in self.skill_contexts:
                # New skill - give it a chance
                score = 0.5 + self.alpha * np.random.normal(0, 0.1)
            else:
                # Compute similarity to past contexts where this skill was used
                past_contexts = np.array(self.skill_contexts[skill.id])
                past_rewards = np.array(self.skill_rewards[skill.id])
                
                # Compute context similarity (cosine similarity)
                similarities = np.dot(past_contexts, context) / (
                    np.linalg.norm(past_contexts, axis=1) * np.linalg.norm(context) + 1e-8
                )
                
                # Weight by similarity and compute expected reward
                weights = np.exp(similarities)  # Exponential weighting
                weights /= weights.sum()
                
                expected_reward = np.dot(weights, past_rewards)
                uncertainty = np.sqrt(np.dot(weights, (past_rewards - expected_reward) ** 2))
                
                # Upper confidence bound with contextual bonus
                score = expected_reward + self.alpha * uncertainty
            
            if score > best_score:
                best_score = score
                best_skill = skill
        
        return best_skill
    
    def update_contextual(
        self, 
        skill: Skill, 
        context: np.ndarray, 
        reward: float
    ) -> None:
        """Update contextual bandit with skill outcome."""
        if skill.id not in self.skill_contexts:
            self.skill_contexts[skill.id] = []
            self.skill_rewards[skill.id] = []
        
        self.skill_contexts[skill.id].append(context.tolist())
        self.skill_rewards[skill.id].append(reward)
        
        # Keep only recent contexts (sliding window)
        max_history = 100
        if len(self.skill_contexts[skill.id]) > max_history:
            self.skill_contexts[skill.id] = self.skill_contexts[skill.id][-max_history:]
            self.skill_rewards[skill.id] = self.skill_rewards[skill.id][-max_history:]