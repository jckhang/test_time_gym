"""Learning agent that uses skills and Thompson Sampling."""

import json
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..utils.models import (
    Action, ObservationState, Skill, SkillStep, SearchForm, 
    Constraints, Flight, Cart, PaymentState
)
from ..utils.bandit import ThompsonSamplingBandit, ContextualBandit
from ..utils.logger import TrajectoryLogger, SkillExtractor
from ..evaluation.ood_detection import ActionShield


class LearningAgent:
    """
    A learning agent that extracts and reuses skills using Thompson Sampling.
    
    This agent demonstrates the core learning mechanisms:
    1. Skill extraction from successful trajectories
    2. Thompson Sampling for exploration/exploitation
    3. Contextual bandit for skill selection
    4. Safety shield for action validation
    """
    
    def __init__(
        self,
        enable_skills: bool = True,
        enable_bandit: bool = True,
        enable_shield: bool = True,
        exploration_rate: float = 0.2,
        skill_update_frequency: int = 50,
        log_dir: str = "logs",
        seed: Optional[int] = None
    ):
        self.enable_skills = enable_skills
        self.enable_bandit = enable_bandit
        self.enable_shield = enable_shield
        self.exploration_rate = exploration_rate
        self.skill_update_frequency = skill_update_frequency
        
        # Initialize components
        self.logger = TrajectoryLogger(log_dir)
        self.skill_extractor = SkillExtractor()
        self.bandit = ThompsonSamplingBandit()
        self.contextual_bandit = ContextualBandit()
        self.shield = ActionShield()
        
        # State tracking
        self.skills: List[Skill] = []
        self.current_skill: Optional[Skill] = None
        self.current_skill_step: int = 0
        self.episode_count: int = 0
        self.action_history: List[Action] = []
        
        if seed is not None:
            self.seed(seed)
        
        # Load existing skills
        if self.enable_skills:
            self.skills = self.logger.load_skills()
            print(f"Loaded {len(self.skills)} existing skills")
    
    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        self.bandit.seed(seed)
    
    def select_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Select action using learned skills or exploration."""
        # Parse observation
        obs_state = self._parse_observation(observation)
        
        # Reset skill if view changed unexpectedly
        if self.current_skill and len(self.current_skill.steps) > self.current_skill_step:
            expected_view = self.current_skill.steps[self.current_skill_step].expected_view
            if expected_view and obs_state.view != expected_view:
                if random.random() < 0.7:  # 70% chance to abandon skill on unexpected state
                    self.current_skill = None
                    self.current_skill_step = 0
        
        selected_action = None
        skill_used = None
        
        # Try to use current skill
        if self.current_skill and self.current_skill_step < len(self.current_skill.steps):
            skill_step = self.current_skill.steps[self.current_skill_step]
            selected_action = skill_step.action
            skill_used = self.current_skill.id
            self.current_skill_step += 1
            
            # If skill is complete, mark it
            if self.current_skill_step >= len(self.current_skill.steps):
                self.current_skill = None
                self.current_skill_step = 0
        
        # Try to select new skill
        elif self.enable_skills and self.enable_bandit and random.random() > self.exploration_rate:
            if self.skills:
                selected_skill = self.bandit.select_skill(self.skills, obs_state)
                if selected_skill:
                    self.current_skill = selected_skill
                    self.current_skill_step = 0
                    
                    if self.current_skill.steps:
                        selected_action = self.current_skill.steps[0].action
                        skill_used = self.current_skill.id
                        self.current_skill_step = 1
        
        # Fallback to heuristic action selection
        if selected_action is None:
            selected_action = self._heuristic_action_selection(obs_state)
        
        # Safety check
        if self.enable_shield:
            is_safe, reason = self.shield.is_action_safe(selected_action, obs_state, self.action_history)
            if not is_safe:
                print(f"Shield blocked action {selected_action.verb}: {reason}")
                # Get safe alternatives
                safe_actions = self.shield.get_safe_action_suggestions(obs_state)
                if safe_actions:
                    safe_verb = random.choice(safe_actions)
                    selected_action = Action(verb=safe_verb, payload={})
                else:
                    selected_action = Action(verb="restart", payload={})
        
        # Record action
        self.action_history.append(selected_action)
        
        # Convert to environment format
        verb_idx = self.action_verbs.index(selected_action.verb)
        payload_str = json.dumps(selected_action.payload)
        
        result = {"verb": verb_idx, "payload": payload_str}
        
        if self.verbose:
            skill_info = f" (using skill: {skill_used})" if skill_used else ""
            print(f"Agent action: {selected_action.verb}{skill_info}")
        
        return result
    
    def end_episode(self, episode_reward: float, episode_success: bool, episode_data: Optional[Any] = None) -> None:
        """Called at the end of each episode for learning updates."""
        self.episode_count += 1
        
        # Update skill success rates if we used skills
        if self.enable_skills and episode_data:
            # This would typically be called by the environment or experiment runner
            # with episode trajectory data to update skill statistics
            pass
        
        # Periodically extract new skills
        if (self.enable_skills and 
            self.episode_count % self.skill_update_frequency == 0 and 
            self.episode_count > 0):
            self._update_skills()
        
        # Reset episode state
        self.current_skill = None
        self.current_skill_step = 0
        self.action_history = []
    
    def _parse_observation(self, observation: Dict[str, Any]) -> ObservationState:
        """Parse raw observation into structured format."""
        try:
            forms_data = json.loads(observation.get("forms", "{}"))
            flights_data = json.loads(observation.get("flights", "[]"))
            cart_data = json.loads(observation.get("cart", '{"items": [], "total": 0}'))
            payment_data = json.loads(observation.get("payment_state", "{}"))
            constraints_data = json.loads(observation.get("constraints", "{}"))
            messages_data = json.loads(observation.get("messages", "[]"))
            
            # Convert to proper models
            forms = SearchForm(**forms_data)
            flights = [Flight(**f) for f in flights_data]
            cart = Cart(**cart_data)
            payment_state = PaymentState(**payment_data)
            constraints = Constraints(**constraints_data)
            
            return ObservationState(
                view=observation.get("view", "search_form"),
                forms=forms,
                flights=flights,
                cart=cart,
                payment_state=payment_state,
                constraints=constraints,
                messages=messages_data,
                step_count=observation.get("step_count", 0)
            )
        except Exception as e:
            if self.verbose:
                print(f"Error parsing observation: {e}")
            # Return minimal state
            return ObservationState(
                view="error",
                constraints=Constraints(budget=1000)
            )
    
    def _heuristic_action_selection(self, obs_state: ObservationState) -> Action:
        """Fallback heuristic action selection."""
        view = obs_state.view
        
        if view == "search_form":
            # Search for flights
            return Action(
                verb="search_flights",
                payload={
                    "from": random.choice(["SFO", "LAX", "JFK"]),
                    "to": random.choice(["MAD", "LHR", "CDG"]),
                    "date": "2025-10-15"
                }
            )
        
        elif view == "search_results":
            if obs_state.flights:
                # Find flights that meet constraints
                valid_flights = []
                for flight in obs_state.flights:
                    if flight.price <= obs_state.constraints.budget:
                        if (obs_state.constraints.max_stops is None or 
                            flight.stops <= obs_state.constraints.max_stops):
                            if (obs_state.constraints.depart_after is None or
                                flight.depart_time >= obs_state.constraints.depart_after):
                                valid_flights.append(flight)
                
                if valid_flights:
                    # Select cheapest valid flight
                    best_flight = min(valid_flights, key=lambda f: f.price)
                    return Action(verb="add_to_cart", payload={"flight_id": best_flight.id})
                else:
                    # Apply filters to reduce results
                    filter_payload = {}
                    if obs_state.constraints.depart_after:
                        filter_payload["depart_after"] = obs_state.constraints.depart_after
                    if obs_state.constraints.max_stops is not None:
                        filter_payload["max_stops"] = obs_state.constraints.max_stops
                    
                    return Action(verb="filter_results", payload=filter_payload)
            else:
                return Action(verb="restart", payload={})
        
        elif view == "cart":
            if obs_state.cart.items:
                # Check if we should apply coupon
                if not obs_state.cart.coupon_applied and random.random() < 0.3:
                    return Action(verb="apply_coupon", payload={"code": "SAVE10"})
                else:
                    return Action(verb="proceed_to_payment", payload={})
            else:
                return Action(verb="restart", payload={})
        
        elif view == "payment":
            if not obs_state.payment_state.card_entered:
                return Action(verb="enter_card", payload={"card_token": "card_1234"})
            else:
                return Action(verb="confirm_payment", payload={})
        
        else:  # error, receipt, or unknown
            return Action(verb="restart", payload={})
    
    def _update_skills(self) -> None:
        """Update skills based on recent episodes."""
        if not self.enable_skills:
            return
        
        print(f"Updating skills at episode {self.episode_count}...")
        
        # Load recent episodes for skill extraction
        recent_episodes = self.logger.load_episodes(limit=200)
        
        if len(recent_episodes) >= 10:  # Need minimum episodes
            # Extract new skills
            new_skills = self.skill_extractor.extract_skills_from_episodes(recent_episodes)
            
            # Merge with existing skills (avoid duplicates)
            existing_skill_names = {skill.name for skill in self.skills}
            for new_skill in new_skills:
                if new_skill.name not in existing_skill_names:
                    self.skills.append(new_skill)
                    self.logger.save_skill(new_skill)
            
            print(f"Skills updated. Total skills: {len(self.skills)}")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's learning."""
        return {
            "total_skills": len(self.skills),
            "episode_count": self.episode_count,
            "current_skill": self.current_skill.id if self.current_skill else None,
            "current_skill_step": self.current_skill_step,
            "shield_stats": self.shield.get_shield_stats() if self.enable_shield else {},
            "top_skills": [
                {"id": skill.id, "name": skill.name, "confidence": skill.confidence}
                for skill in sorted(self.skills, key=lambda s: s.confidence, reverse=True)[:5]
            ]
        }