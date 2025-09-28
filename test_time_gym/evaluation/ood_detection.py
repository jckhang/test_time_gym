"""Out-of-distribution detection and safety mechanisms."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
from datetime import datetime, timedelta

from ..utils.models import ObservationState, Action, Episode


class OODDetector:
    """Detect out-of-distribution states and behaviors."""
    
    def __init__(self, contamination: float = 0.1, n_neighbors: int = 5):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.is_fitted = False
        self.baseline_distances = []
        self.threshold = None
    
    def fit(self, episodes: List[Episode]) -> None:
        """Fit the OOD detector on baseline episodes."""
        if not episodes:
            return
        
        # Extract features from all states in episodes
        features = []
        for episode in episodes:
            for step in episode.steps:
                feature_vector = self._extract_state_features(step.observation)
                features.append(feature_vector)
        
        if not features:
            return
        
        features = np.array(features)
        
        # Fit scaler and kNN
        features_scaled = self.scaler.fit_transform(features)
        self.knn.fit(features_scaled)
        
        # Calculate baseline distances
        distances, _ = self.knn.kneighbors(features_scaled)
        self.baseline_distances = np.mean(distances, axis=1)
        
        # Set threshold based on percentile
        self.threshold = np.percentile(self.baseline_distances, (1 - self.contamination) * 100)
        self.is_fitted = True
    
    def detect_ood_state(self, observation: ObservationState) -> Tuple[bool, float]:
        """
        Detect if current state is out-of-distribution.
        
        Returns:
            (is_ood, anomaly_score)
        """
        if not self.is_fitted:
            return False, 0.0
        
        feature_vector = self._extract_state_features(observation)
        feature_scaled = self.scaler.transform([feature_vector])
        
        distances, _ = self.knn.kneighbors(feature_scaled)
        anomaly_score = np.mean(distances[0])
        
        is_ood = anomaly_score > self.threshold
        
        return is_ood, anomaly_score
    
    def detect_policy_drift(
        self, 
        current_action_probs: Dict[str, float],
        baseline_action_probs: Dict[str, float],
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Detect policy drift using KL divergence.
        
        Args:
            current_action_probs: Current action probability distribution
            baseline_action_probs: Baseline action probability distribution
            threshold: KL divergence threshold for drift detection
            
        Returns:
            (is_drift, kl_divergence)
        """
        # Ensure same action space
        all_actions = set(current_action_probs.keys()) | set(baseline_action_probs.keys())
        
        current_probs = []
        baseline_probs = []
        
        for action in sorted(all_actions):
            current_probs.append(current_action_probs.get(action, 1e-8))
            baseline_probs.append(baseline_action_probs.get(action, 1e-8))
        
        # Normalize to probabilities
        current_probs = np.array(current_probs)
        baseline_probs = np.array(baseline_probs)
        
        current_probs = current_probs / np.sum(current_probs)
        baseline_probs = baseline_probs / np.sum(baseline_probs)
        
        # Calculate KL divergence
        kl_div = stats.entropy(current_probs, baseline_probs)
        
        is_drift = kl_div > threshold
        
        return is_drift, kl_div
    
    def _extract_state_features(self, observation: ObservationState) -> np.ndarray:
        """Extract numerical features from observation state."""
        features = np.zeros(15)  # 15-dimensional feature vector
        
        # View encoding
        view_mapping = {
            "search_form": 0, "search_results": 1, "cart": 2,
            "payment": 3, "receipt": 4, "error": 5
        }
        if observation.view in view_mapping:
            features[view_mapping[observation.view]] = 1.0
        
        # Flight statistics
        if observation.flights:
            prices = [f.price for f in observation.flights]
            features[6] = len(observation.flights) / 20.0  # Normalized count
            features[7] = np.mean(prices) / 1000.0  # Normalized avg price
            features[8] = np.std(prices) / 200.0 if len(prices) > 1 else 0  # Price variance
            features[9] = np.mean([f.stops for f in observation.flights]) / 3.0  # Avg stops
        
        # Cart features
        features[10] = len(observation.cart.items) / 5.0  # Normalized item count
        features[11] = observation.cart.total / 2000.0  # Normalized total
        features[12] = observation.cart.discount / 200.0  # Normalized discount
        
        # Constraint features
        features[13] = observation.constraints.budget / 2000.0  # Normalized budget
        features[14] = observation.step_count / 50.0  # Normalized step count
        
        return features


class ActionShield:
    """Safety shield to prevent dangerous or invalid actions."""
    
    def __init__(self, max_budget: float = 2000, max_attempts: int = 10):
        self.max_budget = max_budget
        self.max_attempts = max_attempts
        self.action_counts = {}
        self.session_start = datetime.now()
    
    def is_action_safe(
        self, 
        action: Action, 
        observation: ObservationState,
        episode_history: List[Action]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an action is safe to execute.
        
        Returns:
            (is_safe, reason_if_unsafe)
        """
        verb = action.verb
        payload = action.payload
        
        # Count action frequency
        if verb not in self.action_counts:
            self.action_counts[verb] = 0
        self.action_counts[verb] += 1
        
        # Check for excessive action repetition
        if self.action_counts[verb] > self.max_attempts:
            return False, f"Action {verb} attempted too many times ({self.action_counts[verb]})"
        
        # Budget safety checks
        if verb == "confirm_payment":
            if observation.cart.total > self.max_budget:
                return False, f"Payment amount ${observation.cart.total} exceeds safety limit ${self.max_budget}"
            
            if observation.cart.total > observation.constraints.budget * 2:
                return False, f"Payment amount significantly exceeds stated budget"
        
        # Payment attempt limits
        if verb == "confirm_payment" and observation.payment_state.attempts >= 5:
            return False, "Too many payment attempts"
        
        # Cart manipulation safety
        if verb == "add_to_cart":
            flight_id = payload.get("flight_id", "")
            if not flight_id:
                return False, "No flight ID specified for cart addition"
            
            # Check if flight exists in current results
            if not any(f.id == flight_id for f in observation.flights):
                return False, "Attempting to add non-existent flight to cart"
        
        # Search parameter validation
        if verb == "search_flights":
            from_airport = payload.get("from", "")
            to_airport = payload.get("to", "")
            
            if from_airport == to_airport:
                return False, "Source and destination airports cannot be the same"
            
            # Airport code validation (basic)
            valid_airports = ["SFO", "LAX", "JFK", "MAD", "LHR", "CDG", "FRA", "NRT"]
            if from_airport not in valid_airports or to_airport not in valid_airports:
                return False, f"Invalid airport codes: {from_airport}, {to_airport}"
        
        # Prevent infinite loops
        recent_actions = episode_history[-5:] if len(episode_history) >= 5 else episode_history
        if len(recent_actions) >= 3 and all(a.verb == verb for a in recent_actions):
            return False, f"Potential infinite loop detected with action {verb}"
        
        return True, None
    
    def get_safe_action_suggestions(self, observation: ObservationState) -> List[str]:
        """Suggest safe actions for the current state."""
        suggestions = []
        
        view = observation.view
        
        if view == "search_form":
            suggestions = ["search_flights"]
        elif view == "search_results":
            if observation.flights:
                suggestions = ["filter_results", "add_to_cart"]
            else:
                suggestions = ["search_flights", "restart"]
        elif view == "cart":
            if observation.cart.items:
                suggestions = ["proceed_to_payment", "apply_coupon"]
            else:
                suggestions = ["search_flights", "restart"]
        elif view == "payment":
            if not observation.payment_state.card_entered:
                suggestions = ["enter_card"]
            else:
                suggestions = ["confirm_payment"]
        elif view == "error":
            suggestions = ["restart", "search_flights"]
        elif view == "receipt":
            suggestions = []  # Episode should be done
        
        return suggestions
    
    def reset_session(self) -> None:
        """Reset action counts for a new session."""
        self.action_counts = {}
        self.session_start = datetime.now()
    
    def get_shield_stats(self) -> Dict[str, Any]:
        """Get statistics about shield interventions."""
        total_actions = sum(self.action_counts.values())
        
        return {
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "total_actions": total_actions,
            "action_distribution": self.action_counts.copy(),
            "max_attempts_reached": {
                action: count for action, count in self.action_counts.items() 
                if count >= self.max_attempts
            }
        }