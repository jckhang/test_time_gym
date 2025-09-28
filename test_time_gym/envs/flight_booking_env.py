"""Flight booking environment for Test-Time Gym."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
import uuid

from ..utils.models import (
    ObservationState, Action, Flight, Cart, CartItem, PaymentState, 
    Constraints, SearchForm, TrajectoryStep, Episode
)


class FlightBookingEnv(gym.Env):
    """
    A flight booking simulation environment for LLM agent test-time learning.
    
    This environment simulates a flight booking website where agents need to:
    1. Search for flights
    2. Filter results based on constraints
    3. Add flights to cart
    4. Complete payment process
    5. Handle various random perturbations (3DS, payment failures, etc.)
    """
    
    metadata = {"render_modes": ["human", "json"]}
    
    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = 50,
        enable_3ds: bool = True,
        enable_payment_failures: bool = True,
        enable_sold_out: bool = True,
        price_noise_std: float = 0.1,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.max_steps = max_steps
        self.enable_3ds = enable_3ds
        self.enable_payment_failures = enable_payment_failures
        self.enable_sold_out = enable_sold_out
        self.price_noise_std = price_noise_std
        self.render_mode = render_mode
        
        # Initialize random state
        if seed is not None:
            self.seed(seed)
        
        # Action space - discrete actions with payload
        self.action_space = spaces.Dict({
            "verb": spaces.Discrete(9),  # 9 different action types
            "payload": spaces.Text(1000)  # JSON string payload
        })
        
        # Observation space - structured JSON
        self.observation_space = spaces.Dict({
            "view": spaces.Text(20),
            "forms": spaces.Text(200),
            "flights": spaces.Text(5000),
            "cart": spaces.Text(1000),
            "payment_state": spaces.Text(200),
            "constraints": spaces.Text(300),
            "messages": spaces.Text(1000),
            "step_count": spaces.Box(low=0, high=max_steps, shape=(), dtype=np.int32)
        })
        
        # Action mapping
        self.action_verbs = [
            "search_flights",
            "filter_results", 
            "select_flight",
            "add_to_cart",
            "proceed_to_payment",
            "enter_card",
            "confirm_payment",
            "apply_coupon",
            "restart"
        ]
        
        # Initialize state
        self.state = None
        self.episode = None
        self.current_step = 0
        self.flight_database = self._create_flight_database()
        
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed for reproducibility."""
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        self._np_random = np.random.RandomState(seed)
        random.seed(seed)
        return [seed]
    
    def _create_flight_database(self) -> List[Flight]:
        """Create a database of available flights."""
        airports = ["SFO", "LAX", "JFK", "MAD", "LHR", "CDG", "FRA", "NRT"]
        airlines = ["AA", "UA", "DL", "IB", "BA", "AF", "LH", "JL"]
        
        flights = []
        for i in range(200):  # Generate 200 flights
            from_airport = random.choice(airports)
            to_airport = random.choice([a for a in airports if a != from_airport])
            
            # Generate realistic flight times
            depart_hour = random.randint(6, 23)
            depart_min = random.choice([0, 15, 30, 45])
            duration = random.randint(180, 900)  # 3-15 hours
            
            arrive_hour = (depart_hour + duration // 60) % 24
            arrive_min = (depart_min + duration % 60) % 60
            next_day = (depart_hour + duration // 60) >= 24
            
            arrive_time = f"{arrive_hour:02d}:{arrive_min:02d}"
            if next_day:
                arrive_time += "+1"
            
            flight = Flight(
                id=f"{random.choice(airlines)}{random.randint(1000, 9999)}",
                price=random.randint(300, 1500),
                depart_time=f"{depart_hour:02d}:{depart_min:02d}",
                arrive_time=arrive_time,
                stops=random.choice([0, 1, 2]),
                airline=random.choice(airlines),
                departure_airport=from_airport,
                arrival_airport=to_airport,
                duration_minutes=duration
            )
            flights.append(flight)
        
        return flights
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed(seed)
        
        # Generate random task constraints
        budget = self._np_random.randint(400, 1200)
        depart_after = f"{self._np_random.randint(8, 18):02d}:00" if self._np_random.random() < 0.7 else None
        max_stops = self._np_random.choice([None, 0, 1, 2]) 
        
        constraints = Constraints(
            budget=budget,
            depart_after=depart_after,
            max_stops=max_stops
        )
        
        # Initialize state
        self.state = ObservationState(
            view="search_form",
            constraints=constraints,
            messages=["Welcome! Please search for flights."]
        )
        
        # Initialize episode tracking
        self.episode = Episode(env_seed=seed if seed else 0)
        self.current_step = 0
        
        obs = self._get_observation()
        info = {"constraints": constraints.dict(), "episode_id": self.episode.episode_id}
        
        return obs, info
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Parse action
        verb_idx = action.get("verb", 0)
        payload_str = action.get("payload", "{}")
        
        if verb_idx >= len(self.action_verbs):
            return self._invalid_action("Invalid action verb")
        
        verb = self.action_verbs[verb_idx]
        
        try:
            payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
        except json.JSONDecodeError:
            return self._invalid_action("Invalid JSON payload")
        
        action_obj = Action(verb=verb, payload=payload)
        
        # Execute action
        reward, done, truncated, info = self._execute_action(action_obj)
        
        # Apply time cost
        reward -= 0.01
        
        # Check truncation
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Update state
        self.state.step_count = self.current_step
        
        # Record trajectory step
        step = TrajectoryStep(
            step_id=self.current_step,
            observation=self.state.copy(deep=True),
            action=action_obj,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info
        )
        self.episode.steps.append(step)
        self.episode.total_reward += reward
        
        obs = self._get_observation()
        
        return obs, reward, done, truncated, info
    
    def _execute_action(self, action: Action) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Execute the given action and return reward, done, truncated, info."""
        verb = action.verb
        payload = action.payload
        
        if verb == "search_flights":
            return self._search_flights(payload)
        elif verb == "filter_results":
            return self._filter_results(payload)
        elif verb == "select_flight":
            return self._select_flight(payload)
        elif verb == "add_to_cart":
            return self._add_to_cart(payload)
        elif verb == "proceed_to_payment":
            return self._proceed_to_payment()
        elif verb == "enter_card":
            return self._enter_card(payload)
        elif verb == "confirm_payment":
            return self._confirm_payment()
        elif verb == "apply_coupon":
            return self._apply_coupon(payload)
        elif verb == "restart":
            return self._restart()
        else:
            return self._invalid_action(f"Unknown action: {verb}")
    
    def _search_flights(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Search for flights."""
        from_airport = payload.get("from", "")
        to_airport = payload.get("to", "")
        date = payload.get("date", "")
        
        if not all([from_airport, to_airport, date]):
            self.state.messages = ["Error: Please provide from, to, and date"]
            return -0.05, False, False, {"error": "Missing required fields"}
        
        # Update form data
        self.state.forms.from_airport = from_airport
        self.state.forms.to_airport = to_airport
        self.state.forms.date = date
        
        # Find matching flights
        matching_flights = [
            f for f in self.flight_database
            if f.departure_airport == from_airport and f.arrival_airport == to_airport
        ]
        
        if not matching_flights:
            self.state.view = "error"
            self.state.messages = [f"No flights found from {from_airport} to {to_airport}"]
            return -0.05, False, False, {"error": "No flights found"}
        
        # Apply price noise
        for flight in matching_flights:
            noise = self._np_random.normal(0, self.price_noise_std * flight.price)
            flight.price = max(50, flight.price + noise)  # Minimum price of 50
        
        # Randomly remove some flights (sold out)
        if self.enable_sold_out:
            num_to_remove = self._np_random.binomial(len(matching_flights), 0.1)
            if num_to_remove > 0:
                to_remove = self._np_random.choice(len(matching_flights), num_to_remove, replace=False)
                matching_flights = [f for i, f in enumerate(matching_flights) if i not in to_remove]
        
        # Random sorting
        random.shuffle(matching_flights)
        
        self.state.flights = matching_flights[:20]  # Show max 20 results
        self.state.view = "search_results"
        self.state.messages = [f"Found {len(self.state.flights)} flights"]
        
        return 0.02, False, False, {"flights_found": len(self.state.flights)}
    
    def _filter_results(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Filter flight results."""
        if self.state.view != "search_results":
            return -0.05, False, False, {"error": "No search results to filter"}
        
        filtered_flights = self.state.flights.copy()
        
        # Apply filters
        if "depart_after" in payload:
            depart_after = payload["depart_after"]
            filtered_flights = [f for f in filtered_flights if f.depart_time >= depart_after]
        
        if "max_stops" in payload:
            max_stops = payload["max_stops"]
            filtered_flights = [f for f in filtered_flights if f.stops <= max_stops]
        
        if "airline" in payload:
            airline = payload["airline"]
            filtered_flights = [f for f in filtered_flights if f.airline == airline]
        
        if "max_price" in payload:
            max_price = payload["max_price"]
            filtered_flights = [f for f in filtered_flights if f.price <= max_price]
        
        self.state.flights = filtered_flights
        self.state.messages = [f"Filtered to {len(filtered_flights)} flights"]
        
        return 0.01, False, False, {"flights_remaining": len(filtered_flights)}
    
    def _select_flight(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Select a specific flight."""
        flight_id = payload.get("flight_id", "")
        
        if not flight_id:
            return -0.05, False, False, {"error": "No flight_id provided"}
        
        selected_flight = None
        for flight in self.state.flights:
            if flight.id == flight_id:
                selected_flight = flight
                break
        
        if not selected_flight:
            return -0.05, False, False, {"error": "Flight not found"}
        
        self.state.messages = [f"Selected flight {flight_id} - ${selected_flight.price}"]
        return 0.01, False, False, {"selected_flight": selected_flight.dict()}
    
    def _add_to_cart(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Add flight to cart."""
        flight_id = payload.get("flight_id", "")
        
        if not flight_id:
            return -0.05, False, False, {"error": "No flight_id provided"}
        
        # Find the flight
        selected_flight = None
        for flight in self.state.flights:
            if flight.id == flight_id:
                selected_flight = flight
                break
        
        if not selected_flight:
            return -0.05, False, False, {"error": "Flight not found"}
        
        # Check if already in cart
        for item in self.state.cart.items:
            if item.flight_id == flight_id:
                return -0.02, False, False, {"error": "Flight already in cart"}
        
        # Add to cart
        cart_item = CartItem(flight_id=flight_id, price=selected_flight.price)
        self.state.cart.items.append(cart_item)
        self.state.cart.total = sum(item.price * item.quantity for item in self.state.cart.items)
        
        self.state.view = "cart"
        self.state.messages = [f"Added {flight_id} to cart. Total: ${self.state.cart.total:.2f}"]
        
        return 0.05, False, False, {"cart_total": self.state.cart.total}
    
    def _proceed_to_payment(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Proceed to payment page."""
        if not self.state.cart.items:
            return -0.05, False, False, {"error": "Cart is empty"}
        
        # Check budget constraint
        if self.state.cart.total > self.state.constraints.budget:
            violation_reward = -0.3
            self.state.messages = [f"Cart total ${self.state.cart.total:.2f} exceeds budget ${self.state.constraints.budget:.2f}"]
            return violation_reward, False, False, {"constraint_violation": "budget_exceeded"}
        
        # Randomly require 3DS
        if self.enable_3ds and self._np_random.random() < 0.3:
            self.state.payment_state.needs_3ds = True
        
        self.state.view = "payment"
        self.state.messages = ["Enter payment details to complete booking"]
        
        return 0.05, False, False, {"payment_ready": True}
    
    def _enter_card(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Enter credit card information."""
        if self.state.view != "payment":
            return -0.05, False, False, {"error": "Not on payment page"}
        
        card_token = payload.get("card_token", "")
        if not card_token:
            return -0.05, False, False, {"error": "No card token provided"}
        
        self.state.payment_state.card_entered = True
        self.state.messages = ["Card details entered. Ready to confirm payment."]
        
        return 0.02, False, False, {"card_ready": True}
    
    def _confirm_payment(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Confirm payment and complete booking."""
        if self.state.view != "payment":
            return -0.05, False, False, {"error": "Not on payment page"}
        
        if not self.state.payment_state.card_entered:
            return -0.05, False, False, {"error": "Card not entered"}
        
        self.state.payment_state.attempts += 1
        
        # Random payment failure
        failure_prob = 0.1
        if self.state.payment_state.needs_3ds:
            failure_prob = 0.2
        
        if self.enable_payment_failures and self._np_random.random() < failure_prob:
            self.state.payment_state.payment_failed = True
            self.state.payment_state.failure_reason = "Payment declined"
            self.state.messages = ["Payment failed. Please try again."]
            
            if self.state.payment_state.attempts >= self.state.payment_state.max_attempts:
                return -0.1, False, True, {"payment_failed": "max_attempts_reached"}
            
            return -0.05, False, False, {"payment_failed": "retry_possible"}
        
        # Payment successful
        self.state.payment_state.payment_confirmed = True
        self.state.view = "receipt"
        self.state.messages = ["Payment successful! Booking confirmed."]
        
        # Calculate final reward
        final_reward = 1.0
        
        # Check all constraints
        constraint_violations = 0
        regret = 0.0
        
        # Budget constraint
        if self.state.cart.total > self.state.constraints.budget:
            constraint_violations += 1
            final_reward -= 0.3
        
        # Time constraints
        for item in self.state.cart.items:
            flight = next((f for f in self.state.flights if f.id == item.flight_id), None)
            if flight:
                if self.state.constraints.depart_after and flight.depart_time < self.state.constraints.depart_after:
                    constraint_violations += 1
                    final_reward -= 0.2
                
                if self.state.constraints.max_stops is not None and flight.stops > self.state.constraints.max_stops:
                    constraint_violations += 1
                    final_reward -= 0.2
        
        # Calculate regret (difference from optimal price)
        if self.state.flights:
            valid_flights = self._get_constraint_valid_flights()
            if valid_flights:
                optimal_price = min(f.price for f in valid_flights)
                regret = self.state.cart.total - optimal_price
        
        # Update episode stats
        self.episode.success = final_reward > 0
        self.episode.constraint_violations = constraint_violations
        self.episode.final_regret = regret
        
        info = {
            "success": True,
            "constraint_violations": constraint_violations,
            "regret": regret,
            "final_price": self.state.cart.total
        }
        
        return self._get_observation(), final_reward, True, False, info
    
    def _apply_coupon(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Apply a coupon code."""
        if self.state.view not in ["cart", "payment"]:
            return -0.05, False, False, {"error": "Cannot apply coupon in current view"}
        
        coupon_code = payload.get("code", "")
        if not coupon_code:
            return -0.05, False, False, {"error": "No coupon code provided"}
        
        # Simulate coupon validation
        valid_coupons = ["SAVE10", "WELCOME", "STUDENT", "SUMMER"]
        
        if coupon_code in valid_coupons and not self.state.cart.coupon_applied:
            discount_rate = 0.1  # 10% discount
            discount = self.state.cart.total * discount_rate
            self.state.cart.discount = discount
            self.state.cart.total -= discount
            self.state.cart.coupon_applied = coupon_code
            
            self.state.messages = [f"Coupon {coupon_code} applied! Saved ${discount:.2f}"]
            return 0.03, False, False, {"coupon_applied": coupon_code, "discount": discount}
        else:
            self.state.messages = ["Invalid coupon code"]
            return -0.02, False, False, {"error": "Invalid coupon"}
    
    def _restart(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Restart the booking process."""
        self.state.view = "search_form"
        self.state.forms = SearchForm()
        self.state.flights = []
        self.state.cart = Cart()
        self.state.payment_state = PaymentState()
        self.state.messages = ["Restarted booking process"]
        
        return -0.05, False, False, {"restarted": True}
    
    def _invalid_action(self, message: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Handle invalid actions."""
        self.state.messages = [f"Invalid action: {message}"]
        obs = self._get_observation()
        return obs, -0.05, False, False, {"error": message}
    
    def _get_constraint_valid_flights(self) -> List[Flight]:
        """Get flights that satisfy all constraints."""
        valid_flights = []
        
        for flight in self.state.flights:
            valid = True
            
            # Check price constraint
            if flight.price > self.state.constraints.budget:
                valid = False
            
            # Check time constraints
            if self.state.constraints.depart_after and flight.depart_time < self.state.constraints.depart_after:
                valid = False
            
            # Check stops constraint
            if self.state.constraints.max_stops is not None and flight.stops > self.state.constraints.max_stops:
                valid = False
            
            if valid:
                valid_flights.append(flight)
        
        return valid_flights
    
    def _get_observation(self) -> Dict[str, Any]:
        """Convert state to observation format."""
        return {
            "view": self.state.view,
            "forms": self.state.forms.json(),
            "flights": json.dumps([f.dict() for f in self.state.flights]),
            "cart": self.state.cart.json(),
            "payment_state": self.state.payment_state.json(),
            "constraints": self.state.constraints.json(),
            "messages": json.dumps(self.state.messages),
            "step_count": self.state.step_count
        }
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment state."""
        if mode == "json":
            return self.state.json(indent=2)
        elif mode == "human":
            print(f"\n=== Step {self.current_step} - View: {self.state.view} ===")
            print(f"Constraints: Budget=${self.state.constraints.budget}, Max Stops={self.state.constraints.max_stops}")
            
            if self.state.view == "search_results":
                print(f"Flights ({len(self.state.flights)}):")
                for flight in self.state.flights[:5]:  # Show first 5
                    print(f"  {flight.id}: ${flight.price} - {flight.depart_time} ({flight.stops} stops)")
            
            if self.state.cart.items:
                print(f"Cart: ${self.state.cart.total:.2f}")
                for item in self.state.cart.items:
                    print(f"  {item.flight_id}: ${item.price}")
            
            if self.state.messages:
                print(f"Messages: {self.state.messages[-1]}")
            print()
        
        return None
    
    def close(self):
        """Clean up environment resources."""
        pass