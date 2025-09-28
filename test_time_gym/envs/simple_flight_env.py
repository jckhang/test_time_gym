"""Simplified flight booking environment without external dependencies."""

import random
import json
from typing import Dict, List, Any, Tuple, Optional

from ..utils.simple_models import (
    ObservationState, Flight, Cart, PaymentState, Constraints
)


class SimpleFlightBookingEnv:
    """
    Simplified flight booking environment for Test-Time Gym.
    
    This version minimizes external dependencies while maintaining core functionality.
    """
    
    def __init__(self, seed: Optional[int] = None, max_steps: int = 50):
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None
        self.episode_reward = 0.0
        
        if seed is not None:
            random.seed(seed)
        
        # Action mapping
        self.action_verbs = [
            "search_flights",     # 0
            "filter_results",     # 1 
            "select_flight",      # 2
            "add_to_cart",        # 3
            "proceed_to_payment", # 4
            "enter_card",         # 5
            "confirm_payment",    # 6
            "apply_coupon",       # 7
            "restart"             # 8
        ]
        
        # Create flight database
        self.flight_database = self._create_flights()
    
    def _create_flights(self) -> List[Flight]:
        """Create a database of flights."""
        airports = ["SFO", "LAX", "JFK", "MAD", "LHR", "CDG", "FRA"]
        airlines = ["AA", "UA", "DL", "IB", "BA", "AF", "LH"]
        
        flights = []
        for i in range(100):
            from_airport = random.choice(airports)
            to_airport = random.choice([a for a in airports if a != from_airport])
            
            flight = Flight(
                id=f"{random.choice(airlines)}{random.randint(1000, 9999)}",
                price=random.randint(300, 1500),
                depart_time=f"{random.randint(6, 23):02d}:{random.choice([0, 30]):02d}",
                arrive_time=f"{random.randint(6, 23):02d}:{random.choice([0, 30]):02d}",
                stops=random.choice([0, 1, 2]),
                airline=random.choice(airlines),
                departure_airport=from_airport,
                arrival_airport=to_airport
            )
            flights.append(flight)
        
        return flights
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to initial state."""
        # Generate random constraints
        budget = random.randint(400, 1200)
        depart_after = f"{random.randint(8, 18):02d}:00" if random.random() < 0.7 else None
        max_stops = random.choice([None, 0, 1, 2])
        
        constraints = Constraints(
            budget=budget,
            depart_after=depart_after,
            max_stops=max_stops
        )
        
        self.state = ObservationState(view="search_form", constraints=constraints)
        self.current_step = 0
        self.episode_reward = 0.0
        
        obs = self._get_observation()
        info = {"constraints": constraints.to_dict()}
        
        return obs, info
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
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
        
        # Execute action
        reward, done, truncated, info = self._execute_action(verb, payload)
        
        # Apply time cost
        reward -= 0.01
        self.episode_reward += reward
        
        # Check truncation
        if self.current_step >= self.max_steps:
            truncated = True
        
        self.state.step_count = self.current_step
        obs = self._get_observation()
        
        return obs, reward, done, truncated, info
    
    def _execute_action(self, verb: str, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Execute the action and return reward, done, truncated, info."""
        if verb == "search_flights":
            return self._search_flights(payload)
        elif verb == "filter_results":
            return self._filter_results(payload)
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
            return -0.05, False, False, {"error": f"Unknown action: {verb}"}
    
    def _search_flights(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Search for flights."""
        from_airport = payload.get("from", "")
        to_airport = payload.get("to", "")
        date = payload.get("date", "")
        
        if not all([from_airport, to_airport, date]):
            self.state.messages = ["Error: Please provide from, to, and date"]
            return -0.05, False, False, {"error": "Missing fields"}
        
        # Update form
        self.state.forms["from_airport"] = from_airport
        self.state.forms["to_airport"] = to_airport
        self.state.forms["date"] = date
        
        # Find matching flights
        matching = [f for f in self.flight_database 
                   if f.departure_airport == from_airport and f.arrival_airport == to_airport]
        
        if not matching:
            self.state.view = "error"
            self.state.messages = [f"No flights from {from_airport} to {to_airport}"]
            return -0.05, False, False, {"error": "No flights"}
        
        # Add price noise and random soldout
        available_flights = []
        for flight in matching[:20]:  # Max 20 results
            if random.random() > 0.1:  # 10% chance of sold out
                # Add price noise
                noise = random.uniform(-50, 100)
                flight.price = max(100, flight.price + noise)
                available_flights.append(flight)
        
        self.state.flights = available_flights
        self.state.view = "search_results"
        self.state.messages = [f"Found {len(available_flights)} flights"]
        
        return 0.02, False, False, {"flights_found": len(available_flights)}
    
    def _filter_results(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Filter search results."""
        if self.state.view != "search_results":
            return -0.05, False, False, {"error": "No results to filter"}
        
        filtered = self.state.flights.copy()
        
        if "depart_after" in payload:
            depart_after = payload["depart_after"]
            filtered = [f for f in filtered if f.depart_time >= depart_after]
        
        if "max_stops" in payload:
            max_stops = payload["max_stops"]
            filtered = [f for f in filtered if f.stops <= max_stops]
        
        if "max_price" in payload:
            max_price = payload["max_price"]
            filtered = [f for f in filtered if f.price <= max_price]
        
        self.state.flights = filtered
        self.state.messages = [f"Filtered to {len(filtered)} flights"]
        
        return 0.01, False, False, {"flights_remaining": len(filtered)}
    
    def _add_to_cart(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Add flight to cart."""
        flight_id = payload.get("flight_id", "")
        
        if not flight_id:
            return -0.05, False, False, {"error": "No flight_id"}
        
        # Find flight
        selected_flight = None
        for flight in self.state.flights:
            if flight.id == flight_id:
                selected_flight = flight
                break
        
        if not selected_flight:
            return -0.05, False, False, {"error": "Flight not found"}
        
        # Check if already in cart
        for item in self.state.cart.items:
            if item["flight_id"] == flight_id:
                return -0.02, False, False, {"error": "Already in cart"}
        
        # Add to cart
        self.state.cart.add_item(flight_id, selected_flight.price)
        self.state.view = "cart"
        self.state.messages = [f"Added {flight_id} to cart. Total: ${self.state.cart.total:.2f}"]
        
        return 0.05, False, False, {"cart_total": self.state.cart.total}
    
    def _proceed_to_payment(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Proceed to payment."""
        if not self.state.cart.items:
            return -0.05, False, False, {"error": "Cart empty"}
        
        # Check budget
        if self.state.cart.total > self.state.constraints.budget:
            self.state.messages = [f"Total ${self.state.cart.total:.2f} exceeds budget ${self.state.constraints.budget:.2f}"]
            return -0.3, False, False, {"constraint_violation": "budget"}
        
        # Random 3DS requirement
        if random.random() < 0.3:
            self.state.payment_state.needs_3ds = True
        
        self.state.view = "payment"
        self.state.messages = ["Enter payment details"]
        
        return 0.05, False, False, {"payment_ready": True}
    
    def _enter_card(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Enter card details."""
        if self.state.view != "payment":
            return -0.05, False, False, {"error": "Not on payment page"}
        
        card_token = payload.get("card_token", "")
        if not card_token:
            return -0.05, False, False, {"error": "No card token"}
        
        self.state.payment_state.card_entered = True
        self.state.messages = ["Card entered. Ready to confirm."]
        
        return 0.02, False, False, {"card_ready": True}
    
    def _confirm_payment(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Confirm payment."""
        if not self.state.payment_state.card_entered:
            return -0.05, False, False, {"error": "Card not entered"}
        
        self.state.payment_state.attempts += 1
        
        # Random payment failure
        failure_prob = 0.15 if self.state.payment_state.needs_3ds else 0.1
        
        if random.random() < failure_prob:
            self.state.payment_state.payment_failed = True
            self.state.messages = ["Payment failed. Try again."]
            
            if self.state.payment_state.attempts >= 3:
                return -0.1, False, True, {"payment_failed": "max_attempts"}
            
            return -0.05, False, False, {"payment_failed": "retry"}
        
        # Success!
        self.state.payment_state.payment_confirmed = True
        self.state.view = "receipt"
        self.state.messages = ["Payment successful! Booking confirmed."]
        
        # Calculate final reward
        final_reward = 1.0
        constraint_violations = 0
        
        # Check constraints
        if self.state.cart.total > self.state.constraints.budget:
            constraint_violations += 1
            final_reward -= 0.3
        
        # Check flight constraints
        for item in self.state.cart.items:
            flight = next((f for f in self.state.flights if f.id == item["flight_id"]), None)
            if flight:
                if (self.state.constraints.depart_after and 
                    flight.depart_time < self.state.constraints.depart_after):
                    constraint_violations += 1
                    final_reward -= 0.2
                
                if (self.state.constraints.max_stops is not None and 
                    flight.stops > self.state.constraints.max_stops):
                    constraint_violations += 1
                    final_reward -= 0.2
        
        info = {
            "success": final_reward > 0,
            "constraint_violations": constraint_violations,
            "final_price": self.state.cart.total
        }
        
        return final_reward, True, False, info
    
    def _apply_coupon(self, payload: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Apply coupon code."""
        if self.state.view not in ["cart", "payment"]:
            return -0.05, False, False, {"error": "Cannot apply coupon"}
        
        code = payload.get("code", "")
        valid_coupons = ["SAVE10", "WELCOME", "STUDENT"]
        
        if code in valid_coupons and not self.state.cart.coupon_applied:
            discount = self.state.cart.total * 0.1
            self.state.cart.total -= discount
            self.state.cart.discount = discount
            self.state.cart.coupon_applied = code
            self.state.messages = [f"Coupon applied! Saved ${discount:.2f}"]
            return 0.03, False, False, {"discount": discount}
        else:
            self.state.messages = ["Invalid coupon"]
            return -0.02, False, False, {"error": "Invalid coupon"}
    
    def _restart(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """Restart booking process."""
        self.state.view = "search_form"
        self.state.forms = {"from_airport": "", "to_airport": "", "date": ""}
        self.state.flights = []
        self.state.cart = Cart()
        self.state.payment_state = PaymentState()
        self.state.messages = ["Restarted"]
        return -0.05, False, False, {"restarted": True}
    
    def _invalid_action(self, message: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Handle invalid action."""
        self.state.messages = [f"Invalid: {message}"]
        obs = self._get_observation()
        return obs, -0.05, False, False, {"error": message}
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        return {
            "view": self.state.view,
            "forms": json.dumps(self.state.forms),
            "flights": json.dumps([f.to_dict() for f in self.state.flights]),
            "cart": json.dumps(self.state.cart.to_dict()),
            "payment_state": json.dumps(self.state.payment_state.to_dict()),
            "constraints": json.dumps(self.state.constraints.to_dict()),
            "messages": json.dumps(self.state.messages),
            "step_count": self.state.step_count
        }
    
    def render(self, mode: str = "human") -> None:
        """Render environment state."""
        if mode == "human":
            print(f"\n--- Step {self.current_step} - View: {self.state.view} ---")
            print(f"Budget: ${self.state.constraints.budget}")
            
            if self.state.view == "search_results":
                print(f"Flights found: {len(self.state.flights)}")
                for i, flight in enumerate(self.state.flights[:3]):
                    print(f"  {i+1}. {flight.id}: ${flight.price} - {flight.depart_time} ({flight.stops} stops)")
            
            if self.state.cart.items:
                print(f"Cart total: ${self.state.cart.total:.2f}")
            
            if self.state.messages:
                print(f"Message: {self.state.messages[-1]}")
            print()


class SimpleDummyAgent:
    """Simplified dummy agent."""
    
    def __init__(self, seed: Optional[int] = None, verbose: bool = False):
        self.verbose = verbose
        if seed:
            random.seed(seed)
    
    def select_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Select action based on current view."""
        view = observation.get("view", "")
        
        try:
            flights = json.loads(observation.get("flights", "[]"))
            cart = json.loads(observation.get("cart", '{"items": [], "total": 0}'))
            payment_state = json.loads(observation.get("payment_state", "{}"))
            constraints = json.loads(observation.get("constraints", "{}"))
        except:
            return {"verb": 8, "payload": "{}"}  # restart on parse error
        
        if self.verbose:
            print(f"Agent in view: {view}")
        
        if view == "search_form":
            # Search for flights
            payload = {
                "from": random.choice(["SFO", "LAX", "JFK"]),
                "to": random.choice(["MAD", "LHR", "CDG"]),
                "date": "2025-10-15"
            }
            return {"verb": 0, "payload": json.dumps(payload)}
        
        elif view == "search_results":
            if flights:
                # Find cheapest flight within budget
                budget = constraints.get("budget", 1000)
                valid_flights = [f for f in flights if f["price"] <= budget]
                
                if valid_flights:
                    cheapest = min(valid_flights, key=lambda f: f["price"])
                    return {"verb": 3, "payload": json.dumps({"flight_id": cheapest["id"]})}
                else:
                    # Filter by budget
                    return {"verb": 1, "payload": json.dumps({"max_price": budget})}
            else:
                return {"verb": 8, "payload": "{}"}  # restart
        
        elif view == "cart":
            if cart.get("items"):
                # Apply coupon sometimes
                if random.random() < 0.3 and not cart.get("coupon_applied"):
                    return {"verb": 7, "payload": json.dumps({"code": "SAVE10"})}
                else:
                    return {"verb": 4, "payload": "{}"}  # proceed to payment
            else:
                return {"verb": 8, "payload": "{}"}
        
        elif view == "payment":
            if not payment_state.get("card_entered", False):
                return {"verb": 5, "payload": json.dumps({"card_token": "card123"})}
            else:
                return {"verb": 6, "payload": "{}"}  # confirm payment
        
        else:  # error, receipt, unknown
            return {"verb": 8, "payload": "{}"}  # restart