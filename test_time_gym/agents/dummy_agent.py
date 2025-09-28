"""Simple dummy agent for testing the environment."""

import random
import json
from typing import Dict, Any, List, Optional

from ..utils.models import Action, ObservationState


class DummyAgent:
    """
    A simple dummy agent that follows basic heuristics.
    
    This agent demonstrates the environment API and serves as a baseline
    for comparison with learning agents.
    """
    
    def __init__(self, seed: Optional[int] = None, verbose: bool = False):
        self.verbose = verbose
        if seed is not None:
            random.seed(seed)
        
        # Simple action mapping
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
    
    def select_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Select an action based on simple heuristics."""
        # Parse observation
        view = observation.get("view", "")
        
        try:
            forms = json.loads(observation.get("forms", "{}"))
            flights = json.loads(observation.get("flights", "[]"))
            cart = json.loads(observation.get("cart", '{"items": [], "total": 0}'))
            payment_state = json.loads(observation.get("payment_state", "{}"))
            constraints = json.loads(observation.get("constraints", "{}"))
            messages = json.loads(observation.get("messages", "[]"))
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            if self.verbose:
                print("Warning: Failed to parse observation JSON")
            return {"verb": 8, "payload": "{}"}  # restart
        
        if self.verbose:
            print(f"Agent sees view: {view}, {len(flights)} flights, cart total: {cart.get('total', 0)}")
        
        # State-based action selection
        if view == "search_form":
            return self._search_action()
        
        elif view == "search_results":
            if flights:
                return self._handle_search_results(flights, constraints)
            else:
                return {"verb": 0, "payload": json.dumps({"from": "SFO", "to": "MAD", "date": "2025-10-15"})}
        
        elif view == "cart":
            if cart.get("items"):
                # Try to apply coupon first (randomly)
                if random.random() < 0.3 and not cart.get("coupon_applied"):
                    return {"verb": 7, "payload": json.dumps({"code": random.choice(["SAVE10", "WELCOME", "STUDENT"])})}
                else:
                    return {"verb": 4, "payload": "{}"}  # proceed_to_payment
            else:
                return {"verb": 8, "payload": "{}"}  # restart
        
        elif view == "payment":
            if not payment_state.get("card_entered", False):
                return {"verb": 5, "payload": json.dumps({"card_token": "card_1234"})}
            else:
                return {"verb": 6, "payload": "{}"}  # confirm_payment
        
        elif view == "receipt":
            # Episode should end, but just in case
            return {"verb": 8, "payload": "{}"}  # restart
        
        elif view == "error":
            return {"verb": 8, "payload": "{}"}  # restart
        
        else:
            # Unknown view - restart
            return {"verb": 8, "payload": "{}"}
    
    def _search_action(self) -> Dict[str, Any]:
        """Generate a search flights action."""
        # Random search parameters
        airports = ["SFO", "LAX", "JFK", "MAD", "LHR", "CDG", "FRA", "NRT"]
        from_airport = random.choice(airports)
        to_airport = random.choice([a for a in airports if a != from_airport])
        
        # Random date (next 30 days)
        import datetime
        base_date = datetime.date.today()
        random_days = random.randint(1, 30)
        flight_date = (base_date + datetime.timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        payload = {
            "from": from_airport,
            "to": to_airport,
            "date": flight_date
        }
        
        return {"verb": 0, "payload": json.dumps(payload)}
    
    def _handle_search_results(self, flights: List[Dict], constraints: Dict) -> Dict[str, Any]:
        """Handle search results view."""
        if not flights:
            return {"verb": 8, "payload": "{}"}  # restart
        
        budget = constraints.get("budget", 1000)
        max_stops = constraints.get("max_stops")
        depart_after = constraints.get("depart_after")
        
        # Filter flights that meet constraints
        valid_flights = []
        for flight in flights:
            valid = True
            
            # Check budget
            if flight.get("price", 0) > budget:
                valid = False
            
            # Check stops
            if max_stops is not None and flight.get("stops", 0) > max_stops:
                valid = False
            
            # Check departure time
            if depart_after and flight.get("depart_time", "") < depart_after:
                valid = False
            
            if valid:
                valid_flights.append(flight)
        
        if not valid_flights:
            # Try to filter to find suitable flights
            filter_payload = {}
            if depart_after:
                filter_payload["depart_after"] = depart_after
            if max_stops is not None:
                filter_payload["max_stops"] = max_stops
            
            if filter_payload:
                return {"verb": 1, "payload": json.dumps(filter_payload)}
            else:
                return {"verb": 8, "payload": "{}"}  # restart
        
        # Select cheapest valid flight
        cheapest_flight = min(valid_flights, key=lambda f: f.get("price", float('inf')))
        
        return {"verb": 3, "payload": json.dumps({"flight_id": cheapest_flight["id"]})}


class RandomAgent:
    """Completely random agent for baseline comparison."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        self.action_verbs = [
            "search_flights", "filter_results", "select_flight", "add_to_cart",
            "proceed_to_payment", "enter_card", "confirm_payment", "apply_coupon", "restart"
        ]
    
    def select_action(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Select a completely random action."""
        verb_idx = random.randint(0, len(self.action_verbs) - 1)
        
        # Generate random payload
        payload = {}
        verb = self.action_verbs[verb_idx]
        
        if verb == "search_flights":
            airports = ["SFO", "LAX", "JFK", "MAD", "LHR", "CDG"]
            payload = {
                "from": random.choice(airports),
                "to": random.choice(airports),
                "date": "2025-10-15"
            }
        elif verb == "add_to_cart" or verb == "select_flight":
            try:
                flights = json.loads(observation.get("flights", "[]"))
                if flights:
                    payload = {"flight_id": random.choice(flights)["id"]}
            except:
                payload = {"flight_id": "DUMMY123"}
        elif verb == "enter_card":
            payload = {"card_token": "random_card_123"}
        elif verb == "apply_coupon":
            payload = {"code": random.choice(["SAVE10", "INVALID", "WELCOME"])}
        
        return {"verb": verb_idx, "payload": json.dumps(payload)}