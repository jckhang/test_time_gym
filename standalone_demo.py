"""Standalone demo of Test-Time Gym framework without external dependencies."""

import random
import json
import uuid
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime


class Flight:
    """Represents a flight option."""
    def __init__(self, id: str, price: float, depart_time: str, arrive_time: str, 
                 stops: int, airline: str, departure_airport: str, arrival_airport: str):
        self.id = id
        self.price = price
        self.depart_time = depart_time
        self.arrive_time = arrive_time
        self.stops = stops
        self.airline = airline
        self.departure_airport = departure_airport
        self.arrival_airport = arrival_airport
    
    def to_dict(self):
        return {
            "id": self.id, "price": self.price, "depart_time": self.depart_time,
            "arrive_time": self.arrive_time, "stops": self.stops, "airline": self.airline,
            "departure_airport": self.departure_airport, "arrival_airport": self.arrival_airport
        }


class Cart:
    """Shopping cart."""
    def __init__(self):
        self.items = []
        self.total = 0.0
        self.coupon_applied = None
        self.discount = 0.0
    
    def add_item(self, flight_id: str, price: float):
        self.items.append({"flight_id": flight_id, "price": price})
        self.total = sum(item["price"] for item in self.items)
    
    def to_dict(self):
        return {
            "items": self.items,
            "total": self.total,
            "coupon_applied": self.coupon_applied,
            "discount": self.discount
        }


class TestTimeGymEnv:
    """Simplified Test-Time Gym Environment."""
    
    def __init__(self, seed: Optional[int] = None, max_steps: int = 30):
        self.max_steps = max_steps
        self.current_step = 0
        self.view = "search_form"
        self.forms = {"from": "", "to": "", "date": ""}
        self.flights = []
        self.cart = Cart()
        self.payment_state = {"card_entered": False, "attempts": 0, "needs_3ds": False}
        self.constraints = {"budget": 1000, "depart_after": None, "max_stops": None}
        self.messages = []
        
        if seed:
            random.seed(seed)
        
        # Create flight database
        self.flight_db = self._create_flights()
        
        self.action_verbs = [
            "search_flights", "filter_results", "select_flight", "add_to_cart",
            "proceed_to_payment", "enter_card", "confirm_payment", "apply_coupon", "restart"
        ]
    
    def _create_flights(self) -> List[Flight]:
        """Create flight database."""
        airports = ["SFO", "LAX", "JFK", "MAD", "LHR", "CDG"]
        airlines = ["AA", "UA", "DL", "IB", "BA", "AF"]
        
        flights = []
        for i in range(50):
            from_apt = random.choice(airports)
            to_apt = random.choice([a for a in airports if a != from_apt])
            
            flight = Flight(
                id=f"{random.choice(airlines)}{random.randint(1000, 9999)}",
                price=random.randint(300, 1200),
                depart_time=f"{random.randint(6, 22):02d}:{random.choice([0, 30]):02d}",
                arrive_time=f"{random.randint(8, 23):02d}:{random.choice([0, 30]):02d}",
                stops=random.choice([0, 1, 2]),
                airline=random.choice(airlines),
                departure_airport=from_apt,
                arrival_airport=to_apt
            )
            flights.append(flight)
        
        return flights
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment."""
        # Random constraints
        budget = random.randint(400, 1000)
        depart_after = f"{random.randint(8, 16):02d}:00" if random.random() < 0.6 else None
        max_stops = random.choice([None, 0, 1, 2])
        
        self.constraints = {"budget": budget, "depart_after": depart_after, "max_stops": max_stops}
        self.view = "search_form"
        self.forms = {"from": "", "to": "", "date": ""}
        self.flights = []
        self.cart = Cart()
        self.payment_state = {"card_entered": False, "attempts": 0, "needs_3ds": False}
        self.messages = ["Welcome! Search for flights."]
        self.current_step = 0
        
        return self._get_obs(), {"constraints": self.constraints}
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute step."""
        self.current_step += 1
        
        verb_idx = action.get("verb", 0)
        payload_str = action.get("payload", "{}")
        
        if verb_idx >= len(self.action_verbs):
            return self._get_obs(), -0.05, False, False, {"error": "Invalid verb"}
        
        verb = self.action_verbs[verb_idx]
        
        try:
            payload = json.loads(payload_str)
        except:
            return self._get_obs(), -0.05, False, False, {"error": "Invalid JSON"}
        
        # Execute action
        reward, done, info = self._execute(verb, payload)
        
        # Time cost
        reward -= 0.01
        
        # Check truncation
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, truncated, info
    
    def _execute(self, verb: str, payload: Dict) -> Tuple[float, bool, Dict[str, Any]]:
        """Execute action and return reward, done, info."""
        if verb == "search_flights":
            from_apt = payload.get("from", "")
            to_apt = payload.get("to", "")
            
            if not from_apt or not to_apt:
                self.messages = ["Missing from/to airports"]
                return -0.05, False, {"error": "missing_params"}
            
            # Find flights
            matching = [f for f in self.flight_db 
                       if f.departure_airport == from_apt and f.arrival_airport == to_apt]
            
            if not matching:
                self.view = "error"
                self.messages = ["No flights found"]
                return -0.05, False, {"error": "no_flights"}
            
            # Show results with price noise
            results = []
            for flight in matching[:15]:
                noise = random.uniform(-30, 50)
                flight.price = max(100, flight.price + noise)
                if random.random() > 0.1:  # 10% sold out
                    results.append(flight)
            
            self.flights = results
            self.view = "search_results"
            self.messages = [f"Found {len(results)} flights"]
            return 0.02, False, {"flights_found": len(results)}
        
        elif verb == "add_to_cart":
            flight_id = payload.get("flight_id", "")
            
            if not flight_id:
                return -0.05, False, {"error": "no_flight_id"}
            
            flight = next((f for f in self.flights if f.id == flight_id), None)
            if not flight:
                return -0.05, False, {"error": "flight_not_found"}
            
            # Check if already in cart
            if any(item["flight_id"] == flight_id for item in self.cart.items):
                return -0.02, False, {"error": "already_in_cart"}
            
            self.cart.add_item(flight_id, flight.price)
            self.view = "cart"
            self.messages = [f"Added to cart. Total: ${self.cart.total:.2f}"]
            return 0.05, False, {"cart_total": self.cart.total}
        
        elif verb == "proceed_to_payment":
            if not self.cart.items:
                return -0.05, False, {"error": "cart_empty"}
            
            if self.cart.total > self.constraints["budget"]:
                self.messages = [f"Total ${self.cart.total:.2f} > budget ${self.constraints['budget']}"]
                return -0.3, False, {"constraint_violation": "budget"}
            
            # Random 3DS
            if random.random() < 0.25:
                self.payment_state["needs_3ds"] = True
            
            self.view = "payment"
            self.messages = ["Enter payment details"]
            return 0.05, False, {"payment_ready": True}
        
        elif verb == "enter_card":
            if self.view != "payment":
                return -0.05, False, {"error": "not_payment_page"}
            
            self.payment_state["card_entered"] = True
            self.messages = ["Card entered. Confirm payment."]
            return 0.02, False, {"card_ready": True}
        
        elif verb == "confirm_payment":
            if not self.payment_state["card_entered"]:
                return -0.05, False, {"error": "card_not_entered"}
            
            self.payment_state["attempts"] += 1
            
            # Random failure
            failure_prob = 0.15 if self.payment_state["needs_3ds"] else 0.08
            
            if random.random() < failure_prob:
                self.messages = ["Payment failed"]
                if self.payment_state["attempts"] >= 3:
                    return -0.1, True, {"payment_failed": "max_attempts"}
                return -0.05, False, {"payment_failed": "retry"}
            
            # Success!
            self.view = "receipt"
            self.messages = ["Payment successful!"]
            
            # Final reward calculation
            reward = 1.0
            violations = 0
            
            # Check constraints
            if self.cart.total > self.constraints["budget"]:
                violations += 1
                reward -= 0.3
            
            # Check flight constraints
            for item in self.cart.items:
                flight = next((f for f in self.flights if f.id == item["flight_id"]), None)
                if flight:
                    if (self.constraints["depart_after"] and 
                        flight.depart_time < self.constraints["depart_after"]):
                        violations += 1
                        reward -= 0.2
                    
                    if (self.constraints["max_stops"] is not None and 
                        flight.stops > self.constraints["max_stops"]):
                        violations += 1
                        reward -= 0.2
            
            return reward, True, {
                "success": reward > 0,
                "constraint_violations": violations,
                "final_price": self.cart.total
            }
        
        elif verb == "apply_coupon":
            code = payload.get("code", "")
            valid_codes = ["SAVE10", "WELCOME", "STUDENT"]
            
            if code in valid_codes and not self.cart.coupon_applied:
                discount = self.cart.total * 0.1
                self.cart.total -= discount
                self.cart.discount = discount
                self.cart.coupon_applied = code
                self.messages = [f"Coupon applied! Saved ${discount:.2f}"]
                return 0.03, False, {"discount": discount}
            else:
                self.messages = ["Invalid coupon"]
                return -0.02, False, {"error": "invalid_coupon"}
        
        elif verb == "restart":
            self.view = "search_form"
            self.forms = {"from": "", "to": "", "date": ""}
            self.flights = []
            self.cart = Cart()
            self.payment_state = {"card_entered": False, "attempts": 0, "needs_3ds": False}
            self.messages = ["Restarted"]
            return -0.05, False, {"restarted": True}
        
        else:
            return -0.05, False, {"error": f"unknown_action_{verb}"}
    
    def _get_obs(self) -> Dict[str, Any]:
        """Get observation."""
        return {
            "view": self.view,
            "forms": json.dumps(self.forms),
            "flights": json.dumps([f.to_dict() for f in self.flights]),
            "cart": json.dumps(self.cart.to_dict()),
            "payment_state": json.dumps(self.payment_state),
            "constraints": json.dumps(self.constraints),
            "messages": json.dumps(self.messages),
            "step_count": self.current_step
        }
    
    def render(self):
        """Render state."""
        print(f"\n--- Step {self.current_step} - View: {self.view} ---")
        print(f"Budget: ${self.constraints['budget']}")
        
        if self.view == "search_results" and self.flights:
            print(f"Flights ({len(self.flights)}):")
            for i, f in enumerate(self.flights[:3]):
                print(f"  {i+1}. {f.id}: ${f.price} - {f.depart_time} ({f.stops} stops)")
        
        if self.cart.items:
            print(f"Cart: ${self.cart.total:.2f}")
        
        if self.messages:
            print(f"Message: {self.messages[-1]}")


class SimpleDummyAgent:
    """Simple agent for demo."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def select_action(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Select action."""
        view = obs.get("view", "")
        
        try:
            flights = json.loads(obs.get("flights", "[]"))
            cart = json.loads(obs.get("cart", '{"items": [], "total": 0}'))
            payment = json.loads(obs.get("payment_state", "{}"))
            constraints = json.loads(obs.get("constraints", "{}"))
        except:
            return {"verb": 8, "payload": "{}"}  # restart
        
        if self.verbose:
            print(f"🤖 Agent choosing action for view: {view}")
        
        if view == "search_form":
            payload = {"from": "SFO", "to": "MAD", "date": "2025-10-15"}
            return {"verb": 0, "payload": json.dumps(payload)}
        
        elif view == "search_results":
            if flights:
                # Find cheapest flight within budget
                budget = constraints.get("budget", 1000)
                valid = [f for f in flights if f["price"] <= budget]
                
                if valid:
                    # Apply constraint filters first
                    if constraints.get("depart_after"):
                        valid = [f for f in valid if f["depart_time"] >= constraints["depart_after"]]
                    
                    if constraints.get("max_stops") is not None:
                        valid = [f for f in valid if f["stops"] <= constraints["max_stops"]]
                    
                    if valid:
                        cheapest = min(valid, key=lambda f: f["price"])
                        return {"verb": 3, "payload": json.dumps({"flight_id": cheapest["id"]})}
                    else:
                        # Apply filters
                        filter_payload = {}
                        if constraints.get("depart_after"):
                            filter_payload["depart_after"] = constraints["depart_after"]
                        if constraints.get("max_stops") is not None:
                            filter_payload["max_stops"] = constraints["max_stops"]
                        return {"verb": 1, "payload": json.dumps(filter_payload)}
                else:
                    return {"verb": 1, "payload": json.dumps({"max_price": budget})}
            return {"verb": 8, "payload": "{}"}
        
        elif view == "cart":
            if cart.get("items"):
                # 30% chance to try coupon
                if random.random() < 0.3 and not cart.get("coupon_applied"):
                    return {"verb": 7, "payload": json.dumps({"code": "SAVE10"})}
                else:
                    return {"verb": 4, "payload": "{}"}  # proceed to payment
            return {"verb": 8, "payload": "{}"}
        
        elif view == "payment":
            if not payment.get("card_entered", False):
                return {"verb": 5, "payload": json.dumps({"card_token": "card123"})}
            else:
                return {"verb": 6, "payload": "{}"}  # confirm
        
        else:  # error, receipt, unknown
            return {"verb": 8, "payload": "{}"}  # restart


def main():
    """Run the standalone demo."""
    print("🚀 Test-Time Gym - Standalone Demo")
    print("=" * 40)
    
    # Create environment and agent
    env = TestTimeGymEnv(seed=42)
    agent = SimpleDummyAgent(verbose=True)
    
    # Run episode
    obs, info = env.reset()
    
    print(f"📋 Task: Book flight within budget ${info['constraints']['budget']}")
    if info['constraints'].get('depart_after'):
        print(f"   Must depart after: {info['constraints']['depart_after']}")
    if info['constraints'].get('max_stops') is not None:
        print(f"   Max stops: {info['constraints']['max_stops']}")
    
    total_reward = 0.0
    step = 0
    
    done = False
    truncated = False
    
    while not (done or truncated) and step < 15:
        # Render current state
        env.render()
        
        # Agent acts
        action = agent.select_action(obs, info)
        action_name = env.action_verbs[action["verb"]]
        
        print(f"🎯 Action: {action_name}")
        
        # Environment step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"💰 Reward: {reward:.3f} (Total: {total_reward:.3f})")
        
        if done:
            print(f"\n🏁 Episode completed after {step} steps!")
            if info.get("success"):
                print(f"✅ SUCCESS! Booked flight for ${info.get('final_price', 0):.2f}")
                if info.get("constraint_violations", 0) == 0:
                    print("🎉 All constraints satisfied!")
                else:
                    print(f"⚠️  {info['constraint_violations']} constraint violations")
            else:
                print("❌ FAILED to complete booking")
            break
        
        if truncated:
            print("\n⏰ Episode truncated (max steps)")
    
    print(f"\n📊 Final Results:")
    print(f"   Total Reward: {total_reward:.3f}")
    print(f"   Steps Taken: {step}")
    print(f"   Success: {'Yes' if info.get('success', False) else 'No'}")
    
    return total_reward


def run_experiment():
    """Run multiple episodes to demonstrate learning potential."""
    print(f"\n🔬 Running Multi-Episode Experiment")
    print("-" * 40)
    
    results = []
    
    for episode_num in range(10):
        env = TestTimeGymEnv(seed=42 + episode_num)
        agent = SimpleDummyAgent(verbose=False)
        
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        
        done = False
        truncated = False
        
        while not (done or truncated) and steps < 20:
            action = agent.select_action(obs, info)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        success = info.get("success", False) if done else False
        results.append({
            "episode": episode_num + 1,
            "success": success,
            "reward": total_reward,
            "steps": steps,
            "violations": info.get("constraint_violations", 0)
        })
        
        status = "✅" if success else "❌"
        print(f"Episode {episode_num + 1:2d}: {status} Reward={total_reward:6.3f}, Steps={steps:2d}")
    
    # Calculate statistics
    successes = sum(1 for r in results if r["success"])
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results if r["success"]) / max(1, successes)
    
    print(f"\n📈 Experiment Results:")
    print(f"   Success Rate: {successes}/10 = {successes/10:.1%}")
    print(f"   Average Reward: {avg_reward:.3f}")
    print(f"   Avg Steps (Successful): {avg_steps:.1f}")
    
    return results


if __name__ == "__main__":
    print("🎮 Test-Time Gym Framework")
    print("A safe simulation environment for LLM agent test-time learning")
    print()
    
    # Single episode demo
    main()
    
    # Multi-episode experiment
    experiment_results = run_experiment()
    
    print(f"\n🎉 Framework Demo Complete!")
    print(f"✨ The Test-Time Gym is working and ready for development!")
    print(f"\nNext steps:")
    print(f"  1. Install full dependencies: pip install -e .")
    print(f"  2. Run advanced demos: python examples/basic_usage.py")
    print(f"  3. Implement your own learning agents!")
    print(f"  4. Experiment with skill extraction and Thompson Sampling!")