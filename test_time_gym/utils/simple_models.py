"""Simplified data models without external dependencies."""

from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
import uuid
import json


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
            "id": self.id,
            "price": self.price,
            "depart_time": self.depart_time,
            "arrive_time": self.arrive_time,
            "stops": self.stops,
            "airline": self.airline,
            "departure_airport": self.departure_airport,
            "arrival_airport": self.arrival_airport
        }


class Cart:
    """Shopping cart state."""
    def __init__(self):
        self.items = []  # List of {"flight_id": str, "price": float}
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


class PaymentState:
    """Payment process state."""
    def __init__(self):
        self.needs_3ds = False
        self.attempts = 0
        self.max_attempts = 3
        self.card_entered = False
        self.payment_confirmed = False
        self.payment_failed = False
        self.failure_reason = None
    
    def to_dict(self):
        return {
            "needs_3ds": self.needs_3ds,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "card_entered": self.card_entered,
            "payment_confirmed": self.payment_confirmed,
            "payment_failed": self.payment_failed,
            "failure_reason": self.failure_reason
        }


class Constraints:
    """Task constraints."""
    def __init__(self, budget: float, depart_after: str = None, max_stops: int = None):
        self.budget = budget
        self.depart_after = depart_after
        self.max_stops = max_stops
    
    def to_dict(self):
        return {
            "budget": self.budget,
            "depart_after": self.depart_after,
            "max_stops": self.max_stops
        }


class ObservationState:
    """Complete observation state."""
    def __init__(self, view: str = "search_form", constraints: Constraints = None):
        self.view = view
        self.forms = {"from_airport": "", "to_airport": "", "date": ""}
        self.flights = []
        self.cart = Cart()
        self.payment_state = PaymentState()
        self.constraints = constraints or Constraints(budget=1000)
        self.messages = []
        self.step_count = 0
        self.session_id = str(uuid.uuid4())
    
    def to_dict(self):
        return {
            "view": self.view,
            "forms": self.forms,
            "flights": [f.to_dict() for f in self.flights],
            "cart": self.cart.to_dict(),
            "payment_state": self.payment_state.to_dict(),
            "constraints": self.constraints.to_dict(),
            "messages": self.messages,
            "step_count": self.step_count,
            "session_id": self.session_id
        }