"""Data models for the Test-Time Gym framework."""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime, time
import uuid


class Flight(BaseModel):
    """Represents a flight option."""
    id: str
    price: float
    depart_time: str  # Format: "HH:MM"
    arrive_time: str  # Format: "HH:MM" or "HH:MM+1" for next day
    stops: int
    airline: str
    departure_airport: str
    arrival_airport: str
    duration_minutes: int = 0


class CartItem(BaseModel):
    """Item in the shopping cart."""
    flight_id: str
    price: float
    quantity: int = 1


class Cart(BaseModel):
    """Shopping cart state."""
    items: List[CartItem] = Field(default_factory=list)
    total: float = 0.0
    coupon_applied: Optional[str] = None
    discount: float = 0.0


class PaymentState(BaseModel):
    """Payment process state."""
    needs_3ds: bool = False
    attempts: int = 0
    max_attempts: int = 3
    card_entered: bool = False
    payment_confirmed: bool = False
    payment_failed: bool = False
    failure_reason: Optional[str] = None


class Constraints(BaseModel):
    """Task constraints."""
    budget: float
    depart_after: Optional[str] = None  # "HH:MM"
    depart_before: Optional[str] = None  # "HH:MM"
    max_stops: Optional[int] = None
    preferred_airline: Optional[str] = None
    must_arrive_before: Optional[str] = None


class SearchForm(BaseModel):
    """Flight search form data."""
    from_airport: str = ""
    to_airport: str = ""
    date: str = ""  # YYYY-MM-DD
    passengers: int = 1


class ObservationState(BaseModel):
    """Complete observation state."""
    view: Literal["search_form", "search_results", "cart", "payment", "receipt", "error"]
    forms: SearchForm = Field(default_factory=SearchForm)
    flights: List[Flight] = Field(default_factory=list)
    cart: Cart = Field(default_factory=Cart)
    payment_state: PaymentState = Field(default_factory=PaymentState)
    constraints: Constraints
    messages: List[str] = Field(default_factory=list)
    step_count: int = 0
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Action(BaseModel):
    """Action representation."""
    verb: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class SkillStep(BaseModel):
    """A single step in a skill sequence."""
    action: Action
    expected_view: Optional[str] = None
    success_condition: Optional[Dict[str, Any]] = None


class Skill(BaseModel):
    """A reusable skill/macro-action."""
    id: str
    name: str
    description: str
    steps: List[SkillStep]
    preconditions: Dict[str, Any] = Field(default_factory=dict)
    postconditions: Dict[str, Any] = Field(default_factory=dict)
    success_count: int = 0
    attempt_count: int = 0
    confidence: float = 0.5  # Beta distribution posterior mean
    alpha: float = 1.0  # Beta prior successes
    beta: float = 1.0   # Beta prior failures
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None


class TrajectoryStep(BaseModel):
    """A single step in an episode trajectory."""
    step_id: int
    observation: ObservationState
    action: Action
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    skill_used: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class Episode(BaseModel):
    """Complete episode trajectory."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[TrajectoryStep] = Field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    constraint_violations: int = 0
    final_regret: float = 0.0
    exploration_ratio: float = 0.0
    env_seed: Optional[int] = None
    agent_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for the framework."""
    success_rate: float = 0.0
    avg_steps_to_success: float = 0.0
    constraint_violation_rate: float = 0.0
    avg_regret: float = 0.0
    exploration_ratio: float = 0.0
    skill_reuse_rate: float = 0.0
    invalid_action_rate: float = 0.0
    episodes_evaluated: int = 0