"""
Test-Time Gym Environment
一个用于LLM智能体测试时学习的安全仿真环境
"""
__version__ = "0.1.0"

from .envs.flight_booking_env import FlightBookingEnv

__all__ = ["FlightBookingEnv"]