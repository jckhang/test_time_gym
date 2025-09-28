"""Quick demo to test the Test-Time Gym framework."""

import sys
import os

# Add the parent directory to the path so we can import test_time_gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.agents.dummy_agent import DummyAgent


def quick_test():
    """Quick test to verify the framework works."""
    print("🚀 Quick Test of Test-Time Gym Framework")
    print("-" * 40)
    
    # Create environment
    env = FlightBookingEnv(seed=42, max_steps=30)
    agent = DummyAgent(seed=42, verbose=True)
    
    # Reset environment
    obs, info = env.reset()
    print(f"✅ Environment created successfully")
    print(f"Initial constraints: {info['constraints']}")
    
    # Run a few steps
    total_reward = 0.0
    step_count = 0
    max_demo_steps = 15
    
    done = False
    truncated = False
    
    while not (done or truncated) and step_count < max_demo_steps:
        # Get action from agent
        action = agent.select_action(obs, info)
        
        print(f"\nStep {step_count + 1}:")
        print(f"  Action: {env.action_verbs[action['verb']]}")
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"  Reward: {reward:.3f}")
        print(f"  View: {obs['view']}")
        
        if obs.get('messages'):
            try:
                messages = json.loads(obs['messages'])
                if messages:
                    print(f"  Message: {messages[-1]}")
            except:
                pass
        
        if done:
            print(f"\n🎉 Episode completed!")
            print(f"Total reward: {total_reward:.3f}")
            if info.get("success"):
                print(f"✅ Booking successful! Price: ${info.get('final_price', 0):.2f}")
            else:
                print("❌ Booking failed")
            break
        
        if truncated:
            print("\n⏰ Episode truncated (max steps)")
            break
    
    print(f"\nDemo completed after {step_count} steps")
    print(f"Final total reward: {total_reward:.3f}")
    
    return total_reward


if __name__ == "__main__":
    try:
        reward = quick_test()
        print(f"\n✨ Framework test successful! Final reward: {reward:.3f}")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()