"""Minimal demo without external dependencies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_time_gym.envs.simple_flight_env import SimpleFlightBookingEnv, SimpleDummyAgent


def main():
    """Run a minimal demo."""
    print("🚀 Test-Time Gym - Minimal Demo")
    print("=" * 40)
    
    # Create environment and agent
    env = SimpleFlightBookingEnv(seed=42, max_steps=25)
    agent = SimpleDummyAgent(seed=42, verbose=True)
    
    print("✅ Environment and agent created")
    
    # Run episode
    obs, info = env.reset()
    print(f"📋 Task: Find flight within budget ${info['constraints']['budget']}")
    if info['constraints'].get('depart_after'):
        print(f"   Depart after: {info['constraints']['depart_after']}")
    if info['constraints'].get('max_stops') is not None:
        print(f"   Max stops: {info['constraints']['max_stops']}")
    
    total_reward = 0.0
    step = 0
    
    done = False
    truncated = False
    
    while not (done or truncated) and step < 20:
        # Agent selects action
        action = agent.select_action(obs, info)
        
        # Environment executes action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Render current state
        env.render(mode="human")
        
        if done:
            print("🏁 Episode completed!")
            if info.get("success"):
                print(f"✅ Booking successful! Price: ${info.get('final_price', 0):.2f}")
                print(f"🎯 Total reward: {total_reward:.3f}")
            else:
                print("❌ Booking failed")
                print(f"💸 Total reward: {total_reward:.3f}")
            break
        
        if truncated:
            print("⏰ Episode truncated (too many steps)")
            break
    
    print(f"\n📊 Final Statistics:")
    print(f"   Steps taken: {step}")
    print(f"   Total reward: {total_reward:.3f}")
    print(f"   Success: {'Yes' if info.get('success', False) else 'No'}")
    
    return total_reward > 0.5


def run_multiple_episodes():
    """Run multiple episodes to see learning potential."""
    print("\n🔄 Running Multiple Episodes")
    print("-" * 30)
    
    successes = 0
    total_rewards = []
    
    for episode in range(10):
        env = SimpleFlightBookingEnv(seed=42 + episode)
        agent = SimpleDummyAgent(seed=42)
        
        obs, info = env.reset()
        episode_reward = 0.0
        
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < 20:
            action = agent.select_action(obs, info)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        success = info.get("success", False) if done else False
        if success:
            successes += 1
        
        total_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1:2d}: Reward={episode_reward:6.3f}, Success={'✅' if success else '❌'}, Steps={steps:2d}")
    
    print(f"\n📈 Summary after 10 episodes:")
    print(f"   Success rate: {successes}/10 = {successes/10:.1%}")
    print(f"   Average reward: {sum(total_rewards)/len(total_rewards):.3f}")
    print(f"   Best reward: {max(total_rewards):.3f}")
    print(f"   Worst reward: {min(total_rewards):.3f}")


if __name__ == "__main__":
    # Run single episode demo
    success = main()
    
    # Run multiple episodes
    run_multiple_episodes()
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"Framework is working and ready for development!")