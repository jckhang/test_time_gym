"""Basic usage example of the Test-Time Gym framework."""

import sys
sys.path.append('..')

from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.agents.dummy_agent import DummyAgent, RandomAgent
from test_time_gym.agents.learning_agent import LearningAgent
from test_time_gym.evaluation.metrics import MetricsCalculator


def run_dummy_agent_example():
    """Run a simple example with the dummy agent."""
    print("=== Running Dummy Agent Example ===")
    
    # Create environment and agent
    env = FlightBookingEnv(seed=42)
    agent = DummyAgent(seed=42, verbose=True)
    
    # Run one episode
    obs, info = env.reset()
    total_reward = 0.0
    step = 0
    
    print(f"Initial constraints: Budget=${info['constraints']['budget']}")
    
    done = False
    truncated = False
    
    while not (done or truncated) and step < 20:
        action = agent.select_action(obs, info)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        env.render(mode="human")
        
        if done:
            print(f"Episode completed! Total reward: {total_reward:.3f}")
            if info.get("success"):
                print(f"✅ Booking successful! Final price: ${info.get('final_price', 0):.2f}")
            else:
                print("❌ Booking failed")
        
        if truncated:
            print("Episode truncated (max steps reached)")
    
    return total_reward


def run_learning_agent_comparison():
    """Compare learning agent with and without skills."""
    print("\n=== Learning Agent Comparison ===")
    
    # Run multiple episodes with different configurations
    configs = {
        "no_memory": {"enable_skills": False, "enable_bandit": False},
        "with_skills": {"enable_skills": True, "enable_bandit": True}
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nRunning {config_name} configuration...")
        
        episodes_data = []
        total_rewards = []
        
        for episode_idx in range(20):  # Run 20 episodes
            env = FlightBookingEnv(seed=42 + episode_idx)
            agent = LearningAgent(seed=42, **config)
            
            obs, info = env.reset()
            episode_reward = 0.0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.select_action(obs, info)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            # Notify agent of episode end
            success = info.get("success", False) if done else False
            agent.end_episode(episode_reward, success, env.episode if hasattr(env, 'episode') else None)
            
            if episode_idx % 5 == 0:
                print(f"  Episode {episode_idx + 1}: Reward={episode_reward:.3f}, Success={success}")
        
        results[config_name] = {
            "avg_reward": sum(total_rewards) / len(total_rewards),
            "success_rate": sum(1 for r in total_rewards if r > 0.5) / len(total_rewards),
            "rewards": total_rewards
        }
    
    # Print comparison
    print("\n=== Results Comparison ===")
    for config_name, result in results.items():
        print(f"{config_name}:")
        print(f"  Average Reward: {result['avg_reward']:.3f}")
        print(f"  Success Rate: {result['success_rate']:.3f}")
    
    return results


def run_multi_seed_evaluation():
    """Run evaluation across multiple seeds for statistical significance."""
    print("\n=== Multi-Seed Evaluation ===")
    
    calculator = MetricsCalculator()
    
    # Simulate episode data (in real usage, this would come from actual runs)
    all_episodes = []
    
    for seed in range(3):
        print(f"Running seed {seed + 1}/3...")
        
        env = FlightBookingEnv(seed=seed)
        agent = DummyAgent(seed=seed)
        
        seed_episodes = []
        
        for episode_idx in range(10):
            obs, info = env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.select_action(obs, info)
                obs, reward, done, truncated, info = env.step(action)
            
            if hasattr(env, 'episode'):
                seed_episodes.append(env.episode)
        
        all_episodes.extend(seed_episodes)
    
    # Calculate metrics
    metrics = calculator.calculate_metrics(all_episodes)
    
    print("Overall Metrics:")
    print(f"  Success Rate: {metrics.success_rate:.3f}")
    print(f"  Avg Steps to Success: {metrics.avg_steps_to_success:.1f}")
    print(f"  Constraint Violation Rate: {metrics.constraint_violation_rate:.3f}")
    print(f"  Invalid Action Rate: {metrics.invalid_action_rate:.3f}")
    
    return metrics


if __name__ == "__main__":
    print("Test-Time Gym Framework Demo")
    print("="*40)
    
    # Run examples
    run_dummy_agent_example()
    run_learning_agent_comparison()
    run_multi_seed_evaluation()
    
    print("\n🎉 Demo completed! Check the logs/ directory for detailed trajectory data.")