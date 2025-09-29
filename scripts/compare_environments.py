#!/usr/bin/env python3
"""
环境对比测试脚本
比较原始环境和改进环境的性能差异
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.envs.improved_flight_booking_env import ImprovedFlightBookingEnv


def run_episode(env, actions: List[str], verbose: bool = False) -> Dict:
    """运行一个episode并收集统计信息"""
    obs, info = env.reset()

    episode_stats = {
        "total_reward": 0.0,
        "steps": 0,
        "success": False,
        "constraint_violations": 0,
        "rewards": [],
        "actions_taken": [],
        "final_budget_usage": 0.0,
        "reward_breakdown": [] if hasattr(env, '_execute_action_with_detailed_feedback') else None
    }

    for action in actions:
        if env.done or env.truncated:
            break

        try:
            obs, reward, done, trunc, info = env.step(action)

            episode_stats["total_reward"] += reward
            episode_stats["steps"] += 1
            episode_stats["rewards"].append(reward)
            episode_stats["actions_taken"].append(action)

            # 收集详细信息（仅改进环境）
            if hasattr(env, '_execute_action_with_detailed_feedback'):
                if "reward_breakdown" in info:
                    episode_stats["reward_breakdown"].append(info["reward_breakdown"])
                if "constraint_satisfaction" in info:
                    episode_stats["constraint_violations"] = info.get("skill_metrics", {}).get("constraint_violations", 0)

            if verbose:
                print(f"Step {episode_stats['steps']}: {action} -> Reward: {reward:.3f}")
                if "reward_breakdown" in info:
                    breakdown = info["reward_breakdown"]
                    print(f"  Breakdown: base={breakdown['base_action']:.3f}, "
                          f"progress={breakdown['progress']:.3f}, penalty={breakdown['penalty']:.3f}")

            if done:
                episode_stats["success"] = True
                # 计算最终预算使用率
                if hasattr(env, 'cart') and hasattr(env, 'current_task'):
                    budget = env.current_task.get("constraints", {}).get("budget", 1000)
                    total_cost = env.cart.get("total", 0)
                    episode_stats["final_budget_usage"] = total_cost / budget if budget > 0 else 0
                break

        except Exception as e:
            if verbose:
                print(f"Action failed: {action}, Error: {e}")
            episode_stats["rewards"].append(-0.1)
            episode_stats["steps"] += 1
            break

    return episode_stats


def compare_environments(num_episodes: int = 10, seed: int = 42) -> Dict:
    """比较两个环境的性能"""

    # 标准动作序列
    standard_actions = [
        "search_flights",
        "filter_results",
        "select_flight cheapest",
        "add_to_cart",
        "proceed_to_payment",
        "enter_card",
        "confirm_payment"
    ]

    # 优化动作序列
    optimized_actions = [
        "search_flights",
        "filter_results",
        "select_flight cheapest",
        "add_to_cart",
        "proceed_to_payment",
        "enter_card",
        "confirm_payment"
    ]

    results = {
        "original_env": {"episodes": [], "summary": {}},
        "improved_env": {"episodes": [], "summary": {}}
    }

    print("="*60)
    print("环境对比测试")
    print("="*60)

    # 测试原始环境
    print(f"\n测试原始环境 ({num_episodes} episodes)...")
    for i in range(num_episodes):
        env = FlightBookingEnv(seed=seed+i)
        stats = run_episode(env, standard_actions, verbose=(i==0))
        results["original_env"]["episodes"].append(stats)
        env.close()

    # 测试改进环境 - 简单难度
    print(f"\n测试改进环境 - 简单难度 ({num_episodes} episodes)...")
    for i in range(num_episodes):
        env = ImprovedFlightBookingEnv(
            seed=seed+i,
            config={"difficulty": "easy", "max_steps": 20}
        )
        stats = run_episode(env, optimized_actions, verbose=(i==0))
        results["improved_env"]["episodes"].append(stats)
        env.close()

    # 计算统计摘要
    for env_name, env_data in results.items():
        episodes = env_data["episodes"]

        env_data["summary"] = {
            "success_rate": sum(1 for ep in episodes if ep["success"]) / len(episodes),
            "avg_reward": np.mean([ep["total_reward"] for ep in episodes]),
            "reward_std": np.std([ep["total_reward"] for ep in episodes]),
            "avg_steps": np.mean([ep["steps"] for ep in episodes]),
            "avg_constraint_violations": np.mean([ep["constraint_violations"] for ep in episodes]),
            "avg_budget_usage": np.mean([ep["final_budget_usage"] for ep in episodes if ep["success"]])
        }

    return results


def print_comparison_results(results: Dict):
    """打印对比结果"""
    original = results["original_env"]["summary"]
    improved = results["improved_env"]["summary"]

    print("\n" + "="*60)
    print("对比结果")
    print("="*60)

    metrics = [
        ("成功率", "success_rate", "{:.1%}"),
        ("平均奖励", "avg_reward", "{:.3f}"),
        ("奖励标准差", "reward_std", "{:.3f}"),
        ("平均步数", "avg_steps", "{:.1f}"),
        ("平均约束违规", "avg_constraint_violations", "{:.1f}"),
        ("平均预算使用率", "avg_budget_usage", "{:.1%}")
    ]

    print(f"{'指标':<15} {'原始环境':<12} {'改进环境':<12} {'改进幅度':<10}")
    print("-" * 55)

    for name, key, fmt in metrics:
        orig_val = original.get(key, 0)
        impr_val = improved.get(key, 0)

        if orig_val != 0:
            improvement = (impr_val - orig_val) / orig_val * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"

        print(f"{name:<15} {fmt.format(orig_val):<12} {fmt.format(impr_val):<12} {improvement_str:<10}")


def plot_reward_comparison(results: Dict):
    """绘制奖励对比图"""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 奖励分布对比
        original_rewards = [ep["total_reward"] for ep in results["original_env"]["episodes"]]
        improved_rewards = [ep["total_reward"] for ep in results["improved_env"]["episodes"]]

        ax1.hist(original_rewards, alpha=0.7, label="Original Env", bins=10)
        ax1.hist(improved_rewards, alpha=0.7, label="Improved Env", bins=10)
        ax1.set_xlabel("Total Reward")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Reward Distribution Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 成功率对比
        metrics = ["Success Rate", "Avg Reward", "Reward Stability"]
        original_vals = [
            results["original_env"]["summary"]["success_rate"],
            results["original_env"]["summary"]["avg_reward"] / 2,  # 归一化
            1 - results["original_env"]["summary"]["reward_std"]   # 稳定性
        ]
        improved_vals = [
            results["improved_env"]["summary"]["success_rate"],
            results["improved_env"]["summary"]["avg_reward"] / 2,  # 归一化
            1 - results["improved_env"]["summary"]["reward_std"]   # 稳定性
        ]

        x = np.arange(len(metrics))
        width = 0.35

        ax2.bar(x - width/2, original_vals, width, label="Original Env", alpha=0.7)
        ax2.bar(x + width/2, improved_vals, width, label="Improved Env", alpha=0.7)
        ax2.set_ylabel("Metric Value")
        ax2.set_title("Key Metrics Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 确保logs目录存在
        import os
        os.makedirs("logs", exist_ok=True)

        plt.savefig("logs/environment_comparison.png", dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: logs/environment_comparison.png")

    except ImportError:
        print("\nNote: matplotlib not installed, skipping chart generation")


def analyze_learning_patterns(results: Dict):
    """分析学习模式"""
    print("\n" + "="*60)
    print("学习模式分析")
    print("="*60)

    # 分析改进环境的奖励分解
    improved_episodes = results["improved_env"]["episodes"]

    if improved_episodes and improved_episodes[0]["reward_breakdown"]:
        print("\n改进环境奖励组成分析:")

        total_breakdowns = {
            "base_action": [],
            "progress": [],
            "constraint_satisfaction": [],
            "efficiency": [],
            "optimization": [],
            "penalty": []
        }

        for episode in improved_episodes:
            if episode["reward_breakdown"]:
                episode_totals = {key: 0 for key in total_breakdowns.keys()}
                for step_breakdown in episode["reward_breakdown"]:
                    for key in episode_totals.keys():
                        episode_totals[key] += step_breakdown.get(key, 0)

                for key in total_breakdowns.keys():
                    total_breakdowns[key].append(episode_totals[key])

        print(f"{'奖励类型':<20} {'平均值':<10} {'标准差':<10}")
        print("-" * 45)
        for key, values in total_breakdowns.items():
            if values:
                avg_val = np.mean(values)
                std_val = np.std(values)
                print(f"{key:<20} {avg_val:>8.3f} {std_val:>8.3f}")

    # 分析失败模式
    print("\n失败模式分析:")
    original_failures = [ep for ep in results["original_env"]["episodes"] if not ep["success"]]
    improved_failures = [ep for ep in results["improved_env"]["episodes"] if not ep["success"]]

    print(f"原始环境失败率: {len(original_failures)/len(results['original_env']['episodes']):.1%}")
    print(f"改进环境失败率: {len(improved_failures)/len(results['improved_env']['episodes']):.1%}")

    if original_failures:
        avg_steps_fail_orig = np.mean([ep["steps"] for ep in original_failures])
        print(f"原始环境失败时平均步数: {avg_steps_fail_orig:.1f}")

    if improved_failures:
        avg_steps_fail_impr = np.mean([ep["steps"] for ep in improved_failures])
        print(f"改进环境失败时平均步数: {avg_steps_fail_impr:.1f}")


def save_detailed_results(results: Dict, filename: str = "/workspace/environment_comparison_results.json"):
    """保存详细结果到文件"""
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # 递归转换所有numpy类型
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {key: recursive_convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy(obj)

    converted_results = recursive_convert(results)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="环境对比测试")
    parser.add_argument("--episodes", type=int, default=10, help="测试episode数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--save-results", action="store_true", help="保存详细结果")

    args = parser.parse_args()

    # 运行对比测试
    results = compare_environments(args.episodes, args.seed)

    # 输出结果
    print_comparison_results(results)
    analyze_learning_patterns(results)
    plot_reward_comparison(results)

    if args.save_results:
        save_detailed_results(results)

    print("\n对比测试完成!")
