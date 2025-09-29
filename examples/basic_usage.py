#!/usr/bin/env python3
"""
基本使用示例
演示如何使用Test-Time Gym环境和智能体
"""

import os
import sys

# 添加包路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_time_gym.agents.dummy_agent import DummyAgent, RandomAgent, SkillBasedAgent
from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.utils.evaluation import (
    EvaluationMetrics,
    TrajectoryLogger,
    Visualizer,
)
from test_time_gym.utils.skill_system import SkillManager


def basic_environment_test():
    """基础环境测试"""
    print("=== 基础环境测试 ===")

    # 创建环境
    env = FlightBookingEnv(seed=42)
    obs, info = env.reset()

    print("初始观察:")
    env.render()

    # 手动执行几个动作
    actions = [
        "search_flights",
        "filter_results",
        "add_to_cart",
        "proceed_to_payment",
        "enter_card",
        "confirm_payment"
    ]

    total_reward = 0
    for i, action in enumerate(actions):
        if env.done or env.truncated:
            break

        print(f"\n步骤 {i+1}: 执行动作 '{action}'")
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        print(f"奖励: {reward:.3f}")
        print(f"完成: {done}, 截断: {trunc}")
        env.render()

        if done:
            print(f"\n任务完成! 总奖励: {total_reward:.3f}")
            break


def agent_comparison_test():
    """智能体对比测试"""
    print("\n=== 智能体对比测试 ===")

    # 创建不同的智能体
    agents = {
        "贪心智能体": DummyAgent("greedy"),
        "随机智能体": RandomAgent(),
        "保守智能体": DummyAgent("conservative"),
        "技能智能体": SkillBasedAgent(exploration_rate=0.2)
    }

    # 测试每个智能体
    env = FlightBookingEnv()
    results = {}

    for name, agent in agents.items():
        print(f"\n测试 {name}...")

        successes = 0
        total_steps = 0
        total_rewards = 0

        # 运行多个episode
        for episode in range(10):
            obs, info = env.reset(seed=42 + episode)

            episode_reward = 0
            steps = 0

            for step in range(30):  # 最大30步
                action = agent.select_action(obs)
                obs, reward, done, trunc, info = env.step(action)

                episode_reward += reward
                steps += 1

                if done or trunc:
                    break

            if episode_reward > 0.5:  # 成功阈值
                successes += 1
                total_steps += steps

            total_rewards += episode_reward

        # 记录结果
        results[name] = {
            "成功率": successes / 10,
            "平均奖励": total_rewards / 10,
            "平均步数": total_steps / max(successes, 1)
        }

        print(f"  成功率: {results[name]['成功率']:.1%}")
        print(f"  平均奖励: {results[name]['平均奖励']:.3f}")
        print(f"  平均步数: {results[name]['平均步数']:.1f}")

    print("\n=== 对比总结 ===")
    for name, stats in results.items():
        print(f"{name}: 成功率={stats['成功率']:.1%}, 奖励={stats['平均奖励']:.3f}")


def skill_learning_demo():
    """技能学习演示"""
    print("\n=== 技能学习演示 ===")

    skill_manager = SkillManager()
    agent = SkillBasedAgent(exploration_rate=0.3)
    env = FlightBookingEnv(seed=42)

    print("运行多个episode以学习技能...")

    for episode in range(20):
        obs, info = env.reset(seed=42 + episode)

        trajectory = []
        total_reward = 0

        for step in range(30):
            action = agent.select_action(obs)
            obs, reward, done, trunc, info = env.step(action)

            total_reward += reward
            trajectory.append({
                "action": action,
                "obs": obs,
                "reward": reward
            })

            if done or trunc:
                break

        # 更新技能
        skill_manager.add_trajectory(trajectory, total_reward)
        agent.update_memory(trajectory)

        if episode % 5 == 0:
            stats = skill_manager.get_skill_stats()
            print(f"Episode {episode}: 学到 {stats['total_skills']} 个技能")

    print("\n=== 最终技能总结 ===")
    final_stats = skill_manager.get_skill_stats()
    print(f"总技能数: {final_stats['total_skills']}")
    print(f"平均可靠性: {final_stats.get('avg_reliability', 0):.3f}")

    print("\n前5个最佳技能:")
    for skill in skill_manager.get_best_skills(5):
        print(f"  {skill.name}: {' -> '.join(skill.action_sequence)} "
              f"(可靠性: {skill.get_confidence():.3f})")


def evaluation_demo():
    """评估系统演示"""
    print("\n=== 评估系统演示 ===")

    # 创建模拟数据
    metrics = EvaluationMetrics()

    # 添加一些模拟结果
    import random

    from test_time_gym.utils.evaluation import EpisodeResult

    for i in range(50):
        episode = EpisodeResult(
            episode_id=f"demo_{i}",
            agent_type="demo_agent",
            seed=42 + i,
            steps=random.randint(5, 25),
            total_reward=random.uniform(-0.5, 1.2),
            final_reward=random.choice([1.0, -0.3, 0.0]),
            success=random.choice([True, False]),
            constraint_violations=random.randint(0, 2),
            regret=random.uniform(0, 150),
            exploration_steps=random.randint(1, 15),
            exploitation_steps=random.randint(1, 15),
            skill_calls=random.randint(0, 8),
            timestamp=str(i),
            trajectory=[]
        )

        metrics.add_episode(episode)

    # 计算指标
    print(f"成功率: {metrics.calculate_success_at_n():.3f}")
    print(f"平均步数: {metrics.calculate_avg_steps_to_success():.1f}")
    print(f"约束违规率: {metrics.calculate_constraint_violation_rate():.3f}")
    print(f"后悔值: {metrics.calculate_regret():.1f}")

    # 生成学习曲线
    x, y = metrics.get_learning_curve("success_rate")
    if x and y:
        print(f"学习曲线数据点: {len(x)}")
        print(f"最终成功率: {y[-1]:.3f}")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 如果有命令行参数，使用CLI模式
        import test_time_gym.cli as cli_module
        cli_module.main()
    else:
        # 否则运行演示
        print("Test-Time Gym 演示程序")
        print("=" * 50)

        try:
            basic_environment_test()
            agent_comparison_test()
            skill_learning_demo()
            evaluation_demo()

            print("\n=== 演示完成 ===")
            print("要运行更多实验，请使用:")
            print("python -m test_time_gym.cli run --agent-type skill --episodes 100")
            print("python -m test_time_gym.cli compare --episodes 50")

        except Exception as e:
            print(f"演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
