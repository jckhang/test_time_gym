#!/usr/bin/env python3
"""
高级使用示例
演示技能学习、OOD检测和A/B测试等高级功能
"""

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# 添加包路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_time_gym.agents.dummy_agent import DummyAgent, SkillBasedAgent
from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.utils.evaluation import (
    EvaluationMetrics,
    OODDetector,
    SafetyMonitor,
    TrajectoryLogger,
)
from test_time_gym.utils.skill_system import IntrinsicRewardCalculator, SkillManager


class SmartAgent(SkillBasedAgent):
    """增强的智能体，集成技能管理和OOD检测"""

    def __init__(self, exploration_rate: float = 0.1):
        super().__init__(exploration_rate)
        self.skill_manager = SkillManager()
        self.intrinsic_calculator = IntrinsicRewardCalculator()
        self.safety_monitor = SafetyMonitor()
        self.previous_obs = None

    def select_action(self, observation: Dict) -> str:
        """智能动作选择，集成技能和安全检查"""
        # 尝试使用技能
        selected_skill = self.skill_manager.select_skill_thompson_sampling(observation)

        if selected_skill and np.random.random() > self.exploration_rate:
            # 利用已知技能
            next_action = selected_skill.action_sequence[0]  # 取技能的第一个动作
            self.exploitation_steps = getattr(self, 'exploitation_steps', 0) + 1
        else:
            # 探索新动作
            next_action = super().select_action(observation)
            self.exploration_steps = getattr(self, 'exploration_steps', 0) + 1

        # 安全检查
        is_safe, safety_msg = self.safety_monitor.check_action_safety(observation, next_action)
        if not is_safe:
            print(f"安全拦截: {safety_msg}")
            return "restart"  # 安全的备选动作

        self.previous_obs = observation
        return next_action

    def update_with_feedback(self, trajectory: List[Dict], total_reward: float):
        """使用反馈更新智能体"""
        # 更新技能管理器
        self.skill_manager.add_trajectory(trajectory, total_reward)

        # 更新父类记忆
        self.update_memory(trajectory)

        # 定期清理技能
        if len(self.memory) % 20 == 0:
            pruned = self.skill_manager.prune_skills()
            if pruned > 0:
                print(f"清理了 {pruned} 个低效技能")


def ab_testing_experiment():
    """A/B测试实验：对比有无经验学习的效果"""
    print("=== A/B测试实验：经验学习效果验证 ===")

    # 实验配置
    episodes_per_group = 100
    seeds = range(42, 47)  # 5个不同种子

    # 创建对照组
    groups = {
        "无记忆组": {"agent_class": DummyAgent, "params": {"strategy": "greedy"}, "memory": False},
        "有记忆组": {"agent_class": SmartAgent, "params": {"exploration_rate": 0.2}, "memory": True},
        "随机对照": {"agent_class": DummyAgent, "params": {"strategy": "random"}, "memory": False}
    }

    all_results = []

    for group_name, config in groups.items():
        print(f"\n运行 {group_name}...")

        group_results = []

        for seed in seeds:
            env = FlightBookingEnv(seed=seed)
            agent = config["agent_class"](**config["params"])

            seed_results = []

            for episode in range(episodes_per_group):
                obs, info = env.reset(seed=seed * 1000 + episode)

                trajectory = []
                total_reward = 0
                steps = 0

                for step in range(50):
                    action = agent.select_action(obs)
                    next_obs, reward, done, trunc, step_info = env.step(action)

                    total_reward += reward
                    steps += 1

                    trajectory.append({
                        "step": step,
                        "action": action,
                        "obs": obs,
                        "reward": reward,
                        "done": done
                    })

                    obs = next_obs

                    if done or trunc:
                        break

                # 记录结果
                success = total_reward > 0.5
                seed_results.append({
                    "episode": episode,
                    "success": success,
                    "steps": steps,
                    "total_reward": total_reward,
                    "group": group_name,
                    "seed": seed
                })

                # 如果智能体支持学习，更新它
                if hasattr(agent, 'update_with_feedback'):
                    agent.update_with_feedback(trajectory, total_reward)

            group_results.extend(seed_results)

        all_results.extend(group_results)

        # 计算组统计
        success_rate = np.mean([r["success"] for r in group_results])
        avg_steps = np.mean([r["steps"] for r in group_results if r["success"]])
        avg_reward = np.mean([r["total_reward"] for r in group_results])

        print(f"  成功率: {success_rate:.3f}")
        print(f"  平均步数: {avg_steps:.1f}")
        print(f"  平均奖励: {avg_reward:.3f}")

    # 统计分析
    df = pd.DataFrame(all_results)

    print("\n=== 统计分析结果 ===")
    summary = df.groupby("group").agg({
        "success": ["mean", "std"],
        "steps": ["mean", "std"],
        "total_reward": ["mean", "std"]
    }).round(3)

    print(summary)

    # 保存结果
    df.to_csv("logs/ab_test_results.csv", index=False)
    print("\nA/B测试结果已保存到 logs/ab_test_results.csv")

    return df


def ood_detection_demo():
    """OOD检测演示"""
    print("\n=== OOD检测演示 ===")

    # 首先收集一些正常轨迹作为参考
    env = FlightBookingEnv(seed=42)
    agent = DummyAgent("greedy")

    reference_trajectories = []

    print("收集参考轨迹...")
    for episode in range(10):
        obs, info = env.reset(seed=42 + episode)

        trajectory = []
        for step in range(20):
            action = agent.select_action(obs)
            next_obs, reward, done, trunc, step_info = env.step(action)

            trajectory.append({
                "step": step,
                "action": action,
                "observation": obs,
                "reward": reward
            })

            obs = next_obs
            if done or trunc:
                break

        reference_trajectories.append({"trajectory": trajectory})

    # 创建OOD检测器
    ood_detector = OODDetector(reference_trajectories)

    print("测试OOD检测...")

    # 测试正常状态
    normal_obs = {
        "view": "search_results",
        "step": 5,
        "flights": [{"id": "AA123", "price": 500}],
        "cart": {"total": 0},
        "constraints": {"budget": 800},
        "payment_state": {"attempts": 0}
    }

    is_ood_state, state_score = ood_detector.detect_ood_state(normal_obs)
    is_ood_action, action_score = ood_detector.detect_ood_action(normal_obs, "add_to_cart")

    print(f"正常状态 OOD检测: {is_ood_state} (得分: {state_score:.3f})")
    print(f"正常动作 OOD检测: {is_ood_action} (得分: {action_score:.3f})")

    # 测试异常状态
    abnormal_obs = {
        "view": "unknown_view",
        "step": 100,
        "flights": [],
        "cart": {"total": 5000},  # 异常高价格
        "constraints": {"budget": 800},
        "payment_state": {"attempts": 20}  # 异常多次尝试
    }

    is_ood_state, state_score = ood_detector.detect_ood_state(abnormal_obs)
    is_ood_action, action_score = ood_detector.detect_ood_action(abnormal_obs, "weird_action")

    print(f"异常状态 OOD检测: {is_ood_state} (得分: {state_score:.3f})")
    print(f"异常动作 OOD检测: {is_ood_action} (得分: {action_score:.3f})")


def safety_monitoring_demo():
    """安全监控演示"""
    print("\n=== 安全监控演示 ===")

    safety_monitor = SafetyMonitor(max_budget=1000, max_attempts=5)

    # 测试安全动作
    safe_obs = {
        "cart": {"total": 500},
        "payment_state": {"attempts": 1}
    }

    is_safe, msg = safety_monitor.check_action_safety(safe_obs, "confirm_payment")
    print(f"安全动作检查: {is_safe} - {msg}")

    # 测试不安全动作
    unsafe_obs = {
        "cart": {"total": 1500},  # 超过预算限制
        "payment_state": {"attempts": 6}  # 超过尝试次数限制
    }

    is_safe, msg = safety_monitor.check_action_safety(unsafe_obs, "confirm_payment")
    print(f"不安全动作检查: {is_safe} - {msg}")

    print(f"安全统计: {safety_monitor.get_safety_stats()}")


def skill_evolution_tracking():
    """技能演化跟踪"""
    print("\n=== 技能演化跟踪 ===")

    skill_manager = SkillManager()
    env = FlightBookingEnv(seed=42)
    agent = SmartAgent(exploration_rate=0.3)

    # 记录技能演化过程
    skill_evolution = []

    for batch in range(5):  # 5个批次，每批次20个episode
        print(f"\n批次 {batch + 1}:")

        batch_start_skills = len(skill_manager.skills)

        for episode in range(20):
            obs, info = env.reset(seed=42 + batch * 20 + episode)

            trajectory = []
            total_reward = 0

            for step in range(30):
                action = agent.select_action(obs)
                obs, reward, done, trunc, step_info = env.step(action)

                total_reward += reward
                trajectory.append({
                    "action": action,
                    "obs": obs,
                    "reward": reward
                })

                if done or trunc:
                    break

            # 更新智能体
            agent.update_with_feedback(trajectory, total_reward)

        # 记录批次后的技能状态
        stats = skill_manager.get_skill_stats()
        skill_evolution.append({
            "batch": batch + 1,
            "total_skills": stats["total_skills"],
            "avg_reliability": stats.get("avg_reliability", 0),
            "skills_with_support": stats.get("skills_with_support", 0),
            "new_skills": stats["total_skills"] - batch_start_skills
        })

        print(f"  新增技能: {skill_evolution[-1]['new_skills']}")
        print(f"  总技能数: {skill_evolution[-1]['total_skills']}")
        print(f"  平均可靠性: {skill_evolution[-1]['avg_reliability']:.3f}")

    # 展示技能演化趋势
    print("\n技能演化趋势:")
    evo_df = pd.DataFrame(skill_evolution)
    print(evo_df.to_string(index=False))

    # 保存演化数据
    evo_df.to_csv("logs/skill_evolution.csv", index=False)
    print("技能演化数据已保存到 logs/skill_evolution.csv")


def main():
    """运行所有高级示例"""
    print("Test-Time Gym 高级功能演示")
    print("=" * 50)

    try:
        # 创建日志目录
        os.makedirs("logs", exist_ok=True)

        # 运行各种演示
        ab_testing_experiment()
        ood_detection_demo()
        safety_monitoring_demo()
        skill_evolution_tracking()

        print("\n=== 高级演示完成 ===")
        print("检查 logs/ 目录查看生成的数据和报告")

    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
