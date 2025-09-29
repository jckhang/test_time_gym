#!/usr/bin/env python3
"""
运行改进环境的可观测演示
展示新的ImprovedFlightBookingEnv的详细奖励分解和技能跟踪功能
"""

import asyncio
import logging
import os
import random
import sys
import time
import webbrowser
from typing import Dict, List

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from experiments.observation_system import ObservationSystem
from test_time_gym.envs.improved_flight_booking_env import (
    ImprovedFlightBookingEnv,
    SkillType,
)


class ImprovedEnvAgent:
    """改进环境的智能体（多种策略）"""

    def __init__(self, strategy: str = "balanced"):
        self.strategy = strategy
        self.learned_skills = []
        self.action_history = []
        self.performance_memory = {}

    def select_action(self, observation: Dict) -> str:
        """根据策略选择动作"""
        current_view = observation.get('view', 'search_form')
        available_actions = observation.get('available_actions', [])

        if not available_actions:
            return "restart"

        # 根据策略选择动作
        if self.strategy == "aggressive":
            return self._aggressive_strategy(observation, available_actions)
        elif self.strategy == "conservative":
            return self._conservative_strategy(observation, available_actions)
        elif self.strategy == "learning":
            return self._learning_strategy(observation, available_actions)
        else:  # balanced
            return self._balanced_strategy(observation, available_actions)

    def _aggressive_strategy(self, obs: Dict, actions: List[str]) -> str:
        """激进策略：快速决策，优先选择最便宜的选项"""
        view = obs.get('view')

        if view == 'search_form':
            return "search_flights"
        elif view == 'search_results':
            if 'select_flight cheapest' in actions:
                return "select_flight cheapest"
            elif 'add_to_cart' in actions:
                return "add_to_cart"
            else:
                return actions[0]
        elif view == 'cart':
            return "proceed_to_payment"
        elif view == 'payment':
            if 'confirm_payment' in actions:
                return "confirm_payment"
            elif 'enter_card' in actions:
                return "enter_card"

        return random.choice(actions)

    def _conservative_strategy(self, obs: Dict, actions: List[str]) -> str:
        """保守策略：仔细筛选，重视约束满足"""
        view = obs.get('view')

        if view == 'search_form':
            return "search_flights"
        elif view == 'search_results':
            if 'filter_results' in actions:
                return "filter_results"
            elif 'select_flight cheapest' in actions:
                return "select_flight cheapest"
            elif 'add_to_cart' in actions:
                return "add_to_cart"
        elif view == 'cart':
            # 检查预算
            cart_total = obs.get('cart', {}).get('total', 0)
            budget = obs.get('constraints', {}).get('budget', 1000)
            if cart_total <= budget:
                return "proceed_to_payment"
            else:
                return "restart"  # 超预算重新开始
        elif view == 'payment':
            if 'confirm_payment' in actions:
                return "confirm_payment"
            elif 'enter_card' in actions:
                return "enter_card"

        return random.choice(actions)

    def _learning_strategy(self, obs: Dict, actions: List[str]) -> str:
        """学习策略：基于历史表现调整行为"""
        view = obs.get('view')

        # 记录状态-动作历史
        state_key = f"{view}_{len(obs.get('flights', []))}"

        # 如果有历史记录，选择表现最好的动作
        if state_key in self.performance_memory:
            best_action = max(self.performance_memory[state_key].items(),
                            key=lambda x: x[1])[0]
            if best_action in actions:
                return best_action

        # 否则使用平衡策略
        return self._balanced_strategy(obs, actions)

    def _balanced_strategy(self, obs: Dict, actions: List[str]) -> str:
        """平衡策略：综合考虑效率和约束"""
        view = obs.get('view')

        if view == 'search_form':
            return "search_flights"
        elif view == 'search_results':
            flights = obs.get('flights', [])
            if flights and len(flights) > 3:
                return "filter_results"
            elif 'select_flight cheapest' in actions:
                return "select_flight cheapest"
            elif 'add_to_cart' in actions:
                return "add_to_cart"
        elif view == 'cart':
            return "proceed_to_payment"
        elif view == 'payment':
            if 'confirm_payment' in actions:
                return "confirm_payment"
            elif 'enter_card' in actions:
                return "enter_card"

        return random.choice(actions)

    def update_performance(self, state_key: str, action: str, reward: float):
        """更新动作表现记录"""
        if state_key not in self.performance_memory:
            self.performance_memory[state_key] = {}

        if action not in self.performance_memory[state_key]:
            self.performance_memory[state_key][action] = []

        self.performance_memory[state_key][action].append(reward)

    def learn_skill(self, skill_name: str, context: Dict):
        """学习新技能"""
        skill_info = {
            'name': skill_name,
            'learned_at': time.time(),
            'context': context,
            'usage_count': 0,
            'success_rate': 0.0
        }
        self.learned_skills.append(skill_info)
        return skill_info


async def run_single_experiment(obs_system: ObservationSystem,
                              experiment_name: str,
                              agent_strategy: str,
                              difficulty: str = "medium",
                              num_episodes: int = 20):
    """运行单个实验"""

    print(f"🎮 开始实验: {experiment_name} (策略: {agent_strategy}, 难度: {difficulty})")

    # 创建环境和智能体
    env = ImprovedFlightBookingEnv(
        seed=42,
        config={
            "difficulty": difficulty,
            "max_steps": 30,
            "progress_weight": 0.1,
            "constraint_weight": 0.3,
            "efficiency_weight": 0.2,
            "optimization_weight": 0.2,
            "completion_weight": 1.0
        }
    )

    agent = ImprovedEnvAgent(strategy=agent_strategy)

    # 技能发现计数器
    skills_discovered = 0

    for episode in range(num_episodes):
        episode_id = f"ep_{episode:03d}"

        # 重置环境
        obs, info = env.reset()

        # 记录episode开始
        obs_system.log_episode_start(experiment_name, episode_id, obs)

        total_reward = 0.0
        trajectory = []
        step_count = 0

        # 模拟技能学习（基于策略和episode进度）
        if agent_strategy == "learning" and episode > 5 and random.random() < 0.2:
            skill_name = f"OptimizedSearch_{skills_discovered+1}"
            skill_info = agent.learn_skill(skill_name, {
                'episode': episode,
                'difficulty': difficulty,
                'strategy': agent_strategy
            })
            skills_discovered += 1

            # 记录技能学习
            obs_system.log_skill_learned(
                experiment_name,
                skill_name,
                random.uniform(0.6, 0.9),  # 模拟技能成功率
                usage_count=0
            )

        while not env.done and not env.truncated:
            # 智能体选择动作
            action = agent.select_action(obs)

            # 执行动作
            next_obs, reward, done, truncated, step_info = env.step(action)

            # 记录轨迹
            step_data = {
                'step': step_count,
                'action': action,
                'obs': obs,
                'reward': reward,
                'reward_breakdown': step_info.get('reward_breakdown', {}),
                'skill_metrics': step_info.get('skill_metrics', {})
            }
            trajectory.append(step_data)

            # 技能使用检测
            skill_used = None
            if agent.learned_skills and random.random() < 0.3:
                skill_used = random.choice(agent.learned_skills)['name']
                skill_success = random.random() > 0.3
                obs_system.log_skill_usage(experiment_name, skill_used, skill_success)

            # 记录步骤
            obs_system.log_step(
                experiment_name, episode_id, step_count,
                action, next_obs, reward, skill_used
            )

            total_reward += reward
            step_count += 1
            obs = next_obs

            # 模拟真实执行延迟
            await asyncio.sleep(0.05)

        # 判断成功
        success = env.done and not env.truncated
        if step_info and 'constraint_satisfaction' in step_info:
            success = success and step_info['constraint_satisfaction'] > 0.8

        # 记录episode结束
        obs_system.log_episode_end(experiment_name, episode_id, total_reward, success, trajectory)

        # 学习策略更新性能记录
        if agent_strategy == "learning":
            for step_data in trajectory:
                state_key = f"{step_data['obs'].get('view', '')}_{len(step_data['obs'].get('flights', []))}"
                agent.update_performance(state_key, step_data['action'], step_data['reward'])

        # 进度报告
        if (episode + 1) % 5 == 0:
            print(f"  📊 {experiment_name}: 已完成 {episode + 1}/{num_episodes} episodes")

        # episode间隔
        await asyncio.sleep(0.2)

    print(f"✅ {experiment_name} 完成! 发现技能: {skills_discovered} 个")


async def run_difficulty_comparison(obs_system: ObservationSystem):
    """运行不同难度级别的对比实验"""

    print("\n🎯 运行难度级别对比实验...")

    difficulties = ["easy", "medium", "hard"]
    strategy = "balanced"

    tasks = []
    for difficulty in difficulties:
        experiment_name = f"difficulty_{difficulty}_{strategy}"
        task = asyncio.create_task(
            run_single_experiment(
                obs_system,
                experiment_name,
                strategy,
                difficulty,
                num_episodes=15
            )
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


async def run_strategy_comparison(obs_system: ObservationSystem):
    """运行不同策略的对比实验"""

    print("\n🧠 运行策略对比实验...")

    strategies = ["aggressive", "conservative", "balanced", "learning"]
    difficulty = "medium"

    tasks = []
    for strategy in strategies:
        experiment_name = f"strategy_{strategy}_{difficulty}"
        task = asyncio.create_task(
            run_single_experiment(
                obs_system,
                experiment_name,
                strategy,
                difficulty,
                num_episodes=20
            )
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


async def run_comprehensive_demo(obs_system: ObservationSystem):
    """运行综合演示"""

    print("\n🚀 运行综合演示实验...")

    # 基础对比实验
    experiments = [
        ("basic_baseline", "conservative", "medium", 15),
        ("basic_optimized", "learning", "medium", 15),
        ("advanced_strategy", "balanced", "hard", 12)
    ]

    tasks = []
    for exp_name, strategy, difficulty, episodes in experiments:
        task = asyncio.create_task(
            run_single_experiment(
                obs_system,
                exp_name,
                strategy,
                difficulty,
                episodes
            )
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


def print_demo_info():
    """打印演示信息"""
    info = """
🎯 改进环境演示内容:

📊 核心特性验证:
  • 详细奖励分解系统 (base_action, progress, constraint_satisfaction, efficiency, optimization)
  • 技能指标跟踪 (search_efficiency, budget_efficiency, constraint_violations)
  • 多难度级别支持 (easy, medium, hard)
  • 确定性业务逻辑 (减少随机性，提高学习效果)

🤖 智能体策略对比:
  • aggressive: 快速决策，优先最便宜选项
  • conservative: 谨慎决策，重视约束满足
  • balanced: 平衡效率和约束
  • learning: 基于历史表现自适应学习

🔍 观测重点:
  • 奖励分解的实时变化
  • 不同策略的成功率差异
  • 技能学习和使用模式
  • 约束满足分数演进
  • 难度级别对性能的影响

🌐 Web界面功能:
  • 实时奖励分解可视化
  • 技能学习时间线
  • 策略性能对比图表
  • 约束满足度分析

💡 使用建议:
  1. 启动演示后访问 http://localhost:5000
  2. 观察实时数据更新
  3. 关注奖励分解的变化
  4. 比较不同策略的表现
  5. 查看生成的详细报告
"""
    print(info)


async def main():
    """主函数"""

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("🔬 改进环境可观测演示")
    print("=" * 60)
    print_demo_info()

    # 创建观测系统
    obs_system = ObservationSystem(enable_web=True, web_port=5000)

    # 启动监控
    all_experiments = [
        "basic_baseline", "basic_optimized", "advanced_strategy",
        "difficulty_easy_balanced", "difficulty_medium_balanced", "difficulty_hard_balanced",
        "strategy_aggressive_medium", "strategy_conservative_medium",
        "strategy_balanced_medium", "strategy_learning_medium"
    ]

    obs_system.start_monitoring(all_experiments)

    print("\n🌐 Web仪表板已启动: http://localhost:5000")
    print("💡 打开浏览器查看实时监控界面")

    # 自动打开浏览器
    try:
        webbrowser.open("http://localhost:5000")
        print("🌐 已自动打开浏览器")
    except:
        print("⚠️ 无法自动打开浏览器，请手动访问 http://localhost:5000")

    print("\n⏳ 等待5秒后开始实验...")
    await asyncio.sleep(5)

    try:
        # 运行演示实验
        print("\n🚀 开始演示实验...")
        await run_comprehensive_demo(obs_system)

        print("\n🎯 运行难度对比...")
        await run_difficulty_comparison(obs_system)

        print("\n🧠 运行策略对比...")
        await run_strategy_comparison(obs_system)

        print("\n🎉 所有实验完成!")
        print("📊 观测数据已收集完毕")

        # 额外等待时间来观察最终状态
        print("\n⏱️ 保持系统运行60秒以供观察...")
        await asyncio.sleep(60)

    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 生成最终报告
        print("\n📝 生成最终报告...")
        obs_system.generate_final_report("logs/improved_env_reports")

        # 清理
        obs_system.cleanup()
        print("🧹 清理完成")
        print("\n📁 报告文件位置: logs/improved_env_reports/")
        print("📁 详细日志位置: logs/observable_experiment.log")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 感谢使用改进环境演示!")
    except Exception as e:
        print(f"\n❌ 演示启动失败: {e}")
        print("💡 请确保已安装所需依赖和配置")
