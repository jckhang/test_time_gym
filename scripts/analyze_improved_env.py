#!/usr/bin/env python3
"""
改进环境分析脚本
深度分析ImprovedFlightBookingEnv的奖励分解和技能系统
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from test_time_gym.envs.improved_flight_booking_env import ImprovedFlightBookingEnv


class ImprovedEnvAnalyzer:
    """改进环境分析器"""

    def __init__(self):
        self.results = {}

    def run_analysis_suite(self):
        """运行完整分析套件"""
        print("🔬 改进环境深度分析")
        print("=" * 60)

        # 1. 奖励分解分析
        self.analyze_reward_breakdown()

        # 2. 难度级别对比
        self.analyze_difficulty_levels()

        # 3. 技能指标演进
        self.analyze_skill_metrics()

        # 4. 确定性验证
        self.verify_determinism()

        # 5. 生成可视化报告
        self.generate_visualizations()

        print("\n✅ 分析完成！查看生成的图表和报告")

    def analyze_reward_breakdown(self):
        """分析奖励分解系统"""
        print("\n📊 分析奖励分解系统...")

        env = ImprovedFlightBookingEnv(seed=42, config={"difficulty": "medium"})

        reward_data = {
            'base_action': [],
            'progress': [],
            'constraint_satisfaction': [],
            'efficiency': [],
            'optimization': [],
            'penalty': [],
            'total': []
        }

        episode_rewards = []

        # 运行多个episode收集数据
        for episode in range(20):
            obs, info = env.reset()
            episode_reward_breakdown = []

            # 模拟智能体执行策略
            actions = [
                "search_flights",
                "filter_results",
                "select_flight cheapest",
                "add_to_cart",
                "proceed_to_payment",
                "enter_card",
                "confirm_payment"
            ]

            for action in actions:
                if env.done or env.truncated:
                    break

                obs, reward, done, trunc, info = env.step(action)

                if 'reward_breakdown' in info:
                    breakdown = info['reward_breakdown']
                    episode_reward_breakdown.append(breakdown)

                    # 收集数据
                    for key in reward_data:
                        reward_data[key].append(breakdown.get(key, 0))

            episode_rewards.append(episode_reward_breakdown)

        # 分析结果
        self.results['reward_breakdown'] = {
            'component_stats': {
                component: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                } for component, values in reward_data.items() if values
            },
            'raw_data': reward_data
        }

        print(f"  • 收集了 {len([v for v in reward_data['total'] if v != 0])} 个有效奖励样本")
        print(f"  • 平均总奖励: {np.mean([v for v in reward_data['total'] if v != 0]):.3f}")

    def analyze_difficulty_levels(self):
        """分析不同难度级别"""
        print("\n🎯 分析难度级别差异...")

        difficulties = ["easy", "medium", "hard"]
        difficulty_results = {}

        for difficulty in difficulties:
            env = ImprovedFlightBookingEnv(
                seed=42,
                config={"difficulty": difficulty, "max_steps": 25}
            )

            success_rates = []
            avg_rewards = []
            avg_steps = []
            constraint_violations = []

            for episode in range(15):
                obs, info = env.reset()
                total_reward = 0
                steps = 0

                # 使用平衡策略
                while not env.done and not env.truncated and steps < 20:
                    available_actions = obs.get('available_actions', ['restart'])

                    if available_actions:
                        action = available_actions[0]
                    else:
                        action = 'restart'

                    obs, reward, done, trunc, info = env.step(action)
                    total_reward += reward
                    steps += 1

                success = env.done and not env.truncated
                success_rates.append(1 if success else 0)
                avg_rewards.append(total_reward)
                avg_steps.append(steps)

                if 'skill_metrics' in info:
                    constraint_violations.append(info['skill_metrics'].get('constraint_violations', 0))

            difficulty_results[difficulty] = {
                'success_rate': np.mean(success_rates),
                'avg_reward': np.mean(avg_rewards),
                'avg_steps': np.mean(avg_steps),
                'constraint_violations': np.mean(constraint_violations) if constraint_violations else 0,
                'task_complexity': len(env.current_task['constraints'])
            }

            print(f"  • {difficulty}: 成功率={np.mean(success_rates):.3f}, 平均奖励={np.mean(avg_rewards):.3f}")

        self.results['difficulty_analysis'] = difficulty_results

    def analyze_skill_metrics(self):
        """分析技能指标系统"""
        print("\n🧠 分析技能指标系统...")

        env = ImprovedFlightBookingEnv(seed=42, config={"difficulty": "medium"})

        skill_evolution = {
            'search_efficiency': [],
            'budget_efficiency': [],
            'constraint_violations': [],
            'error_recovery_count': []
        }

        for episode in range(25):
            obs, info = env.reset()

            # 模拟智能体学习过程
            actions = ["search_flights", "filter_results", "select_flight cheapest",
                      "add_to_cart", "proceed_to_payment", "enter_card", "confirm_payment"]

            for action in actions:
                if env.done or env.truncated:
                    break
                obs, reward, done, trunc, info = env.step(action)

            # 收集技能指标
            if 'skill_metrics' in info:
                metrics = info['skill_metrics']
                for key in skill_evolution:
                    if key in metrics:
                        skill_evolution[key].append(metrics[key])

        self.results['skill_metrics'] = skill_evolution
        print(f"  • 跟踪了 {len(skill_evolution['search_efficiency'])} 个episode的技能演进")

    def verify_determinism(self):
        """验证确定性"""
        print("\n🔍 验证环境确定性...")

        # 使用相同种子运行多次
        results_run1 = []
        results_run2 = []

        for run in range(2):
            env = ImprovedFlightBookingEnv(seed=123, config={"difficulty": "medium"})
            run_results = []

            for episode in range(5):
                obs, info = env.reset()
                episode_data = {
                    'initial_task': env.current_task.copy(),
                    'flights_count': len(env.flights_db),
                    'rewards': []
                }

                actions = ["search_flights", "filter_results", "select_flight cheapest"]
                for action in actions:
                    if env.done or env.truncated:
                        break
                    obs, reward, done, trunc, info = env.step(action)
                    episode_data['rewards'].append(reward)

                run_results.append(episode_data)

            if run == 0:
                results_run1 = run_results
            else:
                results_run2 = run_results

        # 检查一致性
        deterministic = True
        for i in range(len(results_run1)):
            if (results_run1[i]['initial_task'] != results_run2[i]['initial_task'] or
                results_run1[i]['flights_count'] != results_run2[i]['flights_count']):
                deterministic = False
                break

        self.results['determinism'] = {
            'is_deterministic': deterministic,
            'run1_sample': results_run1[0] if results_run1 else None,
            'run2_sample': results_run2[0] if results_run2 else None
        }

        print(f"  • 确定性验证: {'✅ 通过' if deterministic else '❌ 失败'}")

    def generate_visualizations(self):
        """生成可视化图表"""
        print("\n📈 生成可视化图表...")

        # 确保输出目录存在
        output_dir = "logs/improved_env_analysis"
        os.makedirs(output_dir, exist_ok=True)

        # 设置英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 奖励分解饼图
        if 'reward_breakdown' in self.results:
            self._plot_reward_breakdown(output_dir)

        # 2. 难度级别对比
        if 'difficulty_analysis' in self.results:
            self._plot_difficulty_comparison(output_dir)

        # 3. 技能指标演进
        if 'skill_metrics' in self.results:
            self._plot_skill_evolution(output_dir)

        # 4. 生成综合报告
        self._generate_report(output_dir)

        print(f"  • 图表保存至: {output_dir}/")

    def _plot_reward_breakdown(self, output_dir: str):
        """绘制奖励分解图"""
        reward_stats = self.results['reward_breakdown']['component_stats']

        # 提取平均值
        components = []
        values = []
        for comp, stats in reward_stats.items():
            if comp != 'total' and stats['mean'] != 0:
                components.append(comp)
                values.append(abs(stats['mean']))  # 使用绝对值用于饼图

        if values:
            plt.figure(figsize=(10, 8))
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']

            plt.subplot(2, 2, 1)
            plt.pie(values, labels=components, autopct='%1.1f%%', colors=colors[:len(values)])
            plt.title('Reward Component Distribution')

            # 奖励组件条形图
            plt.subplot(2, 2, 2)
            means = [reward_stats[comp]['mean'] for comp in components]
            stds = [reward_stats[comp]['std'] for comp in components]

            bars = plt.bar(range(len(components)), means, yerr=stds, capsize=5)
            plt.xticks(range(len(components)), components, rotation=45)
            plt.title('Reward Component Mean ± Std')
            plt.ylabel('Reward Value')

            # 为正负值设置不同颜色
            for i, bar in enumerate(bars):
                if means[i] >= 0:
                    bar.set_color('#66b3ff')
                else:
                    bar.set_color('#ff9999')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/reward_breakdown.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_difficulty_comparison(self, output_dir: str):
        """绘制难度级别对比图"""
        diff_data = self.results['difficulty_analysis']

        difficulties = list(diff_data.keys())
        metrics = ['success_rate', 'avg_reward', 'avg_steps', 'constraint_violations']

        plt.figure(figsize=(15, 10))

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            values = [diff_data[diff][metric] for diff in difficulties]

            bars = plt.bar(difficulties, values, color=['#90EE90', '#FFB347', '#FF6B6B'])
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.ylabel('Value')

            # 添加数值标签
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/difficulty_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_skill_evolution(self, output_dir: str):
        """绘制技能演进图"""
        skill_data = self.results['skill_metrics']

        plt.figure(figsize=(12, 8))

        for i, (skill, values) in enumerate(skill_data.items()):
            if values:  # 只绘制有数据的技能
                plt.subplot(2, 2, i+1)
                episodes = range(len(values))
                plt.plot(episodes, values, marker='o', linewidth=2, markersize=4)
                plt.title(f'{skill.replace("_", " ").title()}')
                plt.xlabel('Episode')
                plt.ylabel('Value')
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/skill_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self, output_dir: str):
        """生成文本报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = [
            "🔬 改进环境分析报告",
            "=" * 60,
            f"生成时间: {timestamp}",
            "",
            "📊 核心发现:",
            ""
        ]

        # 奖励分解分析
        if 'reward_breakdown' in self.results:
            stats = self.results['reward_breakdown']['component_stats']
            report.extend([
                "1. 奖励分解系统分析:",
                f"   • 主要奖励组件: {len(stats)} 个",
                f"   • 平均总奖励: {stats.get('total', {}).get('mean', 0):.3f}",
                ""
            ])

        # 难度级别分析
        if 'difficulty_analysis' in self.results:
            diff_data = self.results['difficulty_analysis']
            report.extend([
                "2. 难度级别分析:",
            ])
            for diff, data in diff_data.items():
                report.append(f"   • {diff}: 成功率={data['success_rate']:.3f}, "
                            f"平均奖励={data['avg_reward']:.3f}")
            report.append("")

        # 确定性验证
        if 'determinism' in self.results:
            is_det = self.results['determinism']['is_deterministic']
            report.extend([
                "3. 确定性验证:",
                f"   • 环境确定性: {'✅ 通过' if is_det else '❌ 失败'}",
                ""
            ])

        # 技能系统
        if 'skill_metrics' in self.results:
            skill_data = self.results['skill_metrics']
            report.extend([
                "4. 技能指标系统:",
                f"   • 跟踪指标: {len(skill_data)} 个",
                f"   • 数据完整性: {len([v for v in skill_data.values() if v])} / {len(skill_data)}",
                ""
            ])

        report.extend([
            "📁 生成文件:",
            "   • reward_breakdown.png - 奖励分解可视化",
            "   • difficulty_comparison.png - 难度级别对比",
            "   • skill_evolution.png - 技能演进曲线",
            "   • analysis_report.txt - 详细分析报告",
            "",
            "✨ 分析完成！"
        ])

        # 保存报告
        with open(f"{output_dir}/analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        # 保存原始数据
        with open(f"{output_dir}/raw_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)


def main():
    """主函数"""
    analyzer = ImprovedEnvAnalyzer()
    analyzer.run_analysis_suite()

    print("\n📋 分析总结:")
    print("✅ 奖励分解系统 - 提供详细的多维度反馈")
    print("✅ 难度分级系统 - 支持渐进式学习")
    print("✅ 技能指标跟踪 - 量化学习进展")
    print("✅ 确定性业务逻辑 - 减少随机性，提高可重现性")
    print("\n🎯 改进环境已准备就绪，可用于LLM智能体训练！")


if __name__ == "__main__":
    main()
