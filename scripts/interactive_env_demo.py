#!/usr/bin/env python3
"""
交互式环境演示
让用户直接体验改进环境的功能
"""

import os
import sys
import time
from typing import Dict

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from test_time_gym.envs.improved_flight_booking_env import ImprovedFlightBookingEnv


class InteractiveDemo:
    """交互式演示"""

    def __init__(self):
        self.env = None
        self.obs = None
        self.info = None
        self.episode_count = 0
        self.total_episodes = 0
        self.successful_episodes = 0

    def run(self):
        """运行交互式演示"""
        self.print_welcome()

        while True:
            self.print_menu()
            choice = input("\n请选择操作 (1-6): ").strip()

            if choice == '1':
                self.start_new_episode()
            elif choice == '2':
                self.execute_action()
            elif choice == '3':
                self.show_current_state()
            elif choice == '4':
                self.show_reward_breakdown()
            elif choice == '5':
                self.show_statistics()
            elif choice == '6':
                print("\n👋 感谢使用改进环境演示！")
                break
            else:
                print("\n❌ 无效选择，请重新输入")

            input("\n按回车键继续...")

    def print_welcome(self):
        """打印欢迎信息"""
        welcome = """
╔══════════════════════════════════════════════════════════════╗
║                🎮 改进环境交互式演示                           ║
║                ImprovedFlightBookingEnv                      ║
╚══════════════════════════════════════════════════════════════╝

🚀 核心特性:
  • 详细奖励分解 (base_action, progress, constraint_satisfaction 等)
  • 技能指标跟踪 (search_efficiency, budget_efficiency 等)
  • 多难度级别 (easy, medium, hard)
  • 确定性业务逻辑 (提高学习效果)

💡 体验建议:
  1. 先选择难度级别创建新episode
  2. 逐步执行动作观察奖励变化
  3. 查看详细的状态和奖励分解
  4. 体验不同难度的约束效果
"""
        print(welcome)

    def print_menu(self):
        """打印菜单"""
        menu = """
📋 可用操作:
  1. 🎯 开始新episode (选择难度级别)
  2. 🎮 执行动作
  3. 📊 查看当前状态
  4. 💰 查看奖励分解
  5. 📈 查看统计信息
  6. 🚪 退出演示
"""
        print(menu)

    def start_new_episode(self):
        """开始新episode"""
        print("\n🎯 选择难度级别:")
        print("  1. Easy   - 宽松约束，适合学习基础流程")
        print("  2. Medium - 中等约束，平衡挑战和可行性")
        print("  3. Hard   - 严格约束，考验优化能力")

        difficulty_choice = input("请选择难度 (1-3): ").strip()

        difficulty_map = {'1': 'easy', '2': 'medium', '3': 'hard'}
        difficulty = difficulty_map.get(difficulty_choice, 'medium')

        # 创建环境
        self.env = ImprovedFlightBookingEnv(
            seed=42 + self.episode_count,  # 每个episode使用不同种子
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

        # 重置环境
        self.obs, self.info = self.env.reset()
        self.episode_count += 1

        print(f"\n✅ Episode {self.episode_count} 已开始 (难度: {difficulty})")
        print(f"🎯 任务: {self.obs['task']['from']} → {self.obs['task']['to']}")
        print(f"💰 预算: ${self.obs['constraints']['budget']}")
        print(f"🛑 最大经停: {self.obs['constraints']['max_stops']}")

        self.show_current_state()

    def execute_action(self):
        """执行动作"""
        if self.env is None:
            print("\n❌ 请先开始新episode")
            return

        if self.env.done or self.env.truncated:
            print("\n🏁 当前episode已结束，请开始新episode")
            return

        available_actions = self.obs.get('available_actions', [])

        if not available_actions:
            print("\n❌ 没有可用动作")
            return

        print(f"\n🎮 可用动作:")
        for i, action in enumerate(available_actions, 1):
            print(f"  {i}. {action}")

        choice = input(f"请选择动作 (1-{len(available_actions)}): ").strip()

        try:
            action_idx = int(choice) - 1
            if 0 <= action_idx < len(available_actions):
                action = available_actions[action_idx]

                # 执行动作
                old_view = self.obs.get('view', '')
                self.obs, reward, done, truncated, self.info = self.env.step(action)

                print(f"\n✅ 执行动作: {action}")
                print(f"💰 获得奖励: {reward:.3f}")
                print(f"📍 状态转换: {old_view} → {self.obs.get('view', '')}")

                # 显示奖励分解
                if 'reward_breakdown' in self.info:
                    self.show_reward_breakdown_simple(self.info['reward_breakdown'])

                # 检查episode结束
                if done:
                    self.total_episodes += 1
                    if reward > 0:
                        self.successful_episodes += 1
                    print("\n🎉 Episode 成功完成！")
                elif truncated:
                    self.total_episodes += 1
                    print("\n⏰ Episode 超时结束")

            else:
                print("\n❌ 无效选择")
        except ValueError:
            print("\n❌ 请输入数字")

    def show_current_state(self):
        """显示当前状态"""
        if self.env is None:
            print("\n❌ 请先开始新episode")
            return

        print("\n📊 当前状态:")
        print(f"  🎭 视图: {self.obs.get('view', 'unknown')}")
        print(f"  👣 步数: {self.obs.get('step', 0)}")
        print(f"  🏁 完成: {self.obs.get('done', False)}")

        # 显示购物车
        cart = self.obs.get('cart', {})
        if cart.get('items'):
            print(f"  🛒 购物车: {len(cart['items'])} 项, 总额 ${cart.get('total', 0)}")

        # 显示可用航班
        flights = self.obs.get('flights', [])
        if flights:
            print(f"  ✈️ 可用航班: {len(flights)} 个")
            for i, flight in enumerate(flights[:3], 1):  # 只显示前3个
                print(f"     {i}. {flight['id']}: ${flight['price']} "
                      f"({flight['stops']}经停)")

        # 显示技能指标
        if self.info and 'skill_metrics' in self.info:
            metrics = self.info['skill_metrics']
            print("  🧠 技能指标:")
            print(f"     搜索效率: {metrics.get('search_efficiency', 0):.3f}")
            print(f"     预算效率: {metrics.get('budget_efficiency', 0):.3f}")
            print(f"     约束违规: {metrics.get('constraint_violations', 0)}")

        # 显示可用动作
        actions = self.obs.get('available_actions', [])
        if actions:
            print(f"  🎮 可用动作: {', '.join(actions)}")

    def show_reward_breakdown(self):
        """显示详细奖励分解"""
        if self.info is None or 'reward_breakdown' not in self.info:
            print("\n❌ 没有奖励分解信息")
            return

        breakdown = self.info['reward_breakdown']

        print("\n💰 详细奖励分解:")
        print("=" * 40)

        components = [
            ('基础动作', 'base_action'),
            ('进度奖励', 'progress'),
            ('约束满足', 'constraint_satisfaction'),
            ('效率奖励', 'efficiency'),
            ('优化奖励', 'optimization'),
            ('惩罚', 'penalty'),
            ('总计', 'total')
        ]

        for name, key in components:
            value = breakdown.get(key, 0)
            if value != 0:
                emoji = "📈" if value > 0 else "📉" if value < 0 else "➖"
                print(f"  {emoji} {name}: {value:+.3f}")

        print("=" * 40)

    def show_reward_breakdown_simple(self, breakdown: Dict):
        """显示简化奖励分解"""
        components = []
        for key, value in breakdown.items():
            if value != 0 and key != 'total':
                components.append(f"{key}={value:+.3f}")

        if components:
            print(f"   📋 分解: {', '.join(components)}")

    def show_statistics(self):
        """显示统计信息"""
        print("\n📈 会话统计:")
        print(f"  🎮 总episodes: {self.total_episodes}")
        print(f"  ✅ 成功episodes: {self.successful_episodes}")

        if self.total_episodes > 0:
            success_rate = self.successful_episodes / self.total_episodes
            print(f"  📊 成功率: {success_rate:.1%}")

        print(f"  🎯 当前episode: {self.episode_count}")

        if self.env:
            print(f"  👣 当前步数: {self.obs.get('step', 0)}")

            # 环境配置信息
            config = self.env.config
            print("\n⚙️ 环境配置:")
            print(f"  📊 难度: {self.env.difficulty_level}")
            print(f"  👣 最大步数: {self.env.max_steps}")
            print(f"  ⚖️ 奖励权重:")
            for key, value in self.env.reward_weights.items():
                print(f"     {key}: {value}")


def main():
    """主函数"""
    try:
        demo = InteractiveDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断，感谢使用！")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
