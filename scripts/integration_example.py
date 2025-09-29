#!/usr/bin/env python3
"""
集成示例 - 将观测系统集成到现有实验框架
"""

import asyncio
import logging
from typing import Dict, Optional

# 现有实验框架导入
from experiment_framework import ExperimentRunner
from enhanced_skill_system import EnhancedSkillManager

# 观测系统导入
from observation_system import ObservationSystem


class SimpleObservableExperiment:
    """简单的可观测实验示例"""
    
    def __init__(self, enable_observation: bool = True):
        # 创建观测系统
        self.observation_system = None
        if enable_observation:
            self.observation_system = ObservationSystem(
                enable_web=True, 
                web_port=5000
            )
            print("🔬 观测系统已启用")
        
        # 其他组件
        self.skill_manager = EnhancedSkillManager("logs/skills")
        self.episode_count = 0
    
    async def run_simple_experiment(self, num_episodes: int = 20):
        """运行简单的可观测实验"""
        
        experiment_name = "simple_observable_experiment"
        
        # 启动观测
        if self.observation_system:
            self.observation_system.start_monitoring([experiment_name])
            print("📊 监控已启动，访问 http://localhost:5000 查看Web界面")
        
        print(f"🚀 开始实验: {experiment_name}")
        
        try:
            for episode in range(num_episodes):
                await self._run_episode(experiment_name, episode)
                
                # 定期报告
                if (episode + 1) % 5 == 0:
                    print(f"📈 已完成 {episode + 1}/{num_episodes} episodes")
            
            print("✅ 实验完成!")
            
            # 生成报告
            if self.observation_system:
                self.observation_system.generate_final_report("logs/simple_reports")
                print("📊 观测报告已生成")
                
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            raise
        finally:
            # 清理
            if self.observation_system:
                self.observation_system.cleanup()
    
    async def _run_episode(self, experiment_name: str, episode_num: int):
        """运行单个episode"""
        import random
        import time
        
        episode_id = f"ep_{episode_num:03d}"
        
        # 模拟初始状态
        initial_obs = {
            'view': 'search_form',
            'budget': random.randint(500, 1000),
            'time_preference': random.choice(['morning', 'afternoon', 'evening'])
        }
        
        # 记录episode开始
        if self.observation_system:
            self.observation_system.log_episode_start(experiment_name, episode_id, initial_obs)
        
        # 模拟episode执行
        total_reward = 0.0
        steps = random.randint(5, 15)
        
        for step in range(steps):
            # 模拟动作选择
            action = self._select_action(initial_obs, step)
            
            # 模拟环境响应
            obs, reward = self._simulate_step(action, step)
            total_reward += reward
            
            # 技能使用检查
            skill_used = self._check_skill_usage(action, obs)
            
            # 记录步骤
            if self.observation_system:
                self.observation_system.log_step(
                    experiment_name, episode_id, step,
                    action, obs, reward, skill_used
                )
            
            # 模拟延迟
            await asyncio.sleep(0.05)
        
        # 判断成功
        success = total_reward > 0.5
        
        # 记录episode结束
        if self.observation_system:
            self.observation_system.log_episode_end(
                experiment_name, episode_id, total_reward, success
            )
        
        # 更新技能（模拟）
        if success and random.random() < 0.2:  # 20%概率学习新技能
            self._simulate_skill_learning(experiment_name)
    
    def _select_action(self, obs: Dict, step: int) -> str:
        """模拟动作选择"""
        import random
        
        actions = [
            'search_flights', 'filter_results', 'add_to_cart',
            'proceed_to_payment', 'enter_card', 'confirm_payment'
        ]
        
        # 简单的顺序逻辑
        if step < 2:
            return 'search_flights'
        elif step < 4:
            return random.choice(['filter_results', 'add_to_cart'])
        else:
            return random.choice(['proceed_to_payment', 'enter_card', 'confirm_payment'])
    
    def _simulate_step(self, action: str, step: int) -> tuple:
        """模拟环境步骤"""
        import random
        
        # 模拟观察
        views = ['search_form', 'search_results', 'cart', 'payment', 'receipt']
        obs = {
            'view': views[min(step // 3, len(views) - 1)],
            'step': step,
            'action_result': 'success' if random.random() > 0.1 else 'partial'
        }
        
        # 模拟奖励
        base_reward = 0.1
        if action == 'confirm_payment' and step > 5:
            reward = 1.0  # 完成奖励
        else:
            reward = random.uniform(-0.05, base_reward)
        
        return obs, reward
    
    def _check_skill_usage(self, action: str, obs: Dict) -> Optional[str]:
        """检查是否使用了技能"""
        import random
        
        # 模拟技能使用检测
        if random.random() < 0.3:  # 30%概率使用技能
            skills = ['Quick Search', 'Smart Filter', 'Fast Payment']
            return random.choice(skills)
        
        return None
    
    def _simulate_skill_learning(self, experiment_name: str):
        """模拟技能学习"""
        import random
        
        skill_names = [
            'Efficient Search', 'Smart Filtering', 'Quick Payment',
            'Budget Optimization', 'Error Recovery'
        ]
        
        skill_name = random.choice(skill_names)
        success_rate = random.uniform(0.4, 0.9)
        
        if self.observation_system:
            self.observation_system.log_skill_learned(
                experiment_name, skill_name, success_rate
            )


class EnhancedObservableExperiment:
    """增强版可观测实验 - 集成现有框架"""
    
    def __init__(self, results_dir: str = "logs/enhanced_observable"):
        # 创建增强的实验运行器
        from observable_experiment_runner import ObservableExperimentRunner
        
        self.runner = ObservableExperimentRunner(
            results_dir=results_dir,
            enable_observation=True,
            web_port=5001  # 使用不同端口避免冲突
        )
        
        print("🔬 增强观测系统已创建")
        print("📊 Web界面: http://localhost:5001")
    
    async def run_comparative_experiment(self):
        """运行对比实验"""
        print("🚀 运行增强版可观测对比实验...")
        
        try:
            results = await self.runner.run_comparative_experiment(
                num_episodes=30,  # 较少的episodes用于演示
                models=["gpt-3.5-turbo"],
                strategies=["balanced"]
            )
            
            # 分析结果
            self._analyze_results(results)
            
        except Exception as e:
            print(f"❌ 增强实验失败: {e}")
            raise
        finally:
            self.runner.cleanup()
    
    def _analyze_results(self, results: Dict):
        """分析实验结果"""
        print("\n📊 实验结果分析:")
        
        comparison = results.get("comparison_report", {})
        overall = comparison.get("overall_improvement", {})
        
        if overall:
            print(f"• 配置数量: {overall.get('configurations_tested', 0)}")
            print(f"• 有改进配置: {overall.get('improvements_positive', 0)}")
            print(f"• 成功率改进: {overall.get('avg_success_rate_improvement', 0):.3f}")
            print(f"• 效率改进: {overall.get('avg_steps_improvement', 0):.3f}")
        
        # 详细对比
        for config, details in comparison.get("detailed_comparisons", {}).items():
            print(f"\n🔧 {config}:")
            print(f"  成功率: {details.get('baseline_success_rate', 0):.3f} → "
                  f"{details.get('experience_success_rate', 0):.3f}")
            print(f"  技能数: {details.get('total_skills_learned', 0)}")


async def main():
    """主演示函数"""
    print("🎯 观测系统集成示例")
    print("=" * 50)
    
    # 示例1: 简单集成
    print("\n1️⃣ 简单观测集成示例")
    print("-" * 30)
    
    simple_exp = SimpleObservableExperiment(enable_observation=True)
    await simple_exp.run_simple_experiment(num_episodes=10)
    
    print("\n⏸️ 等待5秒后继续...")
    await asyncio.sleep(5)
    
    # 示例2: 增强集成（可选，避免资源冲突）
    run_enhanced = input("\n运行增强版实验? (y/N): ").lower().strip() == 'y'
    
    if run_enhanced:
        print("\n2️⃣ 增强观测集成示例")
        print("-" * 30)
        
        enhanced_exp = EnhancedObservableExperiment()
        await enhanced_exp.run_comparative_experiment()
    
    print("\n🎉 集成示例完成!")
    print("💡 查看生成的报告和日志文件以了解更多细节")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔬 观测系统集成演示")
    print("=" * 60)
    print("📋 本示例展示如何将观测系统集成到现有实验框架")
    print("🌐 Web界面将在实验运行时自动启动")
    print("💡 建议在运行前安装依赖: ./install_observation_deps.sh")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        print("💡 检查依赖安装和配置文件")