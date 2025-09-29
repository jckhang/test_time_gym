#!/usr/bin/env python3
"""
观测系统演示脚本
快速展示观测系统的核心功能
"""

import asyncio
import logging
import random
import time
from typing import Dict

from observation_system import ObservationSystem


async def simulate_experiment_data(obs_system: ObservationSystem, 
                                 experiment_name: str, 
                                 num_episodes: int = 20):
    """模拟实验数据来演示观测系统"""
    
    print(f"🎮 开始模拟实验: {experiment_name}")
    
    # 模拟实验参数
    base_success_rate = 0.3 if "baseline" in experiment_name else 0.4
    learning_rate = 0.02 if "experience" in experiment_name else 0.005
    
    skills_learned = []
    
    for episode in range(num_episodes):
        episode_id = f"ep_{episode:03d}"
        
        # 模拟学习效果
        current_success_rate = min(0.9, base_success_rate + episode * learning_rate)
        
        # 开始episode
        initial_obs = {
            'view': 'search_form',
            'constraints': {
                'budget': random.randint(500, 1200),
                'max_stops': random.choice([0, 1, 2])
            }
        }
        
        obs_system.log_episode_start(experiment_name, episode_id, initial_obs)
        
        # 模拟episode执行
        total_reward = 0.0
        num_steps = random.randint(8, 20)
        
        # 偶尔学习新技能
        if "experience" in experiment_name and random.random() < 0.15:
            skill_name = f"Skill_{len(skills_learned)+1}"
            if skill_name not in skills_learned:
                skills_learned.append(skill_name)
                success_rate = random.uniform(0.4, 0.8)
                obs_system.log_skill_learned(experiment_name, skill_name, success_rate)
                print(f"  🧠 学到新技能: {skill_name} (成功率: {success_rate:.3f})")
        
        for step in range(num_steps):
            # 模拟动作选择
            actions = ['search_flights', 'filter_results', 'add_to_cart', 'proceed_to_payment', 'enter_card', 'confirm_payment']
            action = random.choice(actions)
            
            # 模拟观察
            observation = {
                'view': random.choice(['search_form', 'search_results', 'cart', 'payment', 'receipt']),
                'step': step
            }
            
            # 模拟奖励
            step_reward = random.uniform(-0.1, 0.3)
            total_reward += step_reward
            
            # 技能使用
            skill_used = None
            if skills_learned and random.random() < 0.3:
                skill_used = random.choice(skills_learned)
                obs_system.log_skill_usage(experiment_name, skill_used, random.random() > 0.3)
            
            # 记录步骤
            obs_system.log_step(
                experiment_name, episode_id, step,
                action, observation, step_reward, skill_used
            )
            
            # 模拟执行延迟
            await asyncio.sleep(0.1)
        
        # 结束episode
        success = random.random() < current_success_rate
        if success:
            total_reward += 1.0  # 成功奖励
        
        obs_system.log_episode_end(experiment_name, episode_id, total_reward, success)
        
        # 进度报告
        if (episode + 1) % 5 == 0:
            print(f"  📊 已完成 {episode + 1}/{num_episodes} episodes, 当前成功率: {current_success_rate:.3f}")
        
        # episode间隔
        await asyncio.sleep(0.5)
    
    print(f"✅ {experiment_name} 模拟完成! 学到 {len(skills_learned)} 个技能")


async def run_observation_demo():
    """运行观测系统演示"""
    
    print("🔬 观测系统演示")
    print("=" * 60)
    
    # 创建观测系统
    obs_system = ObservationSystem(enable_web=True, web_port=5000)
    
    # 实验配置
    experiments = [
        "demo_baseline_experiment",
        "demo_experience_experiment"
    ]
    
    # 启动监控
    obs_system.start_monitoring(experiments)
    
    print("\n🌐 Web仪表板已启动: http://localhost:5000")
    print("💡 打开浏览器查看实时监控界面")
    print("\n⏳ 等待3秒后开始模拟实验...")
    await asyncio.sleep(3)
    
    try:
        # 并行运行多个实验
        tasks = []
        for exp_name in experiments:
            task = asyncio.create_task(
                simulate_experiment_data(obs_system, exp_name, num_episodes=15)
            )
            tasks.append(task)
        
        # 等待所有实验完成
        await asyncio.gather(*tasks)
        
        print("\n🎉 所有演示实验完成!")
        print("📊 观测数据已收集完毕")
        
        # 额外等待时间来观察最终状态
        print("\n⏱️ 保持系统运行30秒以供观察...")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 生成最终报告
        print("\n📝 生成最终报告...")
        obs_system.generate_final_report("logs/demo_reports")
        
        # 清理
        obs_system.cleanup()
        print("🧹 清理完成")


def print_demo_info():
    """打印演示信息"""
    info = """
🎯 演示内容:
  • 模拟两个对比实验: 基线 vs 经验学习
  • 实时监控episode进度和成功率
  • 技能学习过程可视化
  • Web仪表板实时更新

📊 观测要点:
  • 对比两个实验的成功率趋势
  • 观察技能学习的时间线
  • 查看实时统计和图表
  • 体验Web界面的交互功能

🌐 Web界面功能:
  • 实时指标监控
  • 动态图表更新
  • 技能分析面板
  • 实时日志显示

💡 使用建议:
  1. 先启动演示脚本
  2. 打开浏览器访问 http://localhost:5000
  3. 观察实时数据更新
  4. 查看生成的报告文件
"""
    print(info)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print_demo_info()
    
    try:
        asyncio.run(run_observation_demo())
    except KeyboardInterrupt:
        print("\n👋 感谢使用观测系统演示!")
    except Exception as e:
        print(f"\n❌ 演示启动失败: {e}")
        print("💡 请确保已安装所需依赖: pip install flask flask-socketio matplotlib")