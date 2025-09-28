#!/usr/bin/env python3
"""
快速演示脚本
展示Test-Time Gym框架的基本功能
"""

import os
import sys

# 确保能导入我们的包
sys.path.insert(0, os.path.dirname(__file__))

from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.agents.dummy_agent import DummyAgent, RandomAgent


def quick_demo():
    """快速演示"""
    print("🚀 Test-Time Gym 快速演示")
    print("=" * 50)
    
    # 创建环境和智能体
    env = FlightBookingEnv(seed=42)
    agent = DummyAgent("greedy")
    
    print("1. 初始化环境...")
    obs, info = env.reset()
    print(f"   任务: {obs['forms']['from']} → {obs['forms']['to']}")
    print(f"   约束: 预算=${obs['constraints']['budget']}, 最大中转{obs['constraints']['max_stops']}次")
    
    print("\n2. 智能体开始行动...")
    total_reward = 0
    step_count = 0
    
    while not (env.done or env.truncated) and step_count < 15:
        # 智能体选择动作
        action = agent.select_action(obs)
        
        # 执行动作
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"   步骤 {step_count}: {action} → 奖励 {reward:.3f}")
        
        # 显示重要状态变化
        if obs["view"] == "search_results" and obs.get("flights"):
            print(f"      → 找到 {len(obs['flights'])} 个航班")
        elif obs["view"] == "cart" and obs["cart"]["total"] > 0:
            print(f"      → 购物车总额: ${obs['cart']['total']}")
        elif obs["view"] == "payment":
            print(f"      → 进入支付页面")
        elif obs["view"] == "receipt":
            print(f"      → 🎉 支付成功!")
    
    print(f"\n3. 结果总结:")
    print(f"   总步数: {step_count}")
    print(f"   总奖励: {total_reward:.3f}")
    print(f"   任务状态: {'✅ 成功' if total_reward > 0.5 else '❌ 失败'}")
    
    return total_reward > 0.5


def multi_agent_comparison():
    """多智能体对比演示"""
    print("\n🤖 多智能体性能对比")
    print("=" * 50)
    
    agents = {
        "贪心智能体": DummyAgent("greedy"),
        "保守智能体": DummyAgent("conservative"), 
        "随机智能体": RandomAgent()
    }
    
    results = {}
    
    for name, agent in agents.items():
        print(f"\n测试 {name}...")
        
        successes = 0
        total_steps = 0
        total_rewards = 0
        episodes = 5
        
        for episode in range(episodes):
            env = FlightBookingEnv(seed=42 + episode)
            obs, info = env.reset()
            
            episode_reward = 0
            steps = 0
            
            while not (env.done or env.truncated) and steps < 20:
                action = agent.select_action(obs)
                obs, reward, done, trunc, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            if episode_reward > 0.5:
                successes += 1
                total_steps += steps
                
            total_rewards += episode_reward
        
        results[name] = {
            "成功率": successes / episodes,
            "平均奖励": total_rewards / episodes,
            "平均步数": total_steps / max(successes, 1)
        }
        
        print(f"   成功率: {results[name]['成功率']:.1%}")
        print(f"   平均奖励: {results[name]['平均奖励']:.3f}")
        print(f"   平均步数: {results[name]['平均步数']:.1f}")
    
    # 找出最佳智能体
    best_agent = max(results.items(), key=lambda x: x[1]["成功率"])
    print(f"\n🏆 最佳智能体: {best_agent[0]} (成功率: {best_agent[1]['成功率']:.1%})")
    
    return results


def skill_learning_demo():
    """技能学习演示"""
    print("\n🧠 技能学习演示")
    print("=" * 50)
    
    try:
        from test_time_gym.utils.skill_system import SkillManager
        
        skill_manager = SkillManager()
        env = FlightBookingEnv(seed=42)
        agent = DummyAgent("greedy")
        
        print("让智能体通过多次尝试学习技能...")
        
        for episode in range(10):
            obs, info = env.reset(seed=42 + episode)
            
            trajectory = []
            total_reward = 0
            
            while not (env.done or env.truncated) and len(trajectory) < 20:
                action = agent.select_action(obs)
                obs, reward, done, trunc, info = env.step(action)
                
                total_reward += reward
                trajectory.append({
                    "action": action,
                    "obs": obs,
                    "reward": reward
                })
                
                if done:
                    break
            
            # 添加到技能管理器
            skill_manager.add_trajectory(trajectory, total_reward)
            
            if episode % 3 == 0:
                stats = skill_manager.get_skill_stats()
                print(f"   Episode {episode}: 已学习 {stats['total_skills']} 个技能")
        
        # 展示学到的技能
        print(f"\n📚 最终技能库:")
        stats = skill_manager.get_skill_stats()
        print(f"   总技能数: {stats['total_skills']}")
        
        if stats['total_skills'] > 0:
            print(f"   平均可靠性: {stats.get('avg_reliability', 0):.3f}")
            
            best_skills = skill_manager.get_best_skills(3)
            print(f"   前3个最佳技能:")
            for skill in best_skills:
                print(f"     • {' → '.join(skill.action_sequence)} "
                      f"(可靠性: {skill.get_confidence():.3f})")
        
        return len(skill_manager.skills)
        
    except ImportError as e:
        print(f"技能系统导入失败: {e}")
        return 0


def main():
    """主演示函数"""
    print("🎯 Test-Time Gym 完整演示")
    print("=" * 60)
    
    try:
        # 1. 基础功能演示
        success = quick_demo()
        
        # 2. 智能体对比
        comparison_results = multi_agent_comparison()
        
        # 3. 技能学习演示  
        skill_count = skill_learning_demo()
        
        # 4. 总结
        print(f"\n🎉 演示完成!")
        print(f"=" * 60)
        print(f"基础演示: {'✅ 成功' if success else '❌ 失败'}")
        print(f"智能体对比: ✅ 完成 ({len(comparison_results)} 个智能体)")
        print(f"技能学习: ✅ 学习了 {skill_count} 个技能")
        
        print(f"\n📖 后续步骤:")
        print(f"• 查看 /workspace/logs/ 目录的生成文件")
        print(f"• 运行 'python examples/basic_usage.py' 进行更详细的测试")
        print(f"• 运行 'python examples/advanced_usage.py' 查看高级功能")
        print(f"• 使用 'python -m test_time_gym.cli --help' 查看命令行工具")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()