#!/usr/bin/env python3
"""
系统测试脚本 - 验证无监督经验积累系统的基本功能
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试所有必要的导入"""
    print("🧪 测试模块导入...")
    
    try:
        # 测试项目导入
        from test_time_gym.envs.flight_booking_env import FlightBookingEnv
        print("  ✅ 环境模块导入成功")
    except ImportError as e:
        print(f"  ❌ 环境模块导入失败: {e}")
        return False
    
    try:
        from test_time_gym.agents.openai_agent import FlightBookingOpenAIAgent
        print("  ✅ 智能体模块导入成功")
    except ImportError as e:
        print(f"  ❌ 智能体模块导入失败: {e}")
        return False
    
    try:
        from enhanced_skill_system import EnhancedSkillManager, SemanticSkill
        print("  ✅ 增强技能系统导入成功")
    except ImportError as e:
        print(f"  ❌ 增强技能系统导入失败: {e}")
        return False
    
    try:
        from experiment_framework import ExperimentRunner, ExperienceEnhancedAgent
        print("  ✅ 实验框架导入成功")
    except ImportError as e:
        print(f"  ❌ 实验框架导入失败: {e}")
        return False
    
    return True


def test_environment():
    """测试环境基本功能"""
    print("\n🧪 测试环境基本功能...")
    
    try:
        from test_time_gym.envs.flight_booking_env import FlightBookingEnv
        
        # 创建环境
        env = FlightBookingEnv(seed=42)
        print("  ✅ 环境创建成功")
        
        # 重置环境
        obs, info = env.reset()
        print(f"  ✅ 环境重置成功，初始观察: {obs.get('view', 'unknown')}")
        
        # 执行一步
        obs, reward, done, trunc, info = env.step("search_flights")
        print(f"  ✅ 环境步进成功，奖励: {reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 环境测试失败: {e}")
        return False


def test_skill_system():
    """测试技能系统"""
    print("\n🧪 测试技能系统...")
    
    try:
        from enhanced_skill_system import EnhancedSkillManager
        
        # 创建技能管理器
        skill_manager = EnhancedSkillManager("logs/test_skills")
        print("  ✅ 技能管理器创建成功")
        
        # 模拟轨迹
        trajectory = [
            {
                "action": "search_flights",
                "obs": {"view": "search_form", "flights": [], "cart": {"total": 0}},
                "reward": 0.02
            },
            {
                "action": "add_to_cart", 
                "obs": {"view": "search_results", "flights": [{"id": "AA123", "price": 600}]},
                "reward": 0.05
            }
        ]
        
        # 处理轨迹
        skill_manager.process_episode(trajectory, final_reward=1.0, episode_id="test_001")
        print("  ✅ 轨迹处理成功")
        
        # 获取统计
        stats = skill_manager.get_skill_analytics()
        print(f"  ✅ 技能统计: {stats.get('total_skills', 0)} 个技能")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 技能系统测试失败: {e}")
        return False


async def test_agent():
    """测试智能体"""
    print("\n🧪 测试智能体...")
    
    try:
        from test_time_gym.envs.flight_booking_env import FlightBookingEnv
        from experiment_framework import ExperienceEnhancedAgent
        from enhanced_skill_system import EnhancedSkillManager
        
        # 创建环境和智能体
        env = FlightBookingEnv(seed=42)
        skill_manager = EnhancedSkillManager("logs/test_agent_skills")
        agent = ExperienceEnhancedAgent(
            model="claude-3-haiku",
            strategy="balanced",
            skill_manager=skill_manager,
            use_experience=False  # 先测试基础功能
        )
        print("  ✅ 智能体创建成功")
        
        # 重置环境
        obs, info = env.reset()
        
        # 智能体选择动作
        action = await agent.select_action(obs)
        print(f"  ✅ 智能体动作选择成功: {action}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 智能体测试失败: {e}")
        return False


async def test_simple_episode():
    """测试简单episode运行"""
    print("\n🧪 测试简单episode运行...")
    
    try:
        from test_time_gym.envs.flight_booking_env import FlightBookingEnv
        from experiment_framework import ExperienceEnhancedAgent
        from enhanced_skill_system import EnhancedSkillManager
        
        # 创建环境和智能体
        env = FlightBookingEnv(seed=42)
        skill_manager = EnhancedSkillManager("logs/test_episode_skills")
        agent = ExperienceEnhancedAgent(
            model="claude-3-haiku",
            strategy="balanced",
            skill_manager=skill_manager,
            use_experience=True
        )
        
        # 运行简单episode
        obs, info = env.reset()
        total_reward = 0.0
        
        for step in range(5):  # 只运行5步
            try:
                action = await agent.select_action(obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                
                print(f"    步骤 {step}: {action} -> 奖励 {reward:.3f}")
                
                if done or trunc:
                    break
                    
            except Exception as e:
                print(f"    步骤 {step} 出错: {e}")
                break
        
        # 结束episode
        agent.end_episode(total_reward, "test_episode")
        
        print(f"  ✅ Episode 运行成功，总奖励: {total_reward:.3f}")
        return True
        
    except Exception as e:
        print(f"  ❌ Episode 测试失败: {e}")
        return False


def test_data_persistence():
    """测试数据持久化"""
    print("\n🧪 测试数据持久化...")
    
    try:
        from enhanced_skill_system import EnhancedSkillManager
        
        # 创建并保存技能
        skill_manager1 = EnhancedSkillManager("logs/test_persistence")
        
        # 添加一些测试数据
        trajectory = [
            {
                "action": "search_flights",
                "obs": {"view": "search_form", "flights": []},
                "reward": 0.02
            }
        ]
        skill_manager1.process_episode(trajectory, 1.0, "persist_test")
        skill_manager1.save_skills()
        
        # 创建新实例并加载
        skill_manager2 = EnhancedSkillManager("logs/test_persistence")
        
        stats1 = skill_manager1.get_skill_analytics()
        stats2 = skill_manager2.get_skill_analytics()
        
        print(f"  ✅ 保存前技能数: {stats1.get('total_skills', 0)}")
        print(f"  ✅ 加载后技能数: {stats2.get('total_skills', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 持久化测试失败: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("🚀 开始系统测试\n")
    
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    tests = [
        ("模块导入", test_imports),
        ("环境功能", test_environment),
        ("技能系统", test_skill_system),
        ("智能体", test_agent),
        ("Episode运行", test_simple_episode),
        ("数据持久化", test_data_persistence)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ 测试 {test_name} 发生异常: {e}")
            results.append((test_name, False))
    
    # 打印测试摘要
    print(f"\n{'='*60}")
    print("📊 测试结果摘要")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
        print("\n💡 下一步:")
        print("  1. 运行快速实验: python run_experiment.py --experiment quick_test")
        print("  2. 运行完整实验: python run_experiment.py --experiment full_comparison")
    else:
        print("⚠️ 存在失败的测试，请检查相关组件。")
    
    return passed == total


if __name__ == "__main__":
    print("🔧 无监督经验积累系统测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行测试
    success = asyncio.run(run_all_tests())
    
    # 退出码
    sys.exit(0 if success else 1)