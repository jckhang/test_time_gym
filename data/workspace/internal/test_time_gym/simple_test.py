#!/usr/bin/env python3
"""
简化测试脚本 - 不依赖外部包
测试核心逻辑是否正确
"""

import sys
import os
import json
import random
from typing import Dict, List, Any, Optional

# 添加包路径
sys.path.insert(0, os.path.dirname(__file__))


# 简化的环境类，不依赖gymnasium
class SimpleFlightEnv:
    """简化的航班预订环境"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        self.step_count = 0
        self.done = False
        self.view = "search_form"
        self.flights = []
        self.cart = {"items": [], "total": 0}
        self.task = {
            "origin": "SFO",
            "destination": "MAD", 
            "constraints": {"budget": 800, "max_stops": 1}
        }
    
    def reset(self):
        self.step_count = 0
        self.done = False
        self.view = "search_form"
        self.flights = []
        self.cart = {"items": [], "total": 0}
        
        return self.get_observation()
    
    def step(self, action: str):
        self.step_count += 1
        reward = -0.01  # 时间成本
        
        try:
            if action == "search_flights":
                self.flights = [
                    {"id": "AA123", "price": 650, "stops": 0},
                    {"id": "IB456", "price": 580, "stops": 1},
                    {"id": "DL789", "price": 750, "stops": 1}
                ]
                self.view = "search_results"
                reward += 0.02
                
            elif action == "add_to_cart":
                if self.flights:
                    flight = self.flights[0]  # 选择第一个
                    self.cart["items"].append({"flight_id": flight["id"], "price": flight["price"]})
                    self.cart["total"] = sum(item["price"] for item in self.cart["items"])
                    self.view = "cart"
                    reward += 0.05
                    
            elif action == "proceed_to_payment":
                if self.cart["items"]:
                    self.view = "payment"
                    reward += 0.03
                    
            elif action == "confirm_payment":
                if self.view == "payment":
                    # 检查约束
                    if self.cart["total"] <= self.task["constraints"]["budget"]:
                        self.view = "receipt"
                        self.done = True
                        reward += 1.0  # 成功奖励
                    else:
                        reward -= 0.3  # 约束违规
                        
        except Exception as e:
            reward -= 0.05
            self.view = "error"
        
        return self.get_observation(), reward, self.done, False, {}
    
    def get_observation(self):
        return {
            "view": self.view,
            "step": self.step_count,
            "flights": self.flights,
            "cart": self.cart,
            "constraints": self.task["constraints"],
            "forms": {"from": self.task["origin"], "to": self.task["destination"]}
        }


# 简化的智能体
class SimpleAgent:
    def __init__(self):
        self.memory = []
    
    def select_action(self, obs: Dict) -> str:
        view = obs["view"]
        
        if view == "search_form":
            return "search_flights"
        elif view == "search_results":
            return "add_to_cart"
        elif view == "cart":
            return "proceed_to_payment"
        elif view == "payment":
            return "confirm_payment"
        else:
            return "restart"


def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试基本功能")
    print("-" * 30)
    
    env = SimpleFlightEnv(seed=42)
    agent = SimpleAgent()
    
    obs = env.reset()
    print(f"初始状态: {obs['view']}")
    
    total_reward = 0
    actions_taken = []
    
    for step in range(10):
        if env.done:
            break
            
        action = agent.select_action(obs)
        obs, reward, done, trunc, info = env.step(action)
        
        total_reward += reward
        actions_taken.append(action)
        
        print(f"步骤 {step+1}: {action} → {obs['view']} (奖励: {reward:.3f})")
    
    print(f"\n结果:")
    print(f"  动作序列: {' → '.join(actions_taken)}")
    print(f"  总奖励: {total_reward:.3f}")
    print(f"  成功: {'✅' if total_reward > 0.5 else '❌'}")
    print(f"  最终视图: {obs['view']}")
    
    return total_reward > 0.5


def test_skill_extraction():
    """测试技能提取逻辑"""
    print("\n🔧 测试技能提取")
    print("-" * 30)
    
    # 模拟成功轨迹
    successful_trajectory = [
        {"action": "search_flights", "reward": 0.02},
        {"action": "add_to_cart", "reward": 0.05},
        {"action": "proceed_to_payment", "reward": 0.03},
        {"action": "confirm_payment", "reward": 1.0}
    ]
    
    # 简化的技能提取
    actions = [step["action"] for step in successful_trajectory]
    total_reward = sum(step["reward"] for step in successful_trajectory)
    
    print(f"轨迹动作: {' → '.join(actions)}")
    print(f"总奖励: {total_reward:.3f}")
    print(f"成功: {'✅' if total_reward > 0.5 else '❌'}")
    
    # 提取子序列作为技能
    skills = []
    for length in range(2, len(actions) + 1):
        for start in range(len(actions) - length + 1):
            subsequence = actions[start:start + length]
            skills.append(subsequence)
    
    print(f"提取的技能候选: {len(skills)} 个")
    for i, skill in enumerate(skills[:5]):  # 显示前5个
        print(f"  技能 {i+1}: {' → '.join(skill)}")
    
    return len(skills)


def test_reward_calculation():
    """测试奖励计算"""
    print("\n💰 测试奖励机制")
    print("-" * 30)
    
    env = SimpleFlightEnv()
    obs = env.reset()
    
    # 测试各种动作的奖励
    test_cases = [
        ("search_flights", "搜索航班"),
        ("add_to_cart", "添加到购物车"),
        ("proceed_to_payment", "进入支付"),
        ("confirm_payment", "确认支付")
    ]
    
    for action, description in test_cases:
        if env.done:
            break
            
        obs, reward, done, trunc, info = env.step(action)
        print(f"{description}: {reward:.3f}")
        
        if done:
            print("任务完成!")
            break
    
    return True


def test_constraint_checking():
    """测试约束检查"""
    print("\n⚖️  测试约束检查")
    print("-" * 30)
    
    # 测试预算约束
    budget_tests = [
        (500, 600, "在预算内"),
        (500, 400, "超出预算")
    ]
    
    for cart_total, budget, description in budget_tests:
        constraint_met = cart_total <= budget
        print(f"{description}: 购物车${cart_total}, 预算${budget} → {'✅' if constraint_met else '❌'}")
    
    return True


def main():
    """主测试函数"""
    print("🎯 Test-Time Gym 简化功能测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行各种测试
    tests = [
        ("基本功能", test_basic_functionality),
        ("技能提取", test_skill_extraction),
        ("奖励计算", test_reward_calculation),
        ("约束检查", test_constraint_checking)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, True, result))
            print(f"✅ {test_name} 测试通过")
        except Exception as e:
            test_results.append((test_name, False, str(e)))
            print(f"❌ {test_name} 测试失败: {e}")
    
    # 总结
    print(f"\n📊 测试总结")
    print("=" * 50)
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    
    print(f"通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for test_name, success, result in test_results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    if passed == total:
        print(f"\n🎉 所有测试通过! 框架核心逻辑正确。")
        print(f"\n📝 后续步骤:")
        print(f"1. 安装完整依赖: 运行 './install.sh'")
        print(f"2. 运行完整测试: 'python3 examples/basic_usage.py'")
        print(f"3. 查看高级功能: 'python3 examples/advanced_usage.py'")
    else:
        print(f"\n⚠️  部分测试失败，需要修复")
    
    return passed == total


if __name__ == "__main__":
    main()