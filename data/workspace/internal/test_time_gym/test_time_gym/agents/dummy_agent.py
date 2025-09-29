"""
Dummy Agent实现
用于测试环境和演示基本交互流程
"""

import random
import json
from typing import Dict, List, Any, Optional


class DummyAgent:
    """简单的示例智能体，使用固定策略完成订票任务"""
    
    def __init__(self, strategy: str = "greedy"):
        """
        初始化智能体
        
        Args:
            strategy: 策略类型 ("greedy", "random", "conservative")
        """
        self.strategy = strategy
        self.memory = []  # 存储交互历史
        self.skills = {}  # 技能库（暂时为空）
        
    def select_action(self, observation: Dict) -> str:
        """根据观察选择动作"""
        view = observation["view"]
        
        if view == "search_form":
            return self._handle_search_form(observation)
        elif view == "search_results":
            return self._handle_search_results(observation)
        elif view == "cart":
            return self._handle_cart(observation)
        elif view == "payment":
            return self._handle_payment(observation)
        elif view == "error":
            return self._handle_error(observation)
        else:
            return "restart"
    
    def _handle_search_form(self, obs: Dict) -> str:
        """处理搜索表单视图"""
        return "search_flights"
    
    def _handle_search_results(self, obs: Dict) -> str:
        """处理搜索结果视图"""
        flights = obs.get("flights", [])
        constraints = obs.get("constraints", {})
        
        if not flights:
            return "restart"
            
        if self.strategy == "greedy":
            # 贪心策略：选择价格最低的符合约束的航班
            valid_flights = [
                f for f in flights 
                if f["price"] <= constraints.get("budget", float('inf'))
                and f["stops"] <= constraints.get("max_stops", 2)
            ]
            
            if valid_flights:
                # 已经有符合条件的航班，直接添加到购物车
                return "add_to_cart"
            else:
                # 先筛选再看
                return "filter_results"
                
        elif self.strategy == "random":
            return random.choice(["filter_results", "add_to_cart", "select_flight"])
            
        elif self.strategy == "conservative":
            # 保守策略：总是先筛选
            return "filter_results"
    
    def _handle_cart(self, obs: Dict) -> str:
        """处理购物车视图"""
        cart = obs.get("cart", {})
        constraints = obs.get("constraints", {})
        
        # 检查购物车总价是否在预算内
        if cart.get("total", 0) <= constraints.get("budget", float('inf')):
            return "proceed_to_payment"
        else:
            # 超预算，重新搜索
            return "restart"
    
    def _handle_payment(self, obs: Dict) -> str:
        """处理支付视图"""
        payment_state = obs.get("payment_state", {})
        
        if not payment_state.get("card_entered", False):
            return "enter_card"
        elif not payment_state.get("confirmed", False):
            return "confirm_payment"
        else:
            # 支付已确认，应该已经完成
            return "restart"
    
    def _handle_error(self, obs: Dict) -> str:
        """处理错误状态"""
        return "restart"
    
    def update_memory(self, trajectory: List[Dict]):
        """更新智能体记忆"""
        self.memory.append(trajectory)
        
        # 简单的技能提取：如果轨迹成功，记录动作序列
        if trajectory and trajectory[-1].get("reward", 0) > 0.5:
            actions = [step.get("action", "") for step in trajectory]
            skill_key = "->".join(actions)
            
            if skill_key not in self.skills:
                self.skills[skill_key] = {"count": 0, "success": 0}
            
            self.skills[skill_key]["count"] += 1
            self.skills[skill_key]["success"] += 1
    
    def get_stats(self) -> Dict:
        """获取智能体统计信息"""
        return {
            "total_episodes": len(self.memory),
            "skills_learned": len(self.skills),
            "top_skills": sorted(
                self.skills.items(), 
                key=lambda x: x[1]["success"], 
                reverse=True
            )[:5]
        }


class RandomAgent(DummyAgent):
    """随机动作智能体，用作基线对照"""
    
    def select_action(self, observation: Dict) -> str:
        """随机选择动作"""
        possible_actions = [
            "search_flights", "filter_results", "select_flight",
            "add_to_cart", "proceed_to_payment", "enter_card",
            "confirm_payment", "apply_coupon", "restart", "abort"
        ]
        return random.choice(possible_actions)


class SkillBasedAgent(DummyAgent):
    """基于技能的智能体（为将来的Thompson Sampling做准备）"""
    
    def __init__(self, exploration_rate: float = 0.1):
        super().__init__()
        self.exploration_rate = exploration_rate
        self.skill_stats = {}  # 技能的Beta分布参数
        
    def select_action(self, observation: Dict) -> str:
        """基于技能选择动作（简化版）"""
        # 探索vs利用
        if random.random() < self.exploration_rate:
            # 探索：使用随机策略
            return super().select_action(observation)
        else:
            # 利用：使用最佳已知技能
            return self._exploit_best_skill(observation)
    
    def _exploit_best_skill(self, observation: Dict) -> str:
        """利用最佳技能"""
        # 简化实现：返回到基础策略
        return super().select_action(observation)


if __name__ == "__main__":
    # 测试不同智能体
    from test_time_gym.envs.flight_booking_env import FlightBookingEnv
    
    # 创建环境
    env = FlightBookingEnv(seed=42)
    
    # 测试不同策略的智能体
    agents = {
        "greedy": DummyAgent("greedy"),
        "random": DummyAgent("random"),
        "conservative": DummyAgent("conservative")
    }
    
    for name, agent in agents.items():
        print(f"\n=== Testing {name} agent ===")
        
        obs, info = env.reset(seed=42)  # 使用相同种子保证公平
        trajectory = []
        total_reward = 0
        
        for step in range(20):  # 最多20步
            action = agent.select_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            
            total_reward += reward
            trajectory.append({
                "step": step,
                "action": action,
                "reward": reward,
                "obs": obs
            })
            
            print(f"Step {step}: {action} -> reward={reward:.3f}")
            
            if done or trunc:
                break
        
        agent.update_memory(trajectory)
        print(f"Total reward: {total_reward:.3f}")
        print(f"Agent stats: {agent.get_stats()}")