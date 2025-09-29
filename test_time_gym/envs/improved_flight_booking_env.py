"""
改进的机票预订环境实现
减少随机性，提供更好的学习反馈机制
"""

import copy
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class FlightBookingView(Enum):
    """环境视图状态"""
    SEARCH_FORM = "search_form"
    SEARCH_RESULTS = "search_results"
    CART = "cart"
    PAYMENT = "payment"
    RECEIPT = "receipt"
    ERROR = "error"


class SkillType(Enum):
    """技能类型枚举"""
    SEARCH_EFFICIENCY = "search_efficiency"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    COST_OPTIMIZATION = "cost_optimization"
    ERROR_RECOVERY = "error_recovery"
    PROCESS_COMPLETION = "process_completion"


@dataclass
class Flight:
    """航班信息"""
    id: str
    price: float
    depart_time: str
    arrive_time: str
    stops: int
    airline: str
    origin: str
    destination: str
    available: bool = True
    demand_level: float = 0.5  # 需求水平 0-1，影响价格变化

    def to_dict(self):
        return {
            "id": self.id,
            "price": self.price,
            "depart": self.depart_time,
            "arrive": self.arrive_time,
            "stops": self.stops,
            "airline": self.airline,
            "origin": self.origin,
            "destination": self.destination,
            "demand": self.demand_level
        }


@dataclass
class RewardBreakdown:
    """奖励分解"""
    base_action: float = 0.0
    progress: float = 0.0
    constraint_satisfaction: float = 0.0
    efficiency: float = 0.0
    optimization: float = 0.0
    penalty: float = 0.0
    total: float = 0.0
    
    def calculate_total(self):
        self.total = (self.base_action + self.progress + 
                     self.constraint_satisfaction + self.efficiency + 
                     self.optimization + self.penalty)
        return self.total


@dataclass
class SkillMetrics:
    """技能指标跟踪"""
    search_attempts: int = 0
    successful_searches: int = 0
    constraint_violations: int = 0
    budget_efficiency: float = 0.0  # 预算利用效率
    time_efficiency: float = 0.0    # 时间效率
    error_recovery_count: int = 0


class ImprovedFlightBookingEnv(gym.Env):
    """改进的机票预订环境"""

    def __init__(self, seed: Optional[int] = None, config: Optional[Dict] = None):
        super().__init__()

        # 设置随机种子
        if seed is not None:
            self.seed(seed)

        # 环境配置
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 50)
        self.difficulty_level = self.config.get("difficulty", "medium")  # easy, medium, hard
        self.enable_dynamic_pricing = self.config.get("dynamic_pricing", True)
        
        # 奖励系统配置
        self.reward_weights = {
            "progress": self.config.get("progress_weight", 0.1),
            "constraint": self.config.get("constraint_weight", 0.3),
            "efficiency": self.config.get("efficiency_weight", 0.2),
            "optimization": self.config.get("optimization_weight", 0.2),
            "completion": self.config.get("completion_weight", 1.0)
        }

        # 环境状态
        self.current_view = FlightBookingView.SEARCH_FORM
        self.step_count = 0
        self.current_task = None
        self.flights_db = self._generate_flights_db()
        self.current_flights = []
        self.cart = {"items": [], "total": 0}
        self.payment_state = {"attempts": 0, "card_entered": False, "confirmed": False}
        self.messages = []
        self.done = False
        self.truncated = False
        
        # 技能和学习跟踪
        self.skill_metrics = SkillMetrics()
        self.session_history = []
        self.constraint_satisfaction_score = 0.0
        
        # 动作空间和观察空间
        self.action_space = gym.spaces.Text(max_length=1000)
        self.observation_space = gym.spaces.Text(max_length=10000)

    def seed(self, seed: int):
        """设置随机种子"""
        self.np_random = np.random.RandomState(seed)
        random.seed(seed)

    def _generate_flights_db(self) -> List[Flight]:
        """生成确定性的航班数据库"""
        airlines = ["AA", "DL", "UA", "IB", "AF", "LH", "BA"]
        flights = []

        # 定义航线和基础数据
        routes = [
            ("SFO", "MAD", 800, 0.7),   # 基础价格，需求水平
            ("LAX", "LHR", 750, 0.8),
            ("JFK", "CDG", 650, 0.9),
            ("ORD", "FRA", 700, 0.6),
            ("BOS", "BCN", 600, 0.5),
            ("SEA", "AMS", 720, 0.6)
        ]

        for i, (origin, dest, base_price, demand) in enumerate(routes):
            # 为每条航线生成多个时间段的航班
            time_slots = [
                ("06:00", "14:00", 0),     # 早班直飞
                ("08:30", "18:30", 1),     # 早班经停
                ("10:00", "18:00", 0),     # 上午直飞
                ("12:00", "22:00", 1),     # 午间经停
                ("14:30", "22:30", 0),     # 下午直飞
                ("16:00", "02:00+1", 1),   # 下午经停
                ("18:00", "04:00+1", 0),   # 晚班直飞
                ("20:00", "08:00+1", 2),   # 晚班多经停
            ]
            
            for j, (depart, arrive, stops) in enumerate(time_slots):
                airline = airlines[j % len(airlines)]
                flight_id = f"{airline}{1000 + i*10 + j}"
                
                # 确定性价格计算
                price = base_price + stops * 100 + j * 50  # 时间段影响价格
                if demand > 0.7:  # 高需求航线更贵
                    price += 100
                
                flights.append(Flight(
                    id=flight_id,
                    price=price,
                    depart_time=depart,
                    arrive_time=arrive,
                    stops=stops,
                    airline=airline,
                    origin=origin,
                    destination=dest,
                    demand_level=demand,
                    available=True
                ))

        return flights

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """重置环境"""
        if seed is not None:
            self.seed(seed)

        # 重置环境状态
        self.current_view = FlightBookingView.SEARCH_FORM
        self.step_count = 0
        self.done = False
        self.truncated = False
        self.messages = []
        self.cart = {"items": [], "total": 0}
        self.payment_state = {"attempts": 0, "card_entered": False, "confirmed": False}
        self.current_flights = []
        
        # 重置技能指标
        self.skill_metrics = SkillMetrics()
        self.session_history = []
        self.constraint_satisfaction_score = 0.0

        # 生成任务（基于难度等级的确定性生成）
        self.current_task = self._generate_task_by_difficulty()

        obs = self._get_observation()
        info = {
            "step": self.step_count, 
            "task": self.current_task,
            "skill_metrics": asdict(self.skill_metrics)
        }

        return obs, info

    def _generate_task_by_difficulty(self) -> Dict:
        """根据难度等级生成任务"""
        routes = [("SFO", "MAD"), ("LAX", "LHR"), ("JFK", "CDG"), ("ORD", "FRA")]
        
        if self.difficulty_level == "easy":
            # 宽松约束
            budget_multiplier = 1.5
            max_stops = 2
            time_flexibility = "any"
        elif self.difficulty_level == "medium":
            # 中等约束
            budget_multiplier = 1.2
            max_stops = 1
            time_flexibility = "morning_or_afternoon"
        else:  # hard
            # 严格约束
            budget_multiplier = 1.0
            max_stops = 0
            time_flexibility = "morning_only"
        
        # 选择航线（基于种子确定性选择）
        route_idx = hash(str(self.np_random.get_state())) % len(routes)
        origin, destination = routes[route_idx]
        
        # 计算基础预算
        base_price = 700  # 平均基础价格
        budget = int(base_price * budget_multiplier)
        
        return {
            "origin": origin,
            "destination": destination,
            "date": "2025-10-15",
            "constraints": {
                "budget": budget,
                "max_stops": max_stops,
                "time_preference": time_flexibility,
                "depart_after": "06:00" if time_flexibility != "any" else None
            },
            "difficulty": self.difficulty_level
        }

    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步动作"""
        self.step_count += 1
        
        # 初始化奖励分解
        reward_breakdown = RewardBreakdown()
        
        try:
            # 解析和执行动作
            action_dict = self._parse_action(action)
            reward_breakdown = self._execute_action_with_detailed_feedback(action_dict)
            
            # 记录历史
            self.session_history.append({
                "step": self.step_count,
                "action": action_dict,
                "view": self.current_view.value,
                "reward_breakdown": asdict(reward_breakdown)
            })
            
        except Exception as e:
            # 智能错误处理
            reward_breakdown.penalty = -0.1
            reward_breakdown.calculate_total()
            self.messages.append(f"Action error: {str(e)}")
            self._handle_error_recovery()

        # 检查终止条件
        self._check_termination()

        # 检查截断
        if self.step_count >= self.max_steps:
            self.truncated = True
            reward_breakdown.penalty -= 0.2  # 超时惩罚

        # 计算最终奖励
        final_reward = reward_breakdown.calculate_total()

        obs = self._get_observation()
        info = {
            "step": self.step_count,
            "task": self.current_task,
            "reward_breakdown": asdict(reward_breakdown),
            "skill_metrics": asdict(self.skill_metrics),
            "constraint_satisfaction": self.constraint_satisfaction_score,
            "action_parsed": action_dict if 'action_dict' in locals() else None
        }

        return obs, final_reward, self.done, self.truncated, info

    def _parse_action(self, action: str) -> Dict:
        """解析动作（支持更详细的参数）"""
        action = action.strip().lower()
        
        # 基础动作类型识别
        if "search" in action:
            # 可以解析更详细的搜索参数
            params = {}
            if "budget" in action:
                # 简单的预算提取逻辑
                import re
                budget_match = re.search(r'budget[:\s]*(\d+)', action)
                if budget_match:
                    params["budget_filter"] = int(budget_match.group(1))
            return {"type": "search_flights", "params": params}
            
        elif "filter" in action:
            return {"type": "filter_results", "params": {}}
            
        elif "select" in action:
            # 可以解析选择哪个航班
            params = {}
            if "cheapest" in action:
                params["criteria"] = "price"
            elif "fastest" in action or "direct" in action:
                params["criteria"] = "stops"
            elif "best" in action:
                params["criteria"] = "value"
            return {"type": "select_flight", "params": params}
            
        elif "add" in action and "cart" in action:
            return {"type": "add_to_cart", "params": {}}
            
        elif "payment" in action or "pay" in action:
            return {"type": "proceed_to_payment", "params": {}}
            
        elif "card" in action:
            return {"type": "enter_card", "params": {}}
            
        elif "confirm" in action:
            return {"type": "confirm_payment", "params": {}}
            
        elif "restart" in action:
            return {"type": "restart", "params": {}}
            
        elif "abort" in action:
            return {"type": "abort", "params": {}}
            
        else:
            raise ValueError(f"未识别的动作: {action}")

    def _execute_action_with_detailed_feedback(self, action: Dict) -> RewardBreakdown:
        """执行动作并提供详细反馈"""
        action_type = action["type"]
        params = action.get("params", {})
        reward_breakdown = RewardBreakdown()

        if action_type == "search_flights":
            reward_breakdown = self._search_flights_with_feedback(params)
        elif action_type == "filter_results":
            reward_breakdown = self._filter_results_with_feedback(params)
        elif action_type == "select_flight":
            reward_breakdown = self._select_flight_with_feedback(params)
        elif action_type == "add_to_cart":
            reward_breakdown = self._add_to_cart_with_feedback(params)
        elif action_type == "proceed_to_payment":
            reward_breakdown = self._proceed_to_payment_with_feedback(params)
        elif action_type == "enter_card":
            reward_breakdown = self._enter_card_with_feedback(params)
        elif action_type == "confirm_payment":
            reward_breakdown = self._confirm_payment_with_feedback(params)
        elif action_type == "restart":
            reward_breakdown = self._restart_with_feedback(params)
        elif action_type == "abort":
            reward_breakdown = self._abort_with_feedback(params)

        return reward_breakdown

    def _search_flights_with_feedback(self, params: Dict) -> RewardBreakdown:
        """搜索航班的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        if self.current_view != FlightBookingView.SEARCH_FORM:
            reward_breakdown.penalty = -0.05
            self.messages.append("错误：当前不在搜索页面")
            return reward_breakdown

        # 基础动作奖励
        reward_breakdown.base_action = 0.05
        
        # 进度奖励
        reward_breakdown.progress = 0.1 * self.reward_weights["progress"]
        
        # 获取匹配的航班（确定性逻辑）
        origin = self.current_task["origin"]
        destination = self.current_task["destination"]
        
        matching_flights = [
            f for f in self.flights_db
            if f.origin == origin and f.destination == destination and f.available
        ]
        
        self.current_flights = matching_flights
        self.skill_metrics.search_attempts += 1
        
        if matching_flights:
            self.skill_metrics.successful_searches += 1
            self.current_view = FlightBookingView.SEARCH_RESULTS
            self.messages.append(f"找到 {len(matching_flights)} 个航班")
            
            # 效率奖励
            efficiency_score = min(len(matching_flights) / 10.0, 1.0)
            reward_breakdown.efficiency = efficiency_score * self.reward_weights["efficiency"]
        else:
            self.messages.append("未找到匹配的航班")
            reward_breakdown.penalty = -0.1

        return reward_breakdown

    def _filter_results_with_feedback(self, params: Dict) -> RewardBreakdown:
        """筛选结果的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        if self.current_view != FlightBookingView.SEARCH_RESULTS:
            reward_breakdown.penalty = -0.05
            self.messages.append("错误：没有搜索结果可筛选")
            return reward_breakdown

        constraints = self.current_task["constraints"]
        original_count = len(self.current_flights)
        
        # 应用约束筛选
        filtered_flights = []
        constraint_violations = 0
        
        for flight in self.current_flights:
            meets_budget = flight.price <= constraints["budget"]
            meets_stops = flight.stops <= constraints["max_stops"]
            
            if meets_budget and meets_stops:
                filtered_flights.append(flight)
            else:
                if not meets_budget:
                    constraint_violations += 1
                if not meets_stops:
                    constraint_violations += 1

        self.current_flights = filtered_flights
        
        # 基础动作奖励
        reward_breakdown.base_action = 0.03
        
        # 约束满足奖励
        if filtered_flights:
            satisfaction_ratio = len(filtered_flights) / max(original_count, 1)
            reward_breakdown.constraint_satisfaction = (
                satisfaction_ratio * self.reward_weights["constraint"]
            )
            self.messages.append(f"筛选后剩余 {len(filtered_flights)} 个航班")
        else:
            reward_breakdown.penalty = -0.05
            self.messages.append("筛选后无可用航班")
            
        # 更新约束违规计数
        self.skill_metrics.constraint_violations += constraint_violations

        return reward_breakdown

    def _select_flight_with_feedback(self, params: Dict) -> RewardBreakdown:
        """选择航班的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        if not self.current_flights:
            reward_breakdown.penalty = -0.05
            self.messages.append("错误：没有可选择的航班")
            return reward_breakdown

        # 根据选择标准选择航班
        criteria = params.get("criteria", "price")
        
        if criteria == "price":
            selected_flight = min(self.current_flights, key=lambda f: f.price)
            reward_breakdown.optimization = 0.1 * self.reward_weights["optimization"]
        elif criteria == "stops":
            selected_flight = min(self.current_flights, key=lambda f: f.stops)
            reward_breakdown.optimization = 0.08 * self.reward_weights["optimization"]
        elif criteria == "value":
            # 综合考虑价格和便利性的价值评分
            def value_score(flight):
                price_score = 1.0 - (flight.price / 2000.0)  # 归一化价格分数
                stops_score = 1.0 - (flight.stops / 2.0)     # 归一化经停分数
                return price_score * 0.6 + stops_score * 0.4
            
            selected_flight = max(self.current_flights, key=value_score)
            reward_breakdown.optimization = 0.12 * self.reward_weights["optimization"]
        else:
            selected_flight = self.current_flights[0]  # 默认选择第一个

        # 将选中的航班放在列表首位
        self.current_flights = [selected_flight] + [
            f for f in self.current_flights if f.id != selected_flight.id
        ]

        # 基础动作奖励
        reward_breakdown.base_action = 0.05
        
        # 检查选择是否满足约束
        constraints = self.current_task["constraints"]
        if (selected_flight.price <= constraints["budget"] and 
            selected_flight.stops <= constraints["max_stops"]):
            reward_breakdown.constraint_satisfaction = 0.1 * self.reward_weights["constraint"]
            self.messages.append(f"已选择航班 {selected_flight.id} (符合约束)")
        else:
            reward_breakdown.penalty = -0.03
            self.messages.append(f"已选择航班 {selected_flight.id} (约束违规)")

        return reward_breakdown

    def _add_to_cart_with_feedback(self, params: Dict) -> RewardBreakdown:
        """添加到购物车的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        if not self.current_flights:
            reward_breakdown.penalty = -0.05
            self.messages.append("错误：没有选择的航班")
            return reward_breakdown

        flight = self.current_flights[0]
        cart_item = {"flight_id": flight.id, "price": flight.price}
        self.cart["items"].append(cart_item)
        self.cart["total"] = sum(item["price"] for item in self.cart["items"])

        self.current_view = FlightBookingView.CART
        
        # 基础动作奖励
        reward_breakdown.base_action = 0.08
        
        # 进度奖励
        reward_breakdown.progress = 0.15 * self.reward_weights["progress"]
        
        # 预算效率评估
        constraints = self.current_task["constraints"]
        budget_usage = self.cart["total"] / constraints["budget"]
        
        if budget_usage <= 1.0:  # 在预算内
            # 奖励接近但不超预算的选择
            efficiency_bonus = (1.0 - abs(budget_usage - 0.9)) * 0.1
            reward_breakdown.optimization = efficiency_bonus * self.reward_weights["optimization"]
            self.skill_metrics.budget_efficiency = budget_usage
            self.messages.append(f"航班已加入购物车 (预算使用率: {budget_usage:.1%})")
        else:  # 超预算
            reward_breakdown.penalty = -0.1
            self.messages.append(f"航班已加入购物车 (超出预算: ${self.cart['total'] - constraints['budget']})")

        return reward_breakdown

    def _proceed_to_payment_with_feedback(self, params: Dict) -> RewardBreakdown:
        """进入支付的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        if not self.cart["items"]:
            reward_breakdown.penalty = -0.05
            self.messages.append("错误：购物车为空")
            return reward_breakdown

        self.current_view = FlightBookingView.PAYMENT
        
        # 基础动作奖励
        reward_breakdown.base_action = 0.05
        
        # 进度奖励
        reward_breakdown.progress = 0.1 * self.reward_weights["progress"]
        
        self.messages.append("进入支付页面")
        return reward_breakdown

    def _enter_card_with_feedback(self, params: Dict) -> RewardBreakdown:
        """输入卡信息的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        if self.current_view != FlightBookingView.PAYMENT:
            reward_breakdown.penalty = -0.05
            self.messages.append("错误：不在支付页面")
            return reward_breakdown

        self.payment_state["card_entered"] = True
        
        # 基础动作奖励
        reward_breakdown.base_action = 0.03
        
        self.messages.append("信用卡信息已输入")
        return reward_breakdown

    def _confirm_payment_with_feedback(self, params: Dict) -> RewardBreakdown:
        """确认支付的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        if not self.payment_state["card_entered"]:
            reward_breakdown.penalty = -0.05
            self.messages.append("错误：未输入信用卡信息")
            return reward_breakdown

        self.payment_state["attempts"] += 1

        # 确定性支付逻辑（基于任务约束）
        constraints = self.current_task["constraints"]
        payment_success = self.cart["total"] <= constraints["budget"]
        
        if payment_success:
            self.payment_state["confirmed"] = True
            self.current_view = FlightBookingView.RECEIPT
            self.done = True
            
            # 计算完成奖励
            completion_reward = self._calculate_completion_reward()
            reward_breakdown.base_action = 0.1
            reward_breakdown.progress = completion_reward * self.reward_weights["completion"]
            
            self.messages.append("支付成功！预订已确认")
        else:
            reward_breakdown.penalty = -0.1
            self.payment_state["card_entered"] = False
            self.messages.append("支付失败：超出预算限制")

        return reward_breakdown

    def _calculate_completion_reward(self) -> float:
        """计算任务完成奖励"""
        constraints = self.current_task["constraints"]
        
        # 基础完成奖励
        base_reward = 1.0
        
        # 效率加成
        steps_efficiency = max(0, (self.max_steps - self.step_count) / self.max_steps)
        efficiency_bonus = steps_efficiency * 0.3
        
        # 预算优化加成
        budget_efficiency = self.skill_metrics.budget_efficiency
        if budget_efficiency > 0:
            # 奖励充分利用预算（接近但不超过）
            optimal_usage = 0.9  # 理想使用率
            budget_bonus = max(0, 0.2 - abs(budget_efficiency - optimal_usage))
        else:
            budget_bonus = 0
        
        # 约束满足加成
        constraint_bonus = 0.1 if self.skill_metrics.constraint_violations == 0 else 0
        
        total_reward = base_reward + efficiency_bonus + budget_bonus + constraint_bonus
        
        # 更新约束满足分数
        self.constraint_satisfaction_score = (
            1.0 if budget_efficiency <= 1.0 and self.skill_metrics.constraint_violations == 0 else 0.5
        )
        
        return total_reward

    def _restart_with_feedback(self, params: Dict) -> RewardBreakdown:
        """重启的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        self.current_view = FlightBookingView.SEARCH_FORM
        self.cart = {"items": [], "total": 0}
        self.payment_state = {"attempts": 0, "card_entered": False, "confirmed": False}
        self.current_flights = []
        
        # 错误恢复计数
        self.skill_metrics.error_recovery_count += 1
        
        # 轻微惩罚但鼓励恢复
        reward_breakdown.penalty = -0.02
        reward_breakdown.base_action = 0.01  # 恢复尝试的小奖励
        
        self.messages = ["会话已重启"]
        return reward_breakdown

    def _abort_with_feedback(self, params: Dict) -> RewardBreakdown:
        """终止任务的详细反馈"""
        reward_breakdown = RewardBreakdown()
        
        self.done = True
        reward_breakdown.penalty = -0.3
        
        self.messages.append("任务已终止")
        return reward_breakdown

    def _handle_error_recovery(self):
        """智能错误恢复"""
        self.skill_metrics.error_recovery_count += 1
        
        # 根据当前状态提供恢复建议
        if self.current_view == FlightBookingView.SEARCH_FORM:
            self.messages.append("提示：使用 'search_flights' 开始搜索")
        elif self.current_view == FlightBookingView.SEARCH_RESULTS:
            self.messages.append("提示：使用 'filter_results' 或 'select_flight' 继续")
        elif self.current_view == FlightBookingView.CART:
            self.messages.append("提示：使用 'proceed_to_payment' 继续结账")

    def _check_termination(self):
        """检查终止条件"""
        # 在相应的动作中已经设置了done状态
        pass

    def _get_observation(self) -> Dict:
        """获取增强的观察状态"""
        obs = {
            "view": self.current_view.value,
            "step": self.step_count,
            "task": {
                "from": self.current_task["origin"],
                "to": self.current_task["destination"],
                "date": self.current_task["date"],
                "difficulty": self.current_task["difficulty"]
            },
            "constraints": self.current_task["constraints"],
            "flights": [f.to_dict() for f in self.current_flights[:5]],  # 限制显示数量
            "cart": self.cart,
            "payment_state": self.payment_state,
            "messages": self.messages[-5:],  # 显示最近5条消息
            "skill_metrics": {
                "search_efficiency": (
                    self.skill_metrics.successful_searches / 
                    max(self.skill_metrics.search_attempts, 1)
                ),
                "budget_efficiency": self.skill_metrics.budget_efficiency,
                "constraint_violations": self.skill_metrics.constraint_violations,
                "error_recovery_count": self.skill_metrics.error_recovery_count
            },
            "constraint_satisfaction_score": self.constraint_satisfaction_score,
            "available_actions": self._get_available_actions(),
            "done": self.done,
            "truncated": self.truncated
        }

        return obs

    def _get_available_actions(self) -> List[str]:
        """获取当前状态下可用的动作"""
        if self.current_view == FlightBookingView.SEARCH_FORM:
            return ["search_flights", "abort"]
        elif self.current_view == FlightBookingView.SEARCH_RESULTS:
            actions = ["filter_results", "select_flight cheapest", "select_flight fastest", "restart"]
            if self.current_flights:
                actions.append("add_to_cart")
            return actions
        elif self.current_view == FlightBookingView.CART:
            return ["proceed_to_payment", "restart", "abort"]
        elif self.current_view == FlightBookingView.PAYMENT:
            actions = ["enter_card"]
            if self.payment_state["card_entered"]:
                actions.append("confirm_payment")
            actions.extend(["restart", "abort"])
            return actions
        elif self.current_view == FlightBookingView.RECEIPT:
            return []  # 任务完成
        else:
            return ["restart", "abort"]

    def render(self, mode: str = "human") -> Optional[str]:
        """增强的渲染功能"""
        obs = self._get_observation()

        if mode == "human":
            print("=" * 60)
            print(f"步骤: {self.step_count}/{self.max_steps} | 视图: {obs['view']} | 难度: {obs['task']['difficulty']}")
            print(f"任务: {obs['task']['from']} -> {obs['task']['to']}")
            print(f"约束: 预算=${obs['constraints']['budget']}, 最大经停={obs['constraints']['max_stops']}")
            print(f"购物车总额: ${obs['cart']['total']}")
            
            # 显示技能指标
            metrics = obs['skill_metrics']
            print(f"技能指标: 搜索效率={metrics['search_efficiency']:.2f}, "
                  f"预算效率={metrics['budget_efficiency']:.2f}, "
                  f"约束违规={metrics['constraint_violations']}")
            
            # 显示消息
            if obs['messages']:
                print(f"消息: {' | '.join(obs['messages'])}")
            
            # 显示可用航班
            if obs['flights']:
                print(f"可用航班 ({len(obs['flights'])}):")
                for flight in obs['flights']:
                    print(f"  {flight['id']}: ${flight['price']} "
                          f"({flight['stops']}经停, {flight['depart']}-{flight['arrive']})")
            
            # 显示可用动作
            actions = obs['available_actions']
            if actions:
                print(f"可用动作: {', '.join(actions)}")
                
            print("=" * 60)
            
        elif mode == "json":
            return json.dumps(obs, indent=2, ensure_ascii=False)

        return None

    def close(self):
        """关闭环境"""
        pass


# 测试代码
if __name__ == "__main__":
    # 测试改进的环境
    env = ImprovedFlightBookingEnv(seed=42, config={
        "difficulty": "medium",
        "max_steps": 20
    })
    
    obs, info = env.reset()
    print("初始观察:")
    env.render()
    
    # 模拟智能体行为
    actions = [
        "search_flights",
        "filter_results", 
        "select_flight cheapest",
        "add_to_cart",
        "proceed_to_payment",
        "enter_card",
        "confirm_payment"
    ]
    
    total_reward = 0
    for action in actions:
        if env.done or env.truncated:
            break
            
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        
        print(f"\n动作: {action}")
        print(f"奖励: {reward:.3f} (总计: {total_reward:.3f})")
        
        # 显示奖励分解
        if "reward_breakdown" in info:
            breakdown = info["reward_breakdown"]
            print(f"奖励分解: 基础={breakdown['base_action']:.3f}, "
                  f"进度={breakdown['progress']:.3f}, "
                  f"约束={breakdown['constraint_satisfaction']:.3f}, "
                  f"优化={breakdown['optimization']:.3f}, "
                  f"惩罚={breakdown['penalty']:.3f}")
        
        env.render()
        
        if done:
            print(f"\n任务完成! 总奖励: {total_reward:.3f}")
            print(f"约束满足分数: {info['constraint_satisfaction']:.2f}")
            break