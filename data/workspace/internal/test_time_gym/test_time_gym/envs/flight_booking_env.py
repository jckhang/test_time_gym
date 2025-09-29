"""
机票预订环境实现
支持智能体在安全的仿真环境中学习订票流程
"""

import gymnasium as gym
import numpy as np
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import copy


class FlightBookingView(Enum):
    """环境视图状态"""
    SEARCH_FORM = "search_form"
    SEARCH_RESULTS = "search_results"
    CART = "cart"
    PAYMENT = "payment"
    RECEIPT = "receipt"
    ERROR = "error"


@dataclass
class Flight:
    """航班信息"""
    id: str
    price: float
    depart_time: str
    arrive_time: str
    stops: int
    airline: str
    available: bool = True

    def to_dict(self):
        return {
            "id": self.id,
            "price": self.price,
            "depart": self.depart_time,
            "arrive": self.arrive_time,
            "stops": self.stops,
            "airline": self.airline
        }


@dataclass
class CartItem:
    """购物车项目"""
    flight_id: str
    price: float


@dataclass
class PaymentState:
    """支付状态"""
    needs_3ds: bool = False
    attempts: int = 0
    card_entered: bool = False
    confirmed: bool = False


class FlightBookingEnv(gym.Env):
    """机票预订环境"""
    
    def __init__(self, seed: Optional[int] = None, config: Optional[Dict] = None):
        super().__init__()
        
        # 设置随机种子
        if seed is not None:
            self.seed(seed)
        
        # 环境配置
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 50)
        self.enable_3ds = self.config.get("enable_3ds", True)
        self.payment_failure_rate = self.config.get("payment_failure_rate", 0.1)
        self.flight_sellout_rate = self.config.get("flight_sellout_rate", 0.05)
        
        # 环境状态
        self.current_view = FlightBookingView.SEARCH_FORM
        self.step_count = 0
        self.current_task = None
        self.flights_db = self._generate_flights_db()
        self.current_flights = []
        self.cart = {"items": [], "total": 0}
        self.payment_state = PaymentState()
        self.messages = []
        self.done = False
        self.truncated = False
        
        # 动作空间（文本描述）
        self.action_space = gym.spaces.Text(max_length=1000)
        
        # 观察空间（JSON结构）
        self.observation_space = gym.spaces.Text(max_length=10000)
        
    def seed(self, seed: int):
        """设置随机种子"""
        self.np_random = np.random.RandomState(seed)
        random.seed(seed)
        
    def _generate_flights_db(self) -> List[Flight]:
        """生成模拟航班数据库"""
        airlines = ["AA", "DL", "UA", "IB", "AF", "LH", "BA"]
        flights = []
        
        # 为常见航线生成航班
        routes = [
            ("SFO", "MAD"), ("LAX", "LHR"), ("JFK", "CDG"),
            ("ORD", "FRA"), ("BOS", "BCN"), ("SEA", "AMS")
        ]
        
        for i, (origin, dest) in enumerate(routes):
            for j in range(5):  # 每条航线5个航班
                flight_id = f"{random.choice(airlines)}{1000 + i*10 + j}"
                base_price = 500 + random.randint(100, 800)
                stops = random.choice([0, 0, 0, 1, 1, 2])  # 更多直飞
                
                # 生成时间
                depart_hour = random.randint(6, 23)
                depart_min = random.choice([0, 15, 30, 45])
                arrive_hour = (depart_hour + 8 + stops * 2) % 24
                arrive_day_offset = 1 if depart_hour + 8 + stops * 2 >= 24 else 0
                
                flights.append(Flight(
                    id=flight_id,
                    price=base_price + stops * 50,
                    depart_time=f"{depart_hour:02d}:{depart_min:02d}",
                    arrive_time=f"{arrive_hour:02d}:{depart_min:02d}" + (f"+{arrive_day_offset}" if arrive_day_offset else ""),
                    stops=stops,
                    airline=flight_id[:2]
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
        self.payment_state = PaymentState()
        self.current_flights = []
        
        # 生成随机任务
        self.current_task = self._generate_random_task()
        
        # 应用随机扰动
        self._apply_random_perturbations()
        
        obs = self._get_observation()
        info = {"step": self.step_count, "task": self.current_task}
        
        return obs, info
    
    def _generate_random_task(self) -> Dict:
        """生成随机预订任务"""
        routes = [("SFO", "MAD"), ("LAX", "LHR"), ("JFK", "CDG"), ("ORD", "FRA")]
        origin, destination = random.choice(routes)
        
        # 随机约束
        budget = random.choice([700, 800, 900, 1000, 1200])
        max_stops = random.choice([0, 1, 2])
        depart_after = random.choice(["06:00", "08:00", "10:00", "12:00"])
        
        return {
            "origin": origin,
            "destination": destination,
            "date": "2025-10-15",  # 固定日期简化实现
            "constraints": {
                "budget": budget,
                "depart_after": depart_after,
                "max_stops": max_stops
            }
        }
    
    def _apply_random_perturbations(self):
        """应用随机扰动"""
        # 随机设置需要3DS验证
        if self.enable_3ds and random.random() < 0.3:
            self.payment_state.needs_3ds = True
            
        # 随机使部分航班售罄
        for flight in self.flights_db:
            if random.random() < self.flight_sellout_rate:
                flight.available = False
    
    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步动作"""
        self.step_count += 1
        reward = -0.01  # 基础时间成本
        
        try:
            # 解析动作
            action_dict = self._parse_action(action)
            
            # 执行动作
            reward += self._execute_action(action_dict)
            
        except Exception as e:
            # 无效动作惩罚
            reward -= 0.05
            self.messages.append(f"Error: {str(e)}")
            self.current_view = FlightBookingView.ERROR
        
        # 检查是否完成
        self._check_termination()
        
        # 检查是否截断（超过最大步数）
        if self.step_count >= self.max_steps:
            self.truncated = True
        
        obs = self._get_observation()
        info = {
            "step": self.step_count,
            "task": self.current_task,
            "action_parsed": action_dict if 'action_dict' in locals() else None
        }
        
        return obs, reward, self.done, self.truncated, info
    
    def _parse_action(self, action: str) -> Dict:
        """解析文本动作为结构化格式"""
        action = action.strip().lower()
        
        if action.startswith("search_flights"):
            # 简单解析，实际可用更复杂的NLP解析
            return {"type": "search_flights", "params": {}}
        elif action.startswith("filter_results"):
            return {"type": "filter_results", "params": {}}
        elif action.startswith("select_flight"):
            return {"type": "select_flight", "params": {}}
        elif action.startswith("add_to_cart"):
            return {"type": "add_to_cart", "params": {}}
        elif action.startswith("proceed_to_payment"):
            return {"type": "proceed_to_payment", "params": {}}
        elif action.startswith("enter_card"):
            return {"type": "enter_card", "params": {}}
        elif action.startswith("confirm_payment"):
            return {"type": "confirm_payment", "params": {}}
        elif action.startswith("apply_coupon"):
            return {"type": "apply_coupon", "params": {}}
        elif action.startswith("restart"):
            return {"type": "restart", "params": {}}
        elif action.startswith("abort"):
            return {"type": "abort", "params": {}}
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _execute_action(self, action: Dict) -> float:
        """执行具体动作并返回奖励"""
        action_type = action["type"]
        reward = 0.0
        
        if action_type == "search_flights":
            reward += self._search_flights()
        elif action_type == "filter_results":
            reward += self._filter_results()
        elif action_type == "select_flight":
            reward += self._select_flight()
        elif action_type == "add_to_cart":
            reward += self._add_to_cart()
        elif action_type == "proceed_to_payment":
            reward += self._proceed_to_payment()
        elif action_type == "enter_card":
            reward += self._enter_card()
        elif action_type == "confirm_payment":
            reward += self._confirm_payment()
        elif action_type == "apply_coupon":
            reward += self._apply_coupon()
        elif action_type == "restart":
            reward += self._restart()
        elif action_type == "abort":
            reward += self._abort()
        
        return reward
    
    def _search_flights(self) -> float:
        """搜索航班"""
        if self.current_view != FlightBookingView.SEARCH_FORM:
            raise ValueError("Cannot search flights from current view")
            
        # 获取符合航线的航班
        origin = self.current_task["origin"]
        destination = self.current_task["destination"]
        
        matching_flights = [
            f for f in self.flights_db 
            if f.available and random.random() > 0.1  # 10%概率航班临时不可用
        ]
        
        # 随机化排序
        random.shuffle(matching_flights)
        self.current_flights = matching_flights[:10]  # 最多显示10个
        
        self.current_view = FlightBookingView.SEARCH_RESULTS
        self.messages.append(f"Found {len(self.current_flights)} flights")
        
        return 0.02  # 成功搜索的小奖励
    
    def _filter_results(self) -> float:
        """筛选结果"""
        if self.current_view != FlightBookingView.SEARCH_RESULTS:
            raise ValueError("No search results to filter")
            
        constraints = self.current_task["constraints"]
        
        # 应用筛选逻辑
        filtered = []
        for flight in self.current_flights:
            if flight.price <= constraints["budget"]:
                if flight.stops <= constraints["max_stops"]:
                    # 简化时间检查
                    filtered.append(flight)
        
        self.current_flights = filtered
        self.messages.append(f"Filtered to {len(filtered)} flights")
        
        return 0.01 if filtered else -0.02
    
    def _select_flight(self) -> float:
        """选择航班（简化版：选择第一个可用的）"""
        if not self.current_flights:
            raise ValueError("No flights available to select")
            
        self.messages.append(f"Selected flight {self.current_flights[0].id}")
        return 0.01
    
    def _add_to_cart(self) -> float:
        """添加到购物车"""
        if not self.current_flights:
            raise ValueError("No flight selected")
            
        flight = self.current_flights[0]
        cart_item = {"flight_id": flight.id, "price": flight.price}
        self.cart["items"].append(cart_item)
        self.cart["total"] = sum(item["price"] for item in self.cart["items"])
        
        self.current_view = FlightBookingView.CART
        self.messages.append("Flight added to cart")
        
        return 0.05  # 进入购物车的奖励
    
    def _proceed_to_payment(self) -> float:
        """进入支付页面"""
        if not self.cart["items"]:
            raise ValueError("Cart is empty")
            
        self.current_view = FlightBookingView.PAYMENT
        self.messages.append("Proceeding to payment")
        
        return 0.03  # 进入支付的奖励
    
    def _enter_card(self) -> float:
        """输入信用卡信息"""
        if self.current_view != FlightBookingView.PAYMENT:
            raise ValueError("Not in payment view")
            
        self.payment_state.card_entered = True
        self.messages.append("Card information entered")
        
        return 0.01
    
    def _confirm_payment(self) -> float:
        """确认支付"""
        if not self.payment_state.card_entered:
            raise ValueError("Card not entered")
            
        self.payment_state.attempts += 1
        
        # 随机支付失败
        if random.random() < self.payment_failure_rate:
            self.messages.append("Payment failed, please try again")
            self.payment_state.card_entered = False
            return -0.05
        
        # 3DS验证
        if self.payment_state.needs_3ds and self.payment_state.attempts == 1:
            self.messages.append("3DS verification required")
            return 0.0
            
        # 支付成功
        self.payment_state.confirmed = True
        self.current_view = FlightBookingView.RECEIPT
        self.done = True
        self.messages.append("Payment successful! Booking confirmed.")
        
        # 计算终局奖励
        return self._calculate_final_reward()
    
    def _apply_coupon(self) -> float:
        """应用优惠券（简化实现）"""
        if random.random() < 0.5:  # 50%概率优惠券有效
            discount = random.randint(20, 100)
            self.cart["total"] = max(0, self.cart["total"] - discount)
            self.messages.append(f"Coupon applied! Saved ${discount}")
            return 0.02
        else:
            self.messages.append("Coupon invalid or expired")
            return -0.01
    
    def _restart(self) -> float:
        """重新开始"""
        self.current_view = FlightBookingView.SEARCH_FORM
        self.cart = {"items": [], "total": 0}
        self.payment_state = PaymentState()
        self.current_flights = []
        self.messages = ["Session restarted"]
        return 0.0
    
    def _abort(self) -> float:
        """终止任务"""
        self.done = True
        self.messages.append("Task aborted")
        return -0.1  # 放弃任务的惩罚
    
    def _calculate_final_reward(self) -> float:
        """计算最终奖励"""
        constraints = self.current_task["constraints"]
        total_price = self.cart["total"]
        
        # 检查约束
        violations = 0
        
        # 预算约束
        if total_price > constraints["budget"]:
            violations += 1
            
        # 其他约束检查可以在这里添加
        
        if violations == 0:
            # 任务成功完成
            return 1.0
        else:
            # 有约束违规
            return -0.3 * violations
    
    def _check_termination(self):
        """检查是否应该终止"""
        # 已经在其他地方设置了done状态
        pass
    
    def _get_observation(self) -> Dict:
        """获取当前观察状态"""
        obs = {
            "view": self.current_view.value,
            "step": self.step_count,
            "forms": {
                "from": self.current_task["origin"],
                "to": self.current_task["destination"],
                "date": self.current_task["date"]
            },
            "flights": [f.to_dict() for f in self.current_flights if f.available],
            "cart": self.cart,
            "payment_state": {
                "needs_3ds": self.payment_state.needs_3ds,
                "attempts": self.payment_state.attempts,
                "card_entered": self.payment_state.card_entered,
                "confirmed": self.payment_state.confirmed
            },
            "constraints": self.current_task["constraints"],
            "messages": self.messages[-3:],  # 只保留最近3条消息
            "done": self.done,
            "truncated": self.truncated
        }
        
        return obs
    
    def render(self, mode: str = "human") -> Optional[str]:
        """渲染环境状态"""
        obs = self._get_observation()
        
        if mode == "human":
            print("=" * 50)
            print(f"Step: {self.step_count} | View: {obs['view']}")
            print(f"Task: {obs['forms']['from']} -> {obs['forms']['to']}")
            print(f"Constraints: {obs['constraints']}")
            print(f"Cart Total: ${obs['cart']['total']}")
            print(f"Messages: {obs['messages']}")
            if obs['flights']:
                print(f"Available Flights: {len(obs['flights'])}")
                for flight in obs['flights'][:3]:  # 显示前3个
                    print(f"  {flight['id']}: ${flight['price']} ({flight['stops']} stops)")
            print("=" * 50)
        elif mode == "json":
            return json.dumps(obs, indent=2)
        
        return None
    
    def close(self):
        """关闭环境"""
        pass


if __name__ == "__main__":
    # 简单测试
    env = FlightBookingEnv(seed=42)
    obs, info = env.reset()
    
    print("Initial observation:")
    env.render()
    
    # 模拟几步操作
    actions = [
        "search_flights",
        "filter_results", 
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
        
        print(f"\nAction: {action}")
        print(f"Reward: {reward:.3f}")
        env.render()
        
        if done:
            print(f"\nTask completed! Total reward: {total_reward:.3f}")
            break