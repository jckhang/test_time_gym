"""
基于OpenAI的智能体实现
使用OpenAI GPT模型进行机票预订决策
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .llm_agent import OpenAILLMAgent, ToolEnabledLLMAgent
from ..llm.base import OpenAIBackend

logger = logging.getLogger(__name__)


class FlightBookingOpenAIAgent(OpenAILLMAgent):
    """专门用于机票预订的OpenAI智能体"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        strategy: str = "balanced",
        max_retries: int = 3,
        temperature: float = 0.7
    ):
        """
        初始化机票预订OpenAI智能体

        Args:
            model: OpenAI模型名称
            strategy: 策略类型 ("aggressive", "balanced", "conservative")
            max_retries: 最大重试次数
            temperature: 生成温度
        """
        system_prompt = self._get_strategy_prompt(strategy)
        super().__init__(model, system_prompt, max_retries, temperature)
        self.strategy = strategy

    def _get_strategy_prompt(self, strategy: str) -> str:
        """根据策略获取系统提示词"""
        base_prompt = """你是一个专业的机票预订助手。你的目标是帮助用户高效地完成机票预订，同时考虑用户的预算和偏好。

环境状态说明：
- search_form: 搜索表单页面
- search_results: 搜索结果页面，显示可选航班
- cart: 购物车页面
- payment: 支付页面
- receipt: 收据页面（预订完成）
- error: 错误状态

可用动作：
- search_flights: 搜索航班
- filter_results: 筛选搜索结果
- select_flight: 选择特定航班
- add_to_cart: 添加到购物车
- proceed_to_payment: 进入支付页面
- enter_card: 输入支付卡信息
- confirm_payment: 确认支付
- apply_coupon: 应用优惠券
- restart: 重新开始
- abort: 中止预订

决策原则："""

        if strategy == "aggressive":
            return base_prompt + """
- 优先选择价格最低的航班
- 快速决策，减少犹豫
- 在预算范围内尽可能节省成本
- 如果遇到问题，快速重试"""

        elif strategy == "conservative":
            return base_prompt + """
- 优先考虑航班质量和可靠性
- 仔细评估每个选项
- 在不确定时选择更安全的选项
- 遇到问题时先分析再行动"""

        else:  # balanced
            return base_prompt + """
- 平衡价格和质量
- 考虑用户的预算约束和偏好
- 在多个选项间进行合理权衡
- 遇到问题时灵活应对"""

    def _format_observation(self, observation: Dict[str, Any]) -> str:
        """格式化观察为详细的文本描述"""
        view = observation.get("view", "unknown")
        context = f"当前页面: {view}\n"

        if view == "search_form":
            search_params = observation.get("search_params", {})
            context += f"搜索参数: 出发地={search_params.get('origin', '未设置')}, "
            context += f"目的地={search_params.get('destination', '未设置')}, "
            context += f"日期={search_params.get('date', '未设置')}\n"
            context += "需要填写完整的搜索信息"

        elif view == "search_results":
            flights = observation.get("flights", [])
            constraints = observation.get("constraints", {})
            context += f"找到 {len(flights)} 个航班选项\n"
            context += f"预算约束: ¥{constraints.get('budget', '无限制')}\n"
            context += f"最大转机次数: {constraints.get('max_stops', '无限制')}\n"

            if flights:
                context += "航班详情:\n"
                for i, flight in enumerate(flights[:5]):  # 显示前5个航班
                    context += f"  {i+1}. {flight.get('airline', 'Unknown')} - ¥{flight.get('price', 0)} "
                    context += f"- {flight.get('stops', 0)}次转机 - {flight.get('depart', 'N/A')} -> {flight.get('arrive', 'N/A')}\n"

                # 分析航班选项
                prices = [f.get('price', 0) for f in flights]
                if prices:
                    context += f"价格范围: ¥{min(prices)} - ¥{max(prices)}\n"
                    budget = constraints.get('budget', float('inf'))
                    affordable = [f for f in flights if f.get('price', 0) <= budget]
                    context += f"符合预算的航班: {len(affordable)}/{len(flights)}\n"

        elif view == "cart":
            cart = observation.get("cart", {})
            context += f"购物车总价: ¥{cart.get('total', 0)}\n"
            context += f"购物车项目数: {cart.get('items', 0)}\n"
            if cart.get('items', 0) > 0:
                context += "购物车内容:\n"
                for item in cart.get('flights', []):
                    context += f"  - {item.get('airline', 'Unknown')} ¥{item.get('price', 0)}\n"

        elif view == "payment":
            payment_state = observation.get("payment_state", {})
            context += f"支付状态: 卡片已输入={payment_state.get('card_entered', False)}, "
            context += f"已确认={payment_state.get('confirmed', False)}\n"
            context += "需要完成支付流程"

        elif view == "error":
            error_msg = observation.get("error_message", "未知错误")
            context += f"错误信息: {error_msg}\n"
            context += "需要处理错误并继续预订流程"

        elif view == "receipt":
            context += "预订已完成！显示收据信息"

        return context

    def _parse_response(self, response: Dict[str, Any]) -> str:
        """解析LLM响应，提取动作"""
        content = response.get("content", "")

        # 更精确的动作识别
        action_patterns = {
            "search_flights": [
                "搜索航班", "search flights", "查找航班", "开始搜索",
                "search", "查找", "搜索"
            ],
            "filter_results": [
                "筛选结果", "filter results", "过滤结果", "筛选",
                "filter", "过滤", "调整筛选条件"
            ],
            "select_flight": [
                "选择航班", "select flight", "挑选航班", "选择",
                "select", "挑选", "选择第", "选择这个"
            ],
            "add_to_cart": [
                "添加到购物车", "add to cart", "加入购物车", "添加",
                "add", "加入", "放入购物车"
            ],
            "proceed_to_payment": [
                "进入支付", "proceed to payment", "去支付", "支付",
                "payment", "付款", "结账"
            ],
            "enter_card": [
                "输入卡片", "enter card", "填写支付信息", "输入支付",
                "enter", "填写", "输入卡号"
            ],
            "confirm_payment": [
                "确认支付", "confirm payment", "确认", "confirm",
                "确认付款", "完成支付"
            ],
            "apply_coupon": [
                "应用优惠券", "apply coupon", "使用优惠券", "优惠券",
                "coupon", "折扣", "优惠"
            ],
            "restart": [
                "重新开始", "restart", "重试", "重新搜索",
                "重新", "重试", "重新开始搜索"
            ],
            "abort": [
                "中止", "abort", "取消", "停止",
                "cancel", "退出", "放弃"
            ]
        }

        content_lower = content.lower()
        for action, patterns in action_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                logger.info(f"识别到动作: {action} (从响应: {content[:100]}...)")
                return action

        # 如果无法识别，使用策略相关的默认动作
        logger.warning(f"无法从响应中识别动作: {content}")
        return self._get_strategy_default_action()

    def _get_strategy_default_action(self) -> str:
        """根据策略获取默认动作"""
        if self.strategy == "aggressive":
            return "add_to_cart"  # 激进策略：快速添加
        elif self.strategy == "conservative":
            return "filter_results"  # 保守策略：先筛选
        else:  # balanced
            return "search_flights"  # 平衡策略：重新搜索


class ToolEnabledFlightBookingAgent(ToolEnabledLLMAgent):
    """支持工具调用的机票预订智能体"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        strategy: str = "balanced",
        max_retries: int = 3,
        temperature: float = 0.7
    ):
        """
        初始化支持工具的机票预订智能体

        Args:
            model: OpenAI模型名称
            strategy: 策略类型
            max_retries: 最大重试次数
            temperature: 生成温度
        """
        backend = OpenAIBackend(model)
        tools = self._get_available_tools()
        system_prompt = self._get_tool_enabled_prompt(strategy)

        super().__init__(backend, tools, system_prompt, max_retries, temperature)
        self.strategy = strategy

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "description": "搜索航班",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "搜索航班的原因"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "filter_results",
                    "description": "筛选搜索结果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_price": {
                                "type": "number",
                                "description": "最大价格限制"
                            },
                            "max_stops": {
                                "type": "integer",
                                "description": "最大转机次数"
                            },
                            "airline_preference": {
                                "type": "string",
                                "description": "航空公司偏好"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "select_flight",
                    "description": "选择特定航班",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flight_id": {
                                "type": "string",
                                "description": "航班ID"
                            },
                            "reason": {
                                "type": "string",
                                "description": "选择该航班的原因"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_to_cart",
                    "description": "添加航班到购物车",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "添加到购物车的原因"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "proceed_to_payment",
                    "description": "进入支付页面",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "进入支付的原因"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "enter_card",
                    "description": "输入支付卡信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "card_type": {
                                "type": "string",
                                "description": "卡片类型"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "confirm_payment",
                    "description": "确认支付",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "确认支付的原因"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_coupon",
                    "description": "应用优惠券",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coupon_code": {
                                "type": "string",
                                "description": "优惠券代码"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "restart",
                    "description": "重新开始预订流程",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "重新开始的原因"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "abort",
                    "description": "中止预订",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "中止预订的原因"
                            }
                        }
                    }
                }
            }
        ]

    def _get_tool_enabled_prompt(self, strategy: str) -> str:
        """获取支持工具调用的系统提示词"""
        base_prompt = """你是一个专业的机票预订助手，可以使用各种工具来完成预订任务。

你可以使用以下工具：
- search_flights: 搜索航班
- filter_results: 筛选搜索结果
- select_flight: 选择特定航班
- add_to_cart: 添加航班到购物车
- proceed_to_payment: 进入支付页面
- enter_card: 输入支付卡信息
- confirm_payment: 确认支付
- apply_coupon: 应用优惠券
- restart: 重新开始预订流程
- abort: 中止预订

决策原则："""

        if strategy == "aggressive":
            return base_prompt + """
- 优先选择价格最低的航班
- 快速决策，减少犹豫
- 在预算范围内尽可能节省成本
- 如果遇到问题，快速重试"""

        elif strategy == "conservative":
            return base_prompt + """
- 优先考虑航班质量和可靠性
- 仔细评估每个选项
- 在不确定时选择更安全的选项
- 遇到问题时先分析再行动"""

        else:  # balanced
            return base_prompt + """
- 平衡价格和质量
- 考虑用户的预算约束和偏好
- 在多个选项间进行合理权衡
- 遇到问题时灵活应对"""

    def _handle_search_flights(self, args: Dict, obs: Dict) -> str:
        """处理搜索航班工具调用"""
        reason = args.get("reason", "开始搜索航班")
        logger.info(f"搜索航班: {reason}")
        return "search_flights"

    def _handle_filter_results(self, args: Dict, obs: Dict) -> str:
        """处理筛选结果工具调用"""
        max_price = args.get("max_price")
        max_stops = args.get("max_stops")
        airline_preference = args.get("airline_preference")

        logger.info(f"筛选结果: 最大价格={max_price}, 最大转机={max_stops}, 航空公司偏好={airline_preference}")
        return "filter_results"

    def _handle_select_flight(self, args: Dict, obs: Dict) -> str:
        """处理选择航班工具调用"""
        flight_id = args.get("flight_id")
        reason = args.get("reason", "选择航班")
        logger.info(f"选择航班 {flight_id}: {reason}")
        return "select_flight"

    def _handle_add_to_cart(self, args: Dict, obs: Dict) -> str:
        """处理添加到购物车工具调用"""
        reason = args.get("reason", "添加到购物车")
        logger.info(f"添加到购物车: {reason}")
        return "add_to_cart"

    def _handle_proceed_to_payment(self, args: Dict, obs: Dict) -> str:
        """处理进入支付工具调用"""
        reason = args.get("reason", "进入支付")
        logger.info(f"进入支付: {reason}")
        return "proceed_to_payment"

    def _handle_enter_card(self, args: Dict, obs: Dict) -> str:
        """处理输入卡片工具调用"""
        card_type = args.get("card_type", "信用卡")
        logger.info(f"输入卡片信息: {card_type}")
        return "enter_card"

    def _handle_confirm_payment(self, args: Dict, obs: Dict) -> str:
        """处理确认支付工具调用"""
        reason = args.get("reason", "确认支付")
        logger.info(f"确认支付: {reason}")
        return "confirm_payment"

    def _handle_apply_coupon(self, args: Dict, obs: Dict) -> str:
        """处理应用优惠券工具调用"""
        coupon_code = args.get("coupon_code", "")
        logger.info(f"应用优惠券: {coupon_code}")
        return "apply_coupon"

    def _handle_restart(self, args: Dict, obs: Dict) -> str:
        """处理重新开始工具调用"""
        reason = args.get("reason", "重新开始")
        logger.info(f"重新开始: {reason}")
        return "restart"

    def _handle_abort(self, args: Dict, obs: Dict) -> str:
        """处理中止预订工具调用"""
        reason = args.get("reason", "中止预订")
        logger.info(f"中止预订: {reason}")
        return "abort"
