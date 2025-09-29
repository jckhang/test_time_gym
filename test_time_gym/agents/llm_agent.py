"""
基于LLM的智能体实现
使用大语言模型进行决策和动作选择
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..llm.base import ChatBackend, OpenAIBackend

logger = logging.getLogger(__name__)


class LLMAgent:
    """基于大语言模型的智能体基类"""

    def __init__(
        self,
        backend: ChatBackend,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.7
    ):
        """
        初始化LLM智能体

        Args:
            backend: LLM后端实现
            system_prompt: 系统提示词
            max_retries: 最大重试次数
            temperature: 生成温度
        """
        self.backend = backend
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.max_retries = max_retries
        self.temperature = temperature
        self.conversation_history = []
        self.memory = []  # 存储交互历史
        self.skills = {}  # 技能库

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个智能的机票预订助手。你的任务是帮助用户完成机票预订流程。

环境状态说明：
- search_form: 搜索表单页面，需要填写出发地、目的地、日期等信息
- search_results: 搜索结果页面，显示可选的航班
- cart: 购物车页面，显示已选择的航班
- payment: 支付页面，需要填写支付信息
- receipt: 收据页面，预订完成
- error: 错误状态，需要处理错误

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

请根据当前环境状态和用户需求，选择最合适的动作。优先考虑用户的预算约束和偏好。"""

    async def select_action(self, observation: Dict[str, Any]) -> str:
        """
        根据观察选择动作

        Args:
            observation: 环境观察

        Returns:
            选择的动作
        """
        try:
            # 构建消息
            messages = self._build_messages(observation)

            # 调用LLM
            response = await self.backend.chat(messages)

            # 解析响应
            action = self._parse_response(response)

            # 更新对话历史
            self.conversation_history.append({
                "observation": observation,
                "action": action,
                "response": response
            })

            return action

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            # 降级到默认策略
            return self._fallback_action(observation)

    def _build_messages(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """构建发送给LLM的消息"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 添加对话历史
        for entry in self.conversation_history[-5:]:  # 只保留最近5轮对话
            messages.append({
                "role": "user",
                "content": f"环境状态: {json.dumps(entry['observation'], ensure_ascii=False)}"
            })
            messages.append({
                "role": "assistant",
                "content": f"选择动作: {entry['action']}"
            })

        # 添加当前观察
        current_context = self._format_observation(observation)
        messages.append({
            "role": "user",
            "content": f"当前环境状态: {current_context}\n请选择下一步动作。"
        })

        return messages

    def _format_observation(self, observation: Dict[str, Any]) -> str:
        """格式化观察为可读文本"""
        view = observation.get("view", "unknown")
        context = f"当前页面: {view}\n"

        if view == "search_form":
            context += "需要填写搜索表单信息"
        elif view == "search_results":
            flights = observation.get("flights", [])
            constraints = observation.get("constraints", {})
            context += f"找到 {len(flights)} 个航班选项\n"
            context += f"预算约束: {constraints.get('budget', '无限制')}\n"
            context += f"最大转机次数: {constraints.get('max_stops', '无限制')}\n"
            if flights:
                context += "航班选项:\n"
                for i, flight in enumerate(flights[:3]):  # 只显示前3个
                    context += f"  {i+1}. {flight.get('airline', 'Unknown')} - ¥{flight.get('price', 0)} - {flight.get('stops', 0)}次转机\n"
        elif view == "cart":
            cart = observation.get("cart", {})
            context += f"购物车总价: ¥{cart.get('total', 0)}\n"
            context += f"购物车项目数: {cart.get('items', 0)}"
        elif view == "payment":
            payment_state = observation.get("payment_state", {})
            context += f"支付状态: 卡片已输入={payment_state.get('card_entered', False)}, 已确认={payment_state.get('confirmed', False)}"
        elif view == "error":
            error_msg = observation.get("error_message", "未知错误")
            context += f"错误信息: {error_msg}"

        return context

    def _parse_response(self, response: Dict[str, Any]) -> str:
        """解析LLM响应，提取动作"""
        content = response.get("content", "")

        # 尝试从响应中提取动作
        action_keywords = {
            "search_flights": ["搜索", "search", "查找"],
            "filter_results": ["筛选", "filter", "过滤"],
            "select_flight": ["选择", "select", "挑选"],
            "add_to_cart": ["添加", "add", "加入购物车"],
            "proceed_to_payment": ["支付", "payment", "付款"],
            "enter_card": ["输入卡片", "enter card", "填写支付"],
            "confirm_payment": ["确认支付", "confirm", "确认"],
            "apply_coupon": ["优惠券", "coupon", "折扣"],
            "restart": ["重新开始", "restart", "重试"],
            "abort": ["中止", "abort", "取消"]
        }

        content_lower = content.lower()
        for action, keywords in action_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return action

        # 如果无法识别，返回默认动作
        logger.warning(f"无法从响应中识别动作: {content}")
        return "search_flights"

    def _fallback_action(self, observation: Dict[str, Any]) -> str:
        """降级策略：当LLM调用失败时使用简单规则"""
        view = observation.get("view", "unknown")

        if view == "search_form":
            return "search_flights"
        elif view == "search_results":
            flights = observation.get("flights", [])
            if flights:
                return "add_to_cart"
            else:
                return "filter_results"
        elif view == "cart":
            return "proceed_to_payment"
        elif view == "payment":
            payment_state = observation.get("payment_state", {})
            if not payment_state.get("card_entered", False):
                return "enter_card"
            elif not payment_state.get("confirmed", False):
                return "confirm_payment"
            else:
                return "restart"
        elif view == "error":
            return "restart"
        else:
            return "search_flights"

    def update_memory(self, trajectory: List[Dict]):
        """更新智能体记忆"""
        self.memory.append(trajectory)

        # 简单的技能提取
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
            "conversation_turns": len(self.conversation_history),
            "skills_learned": len(self.skills),
            "top_skills": sorted(
                self.skills.items(),
                key=lambda x: x[1]["success"],
                reverse=True
            )[:5]
        }

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []


class OpenAILLMAgent(LLMAgent):
    """使用OpenAI后端的LLM智能体"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.7
    ):
        """
        初始化OpenAI LLM智能体

        Args:
            model: OpenAI模型名称
            system_prompt: 系统提示词
            max_retries: 最大重试次数
            temperature: 生成温度
        """
        backend = OpenAIBackend(model)
        super().__init__(backend, system_prompt, max_retries, temperature)


class ToolEnabledLLMAgent(LLMAgent):
    """支持工具调用的LLM智能体"""

    def __init__(
        self,
        backend: ChatBackend,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        temperature: float = 0.7
    ):
        """
        初始化支持工具的LLM智能体

        Args:
            backend: LLM后端实现
            tools: 可用工具列表
            system_prompt: 系统提示词
            max_retries: 最大重试次数
            temperature: 生成温度
        """
        super().__init__(backend, system_prompt, max_retries, temperature)
        self.tools = tools
        self.tool_handlers = self._register_tool_handlers()

    def _register_tool_handlers(self) -> Dict[str, callable]:
        """注册工具处理器"""
        return {
            "search_flights": self._handle_search_flights,
            "filter_results": self._handle_filter_results,
            "select_flight": self._handle_select_flight,
            "add_to_cart": self._handle_add_to_cart,
            "proceed_to_payment": self._handle_proceed_to_payment,
            "enter_card": self._handle_enter_card,
            "confirm_payment": self._handle_confirm_payment,
            "apply_coupon": self._handle_apply_coupon,
            "restart": self._handle_restart,
            "abort": self._handle_abort
        }

    async def select_action(self, observation: Dict[str, Any]) -> str:
        """支持工具调用的动作选择"""
        try:
            messages = self._build_messages(observation)

            # 调用LLM，支持工具调用
            response = await self.backend.chat(
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            # 处理工具调用
            if response.get("tool_calls"):
                return self._handle_tool_calls(response["tool_calls"], observation)
            else:
                # 普通响应
                action = self._parse_response(response)
                return action

        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            return self._fallback_action(observation)

    def _handle_tool_calls(self, tool_calls: List[Dict], observation: Dict[str, Any]) -> str:
        """处理工具调用"""
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            if function_name in self.tool_handlers:
                try:
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    return self.tool_handlers[function_name](arguments, observation)
                except Exception as e:
                    logger.error(f"工具调用处理失败: {e}")
                    continue

        # 如果没有成功处理任何工具调用，使用降级策略
        return self._fallback_action(observation)

    # 工具处理器方法
    def _handle_search_flights(self, args: Dict, obs: Dict) -> str:
        return "search_flights"

    def _handle_filter_results(self, args: Dict, obs: Dict) -> str:
        return "filter_results"

    def _handle_select_flight(self, args: Dict, obs: Dict) -> str:
        return "select_flight"

    def _handle_add_to_cart(self, args: Dict, obs: Dict) -> str:
        return "add_to_cart"

    def _handle_proceed_to_payment(self, args: Dict, obs: Dict) -> str:
        return "proceed_to_payment"

    def _handle_enter_card(self, args: Dict, obs: Dict) -> str:
        return "enter_card"

    def _handle_confirm_payment(self, args: Dict, obs: Dict) -> str:
        return "confirm_payment"

    def _handle_apply_coupon(self, args: Dict, obs: Dict) -> str:
        return "apply_coupon"

    def _handle_restart(self, args: Dict, obs: Dict) -> str:
        return "restart"

    def _handle_abort(self, args: Dict, obs: Dict) -> str:
        return "abort"
