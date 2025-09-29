#!/usr/bin/env python3
"""
LLM智能体离线测试脚本
验证基于LLM的智能体代码结构，不依赖实际LLM调用
"""

import asyncio
import logging
import sys
import os
from unittest.mock import Mock, AsyncMock

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_time_gym.agents import FlightBookingOpenAIAgent, ToolEnabledFlightBookingAgent
from test_time_gym.envs.flight_booking_env import FlightBookingEnv

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockLLMBackend:
    """模拟LLM后端，用于离线测试"""

    def __init__(self, model: str):
        self.model = model

    async def chat(self, messages, tools=None, tool_choice=None, max_tokens=15000):
        """模拟LLM响应"""
        # 简单的规则响应，不依赖实际LLM
        last_message = messages[-1]["content"] if messages else ""

        if "搜索" in last_message or "search" in last_message.lower():
            return {"content": "我需要搜索航班", "tool_calls": None}
        elif "筛选" in last_message or "filter" in last_message.lower():
            return {"content": "我需要筛选结果", "tool_calls": None}
        elif "选择" in last_message or "select" in last_message.lower():
            return {"content": "我选择这个航班", "tool_calls": None}
        elif "购物车" in last_message or "cart" in last_message.lower():
            return {"content": "添加到购物车", "tool_calls": None}
        elif "支付" in last_message or "payment" in last_message.lower():
            return {"content": "进入支付页面", "tool_calls": None}
        else:
            return {"content": "我需要搜索航班", "tool_calls": None}


def test_agent_initialization():
    """测试智能体初始化"""
    print("=== 测试智能体初始化 ===")

    try:
        # 测试基础智能体初始化
        agent = FlightBookingOpenAIAgent(
            model="gpt-4",
            strategy="balanced",
            temperature=0.7
        )
        print("✓ FlightBookingOpenAIAgent 初始化成功")

        # 测试工具调用智能体初始化
        tool_agent = ToolEnabledFlightBookingAgent(
            model="gpt-4",
            strategy="aggressive",
            temperature=0.5
        )
        print("✓ ToolEnabledFlightBookingAgent 初始化成功")

        # 测试不同策略
        strategies = ["aggressive", "balanced", "conservative"]
        for strategy in strategies:
            agent = FlightBookingOpenAIAgent(
                model="gpt-4",
                strategy=strategy
            )
            print(f"✓ {strategy} 策略智能体初始化成功")

        return True

    except Exception as e:
        print(f"❌ 智能体初始化失败: {e}")
        logger.error(f"初始化失败: {e}", exc_info=True)
        return False


def test_agent_memory_system():
    """测试智能体记忆系统"""
    print("\n=== 测试智能体记忆系统 ===")

    try:
        agent = FlightBookingOpenAIAgent(model="gpt-4", strategy="balanced")

        # 测试初始状态
        stats = agent.get_stats()
        assert stats["total_episodes"] == 0
        assert stats["conversation_turns"] == 0
        assert stats["skills_learned"] == 0
        print("✓ 初始记忆状态正确")

        # 测试记忆更新
        trajectory = [
            {"action": "search_flights", "reward": 0.1},
            {"action": "add_to_cart", "reward": 0.5},
            {"action": "proceed_to_payment", "reward": 1.0}
        ]
        agent.update_memory(trajectory)

        stats = agent.get_stats()
        assert stats["total_episodes"] == 1
        assert stats["skills_learned"] == 1
        print("✓ 记忆更新成功")

        # 测试对话历史
        agent.conversation_history = [
            {"observation": {"view": "search_form"}, "action": "search_flights", "response": {"content": "test"}},
            {"observation": {"view": "search_results"}, "action": "add_to_cart", "response": {"content": "test"}}
        ]

        stats = agent.get_stats()
        assert stats["conversation_turns"] == 2
        print("✓ 对话历史管理正确")

        return True

    except Exception as e:
        print(f"❌ 记忆系统测试失败: {e}")
        logger.error(f"记忆系统测试失败: {e}", exc_info=True)
        return False


def test_observation_formatting():
    """测试观察格式化功能"""
    print("\n=== 测试观察格式化功能 ===")

    try:
        agent = FlightBookingOpenAIAgent(model="gpt-4", strategy="balanced")

        # 测试不同视图的观察格式化
        test_observations = [
            {
                "view": "search_form",
                "search_params": {"origin": "北京", "destination": "上海", "date": "2024-01-01"}
            },
            {
                "view": "search_results",
                "flights": [
                    {"airline": "国航", "price": 500, "stops": 0, "depart": "08:00", "arrive": "10:00"},
                    {"airline": "东航", "price": 600, "stops": 1, "depart": "09:00", "arrive": "12:00"}
                ],
                "constraints": {"budget": 1000, "max_stops": 1}
            },
            {
                "view": "cart",
                "cart": {"total": 500, "items": 1, "flights": [{"airline": "国航", "price": 500}]}
            },
            {
                "view": "payment",
                "payment_state": {"card_entered": False, "confirmed": False}
            },
            {
                "view": "error",
                "error_message": "网络连接失败"
            }
        ]

        for obs in test_observations:
            formatted = agent._format_observation(obs)
            assert isinstance(formatted, str)
            assert len(formatted) > 0
            print(f"✓ {obs['view']} 视图格式化成功")

        return True

    except Exception as e:
        print(f"❌ 观察格式化测试失败: {e}")
        logger.error(f"观察格式化测试失败: {e}", exc_info=True)
        return False


def test_fallback_actions():
    """测试降级动作选择"""
    print("\n=== 测试降级动作选择 ===")

    try:
        agent = FlightBookingOpenAIAgent(model="gpt-4", strategy="balanced")

        # 测试不同视图的降级动作
        test_cases = [
            ({"view": "search_form"}, "search_flights"),
            ({"view": "search_results", "flights": []}, "filter_results"),
            ({"view": "search_results", "flights": [{"price": 500}]}, "add_to_cart"),
            ({"view": "cart"}, "proceed_to_payment"),
            ({"view": "payment", "payment_state": {"card_entered": False}}, "enter_card"),
            ({"view": "payment", "payment_state": {"card_entered": True, "confirmed": False}}, "confirm_payment"),
            ({"view": "error"}, "restart"),
            ({"view": "unknown"}, "search_flights")
        ]

        for obs, expected_action in test_cases:
            action = agent._fallback_action(obs)
            assert action == expected_action
            print(f"✓ {obs['view']} 视图降级动作: {action}")

        return True

    except Exception as e:
        print(f"❌ 降级动作测试失败: {e}")
        logger.error(f"降级动作测试失败: {e}", exc_info=True)
        return False


def test_strategy_differences():
    """测试不同策略的差异"""
    print("\n=== 测试策略差异 ===")

    try:
        strategies = ["aggressive", "balanced", "conservative"]

        for strategy in strategies:
            agent = FlightBookingOpenAIAgent(model="gpt-4", strategy=strategy)

            # 测试策略相关的默认动作
            default_action = agent._get_strategy_default_action()
            expected_actions = {
                "aggressive": "add_to_cart",
                "balanced": "search_flights",
                "conservative": "filter_results"
            }

            assert default_action == expected_actions[strategy]
            print(f"✓ {strategy} 策略默认动作: {default_action}")

        return True

    except Exception as e:
        print(f"❌ 策略差异测试失败: {e}")
        logger.error(f"策略差异测试失败: {e}", exc_info=True)
        return False


def test_tool_enabled_agent():
    """测试工具调用智能体"""
    print("\n=== 测试工具调用智能体 ===")

    try:
        agent = ToolEnabledFlightBookingAgent(model="gpt-4", strategy="balanced")

        # 测试工具定义
        tools = agent._get_available_tools()
        assert len(tools) > 0
        print(f"✓ 定义了 {len(tools)} 个工具")

        # 测试工具处理器
        assert hasattr(agent, '_handle_search_flights')
        assert hasattr(agent, '_handle_filter_results')
        assert hasattr(agent, '_handle_select_flight')
        print("✓ 工具处理器定义完整")

        # 测试工具调用处理
        test_args = {"reason": "测试原因"}
        test_obs = {"view": "search_form"}

        action = agent._handle_search_flights(test_args, test_obs)
        assert action == "search_flights"
        print("✓ 工具调用处理正常")

        return True

    except Exception as e:
        print(f"❌ 工具调用智能体测试失败: {e}")
        logger.error(f"工具调用智能体测试失败: {e}", exc_info=True)
        return False


def test_environment_integration():
    """测试环境集成"""
    print("\n=== 测试环境集成 ===")

    try:
        # 创建环境
        env = FlightBookingEnv(seed=42)
        obs, info = env.reset(seed=42)
        print(f"✓ 环境创建成功，初始状态: {obs['view']}")

        # 测试智能体与环境交互（不调用LLM）
        agent = FlightBookingOpenAIAgent(model="gpt-4", strategy="balanced")

        # 使用降级策略测试几步
        for step in range(3):
            action = agent._fallback_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            print(f"  步骤 {step}: {action} -> 奖励 {reward:.3f}, 状态: {obs['view']}")

            if done or trunc:
                break

        print("✓ 环境集成测试成功")
        return True

    except Exception as e:
        print(f"❌ 环境集成测试失败: {e}")
        logger.error(f"环境集成测试失败: {e}", exc_info=True)
        return False


def main():
    """主测试函数"""
    print("开始LLM智能体离线测试...")

    tests = [
        ("智能体初始化", test_agent_initialization),
        ("记忆系统", test_agent_memory_system),
        ("观察格式化", test_observation_formatting),
        ("降级动作", test_fallback_actions),
        ("策略差异", test_strategy_differences),
        ("工具调用智能体", test_tool_enabled_agent),
        ("环境集成", test_environment_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print('='*50)

        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            logger.error(f"测试 {test_name} 异常: {e}", exc_info=True)

    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 通过")
    print('='*50)

    if passed == total:
        print("🎉 所有离线测试通过！LLM智能体代码结构正确！")
        print("\n注意：这些测试验证了代码结构，但实际的LLM调用需要:")
        print("1. 正确的API密钥配置")
        print("2. 网络连接")
        print("3. 可用的LLM服务")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
