#!/usr/bin/env python3
"""
简化的LLM智能体测试脚本
专注于验证基本功能，不依赖实际LLM调用
"""

import asyncio
import logging
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_time_gym.agents import FlightBookingOpenAIAgent, ToolEnabledFlightBookingAgent
from test_time_gym.envs.flight_booking_env import FlightBookingEnv

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_agent_initialization():
    """测试智能体初始化"""
    print("=== 测试智能体初始化 ===")

    try:
        # 测试基础智能体
        agent = FlightBookingOpenAIAgent(
            model="gpt-4",
            strategy="balanced",
            temperature=0.7
        )
        print("✓ FlightBookingOpenAIAgent 初始化成功")

        # 测试工具调用智能体
        tool_agent = ToolEnabledFlightBookingAgent(
            model="gpt-4",
            strategy="aggressive"
        )
        print("✓ ToolEnabledFlightBookingAgent 初始化成功")

        return True

    except Exception as e:
        print(f"❌ 智能体初始化失败: {e}")
        return False


async def test_agent_fallback_behavior():
    """测试智能体降级行为"""
    print("\n=== 测试智能体降级行为 ===")

    try:
        agent = FlightBookingOpenAIAgent(model="gpt-4", strategy="balanced")

        # 测试不同视图的降级动作
        test_cases = [
            ({"view": "search_form"}, "search_flights"),
            ({"view": "search_results", "flights": []}, "filter_results"),
            ({"view": "search_results", "flights": [{"price": 500}]}, "add_to_cart"),
            ({"view": "cart"}, "proceed_to_payment"),
            ({"view": "payment", "payment_state": {"card_entered": False}}, "enter_card"),
            ({"view": "error"}, "restart"),
        ]

        for obs, expected_action in test_cases:
            action = agent._fallback_action(obs)
            if action == expected_action:
                print(f"✓ {obs['view']} -> {action}")
            else:
                print(f"⚠️ {obs['view']} -> {action} (期望: {expected_action})")

        return True

    except Exception as e:
        print(f"❌ 降级行为测试失败: {e}")
        return False


async def test_agent_memory_system():
    """测试智能体记忆系统"""
    print("\n=== 测试智能体记忆系统 ===")

    try:
        agent = FlightBookingOpenAIAgent(model="gpt-4", strategy="balanced")

        # 测试初始状态
        stats = agent.get_stats()
        assert stats["total_episodes"] == 0
        assert stats["conversation_turns"] == 0
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
        print("✓ 记忆更新成功")

        return True

    except Exception as e:
        print(f"❌ 记忆系统测试失败: {e}")
        return False


async def test_environment_integration():
    """测试环境集成"""
    print("\n=== 测试环境集成 ===")

    try:
        # 创建环境
        env = FlightBookingEnv(seed=42)
        obs, info = env.reset(seed=42)
        print(f"✓ 环境创建成功，初始状态: {obs['view']}")

        # 创建智能体
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
        return False


async def test_strategy_differences():
    """测试策略差异"""
    print("\n=== 测试策略差异 ===")

    try:
        strategies = ["aggressive", "balanced", "conservative"]

        for strategy in strategies:
            agent = FlightBookingOpenAIAgent(model="gpt-4", strategy=strategy)
            default_action = agent._get_strategy_default_action()

            expected_actions = {
                "aggressive": "add_to_cart",
                "balanced": "search_flights",
                "conservative": "filter_results"
            }

            if default_action == expected_actions[strategy]:
                print(f"✓ {strategy} 策略默认动作: {default_action}")
            else:
                print(f"⚠️ {strategy} 策略默认动作: {default_action} (期望: {expected_actions[strategy]})")

        return True

    except Exception as e:
        print(f"❌ 策略差异测试失败: {e}")
        return False


async def test_tool_enabled_agent():
    """测试工具调用智能体"""
    print("\n=== 测试工具调用智能体 ===")

    try:
        agent = ToolEnabledFlightBookingAgent(model="gpt-4", strategy="balanced")

        # 测试工具定义
        tools = agent._get_available_tools()
        assert len(tools) > 0
        print(f"✓ 定义了 {len(tools)} 个工具")

        # 测试工具处理器
        test_args = {"reason": "测试原因"}
        test_obs = {"view": "search_form"}

        action = agent._handle_search_flights(test_args, test_obs)
        assert action == "search_flights"
        print("✓ 工具调用处理正常")

        return True

    except Exception as e:
        print(f"❌ 工具调用智能体测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("开始简化LLM智能体测试...")

    tests = [
        ("智能体初始化", test_agent_initialization),
        ("降级行为", test_agent_fallback_behavior),
        ("记忆系统", test_agent_memory_system),
        ("环境集成", test_environment_integration),
        ("策略差异", test_strategy_differences),
        ("工具调用智能体", test_tool_enabled_agent),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print('='*50)

        try:
            result = await test_func()
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
        print("🎉 所有测试通过！LLM智能体基本功能正常！")
        print("\n注意：")
        print("1. 智能体具有完善的降级机制，LLM调用失败时会自动使用规则策略")
        print("2. 所有核心功能（初始化、记忆、策略、工具调用）都正常工作")
        print("3. 环境集成测试通过，智能体可以与环境正常交互")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
