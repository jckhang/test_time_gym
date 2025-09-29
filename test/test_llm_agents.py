#!/usr/bin/env python3
"""
LLM智能体测试脚本
验证基于LLM的智能体是否正常工作
"""

import asyncio
import logging
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_time_gym.agents import FlightBookingOpenAIAgent
from test_time_gym.envs.flight_booking_env import FlightBookingEnv

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_llm_agent_basic():
    """测试LLM智能体的基本功能"""
    print("=== 测试LLM智能体基本功能 ===")

    try:
        # 创建环境和智能体
        env = FlightBookingEnv(seed=42)
        agent = FlightBookingOpenAIAgent(
            model="gpt-4",
            strategy="balanced",
            temperature=0.7
        )

        print("✓ 环境和智能体创建成功")

        # 重置环境
        obs, info = env.reset(seed=42)
        print(f"✓ 环境重置成功，初始状态: {obs['view']}")

        # 测试智能体动作选择
        action = await agent.select_action(obs)
        print(f"✓ 智能体选择动作: {action}")

        # 执行一步
        obs, reward, done, trunc, info = env.step(action)
        print(f"✓ 环境执行成功，新状态: {obs['view']}, 奖励: {reward}")

        # 测试智能体统计
        stats = agent.get_stats()
        print(f"✓ 智能体统计: {stats}")

        print("✅ LLM智能体基本功能测试通过")
        return True

    except Exception as e:
        print(f"❌ LLM智能体测试失败: {e}")
        logger.error(f"测试失败: {e}", exc_info=True)
        return False


async def test_agent_strategies():
    """测试不同策略的智能体"""
    print("\n=== 测试不同策略 ===")

    strategies = ["aggressive", "balanced", "conservative"]

    for strategy in strategies:
        try:
            print(f"\n--- 测试 {strategy} 策略 ---")

            agent = FlightBookingOpenAIAgent(
                model="gpt-4",
                strategy=strategy
            )

            env = FlightBookingEnv(seed=42)
            obs, info = env.reset(seed=42)

            # 运行几步
            for step in range(3):
                action = await agent.select_action(obs)
                obs, reward, done, trunc, info = env.step(action)
                print(f"  步骤 {step}: {action} -> 奖励 {reward:.3f}")

                if done or trunc:
                    break

            print(f"✓ {strategy} 策略测试通过")

        except Exception as e:
            print(f"❌ {strategy} 策略测试失败: {e}")
            return False

    print("✅ 所有策略测试通过")
    return True


async def test_conversation_memory():
    """测试智能体对话记忆功能"""
    print("\n=== 测试对话记忆功能 ===")

    try:
        agent = FlightBookingOpenAIAgent(model="gpt-4", strategy="balanced")

        # 模拟多轮对话
        env = FlightBookingEnv(seed=42)
        obs, info = env.reset(seed=42)

        for step in range(3):
            action = await agent.select_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            print(f"步骤 {step}: {action}")

            if done or trunc:
                break

        # 检查对话历史
        history_length = len(agent.conversation_history)
        print(f"✓ 对话历史长度: {history_length}")

        # 测试记忆更新
        trajectory = [{"action": "test", "reward": 1.0}]
        agent.update_memory(trajectory)

        stats = agent.get_stats()
        print(f"✓ 记忆更新后统计: {stats}")

        print("✅ 对话记忆功能测试通过")
        return True

    except Exception as e:
        print(f"❌ 对话记忆功能测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("开始LLM智能体测试...")

    tests = [
        ("基本功能", test_llm_agent_basic),
        ("策略测试", test_agent_strategies),
        ("对话记忆", test_conversation_memory),
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
        print("🎉 所有测试通过！LLM智能体实现成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
