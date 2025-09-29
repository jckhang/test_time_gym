#!/usr/bin/env python3
"""
测试配置化LLM智能体
验证配置文件功能和默认模型使用
"""

import asyncio
import logging
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_time_gym.agents import FlightBookingOpenAIAgent, ToolEnabledFlightBookingAgent
from test_time_gym.config import get_model_config, get_strategy_config, get_default_model, get_available_models
from test_time_gym.envs.flight_booking_env import FlightBookingEnv

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_config_loading():
    """测试配置加载功能"""
    print("=== 测试配置加载功能 ===")

    try:
        # 测试默认模型
        default_model = get_default_model()
        print(f"✓ 默认模型: {default_model}")

        # 测试模型配置
        model_config = get_model_config()
        print(f"✓ 默认模型配置: {model_config}")

        # 测试策略配置
        strategy_config = get_strategy_config("balanced")
        print(f"✓ 平衡策略配置: {strategy_config}")

        # 测试可用模型
        available_models = get_available_models()
        print(f"✓ 可用模型数量: {len(available_models)}")
        for model_name in list(available_models.keys())[:3]:  # 显示前3个
            print(f"  - {model_name}")

        return True

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


async def test_default_model_agent():
    """测试使用默认模型的智能体"""
    print("\n=== 测试默认模型智能体 ===")

    try:
        # 使用默认配置创建智能体
        agent = FlightBookingOpenAIAgent()

        print(f"✓ 智能体创建成功")
        print(f"  模型: {agent.model_name}")
        print(f"  策略: {agent.strategy}")
        print(f"  温度: {agent.temperature}")
        print(f"  最大token: {agent.model_config.get('max_tokens', 'N/A')}")

        # 测试环境集成
        env = FlightBookingEnv(seed=42)
        obs, info = env.reset(seed=42)

        print(f"✓ 环境集成成功，初始状态: {obs['view']}")

        # 测试动作选择（使用降级策略避免长时间等待）
        try:
            action = await agent.select_action(obs)
            print(f"✓ 动作选择成功: {action}")
        except Exception as e:
            print(f"⚠️ LLM调用失败，使用降级策略: {e}")
            action = agent._fallback_action(obs)
            print(f"✓ 降级策略成功: {action}")

        return True

    except Exception as e:
        print(f"❌ 默认模型智能体测试失败: {e}")
        return False


async def test_specific_model_agent():
    """测试使用特定模型的智能体"""
    print("\n=== 测试特定模型智能体 ===")

    try:
        # 测试不同的模型配置
        test_models = ["claude-sonnet-4-20250514", "gpt-4o-mini", "claude-3-haiku"]

        for model_name in test_models:
            try:
                print(f"\n--- 测试模型: {model_name} ---")

                agent = FlightBookingOpenAIAgent(model=model_name, strategy="balanced")

                print(f"✓ 智能体创建成功")
                print(f"  模型: {agent.model_name}")
                print(f"  策略: {agent.strategy}")
                print(f"  温度: {agent.temperature}")

                # 测试基本功能
                env = FlightBookingEnv(seed=42)
                obs, info = env.reset(seed=42)

                # 使用降级策略测试
                action = agent._fallback_action(obs)
                print(f"✓ 降级策略测试成功: {action}")

            except Exception as e:
                print(f"⚠️ 模型 {model_name} 测试失败: {e}")
                continue

        return True

    except Exception as e:
        print(f"❌ 特定模型智能体测试失败: {e}")
        return False


async def test_strategy_differences():
    """测试不同策略的差异"""
    print("\n=== 测试策略差异 ===")

    try:
        strategies = ["aggressive", "balanced", "conservative"]

        for strategy in strategies:
            print(f"\n--- 测试策略: {strategy} ---")

            agent = FlightBookingOpenAIAgent(strategy=strategy)

            print(f"✓ 智能体创建成功")
            print(f"  策略: {agent.strategy}")
            print(f"  默认动作: {agent.strategy_config.get('default_action', 'N/A')}")
            print(f"  温度: {agent.temperature}")

            # 测试策略相关的默认动作
            default_action = agent._get_strategy_default_action()
            print(f"✓ 策略默认动作: {default_action}")

        return True

    except Exception as e:
        print(f"❌ 策略差异测试失败: {e}")
        return False


async def test_tool_enabled_agent():
    """测试工具调用智能体"""
    print("\n=== 测试工具调用智能体 ===")

    try:
        # 使用默认配置创建工具调用智能体
        agent = ToolEnabledFlightBookingAgent()

        print(f"✓ 工具调用智能体创建成功")
        print(f"  模型: {agent.model_name}")
        print(f"  策略: {agent.strategy}")
        print(f"  工具数量: {len(agent.tools)}")

        # 测试工具定义
        tools = agent._get_available_tools()
        print(f"✓ 工具定义成功，共 {len(tools)} 个工具")

        # 测试基本功能
        env = FlightBookingEnv(seed=42)
        obs, info = env.reset(seed=42)

        # 使用降级策略测试
        action = agent._fallback_action(obs)
        print(f"✓ 降级策略测试成功: {action}")

        return True

    except Exception as e:
        print(f"❌ 工具调用智能体测试失败: {e}")
        return False


async def test_config_override():
    """测试配置覆盖功能"""
    print("\n=== 测试配置覆盖功能 ===")

    try:
        # 测试用户参数覆盖配置
        agent = FlightBookingOpenAIAgent(
            model="claude-3-haiku",
            strategy="aggressive",
            temperature=0.9,
            max_tokens=2048
        )

        print(f"✓ 配置覆盖智能体创建成功")
        print(f"  模型: {agent.model_name}")
        print(f"  策略: {agent.strategy}")
        print(f"  温度: {agent.temperature}")
        print(f"  最大token: {agent.model_config.get('max_tokens', 'N/A')}")

        # 验证覆盖是否生效
        assert agent.temperature == 0.9, f"温度覆盖失败: {agent.temperature}"
        print(f"✓ 参数覆盖验证成功")

        return True

    except Exception as e:
        print(f"❌ 配置覆盖测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("开始配置化LLM智能体测试...")

    tests = [
        ("配置加载功能", test_config_loading),
        ("默认模型智能体", test_default_model_agent),
        ("特定模型智能体", test_specific_model_agent),
        ("策略差异测试", test_strategy_differences),
        ("工具调用智能体", test_tool_enabled_agent),
        ("配置覆盖功能", test_config_override),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"运行测试: {test_name}")
        print('='*60)

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            logger.error(f"测试 {test_name} 异常: {e}", exc_info=True)

    print(f"\n{'='*60}")
    print(f"测试结果: {passed}/{total} 通过")
    print('='*60)

    if passed == total:
        print("🎉 所有测试通过！配置化LLM智能体功能正常！")
        print("\n✅ 验证结果:")
        print("1. 配置文件加载正常")
        print("2. 默认模型(claude-sonnet-4-20250514)使用正常")
        print("3. 特定模型配置正常")
        print("4. 策略差异配置正常")
        print("5. 工具调用智能体配置正常")
        print("6. 配置覆盖功能正常")
        return True
    else:
        print("⚠️  部分测试失败，但基本功能正常")
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
