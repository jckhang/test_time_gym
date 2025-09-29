"""
配置化LLM智能体使用示例
展示如何使用配置文件管理模型和策略
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_time_gym.agents import FlightBookingOpenAIAgent, ToolEnabledFlightBookingAgent
from test_time_gym.config import get_available_models, get_available_strategies, get_default_model
from test_time_gym.envs.flight_booking_env import FlightBookingEnv


async def example_default_model():
    """示例：使用默认模型"""
    print("=== 示例：使用默认模型 ===")

    # 使用默认配置创建智能体
    agent = FlightBookingOpenAIAgent()

    print(f"智能体信息:")
    print(f"  模型: {agent.model_name}")
    print(f"  策略: {agent.strategy}")
    print(f"  温度: {agent.temperature}")
    print(f"  最大token: {agent.model_config.get('max_tokens', 'N/A')}")

    # 测试环境交互
    env = FlightBookingEnv(seed=42)
    obs, info = env.reset(seed=42)

    print(f"\n环境交互测试:")
    print(f"  初始状态: {obs['view']}")

    # 使用降级策略进行测试
    action = agent._fallback_action(obs)
    print(f"  选择动作: {action}")

    return agent


async def example_specific_model():
    """示例：使用特定模型"""
    print("\n=== 示例：使用特定模型 ===")

    # 使用特定模型创建智能体
    agent = FlightBookingOpenAIAgent(
        model="claude-3-haiku",
        strategy="aggressive"
    )

    print(f"智能体信息:")
    print(f"  模型: {agent.model_name}")
    print(f"  策略: {agent.strategy}")
    print(f"  温度: {agent.temperature}")
    print(f"  策略描述: {agent.strategy_config.get('description', 'N/A')}")

    return agent


async def example_custom_parameters():
    """示例：自定义参数覆盖配置"""
    print("\n=== 示例：自定义参数覆盖配置 ===")

    # 使用自定义参数覆盖配置
    agent = FlightBookingOpenAIAgent(
        model="gpt-4o-mini",
        strategy="conservative",
        temperature=0.3,  # 覆盖配置中的温度
        max_tokens=2048   # 覆盖配置中的最大token数
    )

    print(f"智能体信息:")
    print(f"  模型: {agent.model_name}")
    print(f"  策略: {agent.strategy}")
    print(f"  温度: {agent.temperature} (用户自定义)")
    print(f"  最大token: {agent.model_config.get('max_tokens', 'N/A')}")

    return agent


async def example_tool_enabled_agent():
    """示例：工具调用智能体"""
    print("\n=== 示例：工具调用智能体 ===")

    # 创建工具调用智能体
    agent = ToolEnabledFlightBookingAgent(
        model="claude-sonnet-4-20250514",
        strategy="balanced"
    )

    print(f"工具调用智能体信息:")
    print(f"  模型: {agent.model_name}")
    print(f"  策略: {agent.strategy}")
    print(f"  工具数量: {len(agent.tools)}")

    # 显示部分工具
    print(f"  工具列表:")
    for i, tool in enumerate(agent.tools[:3]):  # 显示前3个工具
        tool_name = tool.get("function", {}).get("name", "Unknown")
        print(f"    {i+1}. {tool_name}")

    return agent


async def example_strategy_comparison():
    """示例：策略比较"""
    print("\n=== 示例：策略比较 ===")

    strategies = ["aggressive", "balanced", "conservative"]

    for strategy in strategies:
        agent = FlightBookingOpenAIAgent(strategy=strategy)

        print(f"\n{strategy.upper()} 策略:")
        print(f"  默认动作: {agent.strategy_config.get('default_action', 'N/A')}")
        print(f"  温度: {agent.temperature}")
        print(f"  描述: {agent.strategy_config.get('description', 'N/A')}")

        # 测试策略相关的默认动作
        default_action = agent._get_strategy_default_action()
        print(f"  实际默认动作: {default_action}")


async def example_available_models():
    """示例：查看可用模型"""
    print("\n=== 示例：查看可用模型 ===")

    # 获取可用模型
    models = get_available_models()
    strategies = get_available_strategies()
    default_model = get_default_model()

    print(f"默认模型: {default_model}")
    print(f"\n可用模型 ({len(models)} 个):")
    for model_name, config in models.items():
        description = config.get("description", "无描述")
        print(f"  - {model_name}: {description}")

    print(f"\n可用策略 ({len(strategies)} 个):")
    for strategy_name, config in strategies.items():
        description = config.get("description", "无描述")
        print(f"  - {strategy_name}: {description}")


async def example_environment_interaction():
    """示例：环境交互"""
    print("\n=== 示例：环境交互 ===")

    # 创建智能体和环境
    agent = FlightBookingOpenAIAgent()
    env = FlightBookingEnv(seed=42)
    obs, info = env.reset(seed=42)

    print(f"环境交互测试:")
    print(f"  智能体模型: {agent.model_name}")
    print(f"  智能体策略: {agent.strategy}")

    # 运行几步
    for step in range(3):
        print(f"\n步骤 {step + 1}:")
        print(f"  当前状态: {obs['view']}")

        # 使用降级策略进行测试
        action = agent._fallback_action(obs)
        print(f"  选择动作: {action}")

        # 执行动作
        obs, reward, done, trunc, info = env.step(action)
        print(f"  执行结果: 奖励={reward:.3f}, 新状态={obs['view']}")

        if done or trunc:
            print(f"  任务{'完成' if done else '被截断'}")
            break


async def main():
    """主函数"""
    print("配置化LLM智能体使用示例")
    print("=" * 50)

    try:
        # 运行各种示例
        await example_default_model()
        await example_specific_model()
        await example_custom_parameters()
        await example_tool_enabled_agent()
        await example_strategy_comparison()
        await example_available_models()
        await example_environment_interaction()

        print("\n" + "=" * 50)
        print("✅ 所有示例运行完成！")
        print("\n配置化LLM智能体的优势:")
        print("1. 统一的配置管理")
        print("2. 灵活的模型切换")
        print("3. 策略参数化")
        print("4. 用户参数覆盖")
        print("5. 默认模型支持")

    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
