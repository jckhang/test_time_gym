"""
LLM智能体使用示例
展示如何使用基于大语言模型的智能体进行机票预订
"""

import asyncio
import logging
from test_time_gym.agents import (
    FlightBookingOpenAIAgent,
    ToolEnabledFlightBookingAgent
)
from test_time_gym.envs.flight_booking_env import FlightBookingEnv

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_llm_agent():
    """测试基础LLM智能体"""
    print("=== 测试基础LLM智能体 ===")

    # 创建环境和智能体
    env = FlightBookingEnv(seed=42)
    agent = FlightBookingOpenAIAgent(
        model="gpt-4",
        strategy="balanced",
        temperature=0.7
    )

    # 重置环境
    obs, info = env.reset(seed=42)
    trajectory = []
    total_reward = 0

    print(f"初始状态: {obs['view']}")

    for step in range(20):  # 最多20步
        try:
            # 智能体选择动作
            action = await agent.select_action(obs)
            print(f"步骤 {step}: 选择动作 '{action}'")

            # 环境执行动作
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward

            trajectory.append({
                "step": step,
                "action": action,
                "reward": reward,
                "obs": obs
            })

            print(f"  奖励: {reward:.3f}, 新状态: {obs['view']}")

            if done or trunc:
                print(f"任务{'完成' if done else '被截断'}")
                break

        except Exception as e:
            logger.error(f"步骤 {step} 出错: {e}")
            break

    # 更新智能体记忆
    agent.update_memory(trajectory)

    print(f"\n总奖励: {total_reward:.3f}")
    print(f"智能体统计: {agent.get_stats()}")

    return total_reward


async def test_tool_enabled_agent():
    """测试支持工具调用的智能体"""
    print("\n=== 测试工具调用智能体 ===")

    # 创建环境和智能体
    env = FlightBookingEnv(seed=42)
    agent = ToolEnabledFlightBookingAgent(
        model="gpt-4",
        strategy="aggressive",
        temperature=0.5
    )

    # 重置环境
    obs, info = env.reset(seed=42)
    trajectory = []
    total_reward = 0

    print(f"初始状态: {obs['view']}")

    for step in range(20):  # 最多20步
        try:
            # 智能体选择动作（支持工具调用）
            action = await agent.select_action(obs)
            print(f"步骤 {step}: 选择动作 '{action}'")

            # 环境执行动作
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward

            trajectory.append({
                "step": step,
                "action": action,
                "reward": reward,
                "obs": obs
            })

            print(f"  奖励: {reward:.3f}, 新状态: {obs['view']}")

            if done or trunc:
                print(f"任务{'完成' if done else '被截断'}")
                break

        except Exception as e:
            logger.error(f"步骤 {step} 出错: {e}")
            break

    # 更新智能体记忆
    agent.update_memory(trajectory)

    print(f"\n总奖励: {total_reward:.3f}")
    print(f"智能体统计: {agent.get_stats()}")

    return total_reward


async def compare_strategies():
    """比较不同策略的智能体性能"""
    print("\n=== 比较不同策略 ===")

    strategies = ["aggressive", "balanced", "conservative"]
    results = {}

    for strategy in strategies:
        print(f"\n--- 测试 {strategy} 策略 ---")

        env = FlightBookingEnv(seed=42)
        agent = FlightBookingOpenAIAgent(
            model="gpt-4",
            strategy=strategy,
            temperature=0.7
        )

        obs, info = env.reset(seed=42)
        trajectory = []
        total_reward = 0

        for step in range(15):  # 限制步数
            try:
                action = await agent.select_action(obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward

                trajectory.append({
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "obs": obs
                })

                if done or trunc:
                    break

            except Exception as e:
                logger.error(f"策略 {strategy} 步骤 {step} 出错: {e}")
                break

        agent.update_memory(trajectory)
        results[strategy] = total_reward

        print(f"{strategy} 策略总奖励: {total_reward:.3f}")

    # 显示比较结果
    print(f"\n=== 策略比较结果 ===")
    for strategy, reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{strategy}: {reward:.3f}")

    return results


async def test_agent_learning():
    """测试智能体学习能力"""
    print("\n=== 测试智能体学习能力 ===")

    agent = FlightBookingOpenAIAgent(
        model="gpt-4",
        strategy="balanced"
    )

    # 运行多个episode
    episodes = 3
    episode_rewards = []

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")

        env = FlightBookingEnv(seed=42 + episode)  # 不同种子
        obs, info = env.reset(seed=42 + episode)
        trajectory = []
        total_reward = 0

        for step in range(15):
            try:
                action = await agent.select_action(obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward

                trajectory.append({
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "obs": obs
                })

                if done or trunc:
                    break

            except Exception as e:
                logger.error(f"Episode {episode} 步骤 {step} 出错: {e}")
                break

        agent.update_memory(trajectory)
        episode_rewards.append(total_reward)

        print(f"Episode {episode + 1} 奖励: {total_reward:.3f}")

    # 显示学习统计
    print(f"\n=== 学习统计 ===")
    print(f"平均奖励: {sum(episode_rewards) / len(episode_rewards):.3f}")
    print(f"奖励趋势: {episode_rewards}")
    print(f"智能体统计: {agent.get_stats()}")

    return episode_rewards


async def main():
    """主函数"""
    print("开始LLM智能体测试...")

    try:
        # 测试基础LLM智能体
        await test_basic_llm_agent()

        # 测试工具调用智能体
        await test_tool_enabled_agent()

        # 比较不同策略
        await compare_strategies()

        # 测试学习能力
        await test_agent_learning()

        print("\n=== 所有测试完成 ===")

    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        raise


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
