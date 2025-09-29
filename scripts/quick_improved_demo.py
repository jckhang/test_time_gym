#!/usr/bin/env python3
"""
快速启动改进环境演示
简化版本，快速展示核心功能
"""

import asyncio
import logging
import os
import sys
import time
import webbrowser

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from experiments.observation_system import ObservationSystem
from test_time_gym.envs.improved_flight_booking_env import ImprovedFlightBookingEnv


async def run_quick_demo():
    """运行快速演示"""

    print("🚀 改进环境快速演示")
    print("=" * 50)

    # 创建观测系统
    obs_system = ObservationSystem(enable_web=True, web_port=5000)

    # 启动监控
    experiments = ["quick_baseline", "quick_learning"]
    obs_system.start_monitoring(experiments)

    print("🌐 Web仪表板: http://localhost:5000")

    # 自动打开浏览器
    try:
        webbrowser.open("http://localhost:5000")
    except:
        pass

    await asyncio.sleep(3)

    # 运行两个对比实验
    await asyncio.gather(
        run_experiment(obs_system, "quick_baseline", "conservative", "easy", 10),
        run_experiment(obs_system, "quick_learning", "learning", "medium", 10)
    )

    print("\n🎉 演示完成! 查看Web界面中的结果")
    print("⏱️ 保持运行30秒...")
    await asyncio.sleep(30)

    # 生成报告
    obs_system.generate_final_report("logs/quick_demo_reports")
    obs_system.cleanup()


async def run_experiment(obs_system, exp_name, strategy, difficulty, episodes):
    """运行简单实验"""

    # 创建环境
    env = ImprovedFlightBookingEnv(
        seed=42,
        config={"difficulty": difficulty, "max_steps": 20}
    )

    # 简单智能体
    for episode in range(episodes):
        episode_id = f"ep_{episode:03d}"

        obs, info = env.reset()
        obs_system.log_episode_start(exp_name, episode_id, obs)

        total_reward = 0
        step = 0

        # 模拟技能学习
        if strategy == "learning" and episode == 3:
            obs_system.log_skill_learned(exp_name, "FastSearch", 0.75)

        while not env.done and not env.truncated:
            # 简单动作选择
            actions = obs.get('available_actions', ['restart'])
            action = actions[0] if actions else 'restart'

            next_obs, reward, done, truncated, step_info = env.step(action)

            # 技能使用
            skill_used = "FastSearch" if strategy == "learning" and episode > 3 and step % 3 == 0 else None
            if skill_used:
                obs_system.log_skill_usage(exp_name, skill_used, True)

            obs_system.log_step(exp_name, episode_id, step, action, next_obs, reward, skill_used)

            total_reward += reward
            step += 1
            obs = next_obs

            await asyncio.sleep(0.02)

        # 判断成功
        success = env.done and total_reward > 0.5
        obs_system.log_episode_end(exp_name, episode_id, total_reward, success)

        if (episode + 1) % 3 == 0:
            print(f"  {exp_name}: {episode + 1}/{episodes} episodes")

        await asyncio.sleep(0.1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("💡 这是一个快速演示版本")
    print("💡 完整演示请运行: python run_improved_env_demo.py")
    print()

    try:
        asyncio.run(run_quick_demo())
    except KeyboardInterrupt:
        print("\n👋 演示结束!")
    except Exception as e:
        print(f"❌ 错误: {e}")
