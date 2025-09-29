"""
命令行界面
提供便捷的运行和评估命令
"""

import argparse
import sys
import os
import json
from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.agents.dummy_agent import DummyAgent, RandomAgent, SkillBasedAgent
from test_time_gym.utils.evaluation import EvaluationMetrics, TrajectoryLogger, Visualizer
from test_time_gym.utils.skill_system import SkillManager


def run_experiment(args):
    """运行实验"""
    print(f"开始运行实验: {args.agent_type}, {args.episodes} episodes")
    
    # 初始化组件
    env = FlightBookingEnv(seed=args.seed)
    
    if args.agent_type == "dummy":
        agent = DummyAgent(strategy="greedy")
    elif args.agent_type == "random":
        agent = RandomAgent()
    elif args.agent_type == "skill":
        agent = SkillBasedAgent(exploration_rate=args.exploration_rate)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")
    
    metrics = EvaluationMetrics()
    logger = TrajectoryLogger()
    skill_manager = SkillManager()
    
    # 运行episodes
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        logger.start_episode(args.agent_type, args.seed + episode, obs)
        
        total_reward = 0
        trajectory = []
        
        for step in range(50):  # 最大步数
            action = agent.select_action(obs)
            next_obs, reward, done, trunc, info = env.step(action)
            
            total_reward += reward
            logger.log_step(step, action, obs, reward, done, info)
            
            trajectory.append({
                "step": step,
                "action": action,
                "obs": obs,
                "reward": reward,
                "done": done
            })
            
            obs = next_obs
            
            if done or trunc:
                break
        
        # 记录结果
        success = total_reward > 0.5
        episode_id = logger.end_episode(total_reward, success)
        
        # 更新技能管理器
        skill_manager.add_trajectory(trajectory, total_reward)
        
        # 添加到评估指标
        from test_time_gym.utils.evaluation import EpisodeResult
        episode_result = EpisodeResult(
            episode_id=episode_id,
            agent_type=args.agent_type,
            seed=args.seed + episode,
            steps=len(trajectory),
            total_reward=total_reward,
            final_reward=trajectory[-1]["reward"] if trajectory else 0,
            success=success,
            constraint_violations=0,  # 简化
            regret=0,  # 简化
            exploration_steps=len(trajectory),  # 简化
            exploitation_steps=0,
            skill_calls=0,
            timestamp=str(episode),
            trajectory=trajectory
        )
        
        metrics.add_episode(episode_result)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: 奖励={total_reward:.3f}, 成功={'是' if success else '否'}")
    
    # 保存结果
    metrics.save_results()
    skill_manager.save_skills()
    
    # 生成报告
    if args.generate_report:
        visualizer = Visualizer(metrics)
        report_path = f"logs/evaluation/report_{args.agent_type}.html"
        visualizer.generate_evaluation_report(report_path)
        print(f"评估报告已保存: {report_path}")
    
    # 打印摘要
    print("\n=== 实验结果摘要 ===")
    print(f"总episodes: {len(metrics.episodes)}")
    print(f"成功率: {metrics.calculate_success_at_n():.3f}")
    print(f"平均步数: {metrics.calculate_avg_steps_to_success():.1f}")
    print(f"技能统计: {skill_manager.get_skill_stats()}")


def compare_agents(args):
    """比较不同智能体的性能"""
    print("开始智能体性能对比实验...")
    
    agent_configs = [
        {"type": "random", "name": "随机智能体"},
        {"type": "dummy", "name": "贪心智能体"},
        {"type": "skill", "name": "技能智能体"}
    ]
    
    all_metrics = EvaluationMetrics()
    
    for config in agent_configs:
        print(f"\n测试 {config['name']}...")
        
        # 使用相同的参数运行实验
        test_args = argparse.Namespace(
            agent_type=config["type"],
            episodes=args.episodes,
            seed=args.seed,
            exploration_rate=0.1,
            generate_report=False
        )
        
        # 这里应该调用run_experiment，但为了避免递归，我们简化实现
        env = FlightBookingEnv(seed=args.seed)
        
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            
            if config["type"] == "random":
                agent = RandomAgent()
            elif config["type"] == "dummy":
                agent = DummyAgent("greedy")
            else:
                agent = SkillBasedAgent()
            
            total_reward = 0
            for step in range(30):
                action = agent.select_action(obs)
                obs, reward, done, trunc, _ = env.step(action)
                total_reward += reward
                
                if done or trunc:
                    break
            
            # 添加结果（简化版）
            from test_time_gym.utils.evaluation import EpisodeResult
            episode_result = EpisodeResult(
                episode_id=f"{config['type']}_{episode}",
                agent_type=config["type"],
                seed=args.seed + episode,
                steps=step + 1,
                total_reward=total_reward,
                final_reward=reward,
                success=total_reward > 0.5,
                constraint_violations=0,
                regret=0,
                exploration_steps=step + 1,
                exploitation_steps=0,
                skill_calls=0,
                timestamp=str(episode),
                trajectory=[]
            )
            
            all_metrics.add_episode(episode_result)
    
    # 生成对比报告
    comparison_df = all_metrics.compare_agents()
    print("\n=== 智能体性能对比 ===")
    print(comparison_df.to_string(index=False))
    
    # 保存对比结果
    comparison_df.to_csv("logs/evaluation/agent_comparison.csv", index=False)
    print("对比结果已保存到 logs/evaluation/agent_comparison.csv")


def main():
    """主命令行入口"""
    parser = argparse.ArgumentParser(description="Test-Time Gym 命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 运行实验命令
    run_parser = subparsers.add_parser("run", help="运行单个智能体实验")
    run_parser.add_argument("--agent-type", choices=["dummy", "random", "skill"], 
                           default="dummy", help="智能体类型")
    run_parser.add_argument("--episodes", type=int, default=100, help="运行的episode数量")
    run_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    run_parser.add_argument("--exploration-rate", type=float, default=0.1, 
                           help="探索率（仅对skill agent有效）")
    run_parser.add_argument("--generate-report", action="store_true", 
                           help="生成评估报告")
    
    # 对比实验命令
    compare_parser = subparsers.add_parser("compare", help="比较不同智能体性能")
    compare_parser.add_argument("--episodes", type=int, default=50, help="每个智能体的episode数量")
    compare_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 可视化命令
    viz_parser = subparsers.add_parser("visualize", help="生成可视化报告")
    viz_parser.add_argument("--data-dir", default="logs/evaluation", 
                           help="数据目录")
    viz_parser.add_argument("--output", default="logs/evaluation/dashboard.html", 
                           help="输出文件路径")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_experiment(args)
    elif args.command == "compare":
        compare_agents(args)
    elif args.command == "visualize":
        # 生成可视化仪表板
        metrics = EvaluationMetrics()
        metrics.load_results()
        
        visualizer = Visualizer(metrics)
        visualizer.generate_evaluation_report(args.output)
        print(f"可视化报告已生成: {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()