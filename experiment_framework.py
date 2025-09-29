"""
无监督经验积累实验框架
设计对比实验来验证经验学习的有效性
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 项目导入
from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from test_time_gym.agents.openai_agent import FlightBookingOpenAIAgent
from test_time_gym.utils.evaluation import EvaluationMetrics, EpisodeResult, TrajectoryLogger
from enhanced_skill_system import EnhancedSkillManager, SemanticSkill

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperienceEnhancedAgent(FlightBookingOpenAIAgent):
    """增强的智能体，集成经验学习系统"""
    
    def __init__(self, model: str = None, strategy: str = "balanced", 
                 skill_manager: Optional[EnhancedSkillManager] = None,
                 use_experience: bool = True):
        super().__init__(model, strategy)
        self.skill_manager = skill_manager or EnhancedSkillManager()
        self.use_experience = use_experience
        self.current_trajectory = []
        self.episode_start_time = None
        
    async def select_action(self, observation: Dict[str, Any]) -> str:
        """选择动作，结合经验学习"""
        # 记录轨迹
        if self.episode_start_time is None:
            self.episode_start_time = time.time()
            self.current_trajectory = []
        
        # 如果启用经验学习，尝试使用学到的技能
        if self.use_experience:
            selected_skill = self.skill_manager.select_skill_with_exploration(observation)
            
            if selected_skill:
                # 使用技能的第一个动作
                if selected_skill.action_pattern:
                    action = selected_skill.action_pattern[0]
                    logger.info(f"使用技能 '{selected_skill.name}': {action}")
                    
                    # 记录技能使用
                    step_data = {
                        'obs': observation,
                        'action': action,
                        'skill_used': selected_skill.name,
                        'timestamp': time.time()
                    }
                    self.current_trajectory.append(step_data)
                    return action
        
        # 否则使用原始LLM决策
        action = await super().select_action(observation)
        
        # 记录步骤
        step_data = {
            'obs': observation,
            'action': action,
            'skill_used': None,
            'timestamp': time.time()
        }
        self.current_trajectory.append(step_data)
        
        return action
    
    def end_episode(self, final_reward: float, episode_id: str):
        """结束episode，更新经验"""
        if self.use_experience and self.current_trajectory:
            # 处理轨迹，提取技能
            self.skill_manager.process_episode(
                self.current_trajectory, 
                final_reward, 
                episode_id
            )
        
        # 重置状态
        self.current_trajectory = []
        self.episode_start_time = None
    
    def get_learning_stats(self) -> Dict:
        """获取学习统计"""
        if self.skill_manager:
            return self.skill_manager.get_skill_analytics()
        return {}


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, results_dir: str = "logs/experiments"):
        self.results_dir = results_dir
        self.metrics = EvaluationMetrics(f"{results_dir}/evaluation")
        self.trajectory_logger = TrajectoryLogger(f"{results_dir}/trajectories")
        
        os.makedirs(results_dir, exist_ok=True)
        
    async def run_comparative_experiment(self, 
                                       num_episodes: int = 200,
                                       models: List[str] = None,
                                       strategies: List[str] = None) -> Dict:
        """运行对比实验"""
        
        if models is None:
            models = ["gpt-3.5-turbo"]  # 默认模型
        if strategies is None:
            strategies = ["balanced"]
        
        results = {}
        
        # 实验配置
        experiment_configs = [
            {"use_experience": False, "name": "baseline"},
            {"use_experience": True, "name": "with_experience"}
        ]
        
        for model in models:
            for strategy in strategies:
                for config in experiment_configs:
                    experiment_name = f"{model}_{strategy}_{config['name']}"
                    logger.info(f"开始实验: {experiment_name}")
                    
                    # 运行实验
                    experiment_results = await self._run_single_experiment(
                        model=model,
                        strategy=strategy,
                        use_experience=config["use_experience"],
                        num_episodes=num_episodes,
                        experiment_name=experiment_name
                    )
                    
                    results[experiment_name] = experiment_results
                    
                    # 保存中间结果
                    self._save_intermediate_results(results)
        
        # 生成对比分析
        comparison_report = self._generate_comparison_report(results)
        
        return {
            "experiment_results": results,
            "comparison_report": comparison_report
        }
    
    async def _run_single_experiment(self, 
                                   model: str,
                                   strategy: str, 
                                   use_experience: bool,
                                   num_episodes: int,
                                   experiment_name: str) -> Dict:
        """运行单个实验"""
        
        # 创建环境
        env = FlightBookingEnv(seed=42)
        
        # 创建智能体
        skill_manager = EnhancedSkillManager(f"{self.results_dir}/skills_{experiment_name}")
        agent = ExperienceEnhancedAgent(
            model=model,
            strategy=strategy,
            skill_manager=skill_manager,
            use_experience=use_experience
        )
        
        # 实验统计
        episode_results = []
        learning_curves = {
            'success_rate': [],
            'avg_reward': [],
            'avg_steps': [],
            'skill_usage': []
        }
        
        # 运行episodes
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # 重置环境
            obs, info = env.reset(seed=42 + episode)
            
            # 开始记录轨迹
            self.trajectory_logger.start_episode(
                agent_type=experiment_name,
                seed=42 + episode,
                initial_obs=obs
            )
            
            total_reward = 0.0
            step_count = 0
            skill_usage_count = 0
            
            # 运行episode
            for step in range(50):  # 最大步数
                try:
                    action = await agent.select_action(obs)
                    
                    # 检查是否使用了技能
                    if (hasattr(agent, 'current_trajectory') and 
                        agent.current_trajectory and 
                        agent.current_trajectory[-1].get('skill_used')):
                        skill_usage_count += 1
                    
                    # 执行动作
                    obs, reward, done, trunc, info = env.step(action)
                    total_reward += reward
                    step_count += 1
                    
                    # 记录步骤
                    self.trajectory_logger.log_step(
                        step=step,
                        action=action,
                        observation=obs,
                        reward=reward,
                        done=done,
                        info=info
                    )
                    
                    if done or trunc:
                        break
                        
                except Exception as e:
                    logger.error(f"Episode {episode} 步骤 {step} 出错: {e}")
                    break
            
            # 结束episode
            episode_duration = time.time() - episode_start_time
            success = obs.get('view') == 'receipt' if 'obs' in locals() else False
            
            # 更新智能体经验
            episode_id = f"{experiment_name}_ep_{episode}"
            agent.end_episode(total_reward, episode_id)
            
            # 结束轨迹记录
            trajectory_id = self.trajectory_logger.end_episode(total_reward, success)
            
            # 记录episode结果
            episode_result = EpisodeResult(
                episode_id=episode_id,
                agent_type=experiment_name,
                seed=42 + episode,
                steps=step_count,
                total_reward=total_reward,
                final_reward=total_reward,
                success=success,
                constraint_violations=0,  # 简化
                regret=0.0,  # 简化
                exploration_steps=step_count - skill_usage_count,
                exploitation_steps=skill_usage_count,
                skill_calls=skill_usage_count,
                timestamp=datetime.now().isoformat(),
                trajectory=[]  # 简化，实际轨迹在trajectory_logger中
            )
            
            episode_results.append(episode_result)
            self.metrics.add_episode(episode_result)
            
            # 更新学习曲线（每10个episode计算一次）
            if (episode + 1) % 10 == 0:
                recent_episodes = episode_results[-10:]
                
                learning_curves['success_rate'].append(
                    sum(1 for ep in recent_episodes if ep.success) / len(recent_episodes)
                )
                learning_curves['avg_reward'].append(
                    np.mean([ep.total_reward for ep in recent_episodes])
                )
                learning_curves['avg_steps'].append(
                    np.mean([ep.steps for ep in recent_episodes])
                )
                learning_curves['skill_usage'].append(
                    np.mean([ep.skill_calls for ep in recent_episodes])
                )
            
            # 定期报告进度
            if (episode + 1) % 50 == 0:
                recent_success_rate = sum(1 for ep in episode_results[-50:] if ep.success) / 50
                logger.info(f"{experiment_name} - Episode {episode + 1}: "
                          f"最近50次成功率 {recent_success_rate:.3f}")
                
                # 如果启用了经验学习，报告技能统计
                if use_experience:
                    skill_stats = agent.get_learning_stats()
                    logger.info(f"技能统计: {skill_stats.get('total_skills', 0)} 个技能, "
                              f"平均成功率 {skill_stats.get('avg_success_rate', 0):.3f}")
        
        # 保存技能
        if use_experience:
            agent.skill_manager.save_skills()
        
        # 计算最终统计
        final_stats = self._calculate_experiment_stats(episode_results)
        learning_stats = agent.get_learning_stats() if use_experience else {}
        
        return {
            "config": {
                "model": model,
                "strategy": strategy,
                "use_experience": use_experience,
                "num_episodes": num_episodes
            },
            "episode_results": episode_results,
            "learning_curves": learning_curves,
            "final_stats": final_stats,
            "learning_stats": learning_stats
        }
    
    def _calculate_experiment_stats(self, episode_results: List[EpisodeResult]) -> Dict:
        """计算实验统计"""
        if not episode_results:
            return {}
        
        # 基础统计
        total_episodes = len(episode_results)
        successful_episodes = [ep for ep in episode_results if ep.success]
        success_rate = len(successful_episodes) / total_episodes
        
        avg_reward = np.mean([ep.total_reward for ep in episode_results])
        avg_steps = np.mean([ep.steps for ep in episode_results])
        avg_steps_to_success = np.mean([ep.steps for ep in successful_episodes]) if successful_episodes else float('inf')
        
        # 学习效果统计
        early_episodes = episode_results[:50] if len(episode_results) >= 50 else episode_results[:len(episode_results)//2]
        late_episodes = episode_results[-50:] if len(episode_results) >= 50 else episode_results[len(episode_results)//2:]
        
        early_success_rate = sum(1 for ep in early_episodes if ep.success) / len(early_episodes)
        late_success_rate = sum(1 for ep in late_episodes if ep.success) / len(late_episodes)
        improvement = late_success_rate - early_success_rate
        
        # 稳定性统计
        success_rates_by_window = []
        window_size = 20
        for i in range(0, len(episode_results) - window_size + 1, window_size):
            window_episodes = episode_results[i:i + window_size]
            window_success_rate = sum(1 for ep in window_episodes if ep.success) / len(window_episodes)
            success_rates_by_window.append(window_success_rate)
        
        stability = 1.0 - np.std(success_rates_by_window) if success_rates_by_window else 0.0
        
        return {
            "total_episodes": total_episodes,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "avg_steps_to_success": avg_steps_to_success,
            "early_success_rate": early_success_rate,
            "late_success_rate": late_success_rate,
            "improvement": improvement,
            "stability": stability,
            "skill_usage_rate": np.mean([ep.skill_calls / ep.steps for ep in episode_results if ep.steps > 0])
        }
    
    def _generate_comparison_report(self, all_results: Dict) -> Dict:
        """生成对比分析报告"""
        
        # 提取基线和经验学习结果
        baseline_results = {}
        experience_results = {}
        
        for exp_name, results in all_results.items():
            if "baseline" in exp_name:
                baseline_results[exp_name] = results
            elif "with_experience" in exp_name:
                experience_results[exp_name] = results
        
        comparisons = {}
        
        # 对每种配置进行对比
        for baseline_name, baseline_data in baseline_results.items():
            # 找到对应的经验学习实验
            config_prefix = baseline_name.replace("_baseline", "")
            experience_name = f"{config_prefix}_with_experience"
            
            if experience_name in experience_results:
                experience_data = experience_results[experience_name]
                
                # 计算改进
                baseline_stats = baseline_data["final_stats"]
                experience_stats = experience_data["final_stats"]
                
                comparison = {
                    "baseline_success_rate": baseline_stats["success_rate"],
                    "experience_success_rate": experience_stats["success_rate"],
                    "success_rate_improvement": experience_stats["success_rate"] - baseline_stats["success_rate"],
                    
                    "baseline_avg_steps": baseline_stats["avg_steps"],
                    "experience_avg_steps": experience_stats["avg_steps"],
                    "steps_improvement": baseline_stats["avg_steps"] - experience_stats["avg_steps"],
                    
                    "baseline_stability": baseline_stats["stability"],
                    "experience_stability": experience_stats["stability"],
                    "stability_improvement": experience_stats["stability"] - baseline_stats["stability"],
                    
                    "skill_usage_rate": experience_stats["skill_usage_rate"],
                    "total_skills_learned": experience_data["learning_stats"].get("total_skills", 0)
                }
                
                comparisons[config_prefix] = comparison
        
        # 计算总体改进
        if comparisons:
            overall_improvement = {
                "avg_success_rate_improvement": np.mean([c["success_rate_improvement"] for c in comparisons.values()]),
                "avg_steps_improvement": np.mean([c["steps_improvement"] for c in comparisons.values()]),
                "avg_stability_improvement": np.mean([c["stability_improvement"] for c in comparisons.values()]),
                "configurations_tested": len(comparisons),
                "improvements_positive": sum(1 for c in comparisons.values() if c["success_rate_improvement"] > 0)
            }
        else:
            overall_improvement = {}
        
        return {
            "detailed_comparisons": comparisons,
            "overall_improvement": overall_improvement,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _save_intermediate_results(self, results: Dict):
        """保存中间结果"""
        results_file = os.path.join(self.results_dir, "intermediate_results.json")
        
        # 将numpy数组转换为列表以便JSON序列化
        serializable_results = {}
        for exp_name, exp_data in results.items():
            serializable_data = {}
            for key, value in exp_data.items():
                if key == "episode_results":
                    # 简化episode结果
                    serializable_data[key] = len(value)
                elif key == "learning_curves":
                    serializable_data[key] = {k: list(v) for k, v in value.items()}
                else:
                    serializable_data[key] = value
            serializable_results[exp_name] = serializable_data
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def visualize_results(self, all_results: Dict, save_dir: str = None):
        """可视化实验结果"""
        if save_dir is None:
            save_dir = self.results_dir
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('无监督经验积累实验结果', fontsize=16)
        
        # 1. 成功率对比
        ax1 = axes[0, 0]
        baseline_names = []
        baseline_success = []
        experience_names = []
        experience_success = []
        
        for exp_name, results in all_results.items():
            success_rate = results["final_stats"]["success_rate"]
            if "baseline" in exp_name:
                baseline_names.append(exp_name.replace("_baseline", ""))
                baseline_success.append(success_rate)
            elif "with_experience" in exp_name:
                experience_names.append(exp_name.replace("_with_experience", ""))
                experience_success.append(success_rate)
        
        x = np.arange(len(baseline_names))
        width = 0.35
        
        ax1.bar(x - width/2, baseline_success, width, label='基线', alpha=0.8)
        ax1.bar(x + width/2, experience_success, width, label='经验学习', alpha=0.8)
        ax1.set_xlabel('实验配置')
        ax1.set_ylabel('成功率')
        ax1.set_title('成功率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(baseline_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 学习曲线
        ax2 = axes[0, 1]
        for exp_name, results in all_results.items():
            if "learning_curves" in results and results["learning_curves"]["success_rate"]:
                success_curve = results["learning_curves"]["success_rate"]
                episodes = range(10, len(success_curve) * 10 + 1, 10)
                ax2.plot(episodes, success_curve, 
                        label=exp_name, marker='o', markersize=4)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('成功率 (滑动平均)')
        ax2.set_title('学习曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 平均步数对比
        ax3 = axes[1, 0]
        baseline_steps = []
        experience_steps = []
        
        for exp_name, results in all_results.items():
            avg_steps = results["final_stats"]["avg_steps"]
            if "baseline" in exp_name:
                baseline_steps.append(avg_steps)
            elif "with_experience" in exp_name:
                experience_steps.append(avg_steps)
        
        ax3.bar(x - width/2, baseline_steps, width, label='基线', alpha=0.8)
        ax3.bar(x + width/2, experience_steps, width, label='经验学习', alpha=0.8)
        ax3.set_xlabel('实验配置')
        ax3.set_ylabel('平均步数')
        ax3.set_title('效率对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(baseline_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 技能学习统计
        ax4 = axes[1, 1]
        skill_counts = []
        skill_success_rates = []
        config_names = []
        
        for exp_name, results in all_results.items():
            if "with_experience" in exp_name and "learning_stats" in results:
                stats = results["learning_stats"]
                skill_counts.append(stats.get("total_skills", 0))
                skill_success_rates.append(stats.get("avg_success_rate", 0))
                config_names.append(exp_name.replace("_with_experience", ""))
        
        if skill_counts:
            ax4_twin = ax4.twinx()
            bars1 = ax4.bar(config_names, skill_counts, alpha=0.7, color='skyblue', label='技能数量')
            bars2 = ax4_twin.bar(config_names, skill_success_rates, alpha=0.7, color='orange', 
                                width=0.5, label='平均成功率')
            
            ax4.set_xlabel('实验配置')
            ax4.set_ylabel('学到的技能数量', color='skyblue')
            ax4_twin.set_ylabel('技能平均成功率', color='orange')
            ax4.set_title('技能学习统计')
            ax4.tick_params(axis='x', rotation=45)
            
            # 组合图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(save_dir, "experiment_results.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化结果已保存到: {save_path}")


# 使用示例
async def main():
    """主函数 - 运行完整实验"""
    
    # 创建实验运行器
    runner = ExperimentRunner("logs/unsupervised_learning_experiments")
    
    # 运行对比实验
    print("开始运行无监督经验积累对比实验...")
    
    results = await runner.run_comparative_experiment(
        num_episodes=100,  # 为了快速测试，可以增加到更多episodes
        models=["gpt-3.5-turbo"],
        strategies=["balanced", "aggressive"]
    )
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("实验结果摘要")
    print("="*60)
    
    comparison_report = results["comparison_report"]
    overall = comparison_report.get("overall_improvement", {})
    
    if overall:
        print(f"测试配置数量: {overall['configurations_tested']}")
        print(f"有改进的配置: {overall['improvements_positive']}")
        print(f"平均成功率改进: {overall['avg_success_rate_improvement']:.3f}")
        print(f"平均步数改进: {overall['avg_steps_improvement']:.3f}")
        print(f"平均稳定性改进: {overall['avg_stability_improvement']:.3f}")
        
        for config, details in comparison_report["detailed_comparisons"].items():
            print(f"\n{config}:")
            print(f"  成功率: {details['baseline_success_rate']:.3f} -> {details['experience_success_rate']:.3f} "
                  f"(+{details['success_rate_improvement']:.3f})")
            print(f"  平均步数: {details['baseline_avg_steps']:.1f} -> {details['experience_avg_steps']:.1f} "
                  f"(-{details['steps_improvement']:.1f})")
            print(f"  学到技能数: {details['total_skills_learned']}")
    
    # 生成可视化
    runner.visualize_results(results["experiment_results"])
    
    print(f"\n详细结果已保存到: {runner.results_dir}")
    

if __name__ == "__main__":
    asyncio.run(main())