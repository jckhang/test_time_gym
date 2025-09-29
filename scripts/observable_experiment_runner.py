#!/usr/bin/env python3
"""
可观测的实验运行器
集成观测系统到实验框架中
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

# 项目导入
from experiment_framework import ExperimentRunner, ExperienceEnhancedAgent
from observation_system import ObservationSystem
from test_time_gym.envs.flight_booking_env import FlightBookingEnv
from enhanced_skill_system import EnhancedSkillManager


class ObservableSkillManager(EnhancedSkillManager):
    """可观测的技能管理器"""
    
    def __init__(self, storage_dir: str, observation_system: Optional[ObservationSystem] = None):
        super().__init__(storage_dir)
        self.observation_system = observation_system
        self.experiment_name = storage_dir.split('/')[-1] if '/' in storage_dir else storage_dir
    
    def process_episode(self, trajectory: List[Dict], final_reward: float, episode_id: str):
        """处理episode并发送观测事件"""
        # 调用原始处理逻辑
        super().process_episode(trajectory, final_reward, episode_id)
        
        # 发送观测事件
        if self.observation_system:
            # 记录技能学习事件
            for skill_id, skill in self.skills.items():
                if hasattr(skill, '_just_updated'):  # 标记刚更新的技能
                    self.observation_system.log_skill_learned(
                        self.experiment_name,
                        skill.name,
                        skill.success_rate.mean(),
                        skill.usage_count
                    )
                    delattr(skill, '_just_updated')
    
    def select_skill_with_exploration(self, observation: Dict):
        """选择技能并记录使用"""
        selected_skill = super().select_skill_with_exploration(observation)
        
        # 记录技能使用
        if selected_skill and self.observation_system:
            self.observation_system.log_skill_usage(
                self.experiment_name,
                selected_skill.name,
                True  # 这里简化为True，实际应该根据后续结果判断
            )
        
        return selected_skill
    
    def update_skill(self, skill_id: str, success: bool, context: Dict = None):
        """更新技能并标记为刚更新"""
        # 调用原始更新逻辑
        if hasattr(super(), 'update_skill'):
            super().update_skill(skill_id, success, context)
        
        # 标记为刚更新
        if skill_id in self.skills:
            self.skills[skill_id]._just_updated = True


class ObservableAgent(ExperienceEnhancedAgent):
    """可观测的智能体"""
    
    def __init__(self, model: str = None, strategy: str = "balanced", 
                 skill_manager: Optional[ObservableSkillManager] = None,
                 use_experience: bool = True,
                 observation_system: Optional[ObservationSystem] = None,
                 experiment_name: str = "unknown"):
        super().__init__(model, strategy, skill_manager, use_experience)
        self.observation_system = observation_system
        self.experiment_name = experiment_name
        self.current_episode_id = None
        self.step_count = 0
    
    def start_episode(self, episode_id: str, initial_obs: Dict):
        """开始新的episode"""
        self.current_episode_id = episode_id
        self.step_count = 0
        
        if self.observation_system:
            self.observation_system.log_episode_start(
                self.experiment_name, 
                episode_id, 
                initial_obs
            )
    
    async def select_action(self, observation: Dict[str, Any]) -> str:
        """选择动作并记录观测"""
        # 调用原始动作选择
        action = await super().select_action(observation)
        
        # 记录步骤（假设奖励会在后续step中给出）
        if self.observation_system and self.current_episode_id:
            skill_used = None
            if (hasattr(self, 'current_trajectory') and 
                self.current_trajectory and 
                self.current_trajectory[-1].get('skill_used')):
                skill_used = self.current_trajectory[-1]['skill_used']
            
            self.observation_system.log_step(
                self.experiment_name,
                self.current_episode_id,
                self.step_count,
                action,
                observation,
                0,  # 奖励稍后更新
                skill_used=skill_used
            )
            
            self.step_count += 1
        
        return action
    
    def log_step_reward(self, reward: float, error: str = None):
        """记录步骤奖励"""
        # 这是一个额外的方法来记录奖励
        if self.observation_system and self.current_episode_id:
            # 更新最后一个步骤的奖励
            pass  # 在实际实现中可以考虑更新事件
    
    def end_episode(self, final_reward: float, episode_id: str):
        """结束episode并记录观测"""
        # 调用原始结束逻辑
        super().end_episode(final_reward, episode_id)
        
        # 记录episode结束
        if self.observation_system:
            success = final_reward > 0.5  # 简化的成功判断
            
            self.observation_system.log_episode_end(
                self.experiment_name,
                self.current_episode_id or episode_id,
                final_reward,
                success,
                self.current_trajectory
            )
        
        # 重置状态
        self.current_episode_id = None
        self.step_count = 0


class ObservableExperimentRunner(ExperimentRunner):
    """可观测的实验运行器"""
    
    def __init__(self, results_dir: str = "logs/experiments", 
                 enable_observation: bool = True,
                 web_port: int = 5000):
        super().__init__(results_dir)
        
        # 初始化观测系统
        self.observation_system = None
        if enable_observation:
            self.observation_system = ObservationSystem(
                enable_web=True, 
                web_port=web_port
            )
        
        self.active_experiments = []
    
    async def run_comparative_experiment(self, 
                                       num_episodes: int = 200,
                                       models: List[str] = None,
                                       strategies: List[str] = None) -> Dict:
        """运行可观测的对比实验"""
        
        if models is None:
            models = ["gpt-3.5-turbo"]
        if strategies is None:
            strategies = ["balanced"]
        
        # 准备实验名称列表
        experiment_names = []
        experiment_configs = [
            {"use_experience": False, "name": "baseline"},
            {"use_experience": True, "name": "with_experience"}
        ]
        
        for model in models:
            for strategy in strategies:
                for config in experiment_configs:
                    experiment_name = f"{model}_{strategy}_{config['name']}"
                    experiment_names.append(experiment_name)
        
        # 启动观测系统
        if self.observation_system:
            self.observation_system.start_monitoring(experiment_names)
            logging.info("🔬 观测系统已启动")
            logging.info(f"📊 Web仪表板: http://localhost:{self.observation_system.web_port}")
        
        # 运行实验
        results = {}
        
        for model in models:
            for strategy in strategies:
                for config in experiment_configs:
                    experiment_name = f"{model}_{strategy}_{config['name']}"
                    logging.info(f"开始可观测实验: {experiment_name}")
                    
                    # 运行实验
                    experiment_results = await self._run_observable_experiment(
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
        
        # 生成观测报告
        if self.observation_system:
            self.observation_system.generate_final_report(
                os.path.join(self.results_dir, "observation_reports")
            )
        
        return {
            "experiment_results": results,
            "comparison_report": comparison_report
        }
    
    async def _run_observable_experiment(self, 
                                       model: str,
                                       strategy: str, 
                                       use_experience: bool,
                                       num_episodes: int,
                                       experiment_name: str) -> Dict:
        """运行单个可观测实验"""
        
        # 创建环境
        env = FlightBookingEnv(seed=42)
        
        # 创建可观测的技能管理器
        skill_manager = ObservableSkillManager(
            f"{self.results_dir}/skills_{experiment_name}",
            self.observation_system
        )
        
        # 创建可观测的智能体
        agent = ObservableAgent(
            model=model,
            strategy=strategy,
            skill_manager=skill_manager,
            use_experience=use_experience,
            observation_system=self.observation_system,
            experiment_name=experiment_name
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
            episode_id = f"ep_{episode:03d}"
            episode_start_time = time.time()
            
            # 重置环境
            obs, info = env.reset(seed=42 + episode)
            
            # 开始episode观测
            agent.start_episode(episode_id, obs)
            
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
                    
                    # 记录奖励到观测系统
                    agent.log_step_reward(reward)
                    
                    # 记录步骤到轨迹
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
                    logging.error(f"Episode {episode} 步骤 {step} 出错: {e}")
                    # 记录错误到观测系统
                    if self.observation_system:
                        self.observation_system.log_step(
                            experiment_name, episode_id, step,
                            "error", obs, 0, error=str(e)
                        )
                    break
            
            # 结束episode
            episode_duration = time.time() - episode_start_time
            success = obs.get('view') == 'receipt' if 'obs' in locals() else False
            
            # 更新智能体经验
            agent.end_episode(total_reward, episode_id)
            
            # 结束轨迹记录
            trajectory_id = self.trajectory_logger.end_episode(total_reward, success)
            
            # 记录episode结果（保持原有格式）
            from test_time_gym.utils.evaluation import EpisodeResult
            episode_result = EpisodeResult(
                episode_id=episode_id,
                agent_type=experiment_name,
                seed=42 + episode,
                steps=step_count,
                total_reward=total_reward,
                final_reward=total_reward,
                success=success,
                constraint_violations=0,
                regret=0.0,
                exploration_steps=step_count - skill_usage_count,
                exploitation_steps=skill_usage_count,
                skill_calls=skill_usage_count,
                timestamp=datetime.now().isoformat(),
                trajectory=[]
            )
            
            episode_results.append(episode_result)
            self.metrics.add_episode(episode_result)
            
            # 更新学习曲线
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
            if (episode + 1) % 25 == 0:
                recent_success_rate = sum(1 for ep in episode_results[-25:] if ep.success) / 25
                logging.info(f"🔬 {experiment_name} - Episode {episode + 1}: "
                          f"最近25次成功率 {recent_success_rate:.3f}")
                
                # 如果启用了经验学习，报告技能统计
                if use_experience:
                    skill_stats = agent.get_learning_stats()
                    logging.info(f"🧠 技能统计: {skill_stats.get('total_skills', 0)} 个技能, "
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
    
    def cleanup(self):
        """清理资源"""
        if self.observation_system:
            self.observation_system.cleanup()


# 配置文件支持
class ObservationConfig:
    """观测配置"""
    
    @staticmethod
    def load_config(config_path: str = "observation_config.yaml") -> Dict:
        """加载观测配置"""
        default_config = {
            'observation': {
                'enabled': True,
                'web_dashboard': {
                    'enabled': True,
                    'port': 5000,
                    'host': '0.0.0.0'
                },
                'console_reporting': {
                    'enabled': True,
                    'interval_seconds': 10
                },
                'visualization': {
                    'real_time_plots': True,
                    'trajectory_analysis': True
                },
                'reporting': {
                    'auto_generate': True,
                    'output_dir': 'logs/observation_reports'
                }
            },
            'monitoring': {
                'episode_metrics': ['success_rate', 'avg_reward', 'avg_steps'],
                'skill_metrics': ['learning_rate', 'usage_frequency', 'success_rate'],
                'real_time_threshold': 0.1  # 秒
            }
        }
        
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                
                # 合并配置
                def merge_dicts(default, user):
                    for key, value in user.items():
                        if isinstance(value, dict) and key in default:
                            merge_dicts(default[key], value)
                        else:
                            default[key] = value
                
                merge_dicts(default_config, user_config)
                return default_config
            except Exception as e:
                logging.warning(f"加载配置文件失败: {e}，使用默认配置")
        
        return default_config


# 使用示例
async def main():
    """主函数 - 运行可观测实验"""
    
    # 加载配置
    config = ObservationConfig.load_config()
    
    # 创建可观测实验运行器
    runner = ObservableExperimentRunner(
        "logs/observable_experiments",
        enable_observation=config['observation']['enabled'],
        web_port=config['observation']['web_dashboard']['port']
    )
    
    # 运行对比实验
    print("🚀 开始可观测的无监督经验积累对比实验...")
    print("📊 打开浏览器访问 http://localhost:5000 查看实时监控")
    
    try:
        results = await runner.run_comparative_experiment(
            num_episodes=50,  # 为了演示，使用较少的episodes
            models=["gpt-3.5-turbo"],
            strategies=["balanced"]
        )
        
        # 打印结果摘要
        print("\n" + "="*60)
        print("🎯 可观测实验结果摘要")
        print("="*60)
        
        comparison_report = results["comparison_report"]
        overall = comparison_report.get("overall_improvement", {})
        
        if overall:
            print(f"📊 测试配置数量: {overall['configurations_tested']}")
            print(f"📈 有改进的配置: {overall['improvements_positive']}")
            print(f"🎯 平均成功率改进: {overall['avg_success_rate_improvement']:.3f}")
            print(f"⚡ 平均步数改进: {overall['avg_steps_improvement']:.3f}")
            print(f"🎪 平均稳定性改进: {overall['avg_stability_improvement']:.3f}")
        
        print(f"\n📁 详细结果已保存到: {runner.results_dir}")
        print("📊 观测报告已生成")
        
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        runner.cleanup()
        print("🧹 清理完成")


if __name__ == "__main__":
    asyncio.run(main())