"""
评估和日志系统
实现多维度指标计算、学习曲线跟踪和可视化
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import os
import time
from datetime import datetime
import random

# 可选依赖 - 如果不存在也能工作
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass
class EpisodeResult:
    """单个episode的结果记录"""
    episode_id: str
    agent_type: str
    seed: int
    steps: int
    total_reward: float
    final_reward: float
    success: bool
    constraint_violations: int
    regret: float
    exploration_steps: int
    exploitation_steps: int
    skill_calls: int
    timestamp: str
    trajectory: List[Dict]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self, results_dir: str = "logs/evaluation"):
        self.results_dir = results_dir
        self.episodes: List[EpisodeResult] = []
        self.window_size = 100  # 滑动窗口大小
        
        os.makedirs(results_dir, exist_ok=True)
    
    def add_episode(self, episode: EpisodeResult):
        """添加episode结果"""
        self.episodes.append(episode)
        
        # 定期保存
        if len(self.episodes) % 50 == 0:
            self.save_results()
    
    def calculate_success_at_n(self, n: int = 10, agent_type: Optional[str] = None) -> float:
        """计算Success@N指标"""
        episodes = self._filter_episodes(agent_type)
        
        if len(episodes) < n:
            return 0.0
            
        # 计算每n个episode中的成功率
        success_counts = []
        for i in range(0, len(episodes) - n + 1, n):
            batch = episodes[i:i + n]
            success_count = sum(1 for ep in batch if ep.success)
            success_counts.append(success_count / n)
        
        return np.mean(success_counts) if success_counts else 0.0
    
    def calculate_avg_steps_to_success(self, agent_type: Optional[str] = None) -> float:
        """计算成功任务的平均步数"""
        episodes = self._filter_episodes(agent_type)
        successful_episodes = [ep for ep in episodes if ep.success]
        
        if not successful_episodes:
            return float('inf')
            
        return np.mean([ep.steps for ep in successful_episodes])
    
    def calculate_constraint_violation_rate(self, agent_type: Optional[str] = None) -> float:
        """计算约束违规率"""
        episodes = self._filter_episodes(agent_type)
        
        if not episodes:
            return 0.0
            
        total_violations = sum(ep.constraint_violations for ep in episodes)
        return total_violations / len(episodes)
    
    def calculate_regret(self, agent_type: Optional[str] = None) -> float:
        """计算平均后悔值"""
        episodes = self._filter_episodes(agent_type)
        
        if not episodes:
            return 0.0
            
        regrets = [ep.regret for ep in episodes if ep.regret is not None]
        return np.mean(regrets) if regrets else 0.0
    
    def calculate_exploration_ratio(self, agent_type: Optional[str] = None) -> float:
        """计算探索/利用比例"""
        episodes = self._filter_episodes(agent_type)
        
        if not episodes:
            return 0.0
            
        total_exploration = sum(ep.exploration_steps for ep in episodes)
        total_exploitation = sum(ep.exploitation_steps for ep in episodes)
        total_steps = total_exploration + total_exploitation
        
        return total_exploration / total_steps if total_steps > 0 else 0.0
    
    def get_learning_curve(self, metric: str = "success_rate", window_size: Optional[int] = None,
                          agent_type: Optional[str] = None) -> Tuple[List[int], List[float]]:
        """获取学习曲线"""
        episodes = self._filter_episodes(agent_type)
        window_size = window_size or self.window_size
        
        if len(episodes) < window_size:
            return [], []
            
        x_values = []
        y_values = []
        
        for i in range(window_size, len(episodes) + 1):
            window_episodes = episodes[i - window_size:i]
            
            if metric == "success_rate":
                value = sum(1 for ep in window_episodes if ep.success) / len(window_episodes)
            elif metric == "avg_steps":
                successful = [ep.steps for ep in window_episodes if ep.success]
                value = np.mean(successful) if successful else float('inf')
            elif metric == "avg_reward":
                value = np.mean([ep.total_reward for ep in window_episodes])
            elif metric == "regret":
                regrets = [ep.regret for ep in window_episodes if ep.regret is not None]
                value = np.mean(regrets) if regrets else 0.0
            else:
                continue
                
            x_values.append(i)
            y_values.append(value)
        
        return x_values, y_values
    
    def compare_agents(self, metrics: List[str] = None):
        """比较不同智能体的性能"""
        if metrics is None:
            metrics = ["success_rate", "avg_steps", "violation_rate", "regret", "exploration_ratio"]
        
        agent_types = set(ep.agent_type for ep in self.episodes)
        
        results = []
        for agent_type in agent_types:
            result = {"agent_type": agent_type}
            
            if "success_rate" in metrics:
                result["success_rate"] = self.calculate_success_at_n(agent_type=agent_type)
            if "avg_steps" in metrics:
                result["avg_steps"] = self.calculate_avg_steps_to_success(agent_type=agent_type)
            if "violation_rate" in metrics:
                result["violation_rate"] = self.calculate_constraint_violation_rate(agent_type=agent_type)
            if "regret" in metrics:
                result["regret"] = self.calculate_regret(agent_type=agent_type)
            if "exploration_ratio" in metrics:
                result["exploration_ratio"] = self.calculate_exploration_ratio(agent_type=agent_type)
                
            results.append(result)
        
        if HAS_PANDAS:
            return pd.DataFrame(results)
        else:
            return results
    
    def _filter_episodes(self, agent_type: Optional[str] = None) -> List[EpisodeResult]:
        """筛选episode"""
        if agent_type is None:
            return self.episodes
        return [ep for ep in self.episodes if ep.agent_type == agent_type]
    
    def save_results(self):
        """保存评估结果"""
        results_file = os.path.join(self.results_dir, "evaluation_results.jsonl")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            for episode in self.episodes:
                f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + '\n')
    
    def load_results(self):
        """加载评估结果"""
        results_file = os.path.join(self.results_dir, "evaluation_results.jsonl")
        
        if not os.path.exists(results_file):
            return
            
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        episode = EpisodeResult(**data)
                        self.episodes.append(episode)
        except Exception as e:
            print(f"Failed to load results: {e}")


class TrajectoryLogger:
    """轨迹记录器"""
    
    def __init__(self, log_dir: str = "logs/trajectories"):
        self.log_dir = log_dir
        self.current_trajectory = []
        self.episode_count = 0
        
        os.makedirs(log_dir, exist_ok=True)
    
    def start_episode(self, agent_type: str, seed: int, initial_obs: Dict):
        """开始新episode"""
        self.current_trajectory = []
        self.episode_count += 1
        self.agent_type = agent_type
        self.seed = seed
        self.start_time = time.time()
        
        # 记录初始状态
        self.log_step(
            step=0,
            action="reset",
            observation=initial_obs,
            reward=0.0,
            done=False,
            info={}
        )
    
    def log_step(self, step: int, action: str, observation: Dict, 
                 reward: float, done: bool, info: Dict):
        """记录单步信息"""
        step_record = {
            "step": step,
            "action": action,
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "timestamp": time.time()
        }
        
        self.current_trajectory.append(step_record)
    
    def end_episode(self, total_reward: float, success: bool) -> str:
        """结束episode并保存轨迹"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        episode_data = {
            "episode_id": f"ep_{self.episode_count:06d}",
            "agent_type": self.agent_type,
            "seed": self.seed,
            "total_reward": total_reward,
            "success": success,
            "steps": len(self.current_trajectory),
            "duration": duration,
            "trajectory": self.current_trajectory,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存到文件
        episode_file = os.path.join(
            self.log_dir, 
            f"{episode_data['episode_id']}_{self.agent_type}_{self.seed}.json"
        )
        
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        return episode_data["episode_id"]


class OODDetector:
    """分布外(OOD)检测器"""
    
    def __init__(self, reference_trajectories: List[Dict]):
        self.reference_trajectories = reference_trajectories
        self.state_embeddings = self._extract_state_embeddings()
        self.action_distributions = self._build_action_distributions()
        self.kl_threshold = 0.5  # KL散度阈值
        self.density_threshold = 0.1  # 密度阈值
    
    def _extract_state_embeddings(self) -> np.ndarray:
        """提取状态嵌入（简化版）"""
        embeddings = []
        
        for traj in self.reference_trajectories:
            for step in traj.get("trajectory", []):
                obs = step.get("observation", {})
                
                # 简化的状态编码
                embedding = [
                    float(obs.get("step", 0)),
                    float(len(obs.get("flights", []))),
                    float(obs.get("cart", {}).get("total", 0)),
                    float(obs.get("constraints", {}).get("budget", 0)),
                    float(obs.get("payment_state", {}).get("attempts", 0))
                ]
                
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _build_action_distributions(self) -> Dict[str, Counter]:
        """构建动作分布"""
        distributions = defaultdict(Counter)
        
        for traj in self.reference_trajectories:
            for step in traj.get("trajectory", []):
                view = step.get("observation", {}).get("view", "")
                action = step.get("action", "")
                distributions[view][action] += 1
        
        # 归一化
        for view in distributions:
            total = sum(distributions[view].values())
            for action in distributions[view]:
                distributions[view][action] /= total
        
        return dict(distributions)
    
    def detect_ood_state(self, observation: Dict) -> Tuple[bool, float]:
        """检测状态是否为OOD"""
        # 计算当前状态的嵌入
        current_embedding = np.array([
            float(observation.get("step", 0)),
            float(len(observation.get("flights", []))),
            float(observation.get("cart", {}).get("total", 0)),
            float(observation.get("constraints", {}).get("budget", 0)),
            float(observation.get("payment_state", {}).get("attempts", 0))
        ])
        
        # 计算与参考状态的最小距离
        if len(self.state_embeddings) == 0:
            return False, 0.0
            
        distances = np.linalg.norm(self.state_embeddings - current_embedding, axis=1)
        min_distance = np.min(distances)
        
        # 基于距离的OOD检测
        is_ood = min_distance > self.density_threshold
        
        return is_ood, min_distance
    
    def detect_ood_action(self, observation: Dict, action: str) -> Tuple[bool, float]:
        """检测动作是否为OOD"""
        view = observation.get("view", "")
        
        if view not in self.action_distributions:
            return True, 1.0  # 未见过的视图
            
        expected_dist = self.action_distributions[view]
        
        if action not in expected_dist:
            return True, 1.0  # 未见过的动作
            
        # 基于动作概率的OOD检测
        action_prob = expected_dist.get(action, 0.0)
        is_ood = action_prob < 0.05  # 低概率动作视为OOD
        
        return is_ood, 1.0 - action_prob


class Visualizer:
    """可视化工具"""
    
    def __init__(self, evaluation_metrics: EvaluationMetrics):
        self.metrics = evaluation_metrics
        
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """绘制学习曲线"""
        if not HAS_PLOTLY:
            print("警告: Plotly未安装，无法生成图表")
            return None
            
        agent_types = set(ep.agent_type for ep in self.metrics.episodes)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('成功率', '平均步数', '总奖励', '约束违规率'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, agent_type in enumerate(agent_types):
            color = colors[i % len(colors)]
            
            # 成功率曲线
            x, y = self.metrics.get_learning_curve("success_rate", agent_type=agent_type)
            if x and y:
                fig.add_trace(
                    go.Scatter(x=x, y=y, name=f"{agent_type}_success", 
                              line=dict(color=color), showlegend=True),
                    row=1, col=1
                )
            
            # 平均步数曲线
            x, y = self.metrics.get_learning_curve("avg_steps", agent_type=agent_type)
            if x and y:
                fig.add_trace(
                    go.Scatter(x=x, y=y, name=f"{agent_type}_steps", 
                              line=dict(color=color, dash='dash'), showlegend=False),
                    row=1, col=2
                )
            
            # 总奖励曲线
            x, y = self.metrics.get_learning_curve("avg_reward", agent_type=agent_type)
            if x and y:
                fig.add_trace(
                    go.Scatter(x=x, y=y, name=f"{agent_type}_reward", 
                              line=dict(color=color, dash='dot'), showlegend=False),
                    row=2, col=1
                )
        
        fig.update_layout(
            title="学习曲线对比",
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_skill_reliability(self, skill_manager, save_path: Optional[str] = None):
        """绘制技能可靠性分布"""
        if not HAS_PLOTLY:
            print("警告: Plotly未安装，无法生成图表")
            return None
            
        skills = skill_manager.get_best_skills(20)
        
        if not skills:
            return go.Figure()
        
        skill_names = [f"{skill.name[:10]}" for skill in skills]
        reliabilities = [skill.get_confidence() for skill in skills]
        usage_counts = [skill.usage_count for skill in skills]
        
        # 创建气泡图
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=skill_names,
            y=reliabilities,
            mode='markers',
            marker=dict(
                size=[min(count * 2, 50) for count in usage_counts],
                color=reliabilities,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="可靠性")
            ),
            text=[f"使用次数: {count}" for count in usage_counts],
            hovertemplate="<b>%{x}</b><br>可靠性: %{y:.3f}<br>%{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title="技能可靠性分布",
            xaxis_title="技能",
            yaxis_title="可靠性 (Beta均值)",
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """绘制性能对比图"""
        if not HAS_PLOTLY:
            print("警告: Plotly未安装，无法生成图表")
            return None
            
        comparison_df = self.metrics.compare_agents()
        
        if not comparison_df or (HAS_PANDAS and comparison_df.empty):
            return go.Figure()
        
        # 创建雷达图
        categories = ['success_rate', 'avg_steps_inv', 'violation_rate_inv', 'regret_inv']
        
        fig = go.Figure()
        
        for _, row in comparison_df.iterrows():
            values = [
                row.get('success_rate', 0),
                1 / (row.get('avg_steps', 1) + 1),  # 反转，越小越好
                1 - row.get('violation_rate', 0),   # 反转，越小越好
                1 / (row.get('regret', 1) + 1)      # 反转，越小越好
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=['成功率', '效率', '约束遵守', '价格优化'],
                fill='toself',
                name=row['agent_type']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="智能体性能对比雷达图"
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def generate_evaluation_report(self, output_path: str):
        """生成完整的评估报告"""
        report_dir = os.path.dirname(output_path)
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成各种图表
        learning_curve_fig = self.plot_learning_curves()
        performance_fig = self.plot_performance_comparison()
        
        # 生成统计摘要
        summary = self.metrics.compare_agents()
        
        # 创建HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test-Time Gym 评估报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Test-Time Gym 评估报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric">
                <h2>智能体性能对比</h2>
                {summary.to_html() if not summary.empty else "暂无数据"}
            </div>
            
            <div class="metric">
                <h2>学习曲线</h2>
                {learning_curve_fig.to_html(include_plotlyjs='inline') if learning_curve_fig.data else "暂无数据"}
            </div>
            
            <div class="metric">
                <h2>性能雷达图</h2>
                {performance_fig.to_html(include_plotlyjs='inline') if performance_fig.data else "暂无数据"}
            </div>
            
            <div class="metric">
                <h2>数据统计</h2>
                <p>总episode数: {len(self.metrics.episodes)}</p>
                <p>智能体类型: {', '.join(set(ep.agent_type for ep in self.metrics.episodes))}</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


class SafetyMonitor:
    """安全监控器"""
    
    def __init__(self, max_budget: float = 2000, max_attempts: int = 10):
        self.max_budget = max_budget
        self.max_attempts = max_attempts
        self.violation_count = 0
        self.intervention_count = 0
        
    def check_action_safety(self, observation: Dict, action: str) -> Tuple[bool, str]:
        """检查动作安全性"""
        # 预算检查
        if "confirm_payment" in action:
            cart_total = observation.get("cart", {}).get("total", 0)
            if cart_total > self.max_budget:
                self.intervention_count += 1
                return False, f"支付金额 ${cart_total} 超过安全限制 ${self.max_budget}"
        
        # 支付尝试次数检查
        if "confirm_payment" in action:
            attempts = observation.get("payment_state", {}).get("attempts", 0)
            if attempts >= self.max_attempts:
                self.intervention_count += 1
                return False, f"支付尝试次数 {attempts} 超过限制 {self.max_attempts}"
        
        return True, "安全"
    
    def log_violation(self, violation_type: str, details: Dict):
        """记录安全违规"""
        self.violation_count += 1
        
        # 这里可以添加更详细的日志记录
        print(f"安全违规 #{self.violation_count}: {violation_type}")
        print(f"详情: {details}")
    
    def get_safety_stats(self) -> Dict:
        """获取安全统计"""
        return {
            "total_violations": self.violation_count,
            "total_interventions": self.intervention_count,
            "intervention_rate": self.intervention_count / max(1, self.violation_count + self.intervention_count)
        }


if __name__ == "__main__":
    # 测试评估系统
    metrics = EvaluationMetrics()
    logger = TrajectoryLogger()
    
    # 模拟一些episode结果
    for i in range(10):
        episode = EpisodeResult(
            episode_id=f"test_{i}",
            agent_type="test_agent",
            seed=42 + i,
            steps=random.randint(5, 20),
            total_reward=random.uniform(-0.5, 1.5),
            final_reward=random.choice([1.0, -0.3, 0.0]),
            success=random.choice([True, False]),
            constraint_violations=random.randint(0, 2),
            regret=random.uniform(0, 100),
            exploration_steps=random.randint(1, 10),
            exploitation_steps=random.randint(1, 10),
            skill_calls=random.randint(0, 5),
            timestamp=datetime.now().isoformat(),
            trajectory=[]
        )
        
        metrics.add_episode(episode)
    
    # 打印统计结果
    print("成功率:", metrics.calculate_success_at_n())
    print("平均步数:", metrics.calculate_avg_steps_to_success())
    print("约束违规率:", metrics.calculate_constraint_violation_rate())
    
    # 生成可视化
    visualizer = Visualizer(metrics)
    fig = visualizer.plot_learning_curves()
    
    if fig.data:
        print("学习曲线图已生成")
    else:
        print("数据不足，无法生成学习曲线")