#!/usr/bin/env python3
"""
实验观测系统
提供实时监控、过程可视化和交互式分析功能
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import queue

# 可视化依赖
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

# Web界面依赖
try:
    import flask
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    WEB_ENABLED = True
except ImportError:
    WEB_ENABLED = False
    print("Web功能不可用。安装flask和flask-socketio以启用Web界面")

# 项目导入
from test_time_gym.utils.evaluation import EpisodeResult


@dataclass
class ObservationEvent:
    """观测事件数据结构"""
    timestamp: float
    event_type: str  # 'episode_start', 'step', 'episode_end', 'skill_learn', 'error'
    experiment_name: str
    episode_id: Optional[str] = None
    step_id: Optional[int] = None
    data: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)


class EventBus:
    """事件总线 - 用于组件间通信"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_queue = queue.Queue(maxsize=10000)
        self.is_running = True
        
    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        self.subscribers[event_type].append(callback)
    
    def publish(self, event: ObservationEvent):
        """发布事件"""
        try:
            self.event_queue.put_nowait(event)
            # 同步调用订阅者
            for callback in self.subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logging.error(f"事件处理错误: {e}")
        except queue.Full:
            logging.warning("事件队列已满，丢弃事件")
    
    def get_events(self, timeout: float = 0.1) -> List[ObservationEvent]:
        """获取事件批次"""
        events = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
            except queue.Empty:
                break
        
        return events


class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.current_experiments = {}
        self.episode_stats = defaultdict(list)
        
        # 订阅事件
        self.event_bus.subscribe('episode_start', self._on_episode_start)
        self.event_bus.subscribe('episode_end', self._on_episode_end)
        self.event_bus.subscribe('step', self._on_step)
        self.event_bus.subscribe('skill_learn', self._on_skill_learn)
        
    def _on_episode_start(self, event: ObservationEvent):
        """处理episode开始事件"""
        exp_name = event.experiment_name
        episode_id = event.episode_id
        
        if exp_name not in self.current_experiments:
            self.current_experiments[exp_name] = {}
        
        self.current_experiments[exp_name][episode_id] = {
            'start_time': event.timestamp,
            'steps': 0,
            'rewards': [],
            'actions': [],
            'skills_used': [],
            'errors': 0
        }
        
        logging.info(f"[监控] {exp_name} Episode {episode_id} 开始")
    
    def _on_episode_end(self, event: ObservationEvent):
        """处理episode结束事件"""
        exp_name = event.experiment_name
        episode_id = event.episode_id
        
        if exp_name in self.current_experiments and episode_id in self.current_experiments[exp_name]:
            episode_data = self.current_experiments[exp_name][episode_id]
            duration = event.timestamp - episode_data['start_time']
            
            # 计算统计
            total_reward = sum(episode_data['rewards'])
            success = event.data.get('success', False)
            
            stats = {
                'episode_id': episode_id,
                'duration': duration,
                'steps': episode_data['steps'],
                'total_reward': total_reward,
                'success': success,
                'skills_used': len(episode_data['skills_used']),
                'errors': episode_data['errors'],
                'timestamp': event.timestamp
            }
            
            self.episode_stats[exp_name].append(stats)
            
            # 更新实时指标
            self._update_metrics(exp_name, stats)
            
            # 清理当前episode数据
            del self.current_experiments[exp_name][episode_id]
            
            logging.info(f"[监控] {exp_name} Episode {episode_id} 完成: "
                        f"步数={stats['steps']}, 奖励={total_reward:.3f}, "
                        f"成功={success}, 用时={duration:.2f}s")
    
    def _on_step(self, event: ObservationEvent):
        """处理步骤事件"""
        exp_name = event.experiment_name
        episode_id = event.episode_id
        
        if (exp_name in self.current_experiments and 
            episode_id in self.current_experiments[exp_name]):
            
            episode_data = self.current_experiments[exp_name][episode_id]
            episode_data['steps'] += 1
            
            if event.data:
                episode_data['rewards'].append(event.data.get('reward', 0))
                episode_data['actions'].append(event.data.get('action', ''))
                
                if event.data.get('skill_used'):
                    episode_data['skills_used'].append(event.data['skill_used'])
                
                if event.data.get('error'):
                    episode_data['errors'] += 1
    
    def _on_skill_learn(self, event: ObservationEvent):
        """处理技能学习事件"""
        exp_name = event.experiment_name
        self.metrics[exp_name]['skills_learned'].append({
            'timestamp': event.timestamp,
            'skill_name': event.data.get('skill_name', ''),
            'success_rate': event.data.get('success_rate', 0)
        })
        
        logging.info(f"[监控] {exp_name} 学到新技能: {event.data.get('skill_name', '')}")
    
    def _update_metrics(self, exp_name: str, episode_stats: Dict):
        """更新实时指标"""
        metrics = self.metrics[exp_name]
        
        # 成功率（滑动窗口）
        recent_episodes = self.episode_stats[exp_name][-20:]
        success_rate = sum(1 for ep in recent_episodes if ep['success']) / len(recent_episodes)
        metrics['success_rate'].append((episode_stats['timestamp'], success_rate))
        
        # 平均奖励
        avg_reward = np.mean([ep['total_reward'] for ep in recent_episodes])
        metrics['avg_reward'].append((episode_stats['timestamp'], avg_reward))
        
        # 平均步数
        avg_steps = np.mean([ep['steps'] for ep in recent_episodes])
        metrics['avg_steps'].append((episode_stats['timestamp'], avg_steps))
        
        # 技能使用率
        if recent_episodes:
            skill_usage_rate = np.mean([ep['skills_used'] / max(1, ep['steps']) for ep in recent_episodes])
            metrics['skill_usage_rate'].append((episode_stats['timestamp'], skill_usage_rate))
    
    def get_current_stats(self, exp_name: str = None) -> Dict:
        """获取当前统计"""
        if exp_name:
            return {
                'experiment': exp_name,
                'metrics': dict(self.metrics[exp_name]),
                'episode_stats': self.episode_stats[exp_name][-10:],  # 最近10个episode
                'active_episodes': len(self.current_experiments.get(exp_name, {}))
            }
        else:
            return {
                'all_experiments': list(self.metrics.keys()),
                'active_experiments': {
                    exp: len(episodes) 
                    for exp, episodes in self.current_experiments.items()
                }
            }


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.trajectories = defaultdict(dict)  # {exp_name: {episode_id: trajectory}}
        self.fig = None
        self.axes = None
        
    def start_visualization(self, experiment_names: List[str]):
        """启动可视化"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('实验轨迹实时可视化', fontsize=16)
        
        # 订阅相关事件
        self.event_bus.subscribe('episode_end', self._on_episode_complete)
        
        # 设置子图
        self.axes[0, 0].set_title('Episode奖励轨迹')
        self.axes[0, 0].set_xlabel('步数')
        self.axes[0, 0].set_ylabel('累积奖励')
        
        self.axes[0, 1].set_title('动作分布')
        
        self.axes[1, 0].set_title('状态转换热图')
        
        self.axes[1, 1].set_title('技能使用统计')
        
        plt.tight_layout()
        
        # 启动动画更新
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plots, interval=2000, cache_frame_data=False
        )
        
        plt.show()
    
    def _on_episode_complete(self, event: ObservationEvent):
        """处理完成的episode"""
        exp_name = event.experiment_name
        episode_id = event.episode_id
        
        if event.data and 'trajectory' in event.data:
            self.trajectories[exp_name][episode_id] = event.data['trajectory']
    
    def _update_plots(self, frame):
        """更新图表"""
        if not self.trajectories:
            return
        
        # 清空所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. Episode奖励轨迹
        ax1 = self.axes[0, 0]
        ax1.set_title('Episode奖励轨迹')
        ax1.set_xlabel('步数')
        ax1.set_ylabel('累积奖励')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        color_idx = 0
        
        for exp_name, episodes in self.trajectories.items():
            # 显示最近的几个episode
            recent_episodes = list(episodes.items())[-5:]
            
            for episode_id, trajectory in recent_episodes:
                if trajectory:
                    rewards = [step.get('reward', 0) for step in trajectory]
                    cumulative_rewards = np.cumsum(rewards)
                    steps = range(len(cumulative_rewards))
                    
                    ax1.plot(steps, cumulative_rewards, 
                            color=colors[color_idx % len(colors)], 
                            alpha=0.7, 
                            label=f'{exp_name[-10:]}_{episode_id[-3:]}')
            
            color_idx += 1
        
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 动作分布
        ax2 = self.axes[0, 1]
        ax2.set_title('动作分布')
        
        action_counts = defaultdict(int)
        for episodes in self.trajectories.values():
            for trajectory in list(episodes.values())[-10:]:  # 最近10个episode
                if trajectory:
                    for step in trajectory:
                        action = step.get('action', 'unknown')
                        action_counts[action] += 1
        
        if action_counts:
            actions = list(action_counts.keys())
            counts = list(action_counts.values())
            ax2.bar(actions, counts)
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. 状态转换热图（简化版）
        ax3 = self.axes[1, 0]
        ax3.set_title('视图转换频率')
        
        view_transitions = defaultdict(int)
        for episodes in self.trajectories.values():
            for trajectory in list(episodes.values())[-10:]:
                if trajectory and len(trajectory) > 1:
                    for i in range(len(trajectory) - 1):
                        from_view = trajectory[i].get('obs', {}).get('view', 'unknown')
                        to_view = trajectory[i + 1].get('obs', {}).get('view', 'unknown')
                        view_transitions[f"{from_view}→{to_view}"] += 1
        
        if view_transitions:
            transitions = list(view_transitions.keys())[:10]  # 显示前10个
            counts = [view_transitions[t] for t in transitions]
            ax3.barh(transitions, counts)
        
        # 4. 技能使用统计
        ax4 = self.axes[1, 1]
        ax4.set_title('技能使用统计')
        
        skill_usage = defaultdict(int)
        for episodes in self.trajectories.values():
            for trajectory in list(episodes.values())[-10:]:
                if trajectory:
                    for step in trajectory:
                        skill = step.get('skill_used')
                        if skill:
                            skill_usage[skill] += 1
        
        if skill_usage:
            skills = list(skill_usage.keys())
            counts = list(skill_usage.values())
            ax4.pie(counts, labels=skills, autopct='%1.1f%%')
        
        plt.tight_layout()


class SkillAnalytics:
    """技能学习分析工具"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.skill_history = defaultdict(list)  # {exp_name: [skill_events]}
        self.skill_evolution = defaultdict(lambda: defaultdict(list))  # {exp_name: {skill_name: [(time, success_rate)]}}
        
        self.event_bus.subscribe('skill_learn', self._on_skill_update)
        self.event_bus.subscribe('skill_usage', self._on_skill_usage)
    
    def _on_skill_update(self, event: ObservationEvent):
        """处理技能更新事件"""
        exp_name = event.experiment_name
        skill_name = event.data.get('skill_name', '')
        success_rate = event.data.get('success_rate', 0)
        
        self.skill_history[exp_name].append({
            'timestamp': event.timestamp,
            'skill_name': skill_name,
            'event_type': 'learned',
            'success_rate': success_rate,
            'usage_count': event.data.get('usage_count', 0)
        })
        
        self.skill_evolution[exp_name][skill_name].append((event.timestamp, success_rate))
    
    def _on_skill_usage(self, event: ObservationEvent):
        """处理技能使用事件"""
        exp_name = event.experiment_name
        skill_name = event.data.get('skill_name', '')
        
        self.skill_history[exp_name].append({
            'timestamp': event.timestamp,
            'skill_name': skill_name,
            'event_type': 'used',
            'success': event.data.get('success', False)
        })
    
    def analyze_skill_learning_patterns(self, exp_name: str) -> Dict:
        """分析技能学习模式"""
        if exp_name not in self.skill_history:
            return {}
        
        history = self.skill_history[exp_name]
        
        # 技能发现时间线
        learned_skills = [event for event in history if event['event_type'] == 'learned']
        skill_discovery_timeline = [(event['timestamp'], event['skill_name']) for event in learned_skills]
        
        # 技能成功率演进
        skill_evolution = {}
        for skill_name, evolution in self.skill_evolution[exp_name].items():
            if evolution:
                times, rates = zip(*evolution)
                skill_evolution[skill_name] = {
                    'timestamps': list(times),
                    'success_rates': list(rates),
                    'final_rate': rates[-1],
                    'improvement': rates[-1] - rates[0] if len(rates) > 1 else 0
                }
        
        # 技能使用频率
        skill_usage_freq = defaultdict(int)
        for event in history:
            if event['event_type'] == 'used':
                skill_usage_freq[event['skill_name']] += 1
        
        return {
            'total_skills_learned': len(learned_skills),
            'skill_discovery_timeline': skill_discovery_timeline,
            'skill_evolution': skill_evolution,
            'skill_usage_frequency': dict(skill_usage_freq),
            'learning_efficiency': len(learned_skills) / len(history) if history else 0
        }
    
    def generate_skill_report(self, exp_name: str) -> str:
        """生成技能分析报告"""
        analysis = self.analyze_skill_learning_patterns(exp_name)
        
        if not analysis:
            return f"无技能学习数据: {exp_name}"
        
        report = [
            f"🧠 技能学习分析报告 - {exp_name}",
            "=" * 50,
            f"📊 总计学到技能: {analysis['total_skills_learned']} 个",
            f"⚡ 学习效率: {analysis['learning_efficiency']:.3f}",
            ""
        ]
        
        # 技能演进详情
        if analysis['skill_evolution']:
            report.append("🔄 技能演进:")
            for skill_name, evolution in analysis['skill_evolution'].items():
                improvement = evolution['improvement']
                report.append(f"  • {skill_name}: {evolution['final_rate']:.3f} "
                            f"(改进: {'📈+' if improvement > 0 else '📉'}{improvement:.3f})")
            report.append("")
        
        # 使用频率
        if analysis['skill_usage_frequency']:
            report.append("📈 使用频率:")
            sorted_usage = sorted(analysis['skill_usage_frequency'].items(), 
                                key=lambda x: x[1], reverse=True)
            for skill_name, freq in sorted_usage[:5]:  # 显示前5个
                report.append(f"  • {skill_name}: {freq} 次")
        
        return "\n".join(report)


class WebDashboard:
    """Web仪表板（如果Flask可用）"""
    
    def __init__(self, event_bus: EventBus, monitor: RealTimeMonitor, 
                 visualizer: TrajectoryVisualizer, analytics: SkillAnalytics):
        if not WEB_ENABLED:
            self.enabled = False
            return
        
        self.enabled = True
        self.event_bus = event_bus
        self.monitor = monitor
        self.visualizer = visualizer
        self.analytics = analytics
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'experiment_observation_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._setup_routes()
        self._setup_websocket()
    
    def _setup_routes(self):
        """设置Web路由"""
        
        @self.app.route('/')
        def index():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>实验观测仪表板</title>
                <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                    .panel { border: 1px solid #ccc; padding: 15px; border-radius: 5px; }
                    .metrics { background-color: #f9f9f9; }
                    .realtime-log { height: 300px; overflow-y: auto; background-color: #000; color: #0f0; font-family: monospace; padding: 10px; }
                    .chart { height: 400px; }
                </style>
            </head>
            <body>
                <h1>🔬 实验观测仪表板</h1>
                
                <div class="dashboard">
                    <div class="panel metrics">
                        <h3>📊 实时指标</h3>
                        <div id="metrics-content">等待数据...</div>
                    </div>
                    
                    <div class="panel">
                        <h3>📈 成功率趋势</h3>
                        <div id="success-rate-chart" class="chart"></div>
                    </div>
                    
                    <div class="panel">
                        <h3>💰 奖励趋势</h3>
                        <div id="reward-chart" class="chart"></div>
                    </div>
                    
                    <div class="panel">
                        <h3>🧠 技能分析</h3>
                        <div id="skill-analysis">等待数据...</div>
                    </div>
                </div>
                
                <div class="panel" style="margin-top: 20px;">
                    <h3>📝 实时日志</h3>
                    <div id="realtime-log" class="realtime-log"></div>
                </div>

                <script>
                    const socket = io();
                    
                    socket.on('metrics_update', function(data) {
                        updateMetrics(data);
                    });
                    
                    socket.on('chart_update', function(data) {
                        updateCharts(data);
                    });
                    
                    socket.on('log_message', function(data) {
                        addLogMessage(data.message);
                    });
                    
                    function updateMetrics(data) {
                        const container = document.getElementById('metrics-content');
                        let html = '';
                        
                        for (const [exp, stats] of Object.entries(data)) {
                            html += `<h4>${exp}</h4>`;
                            html += `<p>活跃Episodes: ${stats.active_episodes}</p>`;
                            if (stats.episode_stats && stats.episode_stats.length > 0) {
                                const latest = stats.episode_stats[stats.episode_stats.length - 1];
                                html += `<p>最新成功率: ${latest.success ? '✅' : '❌'}</p>`;
                                html += `<p>最新奖励: ${latest.total_reward.toFixed(3)}</p>`;
                            }
                        }
                        
                        container.innerHTML = html;
                    }
                    
                    function updateCharts(data) {
                        // 更新成功率图表
                        if (data.success_rate) {
                            Plotly.newPlot('success-rate-chart', data.success_rate, {
                                title: '成功率趋势',
                                xaxis: { title: '时间' },
                                yaxis: { title: '成功率' }
                            });
                        }
                        
                        // 更新奖励图表
                        if (data.reward) {
                            Plotly.newPlot('reward-chart', data.reward, {
                                title: '奖励趋势',
                                xaxis: { title: '时间' },
                                yaxis: { title: '平均奖励' }
                            });
                        }
                    }
                    
                    function addLogMessage(message) {
                        const log = document.getElementById('realtime-log');
                        const timestamp = new Date().toLocaleTimeString();
                        log.innerHTML += `[${timestamp}] ${message}\\n`;
                        log.scrollTop = log.scrollHeight;
                    }
                    
                    // 请求初始数据
                    socket.emit('request_data');
                </script>
            </body>
            </html>
            """
        
        @self.app.route('/api/experiments')
        def get_experiments():
            return jsonify(self.monitor.get_current_stats())
        
        @self.app.route('/api/experiment/<exp_name>')
        def get_experiment_details(exp_name):
            return jsonify(self.monitor.get_current_stats(exp_name))
        
        @self.app.route('/api/skills/<exp_name>')
        def get_skill_analysis(exp_name):
            return jsonify(self.analytics.analyze_skill_learning_patterns(exp_name))
    
    def _setup_websocket(self):
        """设置WebSocket事件"""
        
        @self.socketio.on('request_data')
        def handle_data_request():
            # 发送当前指标
            stats = {}
            for exp_name in self.monitor.metrics.keys():
                stats[exp_name] = self.monitor.get_current_stats(exp_name)
            
            emit('metrics_update', stats)
            
            # 发送图表数据
            chart_data = self._prepare_chart_data()
            emit('chart_update', chart_data)
    
    def _prepare_chart_data(self) -> Dict:
        """准备图表数据"""
        chart_data = {}
        
        # 成功率数据
        success_rate_traces = []
        reward_traces = []
        
        for exp_name, metrics in self.monitor.metrics.items():
            if 'success_rate' in metrics and metrics['success_rate']:
                times, rates = zip(*metrics['success_rate'])
                success_rate_traces.append({
                    'x': list(times),
                    'y': list(rates),
                    'name': exp_name,
                    'type': 'scatter'
                })
            
            if 'avg_reward' in metrics and metrics['avg_reward']:
                times, rewards = zip(*metrics['avg_reward'])
                reward_traces.append({
                    'x': list(times),
                    'y': list(rewards),
                    'name': exp_name,
                    'type': 'scatter'
                })
        
        chart_data['success_rate'] = success_rate_traces
        chart_data['reward'] = reward_traces
        
        return chart_data
    
    def send_log_message(self, message: str):
        """发送日志消息到Web界面"""
        if self.enabled:
            self.socketio.emit('log_message', {'message': message})
    
    def run(self, host='localhost', port=5000, debug=False):
        """运行Web服务器"""
        if self.enabled:
            logging.info(f"启动Web仪表板: http://{host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        else:
            logging.warning("Web仪表板不可用，请安装flask和flask-socketio")


class ObservationSystem:
    """观测系统主类"""
    
    def __init__(self, enable_web: bool = True, web_port: int = 5000):
        self.event_bus = EventBus()
        self.monitor = RealTimeMonitor(self.event_bus)
        self.visualizer = TrajectoryVisualizer(self.event_bus)
        self.analytics = SkillAnalytics(self.event_bus)
        
        # Web仪表板
        self.web_dashboard = None
        if enable_web and WEB_ENABLED:
            self.web_dashboard = WebDashboard(
                self.event_bus, self.monitor, self.visualizer, self.analytics
            )
            self.web_port = web_port
        
        # 控制台输出
        self.console_enabled = True
        self.last_report_time = 0
        self.report_interval = 10  # 每10秒报告一次
        
        logging.info("实验观测系统已初始化")
    
    def start_monitoring(self, experiment_names: List[str] = None):
        """开始监控"""
        logging.info("开始实验监控...")
        
        # 启动可视化
        if experiment_names:
            try:
                # 在后台线程启动可视化
                viz_thread = threading.Thread(
                    target=self.visualizer.start_visualization,
                    args=(experiment_names,),
                    daemon=True
                )
                viz_thread.start()
            except Exception as e:
                logging.warning(f"可视化启动失败: {e}")
        
        # 启动Web仪表板
        if self.web_dashboard:
            web_thread = threading.Thread(
                target=self.web_dashboard.run,
                kwargs={'host': '0.0.0.0', 'port': self.web_port, 'debug': False},
                daemon=True
            )
            web_thread.start()
        
        # 启动控制台报告
        if self.console_enabled:
            console_thread = threading.Thread(
                target=self._console_reporter,
                daemon=True
            )
            console_thread.start()
    
    def _console_reporter(self):
        """控制台报告器"""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_report_time >= self.report_interval:
                    self._print_console_report()
                    self.last_report_time = current_time
                
                time.sleep(1)
            except Exception as e:
                logging.error(f"控制台报告错误: {e}")
    
    def _print_console_report(self):
        """打印控制台报告"""
        stats = self.monitor.get_current_stats()
        
        if not stats.get('all_experiments'):
            return
        
        print("\n" + "="*60)
        print(f"📊 实验监控报告 - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        for exp_name in stats['all_experiments']:
            exp_stats = self.monitor.get_current_stats(exp_name)
            print(f"\n🔬 {exp_name}:")
            print(f"  活跃Episodes: {exp_stats['active_episodes']}")
            
            if exp_stats['episode_stats']:
                recent_episodes = exp_stats['episode_stats']
                recent_success_rate = sum(1 for ep in recent_episodes if ep['success']) / len(recent_episodes)
                avg_reward = np.mean([ep['total_reward'] for ep in recent_episodes])
                avg_steps = np.mean([ep['steps'] for ep in recent_episodes])
                
                print(f"  最近成功率: {recent_success_rate:.3f}")
                print(f"  平均奖励: {avg_reward:.3f}")
                print(f"  平均步数: {avg_steps:.1f}")
            
            # 技能分析
            skill_analysis = self.analytics.analyze_skill_learning_patterns(exp_name)
            if skill_analysis.get('total_skills_learned', 0) > 0:
                print(f"  学到技能: {skill_analysis['total_skills_learned']} 个")
                print(f"  学习效率: {skill_analysis['learning_efficiency']:.3f}")
    
    def log_episode_start(self, experiment_name: str, episode_id: str, initial_obs: Dict = None):
        """记录episode开始"""
        event = ObservationEvent(
            timestamp=time.time(),
            event_type='episode_start',
            experiment_name=experiment_name,
            episode_id=episode_id,
            data={'initial_obs': initial_obs} if initial_obs else None
        )
        self.event_bus.publish(event)
    
    def log_step(self, experiment_name: str, episode_id: str, step_id: int, 
                 action: str, observation: Dict, reward: float, 
                 skill_used: str = None, error: str = None):
        """记录步骤"""
        event = ObservationEvent(
            timestamp=time.time(),
            event_type='step',
            experiment_name=experiment_name,
            episode_id=episode_id,
            step_id=step_id,
            data={
                'action': action,
                'observation': observation,
                'reward': reward,
                'skill_used': skill_used,
                'error': error
            }
        )
        self.event_bus.publish(event)
    
    def log_episode_end(self, experiment_name: str, episode_id: str, 
                       total_reward: float, success: bool, trajectory: List[Dict] = None):
        """记录episode结束"""
        event = ObservationEvent(
            timestamp=time.time(),
            event_type='episode_end',
            experiment_name=experiment_name,
            episode_id=episode_id,
            data={
                'total_reward': total_reward,
                'success': success,
                'trajectory': trajectory or []
            }
        )
        self.event_bus.publish(event)
    
    def log_skill_learned(self, experiment_name: str, skill_name: str, 
                         success_rate: float, usage_count: int = 0):
        """记录技能学习"""
        event = ObservationEvent(
            timestamp=time.time(),
            event_type='skill_learn',
            experiment_name=experiment_name,
            data={
                'skill_name': skill_name,
                'success_rate': success_rate,
                'usage_count': usage_count
            }
        )
        self.event_bus.publish(event)
    
    def log_skill_usage(self, experiment_name: str, skill_name: str, success: bool):
        """记录技能使用"""
        event = ObservationEvent(
            timestamp=time.time(),
            event_type='skill_usage',
            experiment_name=experiment_name,
            data={
                'skill_name': skill_name,
                'success': success
            }
        )
        self.event_bus.publish(event)
    
    def generate_final_report(self, output_dir: str = "logs/observation_reports"):
        """生成最终报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成各种报告
        for exp_name in self.monitor.metrics.keys():
            # 统计报告
            stats_report = self._generate_stats_report(exp_name)
            with open(f"{output_dir}/stats_report_{exp_name}_{timestamp}.txt", 'w', encoding='utf-8') as f:
                f.write(stats_report)
            
            # 技能分析报告
            skill_report = self.analytics.generate_skill_report(exp_name)
            with open(f"{output_dir}/skill_report_{exp_name}_{timestamp}.txt", 'w', encoding='utf-8') as f:
                f.write(skill_report)
        
        logging.info(f"观测报告已保存到: {output_dir}")
    
    def _generate_stats_report(self, exp_name: str) -> str:
        """生成统计报告"""
        stats = self.monitor.get_current_stats(exp_name)
        
        report = [
            f"📊 实验统计报告 - {exp_name}",
            "=" * 50,
            f"📈 监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if stats['episode_stats']:
            episodes = stats['episode_stats']
            total_episodes = len(episodes)
            successful_episodes = sum(1 for ep in episodes if ep['success'])
            
            report.extend([
                f"🎮 总Episode数: {total_episodes}",
                f"✅ 成功Episode数: {successful_episodes}",
                f"📊 总体成功率: {successful_episodes / total_episodes:.3f}",
                f"💰 平均奖励: {np.mean([ep['total_reward'] for ep in episodes]):.3f}",
                f"👣 平均步数: {np.mean([ep['steps'] for ep in episodes]):.1f}",
                f"⏱️ 平均用时: {np.mean([ep['duration'] for ep in episodes]):.2f}s",
                ""
            ])
            
            # 趋势分析
            if len(episodes) >= 20:
                early_episodes = episodes[:10]
                late_episodes = episodes[-10:]
                
                early_success_rate = sum(1 for ep in early_episodes if ep['success']) / len(early_episodes)
                late_success_rate = sum(1 for ep in late_episodes if ep['success']) / len(late_episodes)
                improvement = late_success_rate - early_success_rate
                
                report.extend([
                    "📈 学习趋势:",
                    f"  初期成功率: {early_success_rate:.3f}",
                    f"  后期成功率: {late_success_rate:.3f}",
                    f"  改进幅度: {'📈+' if improvement > 0 else '📉'}{improvement:.3f}",
                    ""
                ])
        
        return "\n".join(report)
    
    def cleanup(self):
        """清理资源"""
        logging.info("正在清理观测系统...")
        self.event_bus.is_running = False
        
        # 生成最终报告
        try:
            self.generate_final_report()
        except Exception as e:
            logging.error(f"生成最终报告失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 创建观测系统
    obs_system = ObservationSystem(enable_web=True, web_port=5000)
    
    # 开始监控
    obs_system.start_monitoring(['test_experiment'])
    
    # 模拟一些实验数据
    import random
    
    for episode in range(10):
        episode_id = f"ep_{episode:03d}"
        
        # 开始episode
        obs_system.log_episode_start('test_experiment', episode_id)
        
        total_reward = 0
        for step in range(random.randint(5, 15)):
            action = random.choice(['search_flights', 'filter_results', 'add_to_cart'])
            reward = random.uniform(-0.1, 0.2)
            total_reward += reward
            
            obs_system.log_step(
                'test_experiment', episode_id, step, 
                action, {'view': 'search_results'}, reward
            )
            
            time.sleep(0.1)
        
        # 结束episode
        success = random.random() > 0.3
        obs_system.log_episode_end('test_experiment', episode_id, total_reward, success)
        
        # 偶尔学习技能
        if random.random() > 0.7:
            skill_name = f"skill_{random.randint(1, 5)}"
            obs_system.log_skill_learned('test_experiment', skill_name, random.uniform(0.5, 0.9))
        
        time.sleep(1)
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs_system.cleanup()