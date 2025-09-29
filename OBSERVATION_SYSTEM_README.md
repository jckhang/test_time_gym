# 🔬 实验观测系统使用指南

实验观测系统为您的无监督经验积累实验提供全面的可观测性，包括实时监控、过程可视化和深度分析。

## 🚀 核心特性

### 1. 实时监控
- **Episode级监控**: 成功率、奖励、步数等关键指标
- **技能学习追踪**: 技能发现、演进和使用统计
- **错误监控**: 自动检测和报告实验中的异常
- **性能监控**: 内存使用、处理速度等系统指标

### 2. 可视化界面
- **Web仪表板**: 交互式实时监控界面
- **实时图表**: 动态更新的学习曲线和统计图表
- **轨迹可视化**: Episode执行过程的详细可视化
- **技能演进图**: 技能学习和改进过程的可视化

### 3. 深度分析
- **对比分析**: 不同配置之间的详细对比
- **技能分析**: 技能学习模式和效果分析
- **趋势分析**: 长期学习趋势和稳定性分析
- **报告生成**: 自动生成详细的实验报告

## 📦 安装和设置

### 1. 安装依赖
```bash
# 运行安装脚本
chmod +x install_observation_deps.sh
./install_observation_deps.sh

# 或手动安装
pip install matplotlib numpy pandas flask flask-socketio plotly seaborn pyyaml psutil
```

### 2. 配置系统
观测系统使用 `observation_config.yaml` 进行配置：

```yaml
observation:
  enabled: true
  web_dashboard:
    enabled: true
    port: 5000
  console_reporting:
    enabled: true
    interval_seconds: 10
  visualization:
    real_time_plots: true
    trajectory_analysis: true
```

## 🎮 使用方法

### 1. 快速开始 - 演示实验
```bash
# 运行快速演示（30个episodes）
python run_observable_experiment.py --demo

# 演示并自动打开浏览器
python run_observable_experiment.py --demo --open-browser
```

### 2. 完整实验
```bash
# 运行完整对比实验
python run_observable_experiment.py --full --episodes 200

# 指定模型和策略
python run_observable_experiment.py --full \
  --models gpt-3.5-turbo gpt-4 \
  --strategies balanced aggressive conservative \
  --episodes 150
```

### 3. 自定义配置
```bash
# 使用自定义配置文件
python run_observable_experiment.py --config my_config.yaml
```

### 4. 程序集成
```python
from observable_experiment_runner import ObservableExperimentRunner

# 创建可观测实验运行器
runner = ObservableExperimentRunner(
    "logs/my_experiment",
    enable_observation=True,
    web_port=5000
)

# 运行实验
results = await runner.run_comparative_experiment(
    num_episodes=100,
    models=["gpt-3.5-turbo"],
    strategies=["balanced"]
)
```

## 📊 监控界面

### Web仪表板
访问 `http://localhost:5000` 查看实时监控界面，包括：

- **实时指标面板**: 显示当前实验的关键指标
- **成功率趋势图**: 实时更新的成功率曲线
- **奖励趋势图**: 平均奖励的变化趋势
- **技能分析面板**: 技能学习和使用统计
- **实时日志**: 系统事件和错误日志

### 控制台输出
系统每10秒在控制台输出监控报告：
```
📊 实验监控报告 - 14:30:25
════════════════════════════════════════════════════
🔬 gpt-3.5-turbo_balanced_baseline:
  活跃Episodes: 2
  最近成功率: 0.650
  平均奖励: 0.847
  平均步数: 12.3

🔬 gpt-3.5-turbo_balanced_with_experience:
  活跃Episodes: 1
  最近成功率: 0.750
  平均奖励: 1.023
  平均步数: 10.8
  学到技能: 3 个
  学习效率: 0.180
```

## 📈 观测数据详解

### 1. Episode级指标
- **成功率**: Episode成功完成的比例
- **平均奖励**: Episode的平均总奖励
- **平均步数**: Episode的平均执行步数
- **技能使用率**: 使用技能的步数比例
- **错误率**: 发生错误的步数比例

### 2. 技能学习指标
- **技能数量**: 学到的不同技能总数
- **学习效率**: 技能学习速度（技能数/总事件数）
- **技能成功率**: 各技能的平均成功率
- **使用频率**: 各技能的使用次数统计
- **改进幅度**: 技能成功率的提升情况

### 3. 系统性能指标
- **处理延迟**: 事件处理的平均延迟
- **内存使用**: 系统内存占用情况
- **队列状态**: 事件队列的填充程度

## 🔧 高级配置

### 自定义监控指标
在 `observation_config.yaml` 中配置：

```yaml
monitoring:
  episode_metrics:
    - "success_rate"
    - "avg_reward"
    - "efficiency_score"  # 自定义指标
  
  skill_metrics:
    - "learning_rate"
    - "adaptation_speed"  # 自定义指标
```

### 告警配置
```yaml
alerts:
  enabled: true
  performance_alerts:
    low_success_rate_threshold: 0.1
    high_error_rate_threshold: 0.3
```

### 数据存储配置
```yaml
storage:
  event_queue_size: 10000
  trajectory_retention:
    max_episodes_per_experiment: 1000
    compress_old_trajectories: true
```

## 📁 输出文件结构

```
logs/
├── observable_experiments/           # 主实验目录
│   ├── evaluation/                  # 评估数据
│   ├── trajectories/               # 轨迹数据
│   ├── skills_*experiment*/        # 技能数据
│   └── intermediate_results.json   # 中间结果
├── observation_reports/            # 观测报告
│   ├── stats_report_*.txt         # 统计报告
│   ├── skill_report_*.txt         # 技能分析报告
│   └── *.html                     # HTML格式报告
└── observable_experiment.log      # 系统日志
```

## 🎯 观测策略建议

### 1. 实时监控策略
- **短期实验** (< 50 episodes): 启用所有可视化功能
- **长期实验** (> 200 episodes): 适当降低更新频率
- **大规模实验**: 禁用实时图表，使用报告模式

### 2. 性能优化
- 调整 `event_batch_size` 来平衡实时性和性能
- 使用 `max_memory_mb` 限制内存使用
- 对于生产环境，考虑禁用详细轨迹记录

### 3. 分析重点
- **技能学习**: 关注技能发现时间线和成功率演进
- **对比分析**: 重点观察基线vs经验学习的差异
- **稳定性**: 监控长期趋势和方差
- **效率**: 关注步数和时间的变化

## 🐛 故障排除

### 常见问题

1. **Web界面无法访问**
   - 检查端口是否被占用
   - 确认防火墙设置
   - 验证Flask和SocketIO安装

2. **可视化图表不更新**
   - 检查matplotlib后端设置
   - 确认数据流是否正常
   - 查看控制台错误信息

3. **内存使用过高**
   - 调整 `event_queue_size`
   - 启用轨迹压缩
   - 减少保留的历史数据

### 日志分析
系统日志位于 `logs/observable_experiment.log`，包含：
- 系统启动和配置信息
- 实验进度和关键事件
- 错误和异常详情
- 性能指标

## 🔄 与现有框架集成

观测系统可以无缝集成到现有实验框架中：

```python
# 替换原始的 ExperimentRunner
from observable_experiment_runner import ObservableExperimentRunner

# 使用相同的API，获得额外的观测功能
runner = ObservableExperimentRunner("logs/experiments")
results = await runner.run_comparative_experiment(...)
```

## 📞 支持和扩展

观测系统采用模块化设计，支持：
- 自定义指标添加
- 新的可视化组件
- 外部系统集成
- 自定义报告格式

如需扩展功能，可以：
1. 继承相应的基类
2. 添加新的事件类型
3. 实现自定义分析器
4. 扩展Web界面

---

🎉 享受您的可观测实验之旅！如有问题，请查看日志文件或提交issue。