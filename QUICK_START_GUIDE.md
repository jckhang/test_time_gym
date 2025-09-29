# 🚀 快速启动指南

欢迎使用实验观测系统！这个指南将帮助您在5分钟内上手并运行您的第一个可观测实验。

## ⚡ 30秒快速体验

```bash
# 1. 安装依赖
./install_observation_deps.sh

# 2. 运行演示
python demo_observation.py

# 3. 打开浏览器访问 http://localhost:5000
```

## 📦 完整安装步骤

### 1. 安装Python依赖
```bash
# 使用自动安装脚本（推荐）
chmod +x install_observation_deps.sh
./install_observation_deps.sh

# 或手动安装
pip install matplotlib numpy pandas flask flask-socketio plotly seaborn pyyaml psutil
```

### 2. 验证安装
```bash
python -c "import flask, matplotlib, numpy; print('✅ 依赖安装成功!')"
```

## 🎮 三种使用方式

### 方式1: 快速演示
最简单的方式，立即看到效果：
```bash
python demo_observation.py
```
- 🔍 模拟实验数据
- 🌐 Web界面: http://localhost:5000
- 📊 实时监控和图表
- ⏱️ 运行时间: ~2分钟

### 方式2: 真实实验
使用您的实验框架：
```bash
# 快速测试
python run_observable_experiment.py --demo

# 完整实验
python run_observable_experiment.py --full --episodes 100

# 自定义配置
python run_observable_experiment.py --config example_configs/research_config.yaml
```

### 方式3: 代码集成
在您的代码中集成观测系统：
```python
from observable_experiment_runner import ObservableExperimentRunner

runner = ObservableExperimentRunner("logs/my_experiment")
results = await runner.run_comparative_experiment(num_episodes=50)
```

## 🎯 观测要点

### 📊 Web仪表板功能
访问 http://localhost:5000 查看：
- **实时指标**: 成功率、奖励、步数趋势
- **技能分析**: 学习进度和使用统计
- **对比图表**: 基线vs经验学习
- **实时日志**: 系统事件和错误

### 📈 关键观测指标
- **成功率趋势**: 实验学习效果
- **技能学习曲线**: 技能发现和改进
- **奖励变化**: 性能提升情况
- **步数效率**: 执行效率改进

### 🔍 深度分析
- **轨迹可视化**: 每个episode的详细过程
- **技能演进**: 技能成功率的变化
- **对比分析**: 不同配置的性能差异
- **错误追踪**: 异常和失败模式

## ⚙️ 配置选项

### 快速配置
使用预设配置文件：
```bash
# 最小配置（性能优先）
python run_observable_experiment.py --config example_configs/minimal_config.yaml

# 研究配置（详细监控）
python run_observable_experiment.py --config example_configs/research_config.yaml

# 生产配置（稳定性优先）
python run_observable_experiment.py --config example_configs/production_config.yaml
```

### 自定义配置
复制并修改 `observation_config.yaml`：
```yaml
observation:
  enabled: true
  web_dashboard:
    port: 5000
  console_reporting:
    interval_seconds: 10
```

## 📁 输出文件

运行后查看以下目录：
```
logs/
├── observable_experiments/     # 实验数据
├── observation_reports/        # 分析报告
└── observable_experiment.log   # 系统日志
```

## 🐛 常见问题

### Q: Web界面无法访问？
```bash
# 检查端口占用
netstat -an | grep 5000

# 更换端口
python run_observable_experiment.py --demo --port 5001
```

### Q: 缺少依赖？
```bash
# 重新安装
pip install --upgrade flask flask-socketio matplotlib

# 检查Python版本（需要3.7+）
python --version
```

### Q: 内存使用过高？
使用最小配置：
```bash
python run_observable_experiment.py --config example_configs/minimal_config.yaml
```

### Q: 可视化不显示？
```bash
# 检查matplotlib后端
python -c "import matplotlib; print(matplotlib.get_backend())"

# 设置后端
export MPLBACKEND=TkAgg
```

## 💡 使用建议

### 🔬 研究场景
- 使用 `research_config.yaml`
- 启用所有可视化功能
- 保留详细轨迹数据
- 关注技能学习分析

### 🏭 生产场景  
- 使用 `production_config.yaml`
- 禁用实时图表
- 限制内存使用
- 启用告警监控

### 🎮 演示场景
- 使用 `demo_observation.py`
- 启用浏览器自动打开
- 使用模拟数据
- 重点展示可视化效果

## 🔗 相关文件

- 📖 **详细文档**: `OBSERVATION_SYSTEM_README.md`
- ⚙️ **配置示例**: `example_configs/`
- 🎮 **演示脚本**: `demo_observation.py`
- 🔧 **集成示例**: `integration_example.py`
- 📊 **实验运行器**: `run_observable_experiment.py`

## 🎉 下一步

1. **运行演示**: `python demo_observation.py`
2. **查看Web界面**: http://localhost:5000
3. **阅读详细文档**: `OBSERVATION_SYSTEM_README.md`
4. **尝试真实实验**: `python run_observable_experiment.py --demo`
5. **自定义配置**: 编辑 `observation_config.yaml`

---

🔬 祝您实验观测愉快！如有问题，请查看日志文件或文档。