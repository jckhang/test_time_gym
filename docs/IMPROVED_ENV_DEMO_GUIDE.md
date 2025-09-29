# 🚀 改进环境演示指南

欢迎使用 ImprovedFlightBookingEnv 的完整演示系统！本指南将帮助您全面体验新环境的强大功能。

## 🎯 环境核心改进

### 1. 详细奖励分解系统
- **base_action**: 基础动作奖励
- **progress**: 进度推进奖励
- **constraint_satisfaction**: 约束满足奖励
- **efficiency**: 效率优化奖励
- **optimization**: 选择优化奖励
- **penalty**: 错误行为惩罚

### 2. 技能指标跟踪
- **search_efficiency**: 搜索效率 (成功搜索/总搜索次数)
- **budget_efficiency**: 预算使用效率
- **constraint_violations**: 约束违规次数
- **error_recovery_count**: 错误恢复次数

### 3. 多难度级别支持
- **Easy**: 宽松约束，预算充裕，适合学习基础流程
- **Medium**: 中等约束，平衡挑战和可行性
- **Hard**: 严格约束，预算紧张，考验优化能力

### 4. 确定性业务逻辑
- 基于规则的确定性航班生成
- 可重现的实验结果
- 减少随机性，提高学习效果

## 📁 演示脚本概览

### 1. 🎮 交互式演示 `interactive_env_demo.py`
**最佳入门选择** - 让您亲手体验环境的每个功能

```bash
cd /data/workspace/internal/test_time_gym
uv run python scripts/interactive_env_demo.py
```

**特色功能:**
- 菜单驱动的交互界面
- 逐步执行动作并观察反馈
- 实时查看奖励分解
- 多难度级别体验
- 详细的状态展示

**使用流程:**
1. 选择难度级别开始新episode
2. 逐步执行动作观察变化
3. 查看详细奖励分解
4. 体验不同策略的效果

### 2. 📊 深度分析 `analyze_improved_env.py`
**技术验证工具** - 全面分析环境的各项指标

```bash
cd /data/workspace/internal/test_time_gym
uv run python scripts/analyze_improved_env.py
```

**生成内容:**
- 奖励分解统计分析
- 难度级别性能对比
- 技能指标演进曲线
- 环境确定性验证
- 可视化图表 (PNG格式)

**输出位置:** `logs/improved_env_analysis/`

### 3. 🌐 完整可观测演示 `run_improved_env_demo.py`
**研究级演示** - 集成观测系统的全功能演示

```bash
cd /data/workspace/internal/test_time_gym
uv run python scripts/run_improved_env_demo.py
```

**核心特性:**
- 多策略智能体对比 (aggressive, conservative, balanced, learning)
- 实时Web监控界面 (http://localhost:5000)
- 完整的技能学习模拟
- 详细的观测数据收集
- 自动生成实验报告

**适用场景:**
- 研究开发验证
- 长期实验监控
- 对比分析研究
- 演示展示

### 4. ⚡ 快速演示 `quick_improved_demo.py`
**快速验证工具** - 简化版快速演示

```bash
cd /data/workspace/internal/test_time_gym
uv run python scripts/quick_improved_demo.py
```

**特点:**
- 快速启动 (3分钟内完成)
- 基础功能验证
- 简化的观测界面
- 适合快速测试

## 🎯 推荐使用路径

### 🔰 初学者路径
1. **交互式演示** → 亲手体验环境功能
2. **快速演示** → 观察自动化运行
3. **深度分析** → 了解技术细节

### 🔬 研究者路径
1. **深度分析** → 验证环境设计
2. **完整可观测演示** → 全面功能测试
3. **交互式演示** → 细节功能验证

### 🎪 演示展示路径
1. **完整可观测演示** → 启动Web界面
2. **交互式演示** → 现场操作演示
3. **分析结果展示** → 图表和数据分析

## 📊 观测系统集成

### Web仪表板功能
访问 `http://localhost:5000` 可查看:
- 实时指标监控
- 成功率趋势图
- 奖励分解可视化
- 技能学习时间线
- 策略性能对比

### 自动生成报告
系统自动生成以下报告:
- 统计分析报告 (`stats_report_*.txt`)
- 技能学习报告 (`skill_report_*.txt`)
- 原始数据 (`raw_data.json`)
- 可视化图表 (`*.png`)

## 🔧 环境配置选项

### 基础配置
```python
config = {
    "difficulty": "medium",        # easy/medium/hard
    "max_steps": 30,              # 最大步数限制
    "enable_dynamic_pricing": True # 动态价格机制
}
```

### 奖励权重配置
```python
reward_weights = {
    "progress": 0.1,              # 进度奖励权重
    "constraint": 0.3,            # 约束满足权重
    "efficiency": 0.2,            # 效率奖励权重
    "optimization": 0.2,          # 优化奖励权重
    "completion": 1.0             # 完成奖励权重
}
```

## 📈 性能基准

### 难度级别特征
| 难度 | 预算倍数 | 最大经停 | 时间限制 | 成功率目标 |
|------|----------|----------|----------|------------|
| Easy | 1.5x | 2 | 任意时间 | 80%+ |
| Medium | 1.2x | 1 | 上午/下午 | 60%+ |
| Hard | 1.0x | 0 | 仅上午 | 40%+ |

### 预期学习曲线
- **前10个episodes**: 熟悉环境，成功率 20-40%
- **10-30个episodes**: 策略优化，成功率 40-70%
- **30+个episodes**: 技能掌握，成功率 70%+

## 🛠️ 故障排除

### 常见问题

1. **Web界面无法访问**
   ```bash
   # 检查端口占用
   lsof -i :5000

   # 尝试其他端口
   WEB_PORT=5001 uv run python scripts/run_improved_env_demo.py
   ```

2. **可视化图表显示问题**
   ```bash
   # 安装中文字体支持
   sudo apt-get install fonts-wqy-zenhei

   # 或使用英文界面
   LANG=en_US.UTF-8 uv run python scripts/analyze_improved_env.py
   ```

3. **依赖问题**
   ```bash
   # 检查依赖
   uv sync

   # 安装可视化依赖
   uv add matplotlib seaborn plotly flask flask-socketio
   ```

### 性能优化

1. **长期实验优化**
   ```python
   config = {
       "max_episodes": 1000,
       "batch_size": 50,
       "save_interval": 100
   }
   ```

2. **内存使用优化**
   ```python
   observation_config = {
       "max_trajectory_length": 100,
       "compress_old_data": True,
       "memory_limit_mb": 512
   }
   ```

## 📞 支持信息

### 生成的文件位置
- **日志文件**: `logs/`
- **分析报告**: `logs/improved_env_analysis/`
- **观测报告**: `logs/improved_env_reports/`
- **演示报告**: `logs/demo_reports/`

### 联系支持
如遇到问题，请查看:
1. 生成的日志文件
2. 系统错误输出
3. 内存和CPU使用情况

## 🎉 开始探索

选择适合您需求的演示脚本，开始探索改进环境的强大功能！

```bash
# 快速开始 - 交互式体验
uv run python scripts/interactive_env_demo.py

# 深度分析 - 技术验证
uv run python scripts/analyze_improved_env.py

# 完整演示 - 研究级功能
uv run python scripts/run_improved_env_demo.py
```

🚀 **祝您探索愉快！**
