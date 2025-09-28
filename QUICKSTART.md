# Test-Time Gym 快速开始指南

## 🚀 立即体验

### 方式1：最简演示（无依赖）
```bash
# 克隆项目
git clone <your-repo-url>
cd test-time-gym

# 直接运行最简演示
python3 standalone_demo.py
```

### 方式2：完整功能（需要依赖）
```bash
# 安装依赖
pip install -e .

# 运行基础示例
python3 examples/basic_usage.py

# 运行完整实验
python3 examples/experiment_runner.py
```

## 📋 演示结果说明

运行 `standalone_demo.py` 后，你会看到：

```
🚀 Test-Time Gym - Standalone Demo
========================================
📋 Task: Book flight within budget $860
   Must depart after: 11:00
   Max stops: 1

--- Step 0 - View: search_form ---
🤖 Agent choosing action for view: search_form
🎯 Action: search_flights
💰 Reward: 0.010 (Total: 0.010)

✅ SUCCESS! Booked flight for $650.50
🎉 All constraints satisfied!

📊 Final Results:
   Total Reward: 1.090
   Steps Taken: 5
   Success: Yes
```

## 🎮 核心概念理解

### 1. 环境状态转换
```
search_form → search_results → cart → payment → receipt
     ↓              ↓           ↓        ↓
   搜索航班      筛选/选择    加购物车   支付流程
```

### 2. 奖励机制
- **成功预订**: +1.0
- **满足约束**: 额外奖励
- **过程奖励**: 搜索(+0.02)、加购物车(+0.05)、进入支付(+0.05)
- **约束违规**: -0.3 (超预算)、-0.2 (时间/中转违规)
- **无效操作**: -0.05
- **时间成本**: 每步 -0.01

### 3. 随机扰动
- 10% 航班售罄概率
- 价格随机波动 (±$50-100)
- 25% 需要3DS验证
- 8-15% 支付失败概率

## 🧠 智能体学习机制

### 技能提取流程
1. **轨迹收集**: 记录成功的动作序列
2. **模式挖掘**: 找到高频且成功的子流程
3. **技能抽象**: 转化为可重用的宏动作
4. **置信度更新**: 使用 Beta 分布跟踪成功率

### Thompson Sampling 策略选择
```python
# 每个技能维护 Beta(α, β) 后验分布
confidence = α / (α + β)

# 选择时从后验采样
sampled_rate = Beta(α, β).sample()
selected_skill = argmax(sampled_rate + exploration_bonus)
```

## 📊 评估指标体系

### 核心指标
- **Success@N**: N步内成功率
- **平均步长**: 成功所需平均步数  
- **约束违规率**: 违反预算/时间约束的比例
- **后悔值**: 实际价格 - 最优价格
- **技能复用率**: 使用已学技能的比例

### 学习曲线分析
```python
from test_time_gym.evaluation.metrics import MetricsCalculator

calculator = MetricsCalculator()
episodes, success_rates = calculator.calculate_learning_curve(episodes)

# 应该看到上升趋势：技能积累 → 成功率提升
```

## 🔬 实验设计模板

### A/B 对照实验
```python
conditions = {
    "no_memory": {"enable_skills": False, "enable_bandit": False},
    "with_memory": {"enable_skills": True, "enable_bandit": True},
    "shuffled_memory": {"enable_skills": True, "enable_bandit": False}
}

# 预期结果：with_memory > shuffled_memory > no_memory
```

### 统计显著性检验
- 配对 t 检验（成功率差异）
- Mann-Whitney U（步长分布差异）
- Bootstrap 置信区间（稳健估计）

## 🛡️ 安全机制

### Action Shield
- 预算上限保护 (max_budget=2000)
- 重复操作检测
- 无效参数拦截
- 支付尝试次数限制

### OOD 检测
- 状态密度估计 (kNN距离)
- 策略漂移监控 (KL散度)
- 自动降级为保守策略

## 🔧 自定义扩展

### 1. 新增动作类型
```python
# 在 FlightBookingEnv 中添加
def _cancel_booking(self, payload):
    # 实现取消逻辑
    return reward, done, info

# 更新 action_verbs 列表
self.action_verbs.append("cancel_booking")
```

### 2. 新约束类型
```python
# 在 Constraints 模型中添加
class Constraints:
    airline_preference: Optional[str] = None
    max_duration_hours: Optional[int] = None
```

### 3. 多任务场景
```python
# 继承 FlightBookingEnv 创建新环境
class HotelBookingEnv(FlightBookingEnv):
    def __init__(self):
        super().__init__()
        # 添加酒店特定逻辑
```

## 📈 性能优化建议

### 大规模实验
- 使用多进程并行运行 (`multiprocessing.Pool`)
- 批量处理技能提取 (每100个episode)
- 限制内存使用 (sliding window = 1000 episodes)

### 技能库管理
- 定期清理低效技能 (success_rate < 0.3)
- 合并相似技能 (编辑距离 < 0.2)
- 版本化技能库 (git或时间戳)

## 🎯 下一步开发

1. **集成真实LLM**: 替换DummyAgent为GPT-4/Claude调用
2. **可视化工具**: 轨迹回放、技能演化图表
3. **多智能体**: 竞争/协作预订场景
4. **持续学习**: 在线技能更新与遗忘机制
5. **领域扩展**: 购物、表单填写、客服对话

## 💡 调试建议

### 常见问题
- **成功率低**: 检查约束生成是否过严、航班数据库是否合理
- **无限循环**: 启用ActionShield，限制重复操作
- **技能不生效**: 确认技能提取频率，检查前置条件匹配

### 日志分析
```bash
# 查看详细轨迹
ls logs/episodes_*.jsonl

# 分析技能统计
ls logs/skills/*.json

# 检查评估指标
cat logs/metrics.jsonl
```

---

🎉 **恭喜！你已经成功运行了 Test-Time Gym 框架！**

这个框架为 LLM 智能体提供了一个安全、可控的学习环境，支持：
- ✅ 程序性知识获取 (技能提取)
- ✅ 探索/利用平衡 (Thompson Sampling)  
- ✅ 安全约束保证 (Action Shield + OOD Detection)
- ✅ 可复现实验设计 (随机种子 + 统计检验)

开始构建你的智能体吧！ 🤖✨