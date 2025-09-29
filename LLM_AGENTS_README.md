# LLM智能体使用指南

本项目基于 `/data/workspace/internal/test_time_gym/test_time_gym/llm/base.py` 文件实现了基于大语言模型的智能体，用于机票预订任务。

## 功能特性

### 1. 基础LLM智能体 (`LLMAgent`)
- 支持任意LLM后端的智能体基类
- 自动对话历史管理
- 智能动作解析和降级策略
- 记忆和学习能力

### 2. OpenAI智能体 (`FlightBookingOpenAIAgent`)
- 基于OpenAI GPT模型的专用智能体
- 支持多种策略：aggressive、balanced、conservative
- 智能观察格式化和动作识别
- 策略相关的默认行为

### 3. 工具调用智能体 (`ToolEnabledFlightBookingAgent`)
- 支持OpenAI函数调用功能
- 结构化工具定义和参数验证
- 智能工具选择和参数生成
- 增强的决策能力

## 快速开始

### 基础使用

```python
import asyncio
from test_time_gym.agents import FlightBookingOpenAIAgent
from test_time_gym.envs.flight_booking_env import FlightBookingEnv

async def main():
    # 创建环境和智能体
    env = FlightBookingEnv(seed=42)
    agent = FlightBookingOpenAIAgent(
        model="gpt-3.5-turbo",
        strategy="balanced",
        temperature=0.7
    )

    # 运行一个episode
    obs, info = env.reset(seed=42)

    for step in range(20):
        action = await agent.select_action(obs)
        obs, reward, done, trunc, info = env.step(action)

        print(f"步骤 {step}: {action} -> 奖励 {reward:.3f}")

        if done or trunc:
            break

    # 更新智能体记忆
    agent.update_memory(trajectory)

# 运行
asyncio.run(main())
```

### 使用工具调用智能体

```python
from test_time_gym.agents import ToolEnabledFlightBookingAgent

# 创建支持工具调用的智能体
agent = ToolEnabledFlightBookingAgent(
    model="gpt-3.5-turbo",
    strategy="aggressive",
    temperature=0.5
)

# 使用方式与基础智能体相同
action = await agent.select_action(obs)
```

## 智能体策略

### 1. Aggressive（激进策略）
- 优先选择价格最低的航班
- 快速决策，减少犹豫
- 在预算范围内尽可能节省成本
- 遇到问题时快速重试

### 2. Balanced（平衡策略）
- 平衡价格和质量
- 考虑用户的预算约束和偏好
- 在多个选项间进行合理权衡
- 遇到问题时灵活应对

### 3. Conservative（保守策略）
- 优先考虑航班质量和可靠性
- 仔细评估每个选项
- 在不确定时选择更安全的选项
- 遇到问题时先分析再行动

## 环境状态说明

智能体需要处理以下环境状态：

- **search_form**: 搜索表单页面，需要填写出发地、目的地、日期等信息
- **search_results**: 搜索结果页面，显示可选的航班
- **cart**: 购物车页面，显示已选择的航班
- **payment**: 支付页面，需要填写支付信息
- **receipt**: 收据页面，预订完成
- **error**: 错误状态，需要处理错误

## 可用动作

智能体可以选择以下动作：

- `search_flights`: 搜索航班
- `filter_results`: 筛选搜索结果
- `select_flight`: 选择特定航班
- `add_to_cart`: 添加到购物车
- `proceed_to_payment`: 进入支付页面
- `enter_card`: 输入支付卡信息
- `confirm_payment`: 确认支付
- `apply_coupon`: 应用优惠券
- `restart`: 重新开始
- `abort`: 中止预订

## 工具调用功能

工具调用智能体支持以下结构化工具：

1. **search_flights**: 搜索航班
2. **filter_results**: 筛选结果（支持价格、转机次数、航空公司偏好）
3. **select_flight**: 选择航班（需要航班ID和选择原因）
4. **add_to_cart**: 添加到购物车
5. **proceed_to_payment**: 进入支付
6. **enter_card**: 输入卡片信息
7. **confirm_payment**: 确认支付
8. **apply_coupon**: 应用优惠券
9. **restart**: 重新开始
10. **abort**: 中止预订

## 智能体学习

智能体具有学习和记忆能力：

```python
# 更新智能体记忆
agent.update_memory(trajectory)

# 获取统计信息
stats = agent.get_stats()
print(f"总episodes: {stats['total_episodes']}")
print(f"对话轮数: {stats['conversation_turns']}")
print(f"学习技能数: {stats['skills_learned']}")
print(f"最佳技能: {stats['top_skills']}")
```

## 错误处理

智能体具有完善的错误处理机制：

1. **LLM调用失败**: 自动降级到规则策略
2. **动作解析失败**: 使用策略相关的默认动作
3. **工具调用失败**: 回退到基础动作选择
4. **网络问题**: 自动重试机制

## 性能优化

### 1. 对话历史管理
- 只保留最近5轮对话，避免上下文过长
- 自动清理和压缩历史信息

### 2. 智能降级
- LLM调用失败时自动使用规则策略
- 保证智能体始终能够做出决策

### 3. 策略适配
- 不同策略有不同的默认行为
- 根据策略特点优化决策逻辑

## 测试和验证

运行测试脚本验证实现：

```bash
python test_llm_agents.py
```

运行完整示例：

```bash
python examples/llm_agent_usage.py
```

## 配置选项

### OpenAI智能体配置

```python
agent = FlightBookingOpenAIAgent(
    model="gpt-3.5-turbo",        # 模型名称
    strategy="balanced",           # 策略类型
    max_retries=3,                # 最大重试次数
    temperature=0.7                # 生成温度
)
```

### 工具调用智能体配置

```python
agent = ToolEnabledFlightBookingAgent(
    model="gpt-3.5-turbo",        # 模型名称
    strategy="aggressive",        # 策略类型
    max_retries=3,                # 最大重试次数
    temperature=0.5               # 生成温度
)
```

## 扩展开发

### 添加新的LLM后端

```python
from test_time_gym.agents.llm_agent import LLMAgent

class CustomLLMAgent(LLMAgent):
    def __init__(self, custom_backend):
        super().__init__(custom_backend)
        # 自定义实现
```

### 添加新的工具

```python
# 在 ToolEnabledFlightBookingAgent 中添加新工具
def _handle_custom_tool(self, args: Dict, obs: Dict) -> str:
    # 处理自定义工具
    return "custom_action"
```

## 注意事项

1. **API密钥**: 确保正确配置OpenAI API密钥
2. **网络连接**: 需要稳定的网络连接访问LLM服务
3. **成本控制**: LLM调用会产生费用，建议设置合理的重试次数
4. **性能监控**: 建议监控LLM调用的成功率和响应时间

## 故障排除

### 常见问题

1. **LLM调用失败**
   - 检查API密钥配置
   - 验证网络连接
   - 查看错误日志

2. **动作解析错误**
   - 检查系统提示词
   - 验证观察格式
   - 调整温度参数

3. **工具调用失败**
   - 验证工具定义
   - 检查参数格式
   - 查看调用日志

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查智能体状态
print(f"对话历史: {len(agent.conversation_history)}")
print(f"记忆条目: {len(agent.memory)}")
print(f"技能库: {len(agent.skills)}")
```
