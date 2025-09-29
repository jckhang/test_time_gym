# Test-Time Gym

一个基于强化学习的智能体训练和测试框架，专注于机票预订环境的智能体开发。

## 项目概述

Test-Time Gym 是一个完整的智能体开发框架，支持多种智能体类型，包括传统强化学习智能体和基于大语言模型(LLM)的智能体。项目提供了完整的机票预订环境模拟，支持智能体的训练、测试和评估。

## 主要特性

- **多智能体支持**: 支持传统RL智能体和LLM智能体
- **配置化管理**: 统一的配置文件管理模型和策略
- **工具调用**: 支持LLM智能体的工具调用功能
- **环境模拟**: 完整的机票预订环境模拟
- **评估系统**: 全面的智能体性能评估

## 目录结构

```
test_time_gym/
├── README.md                 # 项目主说明文档
├── CLAUDE.md                 # Claude相关文档
├── config.yaml              # 配置文件
├── docs/                    # 文档目录
│   ├── README.md            # 文档索引
│   ├── DESIGN.md           # 系统设计文档
│   ├── PROJECT_STATUS.md   # 项目状态
│   └── ...                 # 其他文档
├── test/                    # 测试文件目录
│   ├── README.md           # 测试文件索引
│   ├── test_llm_agents.py  # LLM智能体测试
│   └── ...                 # 其他测试文件
├── examples/                # 使用示例
│   ├── basic_usage.py      # 基础使用示例
│   ├── llm_agent_usage.py  # LLM智能体示例
│   └── config_agent_usage.py # 配置化智能体示例
├── test_time_gym/          # 核心代码
│   ├── agents/            # 智能体实现
│   ├── envs/              # 环境实现
│   ├── llm/               # LLM后端
│   ├── config.py          # 配置管理
│   └── utils/             # 工具函数
└── tests/                 # 单元测试
```

## 快速开始

### 安装依赖

```bash
# 使用uv安装依赖
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 基础使用

```python
from test_time_gym.agents import FlightBookingOpenAIAgent
from test_time_gym.envs import FlightBookingEnv

# 创建智能体和环境
agent = FlightBookingOpenAIAgent()
env = FlightBookingEnv()

# 运行智能体
obs, info = env.reset()
action = await agent.select_action(obs)
obs, reward, done, trunc, info = env.step(action)
```

### 配置化使用

```python
# 使用默认配置 (claude-sonnet-4-20250514)
agent = FlightBookingOpenAIAgent()

# 使用特定模型
agent = FlightBookingOpenAIAgent(model="claude-3-haiku")

# 使用特定策略
agent = FlightBookingOpenAIAgent(strategy="aggressive")
```

## 智能体类型

### 传统智能体
- **DummyAgent**: 基础智能体
- **RandomAgent**: 随机智能体
- **SkillBasedAgent**: 基于技能的智能体

### LLM智能体
- **FlightBookingOpenAIAgent**: 基础LLM智能体
- **ToolEnabledFlightBookingAgent**: 支持工具调用的智能体

## 环境特性

- **状态空间**: 搜索表单、搜索结果、购物车、支付等状态
- **动作空间**: 搜索航班、筛选结果、选择航班、添加到购物车等动作
- **奖励函数**: 基于用户满意度和任务完成度的奖励
- **观察空间**: 包含当前状态、搜索参数、可用选项等信息

## 配置系统

项目支持通过`config.yaml`文件统一管理：
- **模型配置**: 支持多种LLM模型
- **策略配置**: 支持aggressive、balanced、conservative策略
- **参数覆盖**: 支持运行时参数覆盖

## 测试和验证

### 运行测试
```bash
# 基础测试
python3 test/test_llm_agent_simple.py

# 完整测试
python3 test/test_llm_agent_final.py

# 配置测试
python3 test/test_config_agents.py
```

### 使用示例
```bash
# 基础使用示例
python3 examples/basic_usage.py

# LLM智能体示例
python3 examples/llm_agent_usage.py

# 配置化智能体示例
python3 examples/config_agent_usage.py
```

## 文档

- **[文档索引](docs/README.md)** - 完整文档导航
- **[系统设计](docs/DESIGN.md)** - 系统架构设计
- **[LLM智能体指南](docs/LLM_AGENTS_README.md)** - LLM智能体使用说明
- **[配置系统](docs/CONFIG_AGENTS_SUMMARY.md)** - 配置化智能体说明

## 开发状态

- ✅ 基础环境实现
- ✅ 传统智能体实现
- ✅ LLM智能体实现
- ✅ 配置化系统
- ✅ 工具调用功能
- ✅ 测试和验证

## 贡献

欢迎贡献代码、文档和测试用例。请查看文档目录了解详细信息。

## 许可证

本项目采用MIT许可证。
