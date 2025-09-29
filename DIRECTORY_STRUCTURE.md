# Test-Time Gym 目录结构说明

## 整理后的目录结构

```
test_time_gym/
├── README.md                    # 项目主说明文档
├── CLAUDE.md                    # Claude相关文档（保留在根目录）
├── config.yaml                  # 配置文件
├── DIRECTORY_STRUCTURE.md       # 本文件
├── docs/                        # 文档目录
│   ├── README.md               # 文档索引
│   ├── DESIGN.md              # 系统设计文档
│   ├── PROJECT_STATUS.md      # 项目状态和进度
│   ├── IMPLEMENTATION_SUMMARY.md # 实现总结
│   ├── LLM_AGENTS_README.md   # LLM智能体使用说明
│   ├── CONFIG_AGENTS_SUMMARY.md # 配置化智能体总结
│   ├── FINAL_LLM_AGENT_REPORT.md # LLM智能体最终调通报告
│   ├── LLM_AGENT_TUNING_SUMMARY.md # LLM智能体调优总结
│   └── Evaluation.md           # 评估方法和结果
├── test/                       # 测试文件目录
│   ├── README.md              # 测试文件索引
│   ├── test_llm_agents.py     # LLM智能体基础测试
│   ├── test_llm_agent_final.py # LLM智能体最终测试
│   ├── test_llm_agent_fixed.py # LLM智能体修复测试
│   ├── test_llm_agent_simple.py # LLM智能体简化测试
│   ├── test_llm_agents_offline.py # LLM智能体离线测试
│   └── test_config_agents.py  # 配置化智能体测试
├── examples/                   # 使用示例
│   ├── basic_usage.py         # 基础使用示例
│   ├── advanced_usage.py      # 高级使用示例
│   ├── llm_agent_usage.py     # LLM智能体示例
│   └── config_agent_usage.py  # 配置化智能体示例
├── test_time_gym/             # 核心代码
│   ├── __init__.py
│   ├── agents/               # 智能体实现
│   │   ├── __init__.py
│   │   ├── dummy_agent.py    # 基础智能体
│   │   ├── llm_agent.py      # LLM智能体基类
│   │   └── openai_agent.py   # OpenAI智能体
│   ├── envs/                 # 环境实现
│   │   ├── __init__.py
│   │   └── flight_booking_env.py # 机票预订环境
│   ├── llm/                  # LLM后端
│   │   └── base.py           # LLM后端基类
│   ├── config.py             # 配置管理
│   ├── cli.py                # 命令行接口
│   └── utils/                # 工具函数
│       ├── __init__.py
│       ├── evaluation.py     # 评估工具
│       └── skill_system.py   # 技能系统
├── tests/                    # 单元测试
│   ├── __init__.py
│   ├── test_agents.py        # 智能体单元测试
│   └── test_environment.py   # 环境单元测试
├── logs/                     # 日志目录
│   └── skills/               # 技能日志
├── test_time_gym.egg-info/   # 包信息
├── pyproject.toml           # 项目配置
├── setup.py                 # 安装脚本
├── install.sh               # 安装脚本
├── run_demo.py              # 演示脚本
├── simple_test.py           # 简单测试
└── uv.lock                  # 依赖锁定文件
```

## 目录说明

### 根目录文件
- **README.md**: 项目主说明文档
- **CLAUDE.md**: Claude相关文档（按要求保留在根目录）
- **config.yaml**: 配置文件
- **DIRECTORY_STRUCTURE.md**: 目录结构说明

### docs/ 目录
包含所有文档文件，除了CLAUDE.md：
- 系统设计文档
- 项目状态文档
- 实现总结文档
- LLM智能体相关文档
- 配置系统文档
- 技术报告文档
- 评估文档

### test/ 目录
包含所有测试文件：
- LLM智能体测试
- 配置化智能体测试
- 各种复杂度的测试
- 测试文件索引

### examples/ 目录
包含使用示例：
- 基础使用示例
- 高级使用示例
- LLM智能体示例
- 配置化智能体示例

### test_time_gym/ 目录
核心代码实现：
- 智能体实现
- 环境实现
- LLM后端
- 配置管理
- 工具函数

### tests/ 目录
单元测试：
- 智能体单元测试
- 环境单元测试

## 文件移动记录

### 移动到 docs/ 目录
- DESIGN.md
- PROJECT_STATUS.md
- IMPLEMENTATION_SUMMARY.md
- LLM_AGENTS_README.md
- CONFIG_AGENTS_SUMMARY.md
- FINAL_LLM_AGENT_REPORT.md
- LLM_AGENT_TUNING_SUMMARY.md
- Evaluation.md

### 移动到 test/ 目录
- test_llm_agents.py
- test_llm_agent_final.py
- test_llm_agent_fixed.py
- test_llm_agent_simple.py
- test_llm_agents_offline.py
- test_config_agents.py

### 保留在根目录
- README.md (项目主文档)
- CLAUDE.md (按要求保留)
- config.yaml (配置文件)

## 索引文件

### docs/README.md
- 文档索引
- 快速导航
- 文档分类

### test/README.md
- 测试文件索引
- 测试分类说明
- 运行指南

## 整理效果

1. **清晰的目录结构**: 文档、测试、示例分离
2. **完整的索引系统**: 每个目录都有README索引
3. **合理的文件组织**: 相关文件归类到对应目录
4. **保留重要文件**: CLAUDE.md按要求保留在根目录
5. **便于维护**: 新文件可以轻松归类到对应目录
