# Test-Time Gym 测试文件索引

## 测试文件说明

本目录包含所有测试文件，用于验证Test-Time Gym项目的各项功能。

## 测试文件列表

### LLM智能体测试
- **[test_llm_agents.py](test_llm_agents.py)** - LLM智能体基础功能测试
- **[test_llm_agent_final.py](test_llm_agent_final.py)** - LLM智能体最终完整测试
- **[test_llm_agent_fixed.py](test_llm_agent_fixed.py)** - LLM智能体修复后测试
- **[test_llm_agent_simple.py](test_llm_agent_simple.py)** - LLM智能体简化测试
- **[test_llm_agents_offline.py](test_llm_agents_offline.py)** - LLM智能体离线测试

### 配置化智能体测试
- **[test_config_agents.py](test_config_agents.py)** - 配置化智能体功能测试

## 测试分类

### 按功能分类
- **基础功能测试**: test_llm_agent_simple.py
- **完整功能测试**: test_llm_agent_final.py
- **离线测试**: test_llm_agents_offline.py
- **配置测试**: test_config_agents.py

### 按复杂度分类
- **简单测试**: test_llm_agent_simple.py, test_llm_agents_offline.py
- **中等测试**: test_llm_agents.py, test_config_agents.py
- **复杂测试**: test_llm_agent_final.py, test_llm_agent_fixed.py

## 运行测试

### 基础测试
```bash
# 运行简化测试
python3 test/test_llm_agent_simple.py

# 运行离线测试
python3 test/test_llm_agents_offline.py
```

### 完整测试
```bash
# 运行完整LLM测试
python3 test/test_llm_agent_final.py

# 运行配置测试
python3 test/test_config_agents.py
```

### 所有测试
```bash
# 运行所有测试
for test_file in test/test_*.py; do
    echo "运行 $test_file"
    python3 "$test_file"
done
```

## 测试结果说明

### 通过标准
- **基础功能**: 智能体创建、环境集成、基本决策
- **LLM调用**: 真实LLM调用成功或降级策略正常
- **配置功能**: 配置文件加载、模型切换、参数覆盖
- **工具调用**: 工具定义和处理正常

### 预期结果
- 所有测试应该通过或显示合理的降级行为
- LLM调用可能失败，但降级策略应该正常工作
- 配置功能应该完全正常
- 环境集成应该稳定

## 故障排除

### 常见问题
1. **LLM调用失败**: 这是正常的，系统有降级机制
2. **模型不可用**: 检查配置文件中的模型名称
3. **环境错误**: 确保依赖已正确安装

### 调试建议
1. 先运行简单测试验证基础功能
2. 检查日志输出了解具体错误
3. 使用离线测试验证代码结构
4. 逐步运行复杂测试

## 更新记录

- **2024-09-29**: 创建测试文件索引
- **2024-09-29**: 整理测试文件结构
- **2024-09-29**: 添加测试说明和运行指南
