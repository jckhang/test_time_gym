#!/bin/bash
# 安装脚本 - 使用uv进行环境管理

set -e

echo "🚀 开始安装 Test-Time Gym..."

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未找到，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ 使用 uv 创建虚拟环境..."
uv venv

echo "✅ 激活虚拟环境并安装依赖..."
source .venv/bin/activate
uv pip install -e .

echo "✅ 创建必要的目录..."
mkdir -p logs/{evaluation,trajectories,skills,memory}

echo "✅ 运行基础测试..."
python run_demo.py

echo ""
echo "🎉 安装完成!"
echo ""
echo "使用方法:"
echo "1. 激活环境: source .venv/bin/activate"
echo "2. 运行演示: python run_demo.py"
echo "3. 基础使用: python examples/basic_usage.py"
echo "4. 高级功能: python examples/advanced_usage.py"
echo "5. 命令行工具: python -m test_time_gym.cli --help"
echo ""