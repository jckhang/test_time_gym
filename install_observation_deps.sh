#!/bin/bash
# 安装观测系统依赖

echo "🔧 安装可观测实验系统依赖..."

# 基础Python包
echo "📦 安装基础依赖..."
pip install matplotlib numpy pandas

# Web界面依赖
echo "🌐 安装Web界面依赖..."
pip install flask flask-socketio

# 可选的高级可视化依赖
echo "📊 安装高级可视化依赖..."
pip install plotly seaborn

# 配置文件处理
echo "⚙️ 安装配置处理依赖..."
pip install pyyaml

# 性能优化依赖
echo "⚡ 安装性能优化依赖..."
pip install psutil

echo "✅ 依赖安装完成!"
echo ""
echo "🚀 现在可以运行可观测实验:"
echo "python run_observable_experiment.py --demo"
echo ""
echo "🌐 或访问Web界面:"
echo "python run_observable_experiment.py --demo --open-browser"