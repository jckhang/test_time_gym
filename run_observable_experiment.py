#!/usr/bin/env python3
"""
启动可观测实验的主脚本
提供简单的命令行界面来运行各种可观测实验
"""

import argparse
import asyncio
import logging
import os
import sys
import webbrowser
import time
from typing import List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from observable_experiment_runner import ObservableExperimentRunner, ObservationConfig


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/observable_experiment.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 减少一些库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🔬 可观测实验系统                          ║
║              无监督经验积累实验的实时监控与分析                ║
╚══════════════════════════════════════════════════════════════╝

🚀 功能特性:
  • 实时监控实验进度和指标
  • 交互式Web仪表板
  • 技能学习过程可视化  
  • 详细的轨迹分析
  • 自动生成实验报告

📊 监控内容:
  • Episode成功率趋势
  • 奖励和步数统计
  • 技能学习效率
  • 错误和异常跟踪
"""
    print(banner)


async def run_quick_demo():
    """运行快速演示"""
    print("🎮 运行快速演示实验...")
    
    runner = ObservableExperimentRunner(
        "logs/demo_experiment",
        enable_observation=True,
        web_port=5000
    )
    
    try:
        results = await runner.run_comparative_experiment(
            num_episodes=30,
            models=["gpt-3.5-turbo"],
            strategies=["balanced"]
        )
        
        print("\n✅ 演示实验完成!")
        print("📊 查看生成的报告和可视化结果")
        
    except Exception as e:
        print(f"❌ 演示实验失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.cleanup()


async def run_full_experiment(models: List[str], strategies: List[str], episodes: int):
    """运行完整实验"""
    print(f"🔬 运行完整对比实验 ({episodes} episodes)...")
    print(f"📋 模型: {models}")
    print(f"📋 策略: {strategies}")
    
    runner = ObservableExperimentRunner(
        "logs/full_observable_experiment",
        enable_observation=True,
        web_port=5000
    )
    
    try:
        results = await runner.run_comparative_experiment(
            num_episodes=episodes,
            models=models,
            strategies=strategies
        )
        
        print("\n✅ 完整实验完成!")
        comparison = results["comparison_report"]
        overall = comparison.get("overall_improvement", {})
        
        if overall:
            print(f"📊 配置测试数: {overall.get('configurations_tested', 0)}")
            print(f"📈 有改进配置: {overall.get('improvements_positive', 0)}")
            print(f"🎯 成功率改进: {overall.get('avg_success_rate_improvement', 0):.3f}")
        
    except Exception as e:
        print(f"❌ 完整实验失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.cleanup()


async def run_custom_experiment(config_file: str):
    """运行自定义实验"""
    print(f"⚙️ 使用配置文件运行实验: {config_file}")
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    try:
        config = ObservationConfig.load_config(config_file)
        
        runner = ObservableExperimentRunner(
            "logs/custom_observable_experiment",
            enable_observation=config['observation']['enabled'],
            web_port=config['observation']['web_dashboard']['port']
        )
        
        # 从配置中读取实验参数（这里简化处理）
        results = await runner.run_comparative_experiment(
            num_episodes=100,
            models=["gpt-3.5-turbo"],
            strategies=["balanced", "aggressive"]
        )
        
        print("\n✅ 自定义实验完成!")
        
    except Exception as e:
        print(f"❌ 自定义实验失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.cleanup()


def open_dashboard(port: int = 5000):
    """尝试打开Web仪表板"""
    try:
        # 等待服务器启动
        time.sleep(2)
        url = f"http://localhost:{port}"
        webbrowser.open(url)
        print(f"🌐 已打开Web仪表板: {url}")
    except Exception as e:
        print(f"⚠️ 无法自动打开浏览器: {e}")
        print(f"请手动访问: http://localhost:{port}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="可观测实验系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 快速演示
  python run_observable_experiment.py --demo
  
  # 完整实验
  python run_observable_experiment.py --full --episodes 200
  
  # 自定义配置
  python run_observable_experiment.py --config my_config.yaml
  
  # 指定模型和策略
  python run_observable_experiment.py --full --models gpt-3.5-turbo gpt-4 --strategies balanced aggressive
        """
    )
    
    parser.add_argument("--demo", action="store_true", 
                       help="运行快速演示实验")
    parser.add_argument("--full", action="store_true", 
                       help="运行完整对比实验")
    parser.add_argument("--config", type=str, 
                       help="使用自定义配置文件")
    
    parser.add_argument("--episodes", type=int, default=100,
                       help="Episode数量 (默认: 100)")
    parser.add_argument("--models", nargs="+", 
                       default=["gpt-3.5-turbo"],
                       help="要测试的模型列表")
    parser.add_argument("--strategies", nargs="+", 
                       default=["balanced"],
                       help="要测试的策略列表")
    
    parser.add_argument("--port", type=int, default=5000,
                       help="Web仪表板端口 (默认: 5000)")
    parser.add_argument("--no-web", action="store_true",
                       help="禁用Web仪表板")
    parser.add_argument("--open-browser", action="store_true",
                       help="自动打开浏览器")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="静默模式")
    
    args = parser.parse_args()
    
    # 设置日志
    if not args.quiet:
        setup_logging(args.verbose)
    
    # 打印横幅
    if not args.quiet:
        print_banner()
    
    # 检查Web依赖
    if not args.no_web:
        try:
            import flask
            import flask_socketio
        except ImportError:
            print("⚠️ Web功能需要安装额外依赖:")
            print("pip install flask flask-socketio")
            print("或使用 --no-web 参数禁用Web功能")
            return
    
    # 启动浏览器（如果需要）
    if args.open_browser and not args.no_web:
        import threading
        browser_thread = threading.Thread(
            target=open_dashboard, 
            args=(args.port,),
            daemon=True
        )
        browser_thread.start()
    
    try:
        # 根据参数运行相应的实验
        if args.demo:
            await run_quick_demo()
        elif args.config:
            await run_custom_experiment(args.config)
        elif args.full:
            await run_full_experiment(args.models, args.strategies, args.episodes)
        else:
            # 默认运行演示
            print("💡 未指定实验类型，运行快速演示")
            print("使用 --help 查看更多选项")
            await run_quick_demo()
            
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    print("\n🎉 感谢使用可观测实验系统!")


if __name__ == "__main__":
    asyncio.run(main())