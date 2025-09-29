#!/usr/bin/env python3
"""
演示启动器 - 统一入口
为用户提供简单的演示选择界面
"""

import os
import subprocess
import sys
import webbrowser


def print_banner():
    """显示横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║            🚀 改进环境演示启动器                              ║
║          ImprovedFlightBookingEnv Demo Launcher              ║
╚══════════════════════════════════════════════════════════════╝

🎯 选择您想要的演示方式:
"""
    print(banner)

def print_options():
    """显示选项"""
    options = """
1. 🎮 交互式演示 - 亲手体验环境功能
   • 逐步执行动作
   • 观察奖励分解
   • 体验不同难度
   • 最佳入门选择

2. 📊 深度分析 - 技术验证与图表生成
   • 奖励分解分析
   • 难度级别对比
   • 技能指标演进
   • 生成可视化图表

3. 🌐 完整可观测演示 - Web界面监控
   • 实时Web仪表板
   • 多策略对比
   • 技能学习模拟
   • 自动报告生成

4. ⚡ 快速演示 - 简化版快速验证
   • 3分钟快速体验
   • 基础功能验证
   • 适合快速测试

5. 📖 查看使用指南 - 详细文档说明
   • 完整使用指南
   • 故障排除帮助
   • 技术细节说明

6. 🚪 退出

请输入选项编号 (1-6): """
    return input(options).strip()

def run_interactive_demo():
    """运行交互式演示"""
    print("\n🎮 启动交互式演示...")
    print("💡 您将能够逐步体验环境的每个功能")
    try:
        subprocess.run(["uv", "run", "python", "scripts/interactive_env_demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请确保在项目根目录下运行此脚本")

def run_analysis():
    """运行深度分析"""
    print("\n📊 启动深度分析...")
    print("💡 将生成详细的技术分析报告和可视化图表")
    try:
        subprocess.run(["uv", "run", "python", "scripts/analyze_improved_env.py"], check=True)
        print("\n✅ 分析完成！")
        print("📁 结果保存在: logs/improved_env_analysis/")

        # 询问是否查看结果
        view = input("\n是否查看分析报告？(y/N): ").strip().lower()
        if view in ['y', 'yes']:
            try:
                subprocess.run(["cat", "logs/improved_env_analysis/analysis_report.txt"], check=True)
            except:
                print("📁 请手动查看: logs/improved_env_analysis/analysis_report.txt")
    except subprocess.CalledProcessError as e:
        print(f"❌ 分析失败: {e}")

def run_observable_demo():
    """运行完整可观测演示"""
    print("\n🌐 启动完整可观测演示...")
    print("💡 将启动Web界面，请在浏览器中访问 http://localhost:5000")
    print("⚠️  此演示需要较长时间运行，建议准备好充足时间")

    confirm = input("确认启动？(y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        try:
            print("🚀 启动中... (可能需要几秒钟)")
            subprocess.run(["uv", "run", "python", "scripts/run_improved_env_demo.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 启动失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 演示被用户中断")

def run_quick_demo():
    """运行快速演示"""
    print("\n⚡ 启动快速演示...")
    print("💡 这是一个简化版本，大约3分钟完成")
    try:
        subprocess.run(["uv", "run", "python", "scripts/quick_improved_demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")

def show_guide():
    """显示使用指南"""
    print("\n📖 查看使用指南...")
    guide_path = "docs/IMPROVED_ENV_DEMO_GUIDE.md"

    if os.path.exists(guide_path):
        print(f"📁 使用指南位置: {guide_path}")

        view = input("是否在终端中查看？(y/N): ").strip().lower()
        if view in ['y', 'yes']:
            try:
                subprocess.run(["cat", guide_path], check=True)
            except:
                print("❌ 无法显示文件，请手动查看")
        else:
            print("💡 您可以使用任何文本编辑器打开此文件")
    else:
        print("❌ 找不到使用指南文件")

def main():
    """主函数"""
    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print_banner()

    while True:
        choice = print_options()

        if choice == '1':
            run_interactive_demo()
        elif choice == '2':
            run_analysis()
        elif choice == '3':
            run_observable_demo()
        elif choice == '4':
            run_quick_demo()
        elif choice == '5':
            show_guide()
        elif choice == '6':
            print("\n👋 感谢使用改进环境演示系统！")
            break
        else:
            print("\n❌ 无效选择，请输入1-6之间的数字")

        input("\n按回车键返回主菜单...")
        print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 感谢使用改进环境演示系统！")
    except Exception as e:
        print(f"\n❌ 启动器出错: {e}")
        print("💡 请确保在test_time_gym项目根目录下运行")
