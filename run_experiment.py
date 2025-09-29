#!/usr/bin/env python3
"""
无监督经验积累实验运行脚本
"""

import asyncio
import argparse
import os
import sys
import json
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_framework import ExperimentRunner

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_experiment_config(config_path: str = "experiment_config.json") -> dict:
    """加载实验配置"""
    default_config = {
        "experiments": {
            "quick_test": {
                "num_episodes": 50,
                "models": ["gpt-3.5-turbo"],
                "strategies": ["balanced"],
                "description": "快速测试实验"
            },
            "full_comparison": {
                "num_episodes": 200,
                "models": ["gpt-3.5-turbo"],
                "strategies": ["balanced", "aggressive", "conservative"],
                "description": "完整对比实验"
            },
            "multi_model": {
                "num_episodes": 150,
                "models": ["gpt-3.5-turbo", "gpt-4"],
                "strategies": ["balanced"],
                "description": "多模型对比实验"
            }
        },
        "general_settings": {
            "results_dir": "logs/experiments",
            "save_intermediate": True,
            "generate_visualizations": True,
            "verbose": True
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
    else:
        # 创建默认配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        logger.info(f"创建默认配置文件: {config_path}")
    
    return default_config


async def run_single_experiment(experiment_name: str, config: dict):
    """运行单个实验"""
    logger.info(f"开始实验: {experiment_name}")
    logger.info(f"实验描述: {config.get('description', '无描述')}")
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"logs/experiments/{experiment_name}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建实验运行器
    runner = ExperimentRunner(experiment_dir)
    
    try:
        # 运行实验
        results = await runner.run_comparative_experiment(
            num_episodes=config["num_episodes"],
            models=config["models"],
            strategies=config["strategies"]
        )
        
        # 保存完整结果
        results_file = os.path.join(experiment_dir, "complete_results.json")
        
        # 简化结果以便保存
        simplified_results = {}
        for exp_name, exp_data in results["experiment_results"].items():
            simplified_results[exp_name] = {
                "config": exp_data["config"],
                "final_stats": exp_data["final_stats"],
                "learning_stats": exp_data["learning_stats"],
                "num_episodes": len(exp_data["episode_results"])
            }
        
        save_data = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "config": config,
            "results": simplified_results,
            "comparison_report": results["comparison_report"]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # 生成可视化
        if config.get("generate_visualizations", True):
            runner.visualize_results(results["experiment_results"], experiment_dir)
        
        # 打印结果摘要
        print_results_summary(results["comparison_report"])
        
        logger.info(f"实验 {experiment_name} 完成，结果保存到: {experiment_dir}")
        return results
        
    except Exception as e:
        logger.error(f"实验 {experiment_name} 失败: {e}")
        raise


def print_results_summary(comparison_report: dict):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)
    
    overall = comparison_report.get("overall_improvement", {})
    
    if overall:
        print(f"📊 测试配置数量: {overall['configurations_tested']}")
        print(f"📈 有改进的配置: {overall['improvements_positive']}")
        print(f"🎯 平均成功率改进: {overall['avg_success_rate_improvement']:.3f}")
        print(f"⚡ 平均步数改进: {overall['avg_steps_improvement']:.3f}")
        print(f"🎪 平均稳定性改进: {overall['avg_stability_improvement']:.3f}")
        
        print("\n📋 详细配置结果:")
        for config, details in comparison_report["detailed_comparisons"].items():
            print(f"\n  🔧 {config}:")
            print(f"    成功率: {details['baseline_success_rate']:.3f} → {details['experience_success_rate']:.3f} "
                  f"({'🔥+' if details['success_rate_improvement'] > 0 else '❄️'}{details['success_rate_improvement']:.3f})")
            print(f"    平均步数: {details['baseline_avg_steps']:.1f} → {details['experience_avg_steps']:.1f} "
                  f"({'⚡-' if details['steps_improvement'] > 0 else '🐌+'}{abs(details['steps_improvement']):.1f})")
            print(f"    学到技能数: {details['total_skills_learned']} 个")
            print(f"    技能使用率: {details['skill_usage_rate']:.3f}")
        
        # 评估结果
        print("\n🎖️ 实验评估:")
        if overall['avg_success_rate_improvement'] > 0.05:
            print("  ✅ 经验学习显著提升了成功率!")
        elif overall['avg_success_rate_improvement'] > 0.01:
            print("  ✴️ 经验学习适度提升了成功率")
        else:
            print("  ❌ 经验学习对成功率改进有限")
        
        if overall['avg_steps_improvement'] > 1.0:
            print("  ✅ 经验学习显著提升了效率!")
        elif overall['avg_steps_improvement'] > 0.5:
            print("  ✴️ 经验学习适度提升了效率")
        else:
            print("  ❌ 经验学习对效率改进有限")
    
    print("\n" + "="*80)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行无监督经验积累实验")
    parser.add_argument("--experiment", "-e", 
                       choices=["quick_test", "full_comparison", "multi_model", "all"],
                       default="quick_test",
                       help="选择实验类型")
    parser.add_argument("--config", "-c", 
                       default="experiment_config.json",
                       help="配置文件路径")
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    # 加载配置
    config = load_experiment_config(args.config)
    
    if args.experiment == "all":
        # 运行所有实验
        for exp_name in config["experiments"]:
            try:
                await run_single_experiment(exp_name, config["experiments"][exp_name])
                print(f"\n{'='*50}\n")
            except Exception as e:
                logger.error(f"实验 {exp_name} 失败: {e}")
    else:
        # 运行指定实验
        if args.experiment in config["experiments"]:
            await run_single_experiment(args.experiment, config["experiments"][args.experiment])
        else:
            logger.error(f"未找到实验配置: {args.experiment}")
            return
    
    print("\n🎉 所有实验完成!")


if __name__ == "__main__":
    asyncio.run(main())