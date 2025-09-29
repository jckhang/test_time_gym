"""
配置管理模块
支持从YAML配置文件加载模型和策略配置
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，默认为项目根目录下的config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config = None
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            else:
                # 如果配置文件不存在，使用默认配置
                self._config = self._get_default_config()
                self._save_default_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "default_model": "claude-sonnet-4-20250514",
            "models": {
                "claude-sonnet-4-20250514": {
                    "name": "claude-sonnet-4-20250514",
                    "strategy": "balanced",
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "description": "Claude Sonnet 4 - 平衡策略，适合大多数任务"
                }
            },
            "strategies": {
                "aggressive": {
                    "description": "激进策略 - 快速决策，优先价格",
                    "default_action": "add_to_cart",
                    "temperature": 0.9
                },
                "balanced": {
                    "description": "平衡策略 - 平衡价格和质量",
                    "default_action": "search_flights",
                    "temperature": 0.7
                },
                "conservative": {
                    "description": "保守策略 - 优先质量，仔细评估",
                    "default_action": "filter_results",
                    "temperature": 0.5
                }
            }
        }

    def _save_default_config(self):
        """保存默认配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"保存默认配置失败: {e}")

    def get_default_model(self) -> str:
        """获取默认模型名称"""
        return self._config.get("default_model", "claude-sonnet-4-20250514")

    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型配置

        Args:
            model_name: 模型名称，如果为None则使用默认模型

        Returns:
            模型配置字典
        """
        if model_name is None:
            model_name = self.get_default_model()

        models = self._config.get("models", {})
        if model_name not in models:
            # 如果模型不存在，返回默认配置
            return {
                "name": model_name,
                "strategy": "balanced",
                "temperature": 0.7,
                "max_tokens": 8192,
                "description": f"模型 {model_name} - 使用默认配置"
            }

        return models[model_name]

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        获取策略配置

        Args:
            strategy_name: 策略名称

        Returns:
            策略配置字典
        """
        strategies = self._config.get("strategies", {})
        if strategy_name not in strategies:
            # 如果策略不存在，返回默认配置
            return {
                "description": f"策略 {strategy_name} - 使用默认配置",
                "default_action": "search_flights",
                "temperature": 0.7
            }

        return strategies[strategy_name]

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """获取所有可用模型"""
        return self._config.get("models", {})

    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """获取所有可用策略"""
        return self._config.get("strategies", {})

    def add_model(self, model_name: str, config: Dict[str, Any]):
        """
        添加新模型配置

        Args:
            model_name: 模型名称
            config: 模型配置
        """
        if "models" not in self._config:
            self._config["models"] = {}

        self._config["models"][model_name] = config
        self._save_config()

    def add_strategy(self, strategy_name: str, config: Dict[str, Any]):
        """
        添加新策略配置

        Args:
            strategy_name: 策略名称
            config: 策略配置
        """
        if "strategies" not in self._config:
            self._config["strategies"] = {}

        self._config["strategies"][strategy_name] = config
        self._save_config()

    def _save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config.copy()

    def update_config(self, new_config: Dict[str, Any]):
        """
        更新配置

        Args:
            new_config: 新配置
        """
        self._config.update(new_config)
        self._save_config()


# 全局配置管理器实例
config_manager = ConfigManager()


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """获取模型配置的便捷函数"""
    return config_manager.get_model_config(model_name)


def get_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """获取策略配置的便捷函数"""
    return config_manager.get_strategy_config(strategy_name)


def get_default_model() -> str:
    """获取默认模型名称的便捷函数"""
    return config_manager.get_default_model()


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """获取所有可用模型的便捷函数"""
    return config_manager.get_available_models()


def get_available_strategies() -> Dict[str, Dict[str, Any]]:
    """获取所有可用策略的便捷函数"""
    return config_manager.get_available_strategies()
