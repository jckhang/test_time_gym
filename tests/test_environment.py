"""
环境功能测试
"""

import os
import sys

import pytest

# 添加包路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_time_gym.envs.flight_booking_env import FlightBookingEnv, FlightBookingView


class TestFlightBookingEnv:
    """FlightBookingEnv测试类"""

    def setup_method(self):
        """测试前准备"""
        self.env = FlightBookingEnv(seed=42)

    def test_environment_initialization(self):
        """测试环境初始化"""
        assert self.env is not None
        assert self.env.max_steps == 50
        assert len(self.env.flights_db) > 0

    def test_reset_functionality(self):
        """测试重置功能"""
        obs, info = self.env.reset(seed=123)

        assert isinstance(obs, dict)
        assert "view" in obs
        assert "constraints" in obs
        assert "forms" in obs
        assert obs["view"] == "search_form"
        assert self.env.step_count == 0
        assert not self.env.done

    def test_step_functionality(self):
        """测试步进功能"""
        obs, info = self.env.reset()

        # 测试搜索航班
        next_obs, reward, done, trunc, info = self.env.step("search_flights")

        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(trunc, bool)
        assert isinstance(next_obs, dict)
        assert self.env.step_count == 1

    def test_action_parsing(self):
        """测试动作解析"""
        # 测试有效动作
        valid_actions = [
            "search_flights",
            "filter_results",
            "add_to_cart",
            "proceed_to_payment",
            "confirm_payment"
        ]

        for action in valid_actions:
            parsed = self.env._parse_action(action)
            assert isinstance(parsed, dict)
            assert "type" in parsed
            assert "params" in parsed

    def test_invalid_action(self):
        """测试无效动作处理"""
        obs, info = self.env.reset()

        # 执行无效动作
        next_obs, reward, done, trunc, info = self.env.step("invalid_action")

        assert reward < 0  # 应该有惩罚
        assert next_obs["view"] == "error"

    def test_full_workflow(self):
        """测试完整工作流"""
        obs, info = self.env.reset()

        actions = [
            "search_flights",
            "filter_results",
            "add_to_cart",
            "proceed_to_payment",
            "enter_card",
            "confirm_payment"
        ]

        total_reward = 0
        for action in actions:
            if self.env.done or self.env.truncated:
                break

            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward

        # 验证最终状态
        assert isinstance(total_reward, float)

    def test_constraint_checking(self):
        """测试约束检查"""
        obs, info = self.env.reset()

        # 检查任务约束
        constraints = obs["constraints"]
        assert "budget" in constraints
        assert "max_stops" in constraints
        assert "depart_after" in constraints

        # 验证约束值合理
        assert constraints["budget"] > 0
        assert constraints["max_stops"] >= 0

    def test_reward_calculation(self):
        """测试奖励计算"""
        obs, info = self.env.reset()

        # 测试搜索奖励
        obs, reward, done, trunc, info = self.env.step("search_flights")
        assert reward > 0  # 成功搜索应有正奖励

        # 测试时间成本
        assert reward < 0.1  # 但不应该太高（包含时间成本）


if __name__ == "__main__":
    # 手动运行测试
    test_env = TestFlightBookingEnv()

    print("运行环境测试...")

    test_methods = [
        test_env.test_environment_initialization,
        test_env.test_reset_functionality,
        test_env.test_step_functionality,
        test_env.test_action_parsing,
        test_env.test_invalid_action,
        test_env.test_constraint_checking,
        test_env.test_reward_calculation
    ]

    passed = 0
    for method in test_methods:
        try:
            test_env.setup_method()
            method()
            print(f"✓ {method.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {method.__name__}: {e}")

    print(f"\n测试结果: {passed}/{len(test_methods)} 通过")

    # 运行完整工作流测试
    print("\n运行完整工作流测试...")
    try:
        test_env.setup_method()
        test_env.test_full_workflow()
        print("✓ 完整工作流测试通过")
    except Exception as e:
        print(f"✗ 完整工作流测试失败: {e}")
