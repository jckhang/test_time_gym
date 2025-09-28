"""
智能体功能测试
"""

import pytest
import sys
import os

# 添加包路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_time_gym.agents.dummy_agent import DummyAgent, RandomAgent, SkillBasedAgent


class TestAgents:
    """智能体测试类"""
    
    def test_dummy_agent_initialization(self):
        """测试DummyAgent初始化"""
        agent = DummyAgent("greedy")
        assert agent.strategy == "greedy"
        assert isinstance(agent.memory, list)
        assert isinstance(agent.skills, dict)
    
    def test_dummy_agent_action_selection(self):
        """测试DummyAgent动作选择"""
        agent = DummyAgent("greedy")
        
        # 测试搜索表单状态
        obs = {"view": "search_form"}
        action = agent.select_action(obs)
        assert action == "search_flights"
        
        # 测试搜索结果状态
        obs = {
            "view": "search_results",
            "flights": [{"id": "AA123", "price": 500, "stops": 1}],
            "constraints": {"budget": 800, "max_stops": 2}
        }
        action = agent.select_action(obs)
        assert action in ["filter_results", "add_to_cart", "select_flight"]
    
    def test_random_agent(self):
        """测试RandomAgent"""
        agent = RandomAgent()
        
        obs = {"view": "search_form"}
        action = agent.select_action(obs)
        
        # 随机智能体应该返回某个有效动作
        assert isinstance(action, str)
        assert len(action) > 0
    
    def test_skill_based_agent(self):
        """测试SkillBasedAgent"""
        agent = SkillBasedAgent(exploration_rate=0.5)
        
        assert agent.exploration_rate == 0.5
        assert hasattr(agent, 'skill_stats')
        
        obs = {"view": "search_form"}
        action = agent.select_action(obs)
        assert isinstance(action, str)
    
    def test_agent_memory_update(self):
        """测试智能体记忆更新"""
        agent = DummyAgent("greedy")
        
        # 创建一个成功轨迹
        successful_trajectory = [
            {"action": "search_flights", "reward": 0.02},
            {"action": "add_to_cart", "reward": 0.05},
            {"action": "confirm_payment", "reward": 1.0}
        ]
        
        initial_memory_count = len(agent.memory)
        agent.update_memory(successful_trajectory)
        
        assert len(agent.memory) == initial_memory_count + 1
        
        # 检查技能是否被提取
        if agent.skills:
            # 应该有一些技能被创建
            assert len(agent.skills) > 0
    
    def test_agent_stats(self):
        """测试智能体统计"""
        agent = DummyAgent("greedy")
        
        stats = agent.get_stats()
        assert isinstance(stats, dict)
        assert "total_episodes" in stats
        assert "skills_learned" in stats
        assert "top_skills" in stats


if __name__ == "__main__":
    # 手动运行测试
    test_agents = TestAgents()
    
    print("运行智能体测试...")
    
    test_methods = [
        test_agents.test_dummy_agent_initialization,
        test_agents.test_dummy_agent_action_selection,
        test_agents.test_random_agent,
        test_agents.test_skill_based_agent,
        test_agents.test_agent_memory_update,
        test_agents.test_agent_stats
    ]
    
    passed = 0
    for method in test_methods:
        try:
            method()
            print(f"✓ {method.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {method.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n测试结果: {passed}/{len(test_methods)} 通过")