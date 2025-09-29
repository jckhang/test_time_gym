"""
技能提取与管理系统
实现技能挖掘、Thompson Sampling选择和记忆更新
"""

import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta


@dataclass
class BetaDistribution:
    """Beta分布参数，用于表示技能的可靠性"""
    alpha: float = 1.0  # 成功次数 + 1
    beta: float = 1.0   # 失败次数 + 1

    def sample(self) -> float:
        """从Beta分布中采样"""
        return np.random.beta(self.alpha, self.beta)

    def mean(self) -> float:
        """计算均值"""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """计算方差"""
        denom = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        return (self.alpha * self.beta) / denom

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        lower = (1 - confidence) / 2
        upper = 1 - lower
        return (beta.ppf(lower, self.alpha, self.beta),
                beta.ppf(upper, self.alpha, self.beta))

    def update(self, success: bool):
        """更新分布参数"""
        if success:
            self.alpha += 1
        else:
            self.beta += 1


@dataclass
class Skill:
    """技能定义"""
    id: str
    name: str
    action_sequence: List[str]
    preconditions: Dict[str, Any]
    postconditions: Dict[str, Any]
    reliability: BetaDistribution
    usage_count: int = 0
    last_used: Optional[str] = None

    def matches_context(self, observation: Dict) -> bool:
        """检查当前上下文是否匹配技能的前置条件"""
        view = observation.get("view", "")

        # 简化的匹配逻辑
        required_view = self.preconditions.get("view", "")
        if required_view and view != required_view:
            return False

        # 检查是否有必要的数据
        if self.preconditions.get("requires_flights", False):
            if not observation.get("flights", []):
                return False

        if self.preconditions.get("requires_cart", False):
            if not observation.get("cart", {}).get("items", []):
                return False

        return True

    def get_confidence(self) -> float:
        """获取技能可靠性的置信度"""
        return self.reliability.mean()

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "action_sequence": self.action_sequence,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "reliability": {
                "alpha": self.reliability.alpha,
                "beta": self.reliability.beta,
                "mean": self.reliability.mean(),
                "variance": self.reliability.variance()
            },
            "usage_count": self.usage_count,
            "last_used": self.last_used
        }


class SkillManager:
    """技能管理器，负责技能提取、存储和选择"""

    def __init__(self, memory_dir: str = "logs/skills"):
        self.skills: Dict[str, Skill] = {}
        self.trajectories: List[Dict] = []
        self.memory_dir = memory_dir
        self.min_skill_length = 2
        self.min_support = 3  # 最少出现次数
        self.max_skills = 100  # 最大技能数量

        # 创建存储目录
        os.makedirs(memory_dir, exist_ok=True)

        # 尝试加载已有技能
        self._load_skills()

    def add_trajectory(self, trajectory: List[Dict], final_reward: float):
        """添加新轨迹并提取技能"""
        self.trajectories.append({
            "trajectory": trajectory,
            "final_reward": final_reward,
            "timestamp": str(np.datetime64('now'))
        })

        # 如果轨迹成功，尝试提取技能
        if final_reward > 0.5:  # 成功阈值
            self._extract_skills_from_trajectory(trajectory, success=True)
        else:
            self._extract_skills_from_trajectory(trajectory, success=False)

    def _extract_skills_from_trajectory(self, trajectory: List[Dict], success: bool):
        """从轨迹中提取技能"""
        if len(trajectory) < self.min_skill_length:
            return

        actions = [step.get("action", "") for step in trajectory if step.get("action")]

        # 提取所有可能的子序列
        for length in range(self.min_skill_length, min(len(actions) + 1, 6)):
            for start in range(len(actions) - length + 1):
                subsequence = actions[start:start + length]
                skill_id = self._generate_skill_id(subsequence)

                # 如果技能不存在，创建新技能
                if skill_id not in self.skills:
                    self._create_skill(skill_id, subsequence, trajectory[start:start + length])

                # 更新技能可靠性
                self.skills[skill_id].reliability.update(success)
                self.skills[skill_id].usage_count += 1

    def _generate_skill_id(self, action_sequence: List[str]) -> str:
        """生成技能ID"""
        sequence_str = "->".join(action_sequence)
        return hashlib.md5(sequence_str.encode()).hexdigest()[:8]

    def _create_skill(self, skill_id: str, action_sequence: List[str], context_steps: List[Dict]):
        """创建新技能"""
        # 分析前置和后置条件
        preconditions = {}
        postconditions = {}

        if context_steps:
            # 从第一步提取前置条件
            first_obs = context_steps[0].get("obs", {})
            preconditions = {
                "view": first_obs.get("view", ""),
                "requires_flights": len(first_obs.get("flights", [])) > 0,
                "requires_cart": len(first_obs.get("cart", {}).get("items", [])) > 0
            }

            # 从最后一步提取后置条件
            if len(context_steps) > 1:
                last_obs = context_steps[-1].get("obs", {})
                postconditions = {
                    "expected_view": last_obs.get("view", ""),
                    "cart_modified": last_obs.get("cart", {}).get("total", 0) > first_obs.get("cart", {}).get("total", 0)
                }

        skill = Skill(
            id=skill_id,
            name=f"skill_{skill_id}",
            action_sequence=action_sequence,
            preconditions=preconditions,
            postconditions=postconditions,
            reliability=BetaDistribution()
        )

        self.skills[skill_id] = skill

    def select_skill_thompson_sampling(self, observation: Dict) -> Optional[Skill]:
        """使用Thompson Sampling选择技能"""
        # 找到所有适用的技能
        applicable_skills = [
            skill for skill in self.skills.values()
            if skill.matches_context(observation) and skill.usage_count >= self.min_support
        ]

        if not applicable_skills:
            return None

        # Thompson Sampling：从每个技能的Beta分布中采样
        sampled_values = []
        for skill in applicable_skills:
            sampled_value = skill.reliability.sample()
            sampled_values.append((sampled_value, skill))

        # 选择采样值最高的技能
        best_skill = max(sampled_values, key=lambda x: x[0])[1]
        best_skill.last_used = str(np.datetime64('now'))

        return best_skill

    def get_best_skills(self, top_k: int = 10) -> List[Skill]:
        """获取可靠性最高的技能"""
        skills_with_confidence = [
            (skill, skill.get_confidence())
            for skill in self.skills.values()
            if skill.usage_count >= self.min_support
        ]

        sorted_skills = sorted(skills_with_confidence, key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in sorted_skills[:top_k]]

    def get_skill_stats(self) -> Dict:
        """获取技能统计信息"""
        if not self.skills:
            return {"total_skills": 0}

        reliabilities = [skill.get_confidence() for skill in self.skills.values()]
        usage_counts = [skill.usage_count for skill in self.skills.values()]

        return {
            "total_skills": len(self.skills),
            "avg_reliability": np.mean(reliabilities),
            "reliability_std": np.std(reliabilities),
            "avg_usage": np.mean(usage_counts),
            "total_trajectories": len(self.trajectories),
            "skills_with_support": len([s for s in self.skills.values() if s.usage_count >= self.min_support])
        }

    def prune_skills(self, min_reliability: float = 0.3, max_age_days: int = 30):
        """清理低效技能"""
        current_time = np.datetime64('now')

        skills_to_remove = []
        for skill_id, skill in self.skills.items():
            # 移除可靠性过低的技能
            if skill.get_confidence() < min_reliability and skill.usage_count >= self.min_support * 2:
                skills_to_remove.append(skill_id)
                continue

            # 移除长期未使用的技能
            if skill.last_used:
                last_used = np.datetime64(skill.last_used)
                if (current_time - last_used) > np.timedelta64(max_age_days, 'D'):
                    skills_to_remove.append(skill_id)

        for skill_id in skills_to_remove:
            del self.skills[skill_id]

        return len(skills_to_remove)

    def save_skills(self):
        """保存技能到磁盘"""
        skills_file = os.path.join(self.memory_dir, "skills.json")

        skills_data = {
            skill_id: skill.to_dict()
            for skill_id, skill in self.skills.items()
        }

        with open(skills_file, 'w', encoding='utf-8') as f:
            json.dump(skills_data, f, indent=2, ensure_ascii=False)

    def _load_skills(self):
        """从磁盘加载技能"""
        skills_file = os.path.join(self.memory_dir, "skills.json")

        if not os.path.exists(skills_file):
            return

        try:
            with open(skills_file, 'r', encoding='utf-8') as f:
                skills_data = json.load(f)

            for skill_id, skill_dict in skills_data.items():
                reliability = BetaDistribution(
                    alpha=skill_dict["reliability"]["alpha"],
                    beta=skill_dict["reliability"]["beta"]
                )

                skill = Skill(
                    id=skill_dict["id"],
                    name=skill_dict["name"],
                    action_sequence=skill_dict["action_sequence"],
                    preconditions=skill_dict["preconditions"],
                    postconditions=skill_dict["postconditions"],
                    reliability=reliability,
                    usage_count=skill_dict["usage_count"],
                    last_used=skill_dict.get("last_used")
                )

                self.skills[skill_id] = skill

        except Exception as e:
            print(f"Failed to load skills: {e}")

    def get_skill_by_id(self, skill_id: str) -> Optional[Skill]:
        """根据ID获取技能"""
        return self.skills.get(skill_id)

    def export_skills_summary(self) -> Dict:
        """导出技能摘要用于分析"""
        summary = {
            "skills": [],
            "stats": self.get_skill_stats(),
            "trajectory_count": len(self.trajectories)
        }

        for skill in self.get_best_skills(20):
            skill_summary = {
                "id": skill.id,
                "name": skill.name,
                "sequence_length": len(skill.action_sequence),
                "reliability_mean": skill.get_confidence(),
                "reliability_ci": skill.reliability.confidence_interval(),
                "usage_count": skill.usage_count,
                "action_sequence": skill.action_sequence
            }
            summary["skills"].append(skill_summary)

        return summary


class IntrinsicRewardCalculator:
    """内在奖励计算器"""

    def __init__(self):
        self.state_history = []

    def calculate_progress_reward(self, obs: Dict, previous_obs: Optional[Dict] = None) -> float:
        """计算进展奖励"""
        reward = 0.0

        # 视图进展奖励
        view_progress = {
            "search_form": 0,
            "search_results": 1,
            "cart": 2,
            "payment": 3,
            "receipt": 4
        }

        current_progress = view_progress.get(obs.get("view", ""), 0)

        if previous_obs:
            previous_progress = view_progress.get(previous_obs.get("view", ""), 0)
            if current_progress > previous_progress:
                reward += 0.01 * (current_progress - previous_progress)

        # 购物车非空奖励
        if obs.get("cart", {}).get("total", 0) > 0:
            reward += 0.005

        return reward

    def calculate_consistency_reward(self, obs: Dict, action: str) -> float:
        """计算一致性奖励"""
        reward = 0.0

        # JSON格式正确性
        try:
            json.dumps(obs)
            reward += 0.001
        except:
            reward -= 0.01

        # 约束一致性检查
        constraints = obs.get("constraints", {})
        cart_total = obs.get("cart", {}).get("total", 0)

        if cart_total > 0 and cart_total <= constraints.get("budget", float('inf')):
            reward += 0.002

        return reward

    def calculate_curiosity_reward(self, obs: Dict, predicted_obs: Optional[Dict] = None) -> float:
        """计算好奇心奖励（基于预测误差）"""
        if not predicted_obs:
            return 0.0

        # 简化实现：比较观察的关键字段
        differences = 0

        key_fields = ["view", "cart", "payment_state"]
        for field in key_fields:
            if obs.get(field) != predicted_obs.get(field):
                differences += 1

        # 预测误差越大，好奇心奖励越高（适度）
        return min(0.01 * differences, 0.05)


class MemoryManager:
    """记忆管理器，处理经验存储和遗忘"""

    def __init__(self, memory_dir: str = "logs/memory"):
        self.memory_dir = memory_dir
        self.trajectories = []
        self.max_trajectories = 10000
        self.forgetting_rate = 0.01

        os.makedirs(memory_dir, exist_ok=True)
        self._load_trajectories()

    def store_trajectory(self, trajectory: List[Dict], metadata: Dict):
        """存储轨迹"""
        trajectory_record = {
            "trajectory": trajectory,
            "metadata": metadata,
            "timestamp": str(np.datetime64('now')),
            "id": hashlib.md5(str(trajectory).encode()).hexdigest()[:12]
        }

        self.trajectories.append(trajectory_record)

        # 定期清理旧轨迹
        if len(self.trajectories) > self.max_trajectories:
            self._apply_forgetting()

    def _apply_forgetting(self):
        """应用遗忘机制"""
        # 保留最近的轨迹和高奖励轨迹
        sorted_trajectories = sorted(
            self.trajectories,
            key=lambda x: (x["metadata"].get("final_reward", 0), x["timestamp"]),
            reverse=True
        )

        # 保留前80%
        keep_count = int(len(sorted_trajectories) * 0.8)
        self.trajectories = sorted_trajectories[:keep_count]

    def get_similar_trajectories(self, current_obs: Dict, k: int = 5) -> List[Dict]:
        """获取相似的历史轨迹"""
        # 简化实现：基于视图和约束相似性
        current_view = current_obs.get("view", "")
        current_constraints = current_obs.get("constraints", {})

        scored_trajectories = []
        for traj_record in self.trajectories:
            traj = traj_record["trajectory"]
            if not traj:
                continue

            # 计算相似度得分
            score = 0.0

            # 视图匹配
            for step in traj:
                step_obs = step.get("obs", {})
                if step_obs.get("view") == current_view:
                    score += 1.0

            # 约束相似性
            if traj and "obs" in traj[0]:
                traj_constraints = traj[0]["obs"].get("constraints", {})
                for key in ["budget", "max_stops"]:
                    if (key in current_constraints and key in traj_constraints and
                        abs(current_constraints[key] - traj_constraints[key]) < 100):
                        score += 0.5

            if score > 0:
                scored_trajectories.append((score, traj_record))

        # 返回得分最高的k个轨迹
        scored_trajectories.sort(key=lambda x: x[0], reverse=True)
        return [traj for _, traj in scored_trajectories[:k]]

    def save_trajectories(self):
        """保存轨迹到磁盘"""
        traj_file = os.path.join(self.memory_dir, "trajectories.jsonl")

        with open(traj_file, 'w', encoding='utf-8') as f:
            for traj_record in self.trajectories:
                f.write(json.dumps(traj_record, ensure_ascii=False) + '\n')

    def _load_trajectories(self):
        """从磁盘加载轨迹"""
        traj_file = os.path.join(self.memory_dir, "trajectories.jsonl")

        if not os.path.exists(traj_file):
            return

        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.trajectories.append(json.loads(line))
        except Exception as e:
            print(f"Failed to load trajectories: {e}")


if __name__ == "__main__":
    # 测试技能系统
    skill_manager = SkillManager()

    # 模拟一些成功轨迹
    successful_trajectory = [
        {"action": "search_flights", "obs": {"view": "search_form"}, "reward": 0.02},
        {"action": "filter_results", "obs": {"view": "search_results", "flights": [{"id": "AA123"}]}, "reward": 0.01},
        {"action": "add_to_cart", "obs": {"view": "search_results"}, "reward": 0.05},
        {"action": "proceed_to_payment", "obs": {"view": "cart", "cart": {"total": 500}}, "reward": 0.03},
        {"action": "confirm_payment", "obs": {"view": "payment"}, "reward": 1.0}
    ]

    skill_manager.add_trajectory(successful_trajectory, final_reward=1.0)

    print("技能统计:", skill_manager.get_skill_stats())
    print("最佳技能:")
    for skill in skill_manager.get_best_skills(3):
        print(f"  {skill.name}: {skill.action_sequence} (可靠性: {skill.get_confidence():.3f})")

    # 测试Thompson Sampling
    test_obs = {"view": "search_results", "flights": [{"id": "test"}]}
    selected_skill = skill_manager.select_skill_thompson_sampling(test_obs)

    if selected_skill:
        print(f"选择的技能: {selected_skill.name}")
    else:
        print("未找到适用的技能")
