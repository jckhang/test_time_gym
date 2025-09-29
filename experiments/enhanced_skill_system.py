"""
增强的无监督技能学习系统
结合语义理解、主动探索和经验迁移
"""

import json
import os
import pickle
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

# 现有模块导入
from test_time_gym.utils.skill_system import SkillManager, Skill, BetaDistribution
from test_time_gym.utils.evaluation import EvaluationMetrics, EpisodeResult
from test_time_gym.agents.openai_agent import FlightBookingOpenAIAgent


@dataclass
class SemanticSkill:
    """语义化技能定义"""
    id: str
    name: str
    description: str  # 技能的语义描述
    action_pattern: List[str]  # 动作模式
    context_embedding: np.ndarray  # 上下文嵌入
    preconditions: Dict[str, Any]  # 前置条件
    postconditions: Dict[str, Any]  # 后置条件
    success_rate: BetaDistribution  # 成功率
    usage_count: int = 0
    generalization_score: float = 0.0  # 泛化得分
    discovery_episode: int = 0  # 发现episode
    last_used: Optional[str] = None


@dataclass
class ExperienceMemory:
    """经验记忆"""
    episode_id: str
    context_state: Dict
    action_taken: str
    outcome_state: Dict
    reward: float
    success: bool
    timestamp: str
    semantic_features: Dict  # 语义特征


class CuriosityDrivenExplorer:
    """好奇心驱动的探索器"""
    
    def __init__(self, exploration_rate: float = 0.3):
        self.exploration_rate = exploration_rate
        self.state_visit_counts = defaultdict(int)
        self.prediction_errors = deque(maxlen=1000)
        self.surprise_threshold = 0.5
    
    def calculate_curiosity_reward(self, state: Dict, predicted_state: Dict = None) -> float:
        """计算好奇心奖励"""
        # 基于状态新颖性的奖励
        state_hash = self._hash_state(state)
        visit_count = self.state_visit_counts[state_hash]
        self.state_visit_counts[state_hash] += 1
        
        # 新颖性奖励（访问次数越少奖励越高）
        novelty_reward = 1.0 / (1.0 + visit_count)
        
        # 预测误差奖励
        prediction_error = 0.0
        if predicted_state:
            prediction_error = self._calculate_prediction_error(state, predicted_state)
            self.prediction_errors.append(prediction_error)
        
        # 综合好奇心奖励
        total_curiosity = 0.6 * novelty_reward + 0.4 * prediction_error
        return min(total_curiosity, 0.2)  # 限制最大奖励
    
    def _hash_state(self, state: Dict) -> str:
        """生成状态哈希"""
        # 提取关键状态特征
        key_features = {
            'view': state.get('view', ''),
            'flights_count': len(state.get('flights', [])),
            'cart_total': state.get('cart', {}).get('total', 0),
            'payment_entered': state.get('payment_state', {}).get('card_entered', False)
        }
        return hashlib.md5(json.dumps(key_features, sort_keys=True).encode()).hexdigest()[:8]
    
    def _calculate_prediction_error(self, actual: Dict, predicted: Dict) -> float:
        """计算预测误差"""
        # 简化的预测误差计算
        differences = 0
        for key in ['view', 'cart', 'payment_state']:
            if actual.get(key) != predicted.get(key):
                differences += 1
        return differences / 3.0
    
    def should_explore(self, state: Dict) -> bool:
        """判断是否应该探索"""
        state_hash = self._hash_state(state)
        visit_count = self.state_visit_counts[state_hash]
        
        # 访问次数少的状态更可能被探索
        explore_prob = self.exploration_rate * (1.0 / (1.0 + visit_count))
        return np.random.random() < explore_prob


class SemanticSkillExtractor:
    """语义技能提取器"""
    
    def __init__(self):
        self.skill_templates = self._load_skill_templates()
    
    def extract_semantic_skills(self, trajectory: List[Dict]) -> List[SemanticSkill]:
        """从轨迹中提取语义技能"""
        skills = []
        
        # 基于模板匹配提取技能
        for template in self.skill_templates:
            extracted_skills = self._match_template(trajectory, template)
            skills.extend(extracted_skills)
        
        # 基于序列模式发现新技能
        pattern_skills = self._discover_patterns(trajectory)
        skills.extend(pattern_skills)
        
        return skills
    
    def _load_skill_templates(self) -> List[Dict]:
        """加载技能模板"""
        return [
            {
                "name": "quick_booking",
                "description": "快速预订 - 直接选择第一个符合预算的航班",
                "pattern": ["search_flights", "add_to_cart", "proceed_to_payment"],
                "context_keywords": ["quick", "fast", "budget"]
            },
            {
                "name": "careful_selection",
                "description": "仔细选择 - 先筛选再选择最优航班",
                "pattern": ["search_flights", "filter_results", "select_flight", "add_to_cart"],
                "context_keywords": ["filter", "compare", "best"]
            },
            {
                "name": "budget_optimization",
                "description": "预算优化 - 寻找最便宜的选项",
                "pattern": ["search_flights", "filter_results", "apply_coupon", "add_to_cart"],
                "context_keywords": ["cheap", "discount", "save"]
            },
            {
                "name": "error_recovery",
                "description": "错误恢复 - 从失败中恢复",
                "pattern": ["restart", "search_flights"],
                "context_keywords": ["error", "retry", "restart"]
            }
        ]
    
    def _match_template(self, trajectory: List[Dict], template: Dict) -> List[SemanticSkill]:
        """匹配技能模板"""
        skills = []
        actions = [step.get('action', '') for step in trajectory]
        pattern = template['pattern']
        
        # 查找模式匹配
        for i in range(len(actions) - len(pattern) + 1):
            if actions[i:i+len(pattern)] == pattern:
                # 创建语义技能
                skill_id = hashlib.md5(f"{template['name']}_{i}".encode()).hexdigest()[:8]
                
                context_embedding = self._extract_context_embedding(
                    trajectory[i:i+len(pattern)]
                )
                
                skill = SemanticSkill(
                    id=skill_id,
                    name=template['name'],
                    description=template['description'],
                    action_pattern=pattern,
                    context_embedding=context_embedding,
                    preconditions=self._extract_preconditions(trajectory[i]),
                    postconditions=self._extract_postconditions(trajectory[i+len(pattern)-1]),
                    success_rate=BetaDistribution(),
                    generalization_score=0.5  # 模板技能有中等泛化性
                )
                skills.append(skill)
        
        return skills
    
    def _discover_patterns(self, trajectory: List[Dict]) -> List[SemanticSkill]:
        """发现新的行为模式"""
        skills = []
        actions = [step.get('action', '') for step in trajectory]
        
        # 发现重复的子序列
        for length in range(2, min(len(actions), 5)):
            for start in range(len(actions) - length + 1):
                subsequence = actions[start:start + length]
                
                # 检查这个子序列是否在其他地方重复出现
                occurrences = 0
                for i in range(len(actions) - length + 1):
                    if actions[i:i + length] == subsequence:
                        occurrences += 1
                
                # 如果重复出现2次以上，视为潜在技能
                if occurrences >= 2:
                    skill_id = hashlib.md5(f"pattern_{'_'.join(subsequence)}".encode()).hexdigest()[:8]
                    
                    skill = SemanticSkill(
                        id=skill_id,
                        name=f"pattern_{skill_id}",
                        description=f"发现的行为模式: {' -> '.join(subsequence)}",
                        action_pattern=subsequence,
                        context_embedding=self._extract_context_embedding(
                            trajectory[start:start + length]
                        ),
                        preconditions={},
                        postconditions={},
                        success_rate=BetaDistribution(),
                        generalization_score=0.3  # 发现的模式泛化性较低
                    )
                    skills.append(skill)
        
        return skills
    
    def _extract_context_embedding(self, steps: List[Dict]) -> np.ndarray:
        """提取上下文嵌入（简化版）"""
        # 提取关键特征
        features = []
        
        for step in steps:
            obs = step.get('obs', {})
            features.extend([
                len(obs.get('flights', [])),
                obs.get('cart', {}).get('total', 0) / 1000.0,  # 归一化
                int(obs.get('view', '') == 'search_results'),
                int(obs.get('view', '') == 'cart'),
                int(obs.get('view', '') == 'payment'),
                obs.get('constraints', {}).get('budget', 1000) / 1000.0,
                obs.get('payment_state', {}).get('attempts', 0)
            ])
        
        # 填充或截断到固定长度
        target_length = 50
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
        
        return np.array(features)
    
    def _extract_preconditions(self, step: Dict) -> Dict:
        """提取前置条件"""
        obs = step.get('obs', {})
        return {
            'view': obs.get('view', ''),
            'has_flights': len(obs.get('flights', [])) > 0,
            'cart_empty': obs.get('cart', {}).get('total', 0) == 0
        }
    
    def _extract_postconditions(self, step: Dict) -> Dict:
        """提取后置条件"""
        obs = step.get('obs', {})
        return {
            'expected_view': obs.get('view', ''),
            'cart_changed': obs.get('cart', {}).get('total', 0) > 0
        }


class EnhancedSkillManager:
    """增强的技能管理器"""
    
    def __init__(self, memory_dir: str = "logs/enhanced_skills"):
        self.skills: Dict[str, SemanticSkill] = {}
        self.experiences: List[ExperienceMemory] = []
        self.skill_extractor = SemanticSkillExtractor()
        self.curiosity_explorer = CuriosityDrivenExplorer()
        self.memory_dir = memory_dir
        self.episode_count = 0
        
        os.makedirs(memory_dir, exist_ok=True)
        self._load_skills()
    
    def process_episode(self, trajectory: List[Dict], final_reward: float, episode_id: str):
        """处理完整episode"""
        self.episode_count += 1
        
        # 存储经验
        self._store_experiences(trajectory, final_reward, episode_id)
        
        # 提取技能
        extracted_skills = self.skill_extractor.extract_semantic_skills(trajectory)
        
        # 更新技能库
        success = final_reward > 0.5
        for skill in extracted_skills:
            self._update_or_add_skill(skill, success)
        
        # 计算技能泛化得分
        self._update_generalization_scores()
        
        # 定期清理和优化
        if self.episode_count % 50 == 0:
            self._optimize_skill_library()
    
    def _store_experiences(self, trajectory: List[Dict], final_reward: float, episode_id: str):
        """存储经验记忆"""
        for i, step in enumerate(trajectory):
            experience = ExperienceMemory(
                episode_id=f"{episode_id}_{i}",
                context_state=step.get('obs', {}),
                action_taken=step.get('action', ''),
                outcome_state=trajectory[i+1].get('obs', {}) if i+1 < len(trajectory) else {},
                reward=step.get('reward', 0.0),
                success=final_reward > 0.5,
                timestamp=datetime.now().isoformat(),
                semantic_features=self._extract_semantic_features(step)
            )
            self.experiences.append(experience)
        
        # 限制经验数量
        if len(self.experiences) > 10000:
            self.experiences = self.experiences[-8000:]  # 保留最新的8000条
    
    def _extract_semantic_features(self, step: Dict) -> Dict:
        """提取语义特征"""
        obs = step.get('obs', {})
        return {
            'complexity': len(obs.get('flights', [])) / 10.0,
            'progress': self._calculate_progress(obs.get('view', '')),
            'constraint_satisfaction': self._check_constraints(obs),
            'decision_difficulty': self._assess_decision_difficulty(obs)
        }
    
    def _calculate_progress(self, view: str) -> float:
        """计算任务进度"""
        progress_map = {
            'search_form': 0.0,
            'search_results': 0.3,
            'cart': 0.6,
            'payment': 0.8,
            'receipt': 1.0,
            'error': 0.1
        }
        return progress_map.get(view, 0.0)
    
    def _check_constraints(self, obs: Dict) -> float:
        """检查约束满足度"""
        constraints = obs.get('constraints', {})
        cart_total = obs.get('cart', {}).get('total', 0)
        budget = constraints.get('budget', float('inf'))
        
        if cart_total == 0:
            return 1.0  # 没有违反约束
        
        return 1.0 if cart_total <= budget else 0.0
    
    def _assess_decision_difficulty(self, obs: Dict) -> float:
        """评估决策难度"""
        flights = obs.get('flights', [])
        
        if not flights:
            return 0.0
        
        # 基于选项数量和价格分散度评估难度
        prices = [f.get('price', 0) for f in flights]
        if not prices:
            return 0.0
        
        price_std = np.std(prices) if len(prices) > 1 else 0
        option_difficulty = len(flights) / 10.0  # 选项越多越难
        price_difficulty = price_std / np.mean(prices) if np.mean(prices) > 0 else 0
        
        return min((option_difficulty + price_difficulty) / 2.0, 1.0)
    
    def _update_or_add_skill(self, skill: SemanticSkill, success: bool):
        """更新或添加技能"""
        if skill.id in self.skills:
            # 更新现有技能
            existing_skill = self.skills[skill.id]
            existing_skill.success_rate.update(success)
            existing_skill.usage_count += 1
            existing_skill.last_used = datetime.now().isoformat()
        else:
            # 添加新技能
            skill.success_rate.update(success)
            skill.usage_count = 1
            skill.discovery_episode = self.episode_count
            skill.last_used = datetime.now().isoformat()
            self.skills[skill.id] = skill
    
    def _update_generalization_scores(self):
        """更新技能泛化得分"""
        for skill in self.skills.values():
            # 基于使用频率和成功率计算泛化得分
            usage_score = min(skill.usage_count / 10.0, 1.0)
            success_score = skill.success_rate.mean()
            diversity_score = self._calculate_context_diversity(skill)
            
            skill.generalization_score = (
                0.4 * usage_score + 
                0.4 * success_score + 
                0.2 * diversity_score
            )
    
    def _calculate_context_diversity(self, skill: SemanticSkill) -> float:
        """计算技能上下文多样性"""
        # 查找使用这个技能的不同上下文
        skill_experiences = [
            exp for exp in self.experiences
            if skill.action_pattern and len(skill.action_pattern) > 0 and 
            exp.action_taken == skill.action_pattern[0]  # 简化匹配
        ]
        
        if len(skill_experiences) < 2:
            return 0.0
        
        # 计算上下文嵌入的多样性
        embeddings = []
        for exp in skill_experiences[-10:]:  # 最近10次使用
            features = list(exp.semantic_features.values())
            embeddings.append(features)
        
        if len(embeddings) < 2:
            return 0.0
        
        # 计算平均欧氏距离作为多样性度量
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(np.array(embeddings[i]) - np.array(embeddings[j]))
                distances.append(dist)
        
        return min(np.mean(distances) if distances else 0.0, 1.0)
    
    def select_skill_with_exploration(self, observation: Dict) -> Optional[SemanticSkill]:
        """结合探索的技能选择"""
        # 检查是否应该探索
        if self.curiosity_explorer.should_explore(observation):
            return None  # 返回None表示应该进行随机探索
        
        # 否则使用智能技能选择
        return self.select_best_skill(observation)
    
    def select_best_skill(self, observation: Dict) -> Optional[SemanticSkill]:
        """选择最佳技能"""
        applicable_skills = self._find_applicable_skills(observation)
        
        if not applicable_skills:
            return None
        
        # 综合考虑成功率、泛化得分和好奇心
        scored_skills = []
        for skill in applicable_skills:
            score = (
                0.5 * skill.success_rate.mean() +
                0.3 * skill.generalization_score +
                0.2 * self._calculate_novelty_bonus(skill, observation)
            )
            scored_skills.append((score, skill))
        
        # 选择得分最高的技能
        scored_skills.sort(reverse=True)
        return scored_skills[0][1]
    
    def _find_applicable_skills(self, observation: Dict) -> List[SemanticSkill]:
        """查找适用的技能"""
        applicable = []
        
        for skill in self.skills.values():
            if skill.usage_count < 2:  # 至少使用过2次
                continue
            
            # 检查前置条件
            if self._check_preconditions(skill, observation):
                applicable.append(skill)
        
        return applicable
    
    def _check_preconditions(self, skill: SemanticSkill, observation: Dict) -> bool:
        """检查技能前置条件"""
        # 简化的前置条件检查
        view = observation.get('view', '')
        
        if skill.preconditions.get('view'):
            if view != skill.preconditions['view']:
                return False
        
        has_flights = len(observation.get('flights', [])) > 0
        if skill.preconditions.get('has_flights', False) and not has_flights:
            return False
        
        return True
    
    def _calculate_novelty_bonus(self, skill: SemanticSkill, observation: Dict) -> float:
        """计算新颖性奖励"""
        # 基于当前上下文与技能历史使用上下文的相似性
        current_embedding = self.skill_extractor._extract_context_embedding([{'obs': observation}])
        
        if len(current_embedding) != len(skill.context_embedding):
            return 0.0
        
        similarity = np.dot(current_embedding, skill.context_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(skill.context_embedding) + 1e-8
        )
        
        # 相似性越低，新颖性奖励越高
        return 1.0 - abs(similarity)
    
    def _optimize_skill_library(self):
        """优化技能库"""
        # 移除低质量技能
        skills_to_remove = []
        for skill_id, skill in self.skills.items():
            # 移除成功率低且使用次数少的技能
            if (skill.success_rate.mean() < 0.2 and skill.usage_count < 5 and 
                self.episode_count - skill.discovery_episode > 50):
                skills_to_remove.append(skill_id)
            
            # 移除泛化得分过低的技能
            if skill.generalization_score < 0.1 and skill.usage_count < 3:
                skills_to_remove.append(skill_id)
        
        for skill_id in skills_to_remove:
            del self.skills[skill_id]
        
        print(f"优化技能库: 移除了 {len(skills_to_remove)} 个低质量技能")
    
    def get_skill_analytics(self) -> Dict:
        """获取技能分析数据"""
        if not self.skills:
            return {"total_skills": 0}
        
        success_rates = [skill.success_rate.mean() for skill in self.skills.values()]
        generalization_scores = [skill.generalization_score for skill in self.skills.values()]
        usage_counts = [skill.usage_count for skill in self.skills.values()]
        
        # 按类型分组技能
        skill_types = defaultdict(int)
        for skill in self.skills.values():
            if skill.name.startswith('pattern_'):
                skill_types['discovered_patterns'] += 1
            else:
                skill_types['template_skills'] += 1
        
        return {
            "total_skills": len(self.skills),
            "avg_success_rate": np.mean(success_rates),
            "avg_generalization": np.mean(generalization_scores),
            "avg_usage": np.mean(usage_counts),
            "skill_types": dict(skill_types),
            "total_experiences": len(self.experiences),
            "episodes_processed": self.episode_count,
            "top_skills": self._get_top_skills(5)
        }
    
    def _get_top_skills(self, n: int) -> List[Dict]:
        """获取top技能"""
        skills_with_scores = [
            (skill.generalization_score, skill) for skill in self.skills.values()
        ]
        skills_with_scores.sort(reverse=True)
        
        top_skills = []
        for score, skill in skills_with_scores[:n]:
            top_skills.append({
                "name": skill.name,
                "description": skill.description,
                "success_rate": skill.success_rate.mean(),
                "generalization_score": skill.generalization_score,
                "usage_count": skill.usage_count,
                "pattern": skill.action_pattern
            })
        
        return top_skills
    
    def save_skills(self):
        """保存技能到磁盘"""
        skills_file = os.path.join(self.memory_dir, "enhanced_skills.pkl")
        experiences_file = os.path.join(self.memory_dir, "experiences.pkl")
        
        # 保存技能
        skills_data = {}
        for skill_id, skill in self.skills.items():
            skills_data[skill_id] = {
                **asdict(skill),
                'context_embedding': skill.context_embedding.tolist(),
                'success_rate': {
                    'alpha': skill.success_rate.alpha,
                    'beta': skill.success_rate.beta
                }
            }
        
        with open(skills_file, 'wb') as f:
            pickle.dump(skills_data, f)
        
        # 保存经验
        with open(experiences_file, 'wb') as f:
            pickle.dump(self.experiences, f)
        
        print(f"已保存 {len(self.skills)} 个技能和 {len(self.experiences)} 条经验")
    
    def _load_skills(self):
        """从磁盘加载技能"""
        skills_file = os.path.join(self.memory_dir, "enhanced_skills.pkl")
        experiences_file = os.path.join(self.memory_dir, "experiences.pkl")
        
        # 加载技能
        if os.path.exists(skills_file):
            try:
                with open(skills_file, 'rb') as f:
                    skills_data = pickle.load(f)
                
                for skill_id, skill_dict in skills_data.items():
                    success_rate = BetaDistribution(
                        alpha=skill_dict['success_rate']['alpha'],
                        beta=skill_dict['success_rate']['beta']
                    )
                    
                    skill = SemanticSkill(
                        id=skill_dict['id'],
                        name=skill_dict['name'],
                        description=skill_dict['description'],
                        action_pattern=skill_dict['action_pattern'],
                        context_embedding=np.array(skill_dict['context_embedding']),
                        preconditions=skill_dict['preconditions'],
                        postconditions=skill_dict['postconditions'],
                        success_rate=success_rate,
                        usage_count=skill_dict['usage_count'],
                        generalization_score=skill_dict['generalization_score'],
                        discovery_episode=skill_dict['discovery_episode'],
                        last_used=skill_dict.get('last_used')
                    )
                    
                    self.skills[skill_id] = skill
                
                print(f"加载了 {len(self.skills)} 个技能")
            except Exception as e:
                print(f"加载技能失败: {e}")
        
        # 加载经验
        if os.path.exists(experiences_file):
            try:
                with open(experiences_file, 'rb') as f:
                    self.experiences = pickle.load(f)
                print(f"加载了 {len(self.experiences)} 条经验")
            except Exception as e:
                print(f"加载经验失败: {e}")


if __name__ == "__main__":
    # 测试增强技能系统
    manager = EnhancedSkillManager()
    
    # 模拟一个成功轨迹
    trajectory = [
        {
            "action": "search_flights",
            "obs": {"view": "search_form", "flights": [], "cart": {"total": 0}, "constraints": {"budget": 800}},
            "reward": 0.02
        },
        {
            "action": "filter_results", 
            "obs": {"view": "search_results", "flights": [{"id": "AA123", "price": 600}], "cart": {"total": 0}},
            "reward": 0.01
        },
        {
            "action": "add_to_cart",
            "obs": {"view": "search_results", "flights": [{"id": "AA123", "price": 600}]},
            "reward": 0.05
        },
        {
            "action": "proceed_to_payment",
            "obs": {"view": "cart", "cart": {"total": 600}},
            "reward": 0.03
        }
    ]
    
    # 处理episode
    manager.process_episode(trajectory, final_reward=1.0, episode_id="test_001")
    
    # 查看分析结果
    analytics = manager.get_skill_analytics()
    print("技能分析结果:")
    for key, value in analytics.items():
        print(f"  {key}: {value}")
    
    # 保存技能
    manager.save_skills()