#!/usr/bin/env python3
"""
简化版无监督经验积累实验
不依赖外部库，演示核心概念
"""

import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import hashlib


class SimpleBetaDistribution:
    """简化的Beta分布实现"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
    
    def sample(self) -> float:
        """简化采样（使用均匀分布近似）"""
        return random.uniform(0, 1) * self.mean()
    
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1


class SimpleSkill:
    """简化的技能类"""
    
    def __init__(self, skill_id: str, name: str, actions: List[str]):
        self.id = skill_id
        self.name = name
        self.actions = actions
        self.success_rate = SimpleBetaDistribution()
        self.usage_count = 0
        self.contexts = []  # 存储使用上下文
    
    def matches_context(self, context: Dict) -> bool:
        """检查是否匹配当前上下文"""
        view = context.get('view', '')
        
        # 简单的匹配逻辑
        if self.name == 'quick_booking' and view == 'search_results':
            return True
        elif self.name == 'careful_selection' and view == 'search_results':
            return True
        elif self.name == 'payment_flow' and view in ['cart', 'payment']:
            return True
        
        return False
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'actions': self.actions,
            'success_rate': self.success_rate.mean(),
            'usage_count': self.usage_count
        }


class SimpleSkillManager:
    """简化的技能管理器"""
    
    def __init__(self, memory_dir: str = "logs/simple_skills"):
        self.skills: Dict[str, SimpleSkill] = {}
        self.episode_count = 0
        self.memory_dir = memory_dir
        self.exploration_rate = 0.3
        self.state_visits = defaultdict(int)
        
        os.makedirs(memory_dir, exist_ok=True)
        self._load_skills()
    
    def process_episode(self, trajectory: List[Dict], final_reward: float):
        """处理episode并提取技能"""
        self.episode_count += 1
        
        # 提取动作序列
        actions = [step.get('action', '') for step in trajectory]
        
        # 提取已知的技能模式
        success = final_reward > 0.5
        
        # 快速预订模式
        if self._contains_pattern(actions, ['search_flights', 'add_to_cart']):
            self._update_skill('quick_booking', 'Quick Booking', 
                             ['search_flights', 'add_to_cart'], success, trajectory)
        
        # 仔细选择模式
        if self._contains_pattern(actions, ['search_flights', 'filter_results', 'add_to_cart']):
            self._update_skill('careful_selection', 'Careful Selection',
                             ['search_flights', 'filter_results', 'add_to_cart'], success, trajectory)
        
        # 支付流程模式
        if self._contains_pattern(actions, ['proceed_to_payment', 'enter_card', 'confirm_payment']):
            self._update_skill('payment_flow', 'Payment Flow',
                             ['proceed_to_payment', 'enter_card', 'confirm_payment'], success, trajectory)
    
    def _contains_pattern(self, actions: List[str], pattern: List[str]) -> bool:
        """检查是否包含特定模式"""
        for i in range(len(actions) - len(pattern) + 1):
            if actions[i:i+len(pattern)] == pattern:
                return True
        return False
    
    def _update_skill(self, skill_id: str, name: str, actions: List[str], success: bool, trajectory: List[Dict]):
        """更新或创建技能"""
        if skill_id not in self.skills:
            self.skills[skill_id] = SimpleSkill(skill_id, name, actions)
        
        skill = self.skills[skill_id]
        skill.success_rate.update(success)
        skill.usage_count += 1
        skill.contexts.append({
            'episode': self.episode_count,
            'success': success,
            'context_summary': self._summarize_context(trajectory)
        })
    
    def _summarize_context(self, trajectory: List[Dict]) -> Dict:
        """总结轨迹上下文"""
        if not trajectory:
            return {}
        
        first_obs = trajectory[0].get('obs', {})
        return {
            'initial_view': first_obs.get('view', ''),
            'flights_available': len(first_obs.get('flights', [])),
            'budget': first_obs.get('constraints', {}).get('budget', 0),
            'trajectory_length': len(trajectory)
        }
    
    def select_skill(self, observation: Dict) -> Optional[SimpleSkill]:
        """选择技能"""
        # 更新状态访问计数
        state_hash = self._hash_state(observation)
        self.state_visits[state_hash] += 1
        
        # 探索vs利用决策
        if random.random() < self.exploration_rate:
            return None  # 探索
        
        # 找到适用的技能
        applicable_skills = [
            skill for skill in self.skills.values()
            if skill.matches_context(observation) and skill.usage_count >= 2
        ]
        
        if not applicable_skills:
            return None
        
        # Thompson Sampling简化版
        best_skill = None
        best_score = -1
        
        for skill in applicable_skills:
            score = skill.success_rate.sample()
            if score > best_score:
                best_score = score
                best_skill = skill
        
        return best_skill
    
    def _hash_state(self, state: Dict) -> str:
        """生成状态哈希"""
        key_features = {
            'view': state.get('view', ''),
            'flights_count': len(state.get('flights', [])),
            'cart_total': state.get('cart', {}).get('total', 0)
        }
        return hashlib.md5(json.dumps(key_features, sort_keys=True).encode()).hexdigest()[:8]
    
    def get_analytics(self) -> Dict:
        """获取分析数据"""
        if not self.skills:
            return {'total_skills': 0}
        
        return {
            'total_skills': len(self.skills),
            'episode_count': self.episode_count,
            'avg_success_rate': sum(s.success_rate.mean() for s in self.skills.values()) / len(self.skills),
            'total_usage': sum(s.usage_count for s in self.skills.values()),
            'skills_details': [s.to_dict() for s in self.skills.values()]
        }
    
    def save_skills(self):
        """保存技能"""
        skills_file = os.path.join(self.memory_dir, "skills.json")
        data = {
            'episode_count': self.episode_count,
            'skills': {sid: skill.to_dict() for sid, skill in self.skills.items()}
        }
        
        with open(skills_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_skills(self):
        """加载技能"""
        skills_file = os.path.join(self.memory_dir, "skills.json")
        
        if os.path.exists(skills_file):
            try:
                with open(skills_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.episode_count = data.get('episode_count', 0)
                
                for skill_id, skill_data in data.get('skills', {}).items():
                    skill = SimpleSkill(skill_id, skill_data['name'], skill_data['actions'])
                    skill.usage_count = skill_data['usage_count']
                    # 重建成功率（简化）
                    skill.success_rate.alpha = max(1, skill_data['success_rate'] * skill_data['usage_count'])
                    skill.success_rate.beta = max(1, (1 - skill_data['success_rate']) * skill_data['usage_count'])
                    self.skills[skill_id] = skill
                
                print(f"加载了 {len(self.skills)} 个技能")
            except Exception as e:
                print(f"加载技能失败: {e}")


class SimpleEnvironment:
    """简化的环境"""
    
    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
        
        self.current_view = 'search_form'
        self.step_count = 0
        self.flights = []
        self.cart_total = 0
        self.budget = random.choice([700, 800, 900, 1000, 1200])
        self.payment_entered = False
        self.done = False
    
    def reset(self) -> Tuple[Dict, Dict]:
        """重置环境"""
        self.current_view = 'search_form'
        self.step_count = 0
        self.flights = []
        self.cart_total = 0
        self.budget = random.choice([700, 800, 900, 1000, 1200])
        self.payment_entered = False
        self.done = False
        
        obs = self._get_observation()
        info = {'step': self.step_count}
        return obs, info
    
    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步"""
        self.step_count += 1
        reward = -0.01  # 时间成本
        
        if action == 'search_flights':
            if self.current_view == 'search_form':
                self.current_view = 'search_results'
                # 生成随机航班
                self.flights = [
                    {'id': f'FL{i}', 'price': random.randint(400, 1200), 'quality': random.uniform(0.5, 1.0)}
                    for i in range(random.randint(3, 8))
                ]
                reward += 0.02
        
        elif action == 'filter_results':
            if self.current_view == 'search_results' and self.flights:
                # 过滤掉超预算的航班
                self.flights = [f for f in self.flights if f['price'] <= self.budget]
                reward += 0.01
        
        elif action == 'add_to_cart':
            if self.current_view == 'search_results' and self.flights:
                # 选择最便宜的航班
                cheapest = min(self.flights, key=lambda x: x['price'])
                self.cart_total = cheapest['price']
                self.current_view = 'cart'
                reward += 0.05
        
        elif action == 'proceed_to_payment':
            if self.current_view == 'cart' and self.cart_total > 0:
                self.current_view = 'payment'
                reward += 0.03
        
        elif action == 'enter_card':
            if self.current_view == 'payment':
                self.payment_entered = True
                reward += 0.01
        
        elif action == 'confirm_payment':
            if self.current_view == 'payment' and self.payment_entered:
                self.current_view = 'receipt'
                self.done = True
                # 检查约束
                if self.cart_total <= self.budget:
                    reward += 1.0  # 成功完成
                else:
                    reward -= 0.3  # 超预算惩罚
        
        obs = self._get_observation()
        info = {'step': self.step_count}
        
        # 超过最大步数
        truncated = self.step_count >= 20
        
        return obs, reward, self.done, truncated, info
    
    def _get_observation(self) -> Dict:
        """获取观察"""
        return {
            'view': self.current_view,
            'step': self.step_count,
            'flights': self.flights,
            'cart': {'total': self.cart_total},
            'constraints': {'budget': self.budget},
            'payment_state': {'card_entered': self.payment_entered},
            'done': self.done
        }


class SimpleAgent:
    """简化的智能体"""
    
    def __init__(self, skill_manager: SimpleSkillManager, use_skills: bool = True):
        self.skill_manager = skill_manager
        self.use_skills = use_skills
        self.current_trajectory = []
    
    def select_action(self, observation: Dict) -> str:
        """选择动作"""
        # 记录轨迹
        self.current_trajectory.append({
            'obs': observation,
            'action': None  # 将在后面设置
        })
        
        # 如果使用技能，尝试技能选择
        if self.use_skills:
            selected_skill = self.skill_manager.select_skill(observation)
            if selected_skill and selected_skill.actions:
                action = selected_skill.actions[0]  # 使用技能的第一个动作
                print(f"    使用技能 '{selected_skill.name}': {action}")
                self.current_trajectory[-1]['action'] = action
                self.current_trajectory[-1]['skill_used'] = selected_skill.name
                return action
        
        # 否则使用简单规则
        action = self._rule_based_action(observation)
        self.current_trajectory[-1]['action'] = action
        return action
    
    def _rule_based_action(self, observation: Dict) -> str:
        """基于规则的动作选择"""
        view = observation.get('view', '')
        
        if view == 'search_form':
            return 'search_flights'
        elif view == 'search_results':
            flights = observation.get('flights', [])
            if flights:
                return 'add_to_cart'
            else:
                return 'search_flights'
        elif view == 'cart':
            return 'proceed_to_payment'
        elif view == 'payment':
            if not observation.get('payment_state', {}).get('card_entered', False):
                return 'enter_card'
            else:
                return 'confirm_payment'
        else:
            return 'search_flights'
    
    def end_episode(self, final_reward: float):
        """结束episode"""
        if self.current_trajectory:
            self.skill_manager.process_episode(self.current_trajectory, final_reward)
        self.current_trajectory = []


class SimpleExperiment:
    """简化的实验类"""
    
    def __init__(self, results_dir: str = "logs/simple_experiment"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def run_comparison(self, num_episodes: int = 50) -> Dict:
        """运行对比实验"""
        results = {}
        
        # 基线智能体（不使用技能）
        print("运行基线实验...")
        baseline_results = self._run_single_experiment(
            "baseline", num_episodes, use_skills=False
        )
        results['baseline'] = baseline_results
        
        # 技能学习智能体
        print("\n运行技能学习实验...")
        skill_results = self._run_single_experiment(
            "with_skills", num_episodes, use_skills=True
        )
        results['with_skills'] = skill_results
        
        # 生成对比报告
        comparison = self._compare_results(baseline_results, skill_results)
        results['comparison'] = comparison
        
        return results
    
    def _run_single_experiment(self, name: str, num_episodes: int, use_skills: bool) -> Dict:
        """运行单个实验"""
        env = SimpleEnvironment(seed=42)
        skill_manager = SimpleSkillManager(f"{self.results_dir}/skills_{name}")
        agent = SimpleAgent(skill_manager, use_skills)
        
        episode_results = []
        success_rates = []  # 滑动平均成功率
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            
            for step in range(20):  # 最大步数
                action = agent.select_action(obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                
                if done or trunc:
                    break
            
            # 记录结果
            success = obs.get('view') == 'receipt'
            episode_results.append({
                'episode': episode,
                'total_reward': total_reward,
                'success': success,
                'steps': info['step']
            })
            
            # 结束episode
            agent.end_episode(total_reward)
            
            # 计算滑动平均成功率
            if episode >= 9:  # 每10个episode计算一次
                recent_success = sum(1 for r in episode_results[-10:] if r['success']) / 10
                success_rates.append(recent_success)
            
            # 进度报告
            if (episode + 1) % 20 == 0:
                recent_success_rate = sum(1 for r in episode_results[-20:] if r['success']) / 20
                print(f"  Episode {episode + 1}: 最近20次成功率 {recent_success_rate:.3f}")
        
        # 计算统计
        total_success = sum(1 for r in episode_results if r['success'])
        avg_reward = sum(r['total_reward'] for r in episode_results) / len(episode_results)
        avg_steps = sum(r['steps'] for r in episode_results) / len(episode_results)
        
        # 获取技能统计
        skill_stats = skill_manager.get_analytics()
        
        # 保存技能
        skill_manager.save_skills()
        
        return {
            'name': name,
            'num_episodes': num_episodes,
            'success_rate': total_success / num_episodes,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rates_curve': success_rates,
            'skill_stats': skill_stats,
            'episode_results': episode_results
        }
    
    def _compare_results(self, baseline: Dict, skills: Dict) -> Dict:
        """对比结果"""
        return {
            'success_rate_improvement': skills['success_rate'] - baseline['success_rate'],
            'reward_improvement': skills['avg_reward'] - baseline['avg_reward'],
            'steps_improvement': baseline['avg_steps'] - skills['avg_steps'],  # 步数减少是好的
            'baseline_success_rate': baseline['success_rate'],
            'skills_success_rate': skills['success_rate'],
            'total_skills_learned': skills['skill_stats']['total_skills']
        }
    
    def save_results(self, results: Dict):
        """保存结果"""
        results_file = os.path.join(self.results_dir, "experiment_results.json")
        
        # 简化结果以便保存
        simplified = {}
        for key, value in results.items():
            if key == 'comparison':
                simplified[key] = value
            else:
                simplified[key] = {
                    'name': value.get('name'),
                    'success_rate': value.get('success_rate'),
                    'avg_reward': value.get('avg_reward'),
                    'avg_steps': value.get('avg_steps'),
                    'skill_stats': value.get('skill_stats', {})
                }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {results_file}")


def main():
    """主函数"""
    print("🚀 简化版无监督经验积累实验")
    print("="*60)
    
    # 运行实验
    experiment = SimpleExperiment()
    results = experiment.run_comparison(num_episodes=100)
    
    # 保存结果
    experiment.save_results(results)
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 实验结果摘要")
    print("="*60)
    
    comparison = results['comparison']
    baseline = results['baseline']
    skills = results['with_skills']
    
    print(f"基线成功率: {baseline['success_rate']:.3f}")
    print(f"技能学习成功率: {skills['success_rate']:.3f}")
    print(f"成功率改进: {comparison['success_rate_improvement']:.3f}")
    print(f"平均奖励改进: {comparison['reward_improvement']:.3f}")
    print(f"步数改进: {comparison['steps_improvement']:.3f}")
    print(f"学到的技能数: {comparison['total_skills_learned']}")
    
    # 技能详情
    if skills['skill_stats']['skills_details']:
        print(f"\n🎯 学到的技能:")
        for skill in skills['skill_stats']['skills_details']:
            print(f"  - {skill['name']}: 成功率 {skill['success_rate']:.3f}, 使用 {skill['usage_count']} 次")
    
    # 评估
    print(f"\n🎖️ 实验评估:")
    if comparison['success_rate_improvement'] > 0.05:
        print("  ✅ 经验学习显著提升了成功率!")
    elif comparison['success_rate_improvement'] > 0.01:
        print("  ✴️ 经验学习适度提升了成功率")
    else:
        print("  ❌ 经验学习对成功率改进有限")
    
    if comparison['steps_improvement'] > 1.0:
        print("  ✅ 经验学习显著提升了效率!")
    elif comparison['steps_improvement'] > 0.5:
        print("  ✴️ 经验学习适度提升了效率")
    else:
        print("  ❌ 经验学习对效率改进有限")
    
    print("\n🎉 实验完成!")


if __name__ == "__main__":
    main()