#!/usr/bin/env python3
"""
更具挑战性的无监督经验积累实验
包含随机失败、复杂约束和动态环境
"""

import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import hashlib


class ChallengingEnvironment:
    """更具挑战性的环境"""
    
    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
        
        self.current_view = 'search_form'
        self.step_count = 0
        self.flights = []
        self.cart_total = 0
        self.budget = 0
        self.max_stops = 0
        self.payment_entered = False
        self.done = False
        self.error_count = 0
        self.payment_failures = 0
        
        # 挑战性设置
        self.flight_availability_rate = 0.7  # 70%概率航班可用
        self.payment_failure_rate = 0.2  # 20%概率支付失败
        self.system_error_rate = 0.1  # 10%概率系统错误
        self.dynamic_pricing = True  # 动态定价
    
    def reset(self) -> Tuple[Dict, Dict]:
        """重置环境"""
        self.current_view = 'search_form'
        self.step_count = 0
        self.flights = []
        self.cart_total = 0
        
        # 随机生成更复杂的约束
        self.budget = random.choice([500, 600, 700, 800, 900, 1000, 1200, 1500])
        self.max_stops = random.choice([0, 1, 2])
        self.preferred_time = random.choice(['morning', 'afternoon', 'evening', 'any'])
        
        self.payment_entered = False
        self.done = False
        self.error_count = 0
        self.payment_failures = 0
        
        obs = self._get_observation()
        info = {'step': self.step_count}
        return obs, info
    
    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步 - 更复杂的逻辑"""
        self.step_count += 1
        reward = -0.02  # 增加时间成本
        
        # 随机系统错误
        if random.random() < self.system_error_rate:
            self.current_view = 'error'
            self.error_count += 1
            return self._get_observation(), -0.1, False, False, {'step': self.step_count, 'error': 'system_error'}
        
        if action == 'search_flights':
            if self.current_view in ['search_form', 'error']:
                self.current_view = 'search_results'
                # 生成更复杂的航班数据
                self.flights = self._generate_complex_flights()
                reward += 0.02
            else:
                reward -= 0.05  # 无效动作惩罚
        
        elif action == 'filter_results':
            if self.current_view == 'search_results' and self.flights:
                # 更智能的过滤
                old_count = len(self.flights)
                self.flights = self._apply_filters()
                new_count = len(self.flights)
                
                if new_count > 0:
                    reward += 0.01 + 0.01 * (old_count - new_count) / old_count  # 过滤效果奖励
                else:
                    reward -= 0.02  # 过滤过严惩罚
            else:
                reward -= 0.05
        
        elif action == 'add_to_cart':
            if self.current_view == 'search_results' and self.flights:
                # 检查航班可用性（动态变化）
                available_flights = [f for f in self.flights if random.random() < self.flight_availability_rate]
                
                if available_flights:
                    # 选择最符合约束的航班
                    best_flight = self._select_best_flight(available_flights)
                    self.cart_total = best_flight['price']
                    
                    # 动态定价
                    if self.dynamic_pricing and random.random() < 0.3:
                        price_change = random.uniform(0.9, 1.1)
                        self.cart_total = int(self.cart_total * price_change)
                    
                    self.current_view = 'cart'
                    reward += 0.05
                    
                    # 约束检查奖励
                    if self.cart_total <= self.budget:
                        reward += 0.03
                    else:
                        reward -= 0.02  # 超预算惩罚
                else:
                    reward -= 0.05  # 航班不可用
            else:
                reward -= 0.05
        
        elif action == 'proceed_to_payment':
            if self.current_view == 'cart' and self.cart_total > 0:
                # 检查预算约束
                if self.cart_total <= self.budget:
                    self.current_view = 'payment'
                    reward += 0.03
                else:
                    reward -= 0.1  # 超预算严重惩罚
            else:
                reward -= 0.05
        
        elif action == 'enter_card':
            if self.current_view == 'payment':
                self.payment_entered = True
                reward += 0.01
            else:
                reward -= 0.05
        
        elif action == 'confirm_payment':
            if self.current_view == 'payment' and self.payment_entered:
                # 支付可能失败
                if random.random() < self.payment_failure_rate:
                    self.payment_failures += 1
                    self.payment_entered = False
                    reward -= 0.1
                    
                    # 多次失败后系统可能锁定
                    if self.payment_failures >= 3:
                        self.current_view = 'error'
                        reward -= 0.2
                else:
                    # 支付成功
                    self.current_view = 'receipt'
                    self.done = True
                    
                    # 最终奖励计算
                    final_reward = self._calculate_final_reward()
                    reward += final_reward
            else:
                reward -= 0.05
        
        elif action == 'restart':
            # 重启到搜索页面
            self.current_view = 'search_form'
            self.cart_total = 0
            self.payment_entered = False
            self.payment_failures = 0
            reward -= 0.05  # 重启惩罚
        
        else:
            reward -= 0.1  # 未知动作惩罚
        
        obs = self._get_observation()
        info = {'step': self.step_count, 'payment_failures': self.payment_failures, 'errors': self.error_count}
        
        # 超过最大步数或过多错误
        truncated = self.step_count >= 25 or self.error_count >= 5
        
        return obs, reward, self.done, truncated, info
    
    def _generate_complex_flights(self) -> List[Dict]:
        """生成复杂的航班数据"""
        flights = []
        num_flights = random.randint(2, 12)  # 更大的变化范围
        
        for i in range(num_flights):
            price = random.randint(300, 1500)
            stops = random.choice([0, 0, 0, 1, 1, 2, 3])  # 更多直飞
            quality = random.uniform(0.3, 1.0)
            time_slot = random.choice(['morning', 'afternoon', 'evening'])
            
            # 价格与质量负相关
            if quality > 0.8:
                price = int(price * random.uniform(1.2, 1.5))
            elif quality < 0.5:
                price = int(price * random.uniform(0.7, 0.9))
            
            flights.append({
                'id': f'FL{i:03d}',
                'price': price,
                'stops': stops,
                'quality': quality,
                'time_slot': time_slot,
                'airline': random.choice(['AA', 'DL', 'UA', 'BA', 'LH'])
            })
        
        return flights
    
    def _apply_filters(self) -> List[Dict]:
        """应用智能过滤"""
        filtered = []
        
        for flight in self.flights:
            # 预算过滤
            if flight['price'] > self.budget:
                continue
            
            # 转机次数过滤
            if flight['stops'] > self.max_stops:
                continue
            
            # 时间偏好过滤
            if self.preferred_time != 'any' and flight['time_slot'] != self.preferred_time:
                # 50%概率仍然保留
                if random.random() < 0.5:
                    continue
            
            filtered.append(flight)
        
        return filtered
    
    def _select_best_flight(self, flights: List[Dict]) -> Dict:
        """选择最优航班"""
        # 综合评分：价格、质量、转机次数
        best_flight = None
        best_score = -1
        
        for flight in flights:
            # 归一化评分
            price_score = max(0, 1 - flight['price'] / self.budget)  # 价格越低越好
            quality_score = flight['quality']  # 质量越高越好
            stops_score = max(0, 1 - flight['stops'] / 3)  # 转机越少越好
            
            # 时间偏好
            time_score = 1.0 if self.preferred_time == 'any' or flight['time_slot'] == self.preferred_time else 0.7
            
            # 综合评分
            total_score = 0.4 * price_score + 0.3 * quality_score + 0.2 * stops_score + 0.1 * time_score
            
            if total_score > best_score:
                best_score = total_score
                best_flight = flight
        
        return best_flight or flights[0]
    
    def _calculate_final_reward(self) -> float:
        """计算最终奖励"""
        reward = 1.0  # 基础完成奖励
        
        # 预算约束奖励
        budget_efficiency = (self.budget - self.cart_total) / self.budget
        reward += 0.5 * max(0, budget_efficiency)  # 节省预算奖励
        
        # 效率奖励（步数越少越好）
        efficiency_bonus = max(0, (25 - self.step_count) / 25) * 0.3
        reward += efficiency_bonus
        
        # 错误惩罚
        reward -= 0.1 * self.error_count
        reward -= 0.05 * self.payment_failures
        
        return max(0, reward)
    
    def _get_observation(self) -> Dict:
        """获取观察"""
        return {
            'view': self.current_view,
            'step': self.step_count,
            'flights': self.flights,
            'cart': {'total': self.cart_total},
            'constraints': {
                'budget': self.budget,
                'max_stops': self.max_stops,
                'preferred_time': self.preferred_time
            },
            'payment_state': {
                'card_entered': self.payment_entered,
                'failures': self.payment_failures
            },
            'system_state': {
                'errors': self.error_count
            },
            'done': self.done
        }


class AdaptiveSkillManager:
    """自适应技能管理器"""
    
    def __init__(self, memory_dir: str = "logs/adaptive_skills"):
        self.skills = {}
        self.episode_count = 0
        self.memory_dir = memory_dir
        self.exploration_rate = 0.4  # 更高的探索率
        self.state_visits = defaultdict(int)
        self.context_memory = deque(maxlen=1000)  # 上下文记忆
        
        os.makedirs(memory_dir, exist_ok=True)
        self._load_skills()
    
    def process_episode(self, trajectory: List[Dict], final_reward: float):
        """处理episode并自适应学习"""
        self.episode_count += 1
        
        # 存储轨迹到上下文记忆
        self.context_memory.append({
            'trajectory': trajectory,
            'reward': final_reward,
            'episode': self.episode_count
        })
        
        # 提取更复杂的技能模式
        actions = [step.get('action', '') for step in trajectory]
        success = final_reward > 0.5
        
        # 错误恢复技能
        if 'restart' in actions:
            restart_pattern = self._extract_restart_pattern(actions)
            if restart_pattern:
                self._update_skill('error_recovery', 'Error Recovery', restart_pattern, success, trajectory)
        
        # 高效搜索技能
        if 'filter_results' in actions and 'add_to_cart' in actions:
            search_pattern = self._extract_search_pattern(actions)
            if search_pattern:
                self._update_skill('efficient_search', 'Efficient Search', search_pattern, success, trajectory)
        
        # 支付处理技能
        payment_actions = ['proceed_to_payment', 'enter_card', 'confirm_payment']
        if all(action in actions for action in payment_actions):
            self._update_skill('payment_handling', 'Payment Handling', payment_actions, success, trajectory)
        
        # 约束优化技能
        if success and self._check_constraint_optimization(trajectory):
            constraint_pattern = self._extract_constraint_pattern(actions)
            if constraint_pattern:
                self._update_skill('constraint_optimization', 'Constraint Optimization', 
                                 constraint_pattern, success, trajectory)
        
        # 自适应探索率
        self._adapt_exploration_rate()
    
    def _extract_restart_pattern(self, actions: List[str]) -> Optional[List[str]]:
        """提取重启模式"""
        restart_idx = actions.index('restart') if 'restart' in actions else -1
        if restart_idx >= 0 and restart_idx < len(actions) - 2:
            return actions[restart_idx:restart_idx + 3]
        return None
    
    def _extract_search_pattern(self, actions: List[str]) -> Optional[List[str]]:
        """提取搜索模式"""
        pattern = []
        if 'search_flights' in actions:
            pattern.append('search_flights')
        if 'filter_results' in actions:
            pattern.append('filter_results')
        if 'add_to_cart' in actions:
            pattern.append('add_to_cart')
        
        return pattern if len(pattern) >= 2 else None
    
    def _extract_constraint_pattern(self, actions: List[str]) -> Optional[List[str]]:
        """提取约束优化模式"""
        # 寻找包含过滤的高效模式
        if 'filter_results' in actions and 'add_to_cart' in actions:
            filter_idx = actions.index('filter_results')
            cart_idx = actions.index('add_to_cart')
            if cart_idx > filter_idx:
                return actions[filter_idx:cart_idx + 1]
        return None
    
    def _check_constraint_optimization(self, trajectory: List[Dict]) -> bool:
        """检查是否进行了约束优化"""
        for step in trajectory:
            obs = step.get('obs', {})
            if obs.get('cart', {}).get('total', 0) > 0:
                budget = obs.get('constraints', {}).get('budget', 0)
                cart_total = obs.get('cart', {}).get('total', 0)
                
                # 如果在预算内且有明显节省
                if cart_total <= budget and (budget - cart_total) / budget > 0.1:
                    return True
        return False
    
    def _update_skill(self, skill_id: str, name: str, actions: List[str], success: bool, trajectory: List[Dict]):
        """更新技能"""
        if skill_id not in self.skills:
            from simplified_experiment import SimpleSkill
            self.skills[skill_id] = SimpleSkill(skill_id, name, actions)
        
        skill = self.skills[skill_id]
        skill.success_rate.update(success)
        skill.usage_count += 1
        
        # 存储上下文信息
        if not hasattr(skill, 'contexts'):
            skill.contexts = []
        
        skill.contexts.append({
            'episode': self.episode_count,
            'success': success,
            'complexity': len(trajectory),
            'final_reward': sum(step.get('reward', 0) for step in trajectory if 'reward' in step)
        })
        
        # 限制上下文历史
        if len(skill.contexts) > 50:
            skill.contexts = skill.contexts[-30:]
    
    def _adapt_exploration_rate(self):
        """自适应调整探索率"""
        # 根据最近的成功率调整探索
        recent_episodes = list(self.context_memory)[-20:]
        if recent_episodes:
            recent_success_rate = sum(1 for ep in recent_episodes if ep['reward'] > 0.5) / len(recent_episodes)
            
            # 成功率低时增加探索
            if recent_success_rate < 0.3:
                self.exploration_rate = min(0.6, self.exploration_rate + 0.05)
            elif recent_success_rate > 0.7:
                self.exploration_rate = max(0.2, self.exploration_rate - 0.05)
    
    def select_skill(self, observation: Dict) -> Optional:
        """选择技能 - 增强版"""
        # 更新状态访问
        state_hash = self._hash_state(observation)
        self.state_visits[state_hash] += 1
        
        # 自适应探索
        visit_count = self.state_visits[state_hash]
        dynamic_exploration = self.exploration_rate * (1.0 / (1.0 + visit_count * 0.1))
        
        if random.random() < dynamic_exploration:
            return None  # 探索
        
        # 上下文感知的技能选择
        applicable_skills = []
        
        for skill in self.skills.values():
            if self._skill_matches_context(skill, observation):
                applicable_skills.append(skill)
        
        if not applicable_skills:
            return None
        
        # 基于历史表现的选择
        best_skill = None
        best_score = -1
        
        for skill in applicable_skills:
            # 基础成功率
            success_score = skill.success_rate.mean()
            
            # 上下文相似性奖励
            context_bonus = self._calculate_context_similarity(skill, observation)
            
            # 最近表现
            recent_bonus = self._calculate_recent_performance(skill)
            
            total_score = 0.5 * success_score + 0.3 * context_bonus + 0.2 * recent_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_skill = skill
        
        return best_skill
    
    def _skill_matches_context(self, skill, observation: Dict) -> bool:
        """检查技能是否匹配当前上下文"""
        view = observation.get('view', '')
        
        if skill.name == 'Error Recovery' and view == 'error':
            return True
        elif skill.name == 'Efficient Search' and view in ['search_form', 'search_results']:
            return True
        elif skill.name == 'Payment Handling' and view in ['cart', 'payment']:
            return True
        elif skill.name == 'Constraint Optimization' and view == 'search_results':
            # 检查是否有复杂约束
            constraints = observation.get('constraints', {})
            return constraints.get('budget', 0) < 1000 or constraints.get('max_stops', 3) < 2
        
        return False
    
    def _calculate_context_similarity(self, skill, observation: Dict) -> float:
        """计算上下文相似性"""
        if not hasattr(skill, 'contexts') or not skill.contexts:
            return 0.5
        
        # 简化的相似性计算
        current_complexity = observation.get('step', 0)
        budget_pressure = observation.get('constraints', {}).get('budget', 1000) / 1000.0
        
        similar_contexts = 0
        for context in skill.contexts[-10:]:  # 最近10次
            context_complexity = context.get('complexity', 0)
            if abs(context_complexity - current_complexity) <= 3:
                similar_contexts += 1
        
        return similar_contexts / min(10, len(skill.contexts))
    
    def _calculate_recent_performance(self, skill) -> float:
        """计算最近表现"""
        if not hasattr(skill, 'contexts') or not skill.contexts:
            return 0.5
        
        recent_contexts = skill.contexts[-5:]  # 最近5次
        if not recent_contexts:
            return 0.5
        
        recent_success = sum(1 for c in recent_contexts if c['success']) / len(recent_contexts)
        return recent_success
    
    def _hash_state(self, state: Dict) -> str:
        """状态哈希"""
        key_features = {
            'view': state.get('view', ''),
            'flights_count': len(state.get('flights', [])),
            'budget_range': state.get('constraints', {}).get('budget', 0) // 200 * 200,  # 范围化
            'has_errors': state.get('system_state', {}).get('errors', 0) > 0
        }
        return hashlib.md5(json.dumps(key_features, sort_keys=True).encode()).hexdigest()[:8]
    
    def get_analytics(self) -> Dict:
        """获取分析数据"""
        if not self.skills:
            return {'total_skills': 0}
        
        # 计算技能多样性
        skill_types = set(skill.name for skill in self.skills.values())
        
        # 计算自适应统计
        recent_exploration_rate = self.exploration_rate
        
        return {
            'total_skills': len(self.skills),
            'skill_types': len(skill_types),
            'episode_count': self.episode_count,
            'avg_success_rate': sum(s.success_rate.mean() for s in self.skills.values()) / len(self.skills),
            'total_usage': sum(s.usage_count for s in self.skills.values()),
            'exploration_rate': recent_exploration_rate,
            'context_memory_size': len(self.context_memory),
            'skills_details': [s.to_dict() for s in self.skills.values()]
        }
    
    def save_skills(self):
        """保存技能"""
        skills_file = os.path.join(self.memory_dir, "adaptive_skills.json")
        data = {
            'episode_count': self.episode_count,
            'exploration_rate': self.exploration_rate,
            'skills': {sid: skill.to_dict() for sid, skill in self.skills.items()}
        }
        
        with open(skills_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_skills(self):
        """加载技能"""
        skills_file = os.path.join(self.memory_dir, "adaptive_skills.json")
        
        if os.path.exists(skills_file):
            try:
                with open(skills_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.episode_count = data.get('episode_count', 0)
                self.exploration_rate = data.get('exploration_rate', 0.4)
                
                for skill_id, skill_data in data.get('skills', {}).items():
                    from simplified_experiment import SimpleSkill
                    skill = SimpleSkill(skill_id, skill_data['name'], skill_data['actions'])
                    skill.usage_count = skill_data['usage_count']
                    skill.success_rate.alpha = max(1, skill_data['success_rate'] * skill_data['usage_count'])
                    skill.success_rate.beta = max(1, (1 - skill_data['success_rate']) * skill_data['usage_count'])
                    self.skills[skill_id] = skill
                
                print(f"加载了 {len(self.skills)} 个自适应技能")
            except Exception as e:
                print(f"加载技能失败: {e}")


def run_challenging_experiment(num_episodes: int = 150):
    """运行挑战性实验"""
    print("🚀 挑战性无监督经验积累实验")
    print("="*60)
    
    results_dir = "logs/challenging_experiment"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}
    
    # 基线智能体
    print("运行基线实验...")
    baseline_results = run_single_challenging_experiment(
        "baseline", num_episodes, use_adaptive_skills=False, results_dir=results_dir
    )
    results['baseline'] = baseline_results
    
    # 自适应技能学习智能体
    print("\n运行自适应技能学习实验...")
    adaptive_results = run_single_challenging_experiment(
        "adaptive_skills", num_episodes, use_adaptive_skills=True, results_dir=results_dir
    )
    results['adaptive_skills'] = adaptive_results
    
    # 保存结果
    results_file = os.path.join(results_dir, "challenging_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': {k: v for k, v in baseline_results.items() if k != 'episode_results'},
            'adaptive_skills': {k: v for k, v in adaptive_results.items() if k != 'episode_results'},
            'comparison': compare_challenging_results(baseline_results, adaptive_results)
        }, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print_challenging_results(baseline_results, adaptive_results)
    
    return results


def run_single_challenging_experiment(name: str, num_episodes: int, use_adaptive_skills: bool, results_dir: str) -> Dict:
    """运行单个挑战性实验"""
    
    class ChallengingAgent:
        def __init__(self, skill_manager, use_skills: bool):
            self.skill_manager = skill_manager
            self.use_skills = use_skills
            self.current_trajectory = []
            
        def select_action(self, observation: Dict) -> str:
            self.current_trajectory.append({'obs': observation, 'action': None})
            
            if self.use_skills:
                selected_skill = self.skill_manager.select_skill(observation)
                if selected_skill and selected_skill.actions:
                    action = selected_skill.actions[0]
                    print(f"    使用技能 '{selected_skill.name}': {action}")
                    self.current_trajectory[-1]['action'] = action
                    self.current_trajectory[-1]['skill_used'] = selected_skill.name
                    return action
            
            action = self._complex_rule_based_action(observation)
            self.current_trajectory[-1]['action'] = action
            return action
        
        def _complex_rule_based_action(self, observation: Dict) -> str:
            view = observation.get('view', '')
            
            if view == 'search_form':
                return 'search_flights'
            elif view == 'search_results':
                flights = observation.get('flights', [])
                constraints = observation.get('constraints', {})
                
                if not flights:
                    return 'search_flights'
                
                # 检查是否需要过滤
                over_budget = any(f['price'] > constraints.get('budget', 0) for f in flights)
                over_stops = any(f['stops'] > constraints.get('max_stops', 3) for f in flights)
                
                if over_budget or over_stops:
                    return 'filter_results'
                else:
                    return 'add_to_cart'
            elif view == 'cart':
                return 'proceed_to_payment'
            elif view == 'payment':
                payment_state = observation.get('payment_state', {})
                if not payment_state.get('card_entered', False):
                    return 'enter_card'
                else:
                    return 'confirm_payment'
            elif view == 'error':
                return 'restart'
            else:
                return 'search_flights'
        
        def end_episode(self, final_reward: float):
            if self.current_trajectory:
                self.skill_manager.process_episode(self.current_trajectory, final_reward)
            self.current_trajectory = []
    
    env = ChallengingEnvironment(seed=42)
    
    if use_adaptive_skills:
        skill_manager = AdaptiveSkillManager(f"{results_dir}/skills_{name}")
    else:
        from simplified_experiment import SimpleSkillManager
        skill_manager = SimpleSkillManager(f"{results_dir}/skills_{name}")
    
    agent = ChallengingAgent(skill_manager, use_adaptive_skills)
    
    episode_results = []
    success_rates = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        
        for step in range(25):
            action = agent.select_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            
            if done or trunc:
                break
        
        success = obs.get('view') == 'receipt'
        episode_results.append({
            'episode': episode,
            'total_reward': total_reward,
            'success': success,
            'steps': info['step'],
            'payment_failures': info.get('payment_failures', 0),
            'errors': info.get('errors', 0)
        })
        
        agent.end_episode(total_reward)
        
        # 滑动平均
        if episode >= 19:
            recent_success = sum(1 for r in episode_results[-20:] if r['success']) / 20
            success_rates.append(recent_success)
        
        if (episode + 1) % 30 == 0:
            recent_success_rate = sum(1 for r in episode_results[-30:] if r['success']) / 30
            print(f"  Episode {episode + 1}: 最近30次成功率 {recent_success_rate:.3f}")
    
    # 统计
    total_success = sum(1 for r in episode_results if r['success'])
    avg_reward = sum(r['total_reward'] for r in episode_results) / len(episode_results)
    avg_steps = sum(r['steps'] for r in episode_results) / len(episode_results)
    avg_payment_failures = sum(r['payment_failures'] for r in episode_results) / len(episode_results)
    avg_errors = sum(r['errors'] for r in episode_results) / len(episode_results)
    
    skill_stats = skill_manager.get_analytics()
    skill_manager.save_skills()
    
    return {
        'name': name,
        'num_episodes': num_episodes,
        'success_rate': total_success / num_episodes,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_payment_failures': avg_payment_failures,
        'avg_errors': avg_errors,
        'success_rates_curve': success_rates,
        'skill_stats': skill_stats,
        'episode_results': episode_results
    }


def compare_challenging_results(baseline: Dict, adaptive: Dict) -> Dict:
    """对比挑战性实验结果"""
    return {
        'success_rate_improvement': adaptive['success_rate'] - baseline['success_rate'],
        'reward_improvement': adaptive['avg_reward'] - baseline['avg_reward'],
        'steps_improvement': baseline['avg_steps'] - adaptive['avg_steps'],
        'payment_failures_improvement': baseline['avg_payment_failures'] - adaptive['avg_payment_failures'],
        'errors_improvement': baseline['avg_errors'] - adaptive['avg_errors'],
        'baseline_stats': {
            'success_rate': baseline['success_rate'],
            'avg_reward': baseline['avg_reward'],
            'avg_steps': baseline['avg_steps']
        },
        'adaptive_stats': {
            'success_rate': adaptive['success_rate'],
            'avg_reward': adaptive['avg_reward'],
            'avg_steps': adaptive['avg_steps']
        },
        'skills_learned': adaptive['skill_stats']['total_skills']
    }


def print_challenging_results(baseline: Dict, adaptive: Dict):
    """打印挑战性实验结果"""
    comparison = compare_challenging_results(baseline, adaptive)
    
    print("\n" + "="*60)
    print("📊 挑战性实验结果摘要")
    print("="*60)
    
    print(f"基线成功率: {baseline['success_rate']:.3f}")
    print(f"自适应技能成功率: {adaptive['success_rate']:.3f}")
    print(f"成功率改进: {comparison['success_rate_improvement']:.3f}")
    print(f"平均奖励改进: {comparison['reward_improvement']:.3f}")
    print(f"步数改进: {comparison['steps_improvement']:.3f}")
    print(f"支付失败减少: {comparison['payment_failures_improvement']:.3f}")
    print(f"系统错误减少: {comparison['errors_improvement']:.3f}")
    print(f"学到的技能数: {comparison['skills_learned']}")
    
    # 自适应技能详情
    if adaptive['skill_stats'].get('skills_details'):
        print(f"\n🎯 学到的自适应技能:")
        for skill in adaptive['skill_stats']['skills_details']:
            print(f"  - {skill['name']}: 成功率 {skill['success_rate']:.3f}, 使用 {skill['usage_count']} 次")
    
    # 评估
    print(f"\n🎖️ 挑战性实验评估:")
    if comparison['success_rate_improvement'] > 0.1:
        print("  ✅ 自适应经验学习显著提升了成功率!")
    elif comparison['success_rate_improvement'] > 0.05:
        print("  ✴️ 自适应经验学习适度提升了成功率")
    else:
        print("  ❌ 自适应经验学习对成功率改进有限")
    
    if comparison['reward_improvement'] > 0.1:
        print("  ✅ 自适应经验学习显著提升了整体表现!")
    elif comparison['reward_improvement'] > 0.05:
        print("  ✴️ 自适应经验学习适度提升了整体表现")
    else:
        print("  ❌ 自适应经验学习对整体表现改进有限")


if __name__ == "__main__":
    run_challenging_experiment(num_episodes=200)