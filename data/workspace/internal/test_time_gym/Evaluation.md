下面给你一套**可复现的验证方案**，用来回答两个核心问题：

1. 在这个 test-time gym 里，持续获取“经验”（trajectories/skills）是否**带来可量化收益**？
2. 模型是否会**逐渐学会分布外（OOD）的行为**，以及如何**检测/约束**？

---

# 一、评估框架总览

把评估拆成三层：

* **即时效益**（短期）：同一分布、同一任务族内，随着 episode 累积，是否更快/更稳地达成目标。
* **稳健泛化**（中期）：轻微分布漂移（价格波动、排序改变、3DS 概率变化）下是否仍提升。
* **分布外泛化**（长期）：目标约束/流程阶段新增或改变（新增退改签、两段联程、强制优惠券等）时的表现与安全。

---

# 二、核心指标（建议至少跟踪这些）

**任务/策略类**

* **Success@N**：N 条尝试内完成并满足约束的比例。
* **Avg Steps to Success**：成功所需步数（越低越好）。
* **Constraint Violations / ep**：每条轨迹中的约束违规数（预算、起飞时间、停留数等）。
* **Regret**：实际票价 − 同步可得最优票价（或理想最优值）。
* **Reliability（技能级）**：每个 skill 的 Beta(α,β) 成功率后验均值与方差随时间变化。
* **Exploration Ratio**：探索（新 plan/new skill）步数占比；**Exploit Yield**：利用已知技能的成功率。

**学习/记忆类**

* **Learning Curve**：累计回报（或 Success@N）随 episode 的曲线。
* **Sample Efficiency**：达到目标成功率阈值所需 episode 数。
* **Knowledge Reuse Rate**：被重复调用的技能/子流程占比。
* **Forgetting Score**：旧任务族上成功率随时间下降幅度。

**安全/一致性类**

* **Invalid Action Rate**：无效动作比例。
* **Safety Violation Rate**：越过风控阈值的危险操作比例（由 shield 定义）。
* **Policy Drift (Probe-KL)**：在固定探针状态集上的动作分布 KL 漂移。

---

# 三、对照实验（A/B 设计）

> 目标：隔离“经验获取”本身的贡献，避免把改进误判为环境随机性或 prompt 调整带来的效果。

至少做以下几组可控对照（同一环境种子分层抽样，固定评估窗口）：

1. **No-Memory vs Memory**

* A：**禁用**经验库（不写入、不检索、不挖掘技能）。
* B：**启用**经验库 + 技能挖掘 + bandit 选择。
* 观察 Learning Curve、Success@N、Regret 差异；做 **paired t-test / bootstrap CI**。

2. **Shuffled-Memory（破坏检索）**

* 把经验检索改成随机近邻或打乱键索引，只保留“有记忆但不相关”的噪声对照。
* 若 B 优于此组，说明“相关经验”而非“额外计算”产生增益。

3. **Skills Off / Bandit Off**（消融）

* 仅使用原子动作规划；或只保留技能但**不用**Thompson Sampling（用贪心/随机）。
* 对比确定**技能抽象**与**策略选择器**的边际贡献。

4. **HER / 自洽检查 Off**

* 关闭 Hindsight relabel 或 JSON/约束自检，验证**内在奖励**与**自检**对稀疏奖励场景的重要性。

5. **OOD 迁移组**

* 训练：分布 D（基础流程）。评估：分布 D′（约束变化 + 随机扰动上调 + 新子流程）。
* 分三档漂移强度：**轻微**（价格/排序噪声↑）、**中等**（3DS 概率↑、支付超时↑）、**强**（新增退改签环节或必须优惠券）。

> 统计建议：每组 ≥3 个随机种子、每种子 ≥1k episodes；主指标报 **均值±95% bootstrap CI**；显著性用 **paired test**。

---

# 四、学习是否发生：可视化 & 统计证据

**1) 学习曲线**

* `Success@N` / `Avg Steps` 对 episode 的 Loess 平滑曲线：应出现**单调上升/下降**趋势。
* 叠加无记忆组作为基线。

**2) 可靠性后验**

* 每个 skill 的 Beta(α,β) 后验均值与 95% 区间随时间，若经验有效，应看到**均值上升、方差收敛**。

**3) 利用收益**

* 统计“使用已知技能”与“从零规划”两类 episode 的**成功率与 Regret 分布**；**技能路径**应显著更优。

**4) 知识复用谱**

* 画技能调用的 Zipf 分布与重用率热力图；经验有效时，应有一批“高产”技能在多任务间共享。

**5) 反事实回放（Counterfactual Replay）**

* 用记录的环境响应与**离线评估**（WIS/IPS/DR/Doubly-Robust）估计“若不调用某技能/换另一个技能”的回报差异，给出技能的**因果贡献**区间。

---

# 五、分布外（OOD）行为：检测、量化与约束

**A. OOD 检测（状态/上下文）**

* 在状态嵌入空间做 **密度估计**（kNN 距离、Mahalanobis、VAE recon-error）；超过阈值标记 **OOD-state**。
* 对 **目标约束/表单 schema** 做一致性校验（JSON Schema / 规则图灵机）：违背即 OOD-goal/constraint。

**B. OOD 行为（动作/策略）**

* 维护固定探针状态集 (S_\text{probe})：周期性对当前策略的动作分布 ( \pi_t(a|s) ) 与历史基线 ( \pi_0 ) 做 **KL/CUSUM** 漂移检测。
* **Action Shield**：基于规则/学习到的风险模型，限制危险参数空间（如超预算支付、重复提交等）。

**C. OOD 量化指标**

* **OOD Episode Rate**：被判为 OOD 的轨迹比例（按状态密度或 schema 违规）。
* **OOD-Success@N / OOD-Regret**：在 OOD 子集上的成功与后悔值。
* **Shield Intervention Rate**：被拦截/降级执行的比例。
* **Recovery Rate**：从 OOD 退回到安全分布并完成任务的比例。

**D. 失控预警**

* 若 `Probe-KL` 或 `OOD Episode Rate` 超阈值：自动降级为**保守策略**（仅使用高置信技能；或只允许只读/模拟动作），并触发日志告警。

---

# 六、最小统计分析配方

* **主效应验证**：

  * 指标：`Success@N`、`Avg Steps`、`Regret`、`Violations`
  * 方法：**配对 bootstrap（1e4 次）**，报 Δ均值与 95%CI；或 **配对 t 检验**（先做正态性/方差齐性检查）。

* **稳健性**：

  * 分层（seed × 随机扰动强度 × 任务子类），做 **两因素 ANOVA** 或分层 bootstrap。

* **学习动态**：

  * 用 **Mann-Kendall 趋势检验** 看学习曲线是否显著上升/下降。

---

# 七、日志与可复现实验规范（落地）

**记录**（每步/每轨迹）：

* `obs_digest`（哈希特征）、`action`、`reward`、`done`、`constraints_ok`、`skill_id`、`posterior_before/after`、`latency`、`errors`
* `env_seed`、`drift_level`、`coupon/3DS/payment` 随机因子
* 保存为 **JSONL** + 元数据 YAML；版本化技能库（含 Beta 统计与前后置条件）。

**随机性**

* 训练与评测严格分 seed；线上/线下回放一致。

**基线**

* 固定 prompt/planner 版本；只改“是否启用经验/技能/带权选择器”。

---

# 八、如何判定“经验带来收益 & 不致失控”

给一条**可操作的验收标准（示例）**：

1. **同分布**（无漂移）：

   * 相比 No-Memory，Memory 组在 10k episodes 内的 `Success@10` 至少 **+8%（绝对值）**；`Avg Steps` **−15%**；两者 95%CI 不含 0。
2. **轻/中度漂移**：

   * `Success@10` 仍 **+5%/ +3%**，`Violations` **不高于**基线；`OOD Episode Rate` 不上升超过 **+2%**。
3. **强漂移**：

   * `Shield Intervention Rate` 上升，但 **Safety Violations 不上升**；`Recovery Rate` ≥ **40%**；触发探针-KL 告警后已**自动降级**策略。
4. **无失控迹象**：

   * `Probe-KL`、`OOD Episode Rate`、`Invalid Action Rate` 都在阈值内（给出红线），且月度回顾无安全事件。

---

# 九、避免“假学习”的常见坑

* 只调 prompt/温度而把增益误认为“经验作用”。→ 用**消融**固定 prompt 与温度。
* 经验检索泄漏评估答案（数据穿越）。→ 严格 **train/test seeds** 与任务拆分，评测时禁写入。
* 过度依赖单一 reward，导致投机行为。→ 用 **多信号奖励** + 约束罚项 + 事后审计。
* 未做 OOD 监控，出现“越权支付/无限重试”。→ 上 **Action Shield** + **预算/次数**上限 + **探针-KL**。

---

# 十、快速落地清单（Checklist）

* [ ] 建立 No-Memory / Memory / Shuffled / Skills-Off / Bandit-Off / HER-Off 六组对照
* [ ] 统一指标与日志 schema；加 episode 回放
* [ ] 实装 Beta(α,β) 技能可靠性与 Thompson Sampling
* [ ] 状态密度/kNN OOD 检测 + Probe-KL + CUSUM
* [ ] Action Shield：预算、次数、敏感动作白/黑名单
* [ ] 统计脚本：bootstrap CI、趋势检验、分层报表
* [ ] 评估报告模板：曲线、箱线图、技能后验、失效案例库
