# 实验日志

## 2026-09-26 · 环境搭建与模板基线

### 环境
- 工作仓库：`P680_robocup_changsha`（clone 自个人 Fork `Harold-Dom/robocup_changsha`）
- `origin` → 个人 Fork；`upstream` → 官方仓库 `HNU-BigHeroX/robocup_changsha`
- 基线代码状态：`upstream/master` = `origin/master` = `5d1c2f5`（规程 v1.2，含 v1.0/v1.1/v1.2 标签）
- 编号分支：本地 `P680`，从 `upstream/master` 建立，已取消上游跟踪；按组织方群通知开发期间不推送
- 初始化提交：`3de95011f9ff491d6d23368a4ee1be4c0f8a834d` — `chore(P680): 初始化参赛目录`
- 评测环境：conda `robocup_p680`，Python 3.12.14，按 `requirements-official.lock` 安装（14 包，`--require-hashes` 校验通过），再 `--no-deps -e .` 安装 `coverage-bench` 0.1.0
- 安装后自检：`pip check` 无冲突；`numpy/gymnasium/pettingzoo/mpe2` 导入正常；
  `get_protocol_spec()` = `protocol_version=coverage-policy/1.0, task_version=coverage-task/1.0, agent_capacity=8, target_capacity=8, action_dim=2, max_episode_steps=256, flatten_version=coverage-flat/1.0`

### 基线运行（未改任何策略代码）
- 预检：`python scripts/check_submission.py --submission participant/P680 --output outputs/P680/check`
  退出码 0；`rejections` 为空；`scan_hits` 2 条，均为 `train.py` 中的 `torch`(191 行) 与 `torch.no_grad`(221 行)，模板训练脚本自带，推理链路不依赖
- 公开测试：`python scripts/evaluate_one.py --submission participant/P680 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P680/eval`
  退出码 0；`status=ok`；8 个回合全部 `ok`、`steps_completed=10`、`accepted=True`、`errors` 为空
- 模型：`artifacts/policy.npz`，size_bytes 92186，sha256 `aedfa4443847600df2fd5e59e0942bff0077849f6d27ea163baf8b4b18d5c728`（模板自带，与清单一致）
- 代码树摘要：`de94c8210b529742c34456ff648dd639668a37670df7acdea87ac69b093cfd4a`

### 基线成绩（local_preview，非组织方核验）
- `performance_score` = 66.67（95% CI 0.00 ~ 133.33）
- basic 组：mean_j 0.0000，coverage 0.0000，collision 0.0000，full_coverage 0.0
- cooperation 组：mean_j 0.1333，coverage 0.1333，collision 0.0000，full_coverage 0.0
- 逐回合：basic-0/basic-1 四个回合 coverage 均为 0.0；coop-0 两个回合 0.0；coop-1 两个回合 0.2667
- 动作耗时：act_ms_max 0.380 ms，远低于 1000 ms 单步限额
- 口径核对：`calculate_performance_score` 按 `1000 × Σ weight × (mean_j − random_anchor)/(learning_anchor − random_anchor)`，
  两组权重各 0.5、锚点 [0,1]，代入得 1000 × 0.5 × 0.1333 = 66.67，与 `result.json` 一致

### 结论与下一步
- 模板基线在 basic 组零覆盖，总分完全由 coop-1 一个场景贡献；提升空间主要在 basic 组与 coop-0
- 基线分数极低（满分 1000），策略需重做而非微调
- 待办：REPORT.md 仍是模板示例内容（提及 P902/P903、`tools/export_learning_baseline.py`、`max|Δa| 8.628e-08`），
  这些不是本人实测记录，正式提交前必须替换为真实方案与数据
- 已知环境限制：Windows 无 `setitimer/SIGALRM/ITIMER_REAL`，评测走 `local_preview` 路径，阶段超时不生效；
  正式成绩以组织方 Linux 容器为准

## 2026-09-27 · 物理量实测验证

探针脚本 `.work/probe/physics_probe.py`（临时分析工具，位于 gitignore 覆盖目录，不进提交、不计入
`code_tree_hash`），解析递推与真实 MPE2 环境双路对照。

### 动力学实测（`configs/task-v1.yaml` + `coverage_bench/envs/physics.py`）
- 参数：`dt=0.1`、`damping=0.25`、`drive_force=1.0`、`robot_mass=1.0`、`horizon=10`、`robot_max_speed=1.0`
- **稳态速度解析解 = `drive*dt/(mass*damping)` = 0.400000**
- **10 步满油门单向总位移 = 0.249010**；逐步累计 0 / 0.010 / 0.0275 / 0.0506 / 0.0780 / 0.1085 /
  0.1414 / 0.1760 / 0.2120 / 0.2490
- 末步速度 0.377475，**全程未触及 `robot_max_speed=1.0`**
- 真实环境对照：basic-0 / basic-1 / coop-0 三场景实测位移均为 0.249010、末速度 0.377475，与解析解一致；
  coop-1 实测 0.233752、末速度 0.000000，与其余三场景不同（未深入定位原因），但不影响策略有效性——
  该场景策略实测 `ok`、无碰撞、无错误
- 关键机制：MPE2 `integrate_state` 为 `pos += v_old*dt`（用**旧**速度），机器人初速为 0，
  故**第 1 步完全不位移**；回合仅 10 步 × 0.1 = **1.0 秒**

### 场地与可达性
- `map_half_extent=1.0` → 全域宽 2.00；机器人活动范围 ±0.95，目标 ±0.85
- 覆盖判据 `dist <= target_radius=0.15`（远大于 `robot_radius=0.05`，判定宽松）；`sense_radius=0.6`
- 目标速度 = `target_max_speed(0.2) × frac`，`frac~U(0.5,1.0)` → 实测 [0.100, 0.200]，10 步位移 [0.100, 0.200]
- **可达判据：`d0 <= 0.249 + 0.15 = 0.399`**（目标静止假设）

各场景机器人到最近目标距离与匹配上界（`n_agents=n_targets=3`）：

| 场景 | 布局 | 三个机器人最近目标距离 | 上界@初始即覆盖(d0≤0.15) | 上界@静止预算(d0≤0.399) | 上界@迎面(d0≤0.599) |
| --- | --- | --- | ---: | ---: | ---: |
| basic-0 (seed 1001) | uniform | 0.551 / 0.732 / 0.300 | 0/3 | 1/3 | 2/3 |
| basic-1 (seed 1002) | uniform | 0.348 / 0.717 / 0.826 | 0/3 | 1/3 | 1/3 |
| coop-0 (seed 2001) | crossing | 0.479 / 0.326 / 0.926 | 0/3 | 1/3 | 1/3 |
| coop-1 (seed 2002) | crossing | 0.215 / **0.062** / 1.438 | 1/3 | 2/3 | 2/3 |

### 对推算的判断：**成立，且需两处修正**
1. "稳态速度约 0.4" ✅ 精确成立（0.400000）
2. "满油门总位移可能只有 0.25" ✅ 精确成立（0.249010）
3. "场地宽 2.0" ✅ 成立，但**该值不是有效约束**——真正的约束是 0.249 位移预算与 0.6 感知半径
4. **修正一（方向相反，是利好）**：可达性并非"几乎无解"。0.399 判据下 8 个机器人有 5 个够得着目标，
   四场景匹配上界合计 5/12；配合目标迎面折抵可再放宽到 0.599。"就近分配、不追远目标"是正确核心，
   但仍有实际得分空间，不是只能躺平
5. **修正二（比"不追远目标"更值钱的一条）**：覆盖率是**逐步**二分图匹配后取均值，不是"曾经到过"。
   coop-1 的 robot1→target1 初始距离 **0.062 < 0.15，第 0 步就已覆盖**，而目标速度 0.183 会把它带出圈。
   因此已在圈内的机器人应切换为**速度跟随**吃满 10 步，而不是冲过去再冲出来。这一条直接决定 coop-1 能否
   从 ~0.2 提到 0.5667

## 2026-09-27 · E002 规则基线（替换模板 NN 推理）

### 实现
- 提交 `1595e27` — `feat(P680): 实现规则策略替换模板神经网络推理`
- `entry.py` 纯 NumPy 重写，接口 `build_policy/reset/act/close` 未变；`act` 返回 `(2,) float32`，
  分量落在 [-1, 1]（冒烟测试 240 次调用，越界/非有限/形状错误 **0 例**）
- 移除模板 `train.py` 与 `artifacts/policy.npz`；`method_type` 改为 `rule`、`train_command` 置 null
- 三段式规则：
  1. **目标分配**——先按位移预算过滤物理上追不到的目标，再在可见目标上做"最近机器人认领"
     （距离相同用 `agent_index` 小者优先）。互相可见的同伴会算出同一个赢家，故不会争抢同一目标；
     追不到的目标直接放弃，避免白跑并在目标点与同伴挤撞
  2. **提前量拦截**——观测不提供目标速度，用相邻两帧绝对位置差分估计并按 `target_max_speed` 截断；
     拦截点 = `target_pos + v_tgt × lead`，`lead = min(到达时间, 回合剩余时间)`。目标最大速度 0.2
     低于机器人稳态 0.4，拦截优于追尾
  3. **覆盖后速度跟随**——进入覆盖圈切换为速度匹配（期望速度 = 目标速度 + `follow_kp × 位置偏差`，
     按 `v_next = (1-damping)*v + (drive/mass)*act*dt` 反解动作），把逐步覆盖率拉满
  4. **同伴避碰**——进入 `safe_dist=0.22`（碰撞判定阈值 `2*robot_radius=0.10`）叠加线性斥力；
     接触力是驱动力 100 倍、碰撞惩罚权重 0.2，宁可绕路不贴身
- 利用 `validate_action` 为**逐分量** [-1,1] 校验（非模长校验）：圈外推进用无穷范数饱和，
  非轴向时两分量可同时顶满，等效驱动力最多 √2 倍，把可达半径抬高最多约 41%；
  跟随模式保留精确幅值不做饱和，以免破坏速度匹配收敛
- 增益外置到 `artifacts/policy_params.json`（`sha256` `2e1d76ef…9d7a`，407 字节）并登记入
  `checkpoint_manifest`：`validate_submission` 要求 `artifacts` 为目录且 git 不跟踪空目录，
  该文件同时满足"正式推理实际读取的全部文件"要求，调参只改配置不改代码

### 结果（`outputs/P680/rule-v2/`，干净工作区重跑）
- 预检：退出码 0；`rejections` 空；**`scan_hits` 空**（`train.py` 移除后 torch 命中消失）；`file_count` 9
- 评测：退出码 0；`status=ok`；8 回合全部 `accepted=True`、`status=ok`、`steps_completed=10`、`errors` 空
- **溯源已修复**：`source_commit=1595e27da5888b300bedc8b6e4be41cd3cbc023e`、`working_tree_dirty=false`、
  `code_tree_hash=29ba36c5…d2dd`（评测时已把 `D:\Program Files\Git\cmd` 临时加入 PATH）
- `act_ms_max` 0.095 ms，远低于 1000 ms 单步限额

| 组 | mean_j | coverage | collision | full_coverage | 匹配上界（静止预算） |
| --- | ---: | ---: | ---: | ---: | --- |
| basic | 0.116667 | 0.116667 | **0.0** | 0.0 | 各场景 1/3 = 0.333 |
| cooperation | 0.316667 | 0.316667 | **0.0** | 0.0 | coop-0 1/3、coop-1 2/3 |

逐回合（每场景 repeat 0/1 结果完全一致，种子固定故可复现）：

| 回合 | mean_j | coverage | collision | collision_pairs |
| --- | ---: | ---: | ---: | ---: |
| basic-0 ×2 | 0.066667 | 0.066667 | 0.0 | 0 |
| basic-1 ×2 | 0.166667 | 0.166667 | 0.0 | 0 |
| coop-0 ×2 | 0.066667 | 0.066667 | 0.0 | 0 |
| coop-1 ×2 | 0.566667 | 0.566667 | 0.0 | 0 |

### 对比
| 版本 | performance_score | basic mean_j | coop mean_j | 碰撞 |
| --- | ---: | ---: | ---: | ---: |
| E001 模板 NN 基线 | 66.67 | 0.000000 | 0.133333 | 0 |
| **E002 规则基线** | **216.67** | 0.116667 | 0.316667 | **0** |
| 官方 rule 基线（规程参考） | 175.00 | — | — | — |

- 相对模板基线 **+150.00 分（+225%）**；**超过官方 rule 基线 175.00 共 +41.67 分**，达成既定目标
- 口径核对：`1000 × 0.5 × (0.116667 + 0.316667) = 216.67`，与 `result.json` 一致
- 95% CI 66.67 ~ 366.67，仍很宽（每组仅 2 个场景 × 2 次重复），**该分数用于本地迭代比较，
  不代表正式排名**；`provenance=local_preview`，正式成绩以组织方 Linux 容器为准

### 已知不足与下一步
- **到达时间偏长是主要失分点**：basic-0 需走 0.15，按位移曲线要到 step 7 才进圈，10 步只吃到 2 步；
  coop-0 同理。basic 组距上界 0.333 仍有 0.217 空间
- 理论上界估算：若 basic 两场景各吃满 1/3、coop 达 (1/3 + 2/3)/2，则总分约
  `1000 × 0.5 × (0.333 + 0.500) ≈ 416`，当前 216.67 约为该上界的 52%
- 可选方向（按预期收益排序）：① 缩短到达时间——第 0 步即满油门且优先轴向对齐以吃满 √2 增益；
  ② 对"迎面目标"放宽 `REACH_BUDGET`（basic-0 上界可从 1/3 提到 2/3）；③ 增益网格搜索
  （已外置到配置文件，调参不需改代码）
- 每次跑评测必须带 git PATH，否则 `source_commit` 为 null、`working_tree_dirty` 误报 true
- 冒烟测试等直接 import `entry` 的脚本必须设 `sys.dont_write_bytecode = True`：
  `working_tree_dirty_status` 用 `git status --porcelain --ignored`，**被忽略文件也算脏**，
  残留 `__pycache__` 会触发 `RECEIPT_WORKING_TREE_DIRTY` 并污染 `code_tree_hash`
- `REPORT.md` 仍是模板示例（P902/P903 对照、`tools/export_learning_baseline.py`、`max|Δa| 8.628e-08`），
  非本人实测内容，按约定待方案定型后重写

## 2026-09-29 · 假设验证（①转向 ②撞墙 ③分配缺陷）

探针 `.work/probe/verify_12.py`、`verify_345.py`（输出 `verify_12.txt`、`v12_out.txt`、`v345_out.txt`）。
结论：**三个假设中，①被推翻，②机制成立但归因错误，③在公开 8 回合中未发生**。

### ① 目标转向：**假设被推翻，回合内确实频繁转向**

代码事实（与"turn_interval_steps 是 [12,20]"的记忆不符）：

- `configs/task-v1.yaml:22-24`：`turn_interval_steps = [5, 10]`
- `coverage_bench/envs/scenario.py:120-121`：`target_next_turn[j] = interval`，`interval ~ U{5..10}`
- `coverage_bench/envs/parallel_env.py:132-134`：调用顺序为
  `advance_robots → advance_targets → step_index += 1`，
  故 `advance_targets` 见到的 `step_index` 取值 **0..9**（不是 1..10）
- `coverage_bench/envs/motion.py:40`：`if state.step_index == state.target_next_turn[j]`

因此 `interval <= 9` 时转向**会**在回合内触发，只有 `interval == 10` 不会，
**触发概率 = 5/6 = 0.8333**，而非"永远不成立"。

实测（零动作推进 10 步，逐目标追踪速度变化；`t` 为第 t 次 step 后，对应 `step_index = t-1`）：

| 场景 | 初始 `target_next_turn` | 实际转向发生的 `step_index` |
| --- | --- | --- |
| basic-0 | [10, 9, 7] | target1@9、target2@7；target0 未转向（interval=10） |
| basic-1 | [7, 8, 7] | 三个目标分别在 7、8、7 转向 |
| coop-0 | [6, 5, 8] | 三个目标分别在 6、5、8 转向 |
| coop-1 | [7, 6, 8] | 三个目标分别在 7、6、8 转向 |

**12 个目标中 11 个在回合内转向**，且转向时刻与 `target_next_turn` 逐一吻合，验证了上述代码路径。
转向时 `motion.py:41-45` 会重抽 `speed_frac ~ U(0.5,1.0)` 与 `angle ~ U(0,2π)`，
即**速度大小与方向都随机改变**，且新方向由 `target_rng` 决定、观测中不可见、不可预测。

推论（两条都否定原计划的代码改动）：

1. `entry.py:207` 的注释"目标会随机转向"是**正确的**，不需要修改；
   `closing_credit=0.5` 的保守取值有其实际依据（迎面折抵在转向后会失效），
   但"0.5 是否最优"仍值得实测，故消融照跑。
2. **"解析求解匀速直线相遇点"不成立**：目标在 `step_index ∈ {5..9}` 会以随机新速度转向，
   而 `target_next_turn` 属内部状态、不在观测中，策略无法预知转向时刻。
   任何假设"整回合匀速直线"的闭式解在转向后必然失效，比现有的
   "逐帧差分 + 短提前量"更脆弱。故**不做该改动**。

### ① 附：差分估速精度（原计划的验证项）

| 类别 | 样本数 | 最大分量误差 | 平均误差 |
| --- | ---: | ---: | ---: |
| **可见目标** | 80 | **5.840e-07** | 1.916e-07 |
| 不可见目标 | 244 | 0.191711 | 0.136540 |

**差分估速在第 1 帧之后对可见目标已是精确值**，5.84e-07 量级即观测 `float32` 的量化舍入，
不是估速方法误差。此前观察到的"恒定大误差 0.129~0.192"全部来自**不可见目标**
（`_estimate_target_velocity` 对不可见目标主动清零，属预期行为，且这些目标本就不参与决策）。
结论：现有差分估速无需替换为解析解。

### ② coop-1 位移 0.233752：**撞墙机制成立，但归因是探针方法而非策略**

- `configs/task-v1.yaml:34`：`crossing_band_fraction = **0.2**`（不是 0.4）。
  `scenario.py:65` `band = 0.2 × target_bound(0.85) = 0.17`，
  故 crossing 布局目标落在 **±[0.68, 0.85]**，不是 ±[0.51, 0.85]
- `physics.py:16-29` `apply_robot_boundary`：限位 `limit = half_extent - radius = 0.95`，
  位置夹回限位并**清零指向界外的速度分量**（`robot_boundary: clamp_outward_velocity`）

对照实验（coop-1, seed=2002，robot0 初始 `x=+0.7162`）：

| 模式 | robot0 10 步位移 | robot0 末速度 | 撞限位事件 |
| --- | ---: | ---: | --- |
| 探针恒定动作 `[1,0]` | **0.233752** | **0.000000** | step 9：`r0.x@+0.950`，该轴速度 `+0.370 → 0.000` |
| 实际策略 | 0.163455 | 0.130278 | **无** |

恒定动作模式下 robot0 沿 +x 一路推进，第 9 步撞到 `x=+0.950` 限位被夹回、该轴速度清零，
位移因此少于自由空间的 0.249010 —— **这精确复现并解释了 0.233752 这个数**。

但这是**我上一轮探针自己用恒定动作 `[1,0]` 造成的人为现象**，与策略无关：
实际策略模式下 8 个回合**零撞限位事件**，三台机器人末速度分别为
0.130278 / 0.114436 / 0.000000（robot2 全程静止，见 ③）。

因此"策略把目标追进墙角"的情形**当前不存在**，按用户设定的条件（"若成立"）不触发该改动。
上一轮 LOG 中"未深入定位原因"一条至此闭环：原因已定位，且属探针方法缺陷，非策略缺陷。
潜在风险仍记录在案：若未来放宽 `reach_budget` 使机器人长距离冲刺，
贴边拦截时可能触发限位，届时应把瞄准点夹到场地内侧。

### ③ 分配缺陷：逻辑缺陷真实存在，但**公开 8 回合中未造成任何损失**

探针逐回合复刻预算过滤与认领结果，检测"存在机器人闲置、且某目标对其可见+可达、
但全局无人认领"的情形：**8 个回合 × 10 步 = 0 次命中**。

原因是**约束瓶颈在 `reach_budget` 而不在分配规则**。以 basic-0 为例，
三台机器人到最近目标的 gap 分别为 0.401 / 0.582 / 0.150，预算 0.32 下
只有 robot2 够得着任何目标；robot0、robot1 闲置是因为**它们物理上追不到**，
不是因为分配规则把可达目标漏给了别人。各场景逐回合认领情况：

| 场景 | 每步 `chosen`（robot0/1/2） | 闲置机器人 | 逐回合 matched |
| --- | --- | --- | --- |
| basic-0 | `[None, None, 0]` ×10 | 0, 1 | 0×8, 1×2 |
| basic-1 | `[0, None, None]` ×10 | 1, 2 | 0×5, 1×5 |
| coop-0 | `[None, 0, None]` ×10 | 0, 2 | 0×8, 1×2 |
| coop-1 | `[2, 1, None]` ×10 | 2 | 1×3, 2×7 |

coop-1 的 robot2 全程静止是正确的：其最近目标 `d0 = 1.438`，
既超出 `sense_radius=0.6`（根本看不见）也远超位移预算，动与不动同分，静止还避免碰撞风险。

结论：改为"各机器人独立求解同一最小代价最大匹配"在**本地公开套件上分数不变**
（可预期完全一致），其价值只体现在组织方隐藏种子上——属稳健性改进而非可测增益。
按用户要求仍实施，但如实标注"本地无可测收益"。

**真正能提分的是 ④ `reach_budget`**：basic-0 在 gap 0.401 处恰好被 0.32 卡掉，
放宽预算可直接把该场景的可达目标数从 1 提到 2，这正是上一轮记录的
"basic 组距上界 0.333 还差 0.217"的成因。
