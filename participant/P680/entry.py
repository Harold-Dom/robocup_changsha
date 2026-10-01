"""P680 规则策略：可达性过滤的最小代价匹配分配 + 提前量拦截 + 覆盖后速度跟随 + 同伴避碰。

只用 NumPy 推理，不引入 torch 或 requirements-infer.lock 之外的任何依赖。
接口保持 build_policy / reset / act / close 不变，act 返回 (2,) float32 且分量落在 [-1, 1]。

设计依据（实测于本仓库公开套件，详见 LOG.md）：
  * 回合仅 10 步 × dt 0.1 = 1.0 秒，机器人初速为 0，MPE2 积分器 pos += v_old*dt
    使第 1 步完全不位移；满油门单向 10 步总位移只有 0.249，稳态速度 0.4 从未触及
    robot_max_speed=1.0。位移预算而非速度上限才是本任务的硬约束。
  * 覆盖判据 dist <= target_radius=0.15，故可达判据为 d0 <= 0.249+0.15 = 0.399。
    公开套件 8 个机器人中只有 5 个满足，追不到的目标必须放弃。
  * 覆盖率按"逐步"二分图最大匹配统计后取均值，不是"曾经到过"。已在圈内的机器人
    应当跟随目标把得分步数拉满，而不是冲过去再冲出来。
"""
import json

import numpy as np

# 策略增益外置到 artifacts/policy_params.json：调参只改配置不改代码，
# 且 submission.yaml 的 checkpoint_manifest 必须覆盖 artifacts 下全部文件，
# 参数文件因此成为可溯源的提交产物（空 artifacts 目录无法被 git 跟踪）。
_PARAMS_FILENAME = "policy_params.json"

# 缺省增益。读取配置文件失败时回退到此值，保证策略始终可运行。
_DEFAULT_PARAMS = {
    "reach_budget": 0.32,
    "closing_credit": 0.5,
    "lead_gain": 1.0,
    "lead_cap": 1.0,
    "speed_est": 0.25,
    "cover_frac": 1.0,
    "follow_kp": 2.5,
    "safe_dist": 0.22,
    "avoid_gain": 1.8,
}


def _load_params(artifact_dir):
    """从产物目录读取增益参数，缺失或损坏时回退到缺省值。"""
    params = dict(_DEFAULT_PARAMS)
    try:
        raw = json.loads((artifact_dir / _PARAMS_FILENAME).read_text(encoding="utf-8"))
        loaded = raw.get("params", {})
        for key in _DEFAULT_PARAMS:
            if key in loaded:
                value = float(loaded[key])
                if np.isfinite(value):
                    params[key] = value
    except (OSError, ValueError, TypeError):
        pass
    return params


def _inf_saturate(vec):
    """把向量按无穷范数放大到 1，方向不变。

    validate_action 是逐分量 [-1, 1] 校验而非模长校验，因此沿非轴向推进时两个
    分量可以合法地同时顶满，拿到比单位向量更大的驱动力（45° 方向最多 √2 倍）。
    在 0.249 的位移预算下，这直接把可达半径抬高最多约 41%。
    """
    ax = float(vec[0])
    ay = float(vec[1])
    m = abs(ax) if abs(ax) >= abs(ay) else abs(ay)
    if m < 1e-12:
        return np.zeros(2, dtype=np.float64)
    return np.array([ax / m, ay / m], dtype=np.float64)


# 不可行边的代价。远大于任何真实净距离（场地全域宽 2.0），
# 用于在方阵上求"最大匹配优先、其次最小代价"。
_BIG_COST = 1e6

# 同分时按 agent_index 升序打破平局的代价偏置。
# 必须高于观测 float32 量化噪声（约 2e-7），才能盖过舍入造成的伪差异；
# 同时远小于可达目标的净距离量级，不会改变正常的距离排序。
_INDEX_EPS = 1e-6


def _min_cost_assignment(cost):
    """方阵最小代价完美匹配，O(n^3) 匈牙利算法（最短增广路 + 势能）。

    cost 为 (n, n) 数组，不可行边填 _BIG_COST；返回 ans，ans[i] 为行 i 分到的列。
    requirements-infer.lock 只有 NumPy，不能用 scipy.optimize.linear_sum_assignment，
    故自行实现。规模上限为协议的 agent_capacity = target_capacity = 8，
    纯 Python 循环的耗时远低于 act_ms 限额。
    """
    n = int(cost.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    INF = float("inf")
    u = np.zeros(n + 1, dtype=np.float64)   # 行势能
    v = np.zeros(n + 1, dtype=np.float64)   # 列势能
    p = np.zeros(n + 1, dtype=np.int64)     # p[j] = 匹配到列 j 的行号（1-indexed，0 表示空闲）
    way = np.zeros(n + 1, dtype=np.int64)   # 增广路前驱
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = INF
            j1 = -1
            row = cost[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = float(row[j - 1]) - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    ans = np.full(n, -1, dtype=np.int64)
    for j in range(1, n + 1):
        if p[j] != 0:
            ans[int(p[j]) - 1] = j - 1
    return ans


class RuleCoveragePolicy:
    """三段式规则策略。

    分配：每台机器人在"自己可见的机器人 × 自己可见的目标"上独立求解**同一个**
      最小代价最大匹配（自实现匈牙利算法，见 `_min_cost_assignment`），再执行判给
      自己的目标。代价为"净需走距离 = d - target_radius - 迎面折抵"，超出
      `REACH_BUDGET` 或超出 `sense_radius` 的边视为不可行。取代了原先的
      "最近者认领"——后者会让一台对所有目标都不是最近的机器人完全静止，
      从而把可达目标白白漏掉（该缺陷逻辑上存在，实测公开 8 回合中 0 次造成损失，
      本改动属对隐藏种子的稳健性改进，本地分数不变，详见 LOG.md）。
    接近：朝拦截点满油门推进。观测不提供目标速度，用相邻两帧绝对位置差分估计
      （实测对可见目标精度 5.8e-07，即 float32 舍入），再按相遇时间做提前量。
      提前量不再用固定单步速度估算，而是扫描剩余步数预算内最小的可行相遇步数：
      机器人沿自身位移曲线（每步增量 0 / 0.010 / 0.0175 / 0.0231 …，10 步总额
      0.249 轴向、按饱和增益最高 0.352 对角）推进，目标按当前速度匀速直线外推，
      迭代修正"方向→饱和增益→相遇点"直至稳定。实测目标以 5/6 概率在回合内
      随机转向且转向时刻不可观测，故外推只用于估算相遇时间，不追求预判转向。
      预算内无可行相遇时回退到固定 speed_est 估算（E007 行为）。
      目标最大速度 0.2 低于机器人稳态 0.4，拦截优于追尾。
    保持：进入覆盖圈后切换为速度跟随，把机器人速度收敛到目标速度，让逐步覆盖率
      尽可能多地累计得分步。
    避碰：同伴进入安全距离时叠加斥力。接触力是驱动力的 100 倍、碰撞惩罚权重 0.2，
      宁可绕路也不贴身。
    """

    # ---- 可调增益 ----
    # 单一数据源：缺省值见模块级 _DEFAULT_PARAMS，实际取值由
    # artifacts/policy_params.json 覆盖，在 __init__ 中绑定为实例属性。
    # 调参只改配置文件，不改代码；消融记录见 LOG.md。

    # 相遇点定点迭代的次数上限。位移曲线增益只依赖瞄准方向，迭代 3~4 次后
    # 方向变化进入观测噪声量级；超出上限则沿用最后一次可行解。
    _LEAD_ITERS = 4

    def __init__(self, context):
        """context 是 BuildContext（spec / artifact_dir / device / limits / rng）。

        增益参数从 artifact_dir/policy_params.json 读取；物理参数（dt、damping、
        target_radius 等）在 reset() 时由 EpisodeContext.task 下发。
        """
        p = _load_params(context.artifact_dir)
        self.LEAD_GAIN = p["lead_gain"]          # 拦截提前量系数
        self.LEAD_CAP = p["lead_cap"]            # 提前时间上限（秒），等于整个回合长度
        self.SPEED_EST = p["speed_est"]          # 回退用的自身平均速度估算
        self.COVER_FRAC = p["cover_frac"]        # 覆盖圈判定余量（相对 target_radius）
        self.FOLLOW_KP = p["follow_kp"]          # 跟随模式的位置修正增益（1/秒）
        self.REACH_BUDGET = p["reach_budget"]    # 可达性预算：超出即认定追不到，不参与认领
        self.CLOSING_CREDIT = p["closing_credit"]  # 目标迎面靠近时折抵的距离比例（防转向高估）
        self.SAFE_DIST = p["safe_dist"]          # 避碰安全距离，碰撞判定阈值是 2*robot_radius=0.10
        self.AVOID_GAIN = p["avoid_gain"]        # 避碰斥力权重
        self._ready = False

    def reset(self, context):
        """context 是 EpisodeContext，携带本回合的物理参数与规模。

        同时清空跨步状态：目标速度估计依赖上一帧绝对位置，回合边界必须重置，
        否则会把上一回合的末帧位置差分成本回合第 0 步的假速度。
        """
        task = context.task
        self.dt = float(task.dt)
        self.damping = float(task.damping)
        self.drive_force = float(task.drive_force)
        self.robot_mass = float(task.robot_mass)
        self.target_radius = float(task.target_radius)
        self.target_max_speed = float(task.target_max_speed)
        self.sense_radius = float(task.sense_radius)
        # v_next = (1-damping)*v + (drive/mass)*act*dt  =>  act 的反解系数
        self._inv_act_scale = self.robot_mass / (self.drive_force * self.dt)

        self.horizon = int(context.horizon)
        self.num_agents = int(context.num_agents)
        self.num_targets = int(context.num_targets)
        self.agent_index = int(context.agent_index)
        m = self.num_targets
        self._prev_tgt_pos = np.zeros((m, 2), dtype=np.float64)
        self._prev_tgt_vis = np.zeros(m, dtype=bool)
        self._tgt_vel = np.zeros((m, 2), dtype=np.float64)
        self._ready = True

    def close(self):
        """无外部资源需要释放。"""

    # ------------------------------------------------------------------
    # 观测解析
    # ------------------------------------------------------------------
    def _read_targets(self, observation, my_pos):
        """把可见目标的相对位置还原为绝对位置（position_scale=1.0）。"""
        raw = observation["targets"]
        visible = observation["target_visible"]
        exists = observation["target_exists"]
        pos = np.zeros((self.num_targets, 2), dtype=np.float64)
        vis = np.zeros(self.num_targets, dtype=bool)
        for j in range(self.num_targets):
            if bool(exists[j]) and bool(visible[j]):
                pos[j, 0] = my_pos[0] + float(raw[j, 0])
                pos[j, 1] = my_pos[1] + float(raw[j, 1])
                vis[j] = True
        return pos, vis

    def _estimate_target_velocity(self, tgt_pos, tgt_vis):
        """观测不提供目标速度，用相邻两帧绝对位置差分估计。

        只在连续两帧都可见时更新；不可见时清零，避免用跨帧跳变污染提前量。
        估计值按 target_max_speed 截断，抑制可见性切换或数值抖动造成的假高速。
        """
        limit = self.target_max_speed
        for j in range(self.num_targets):
            if tgt_vis[j] and self._prev_tgt_vis[j]:
                delta = (tgt_pos[j] - self._prev_tgt_pos[j]) / self.dt
                speed = float(np.hypot(delta[0], delta[1]))
                if limit > 0.0 and speed > limit:
                    delta = delta * (limit / speed)
                self._tgt_vel[j] = delta
            elif not tgt_vis[j]:
                self._tgt_vel[j] = 0.0
        self._prev_tgt_pos = tgt_pos.copy()
        self._prev_tgt_vis = tgt_vis.copy()

    def _read_peers(self, observation, my_pos):
        """把可见同伴的相对位置还原为绝对位置，跳过自己的槽位。"""
        raw = observation["peers"]
        visible = observation["peer_visible"]
        ids = []
        pos = []
        for i in range(self.num_agents):
            if i == self.agent_index:
                continue
            if bool(visible[i]):
                ids.append(i)
                pos.append(np.array(
                    [my_pos[0] + float(raw[i, 0]), my_pos[1] + float(raw[i, 1])],
                    dtype=np.float64,
                ))
        return ids, pos

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------
    def _assign_target(self, tgt_pos, tgt_vis, peer_ids, peer_pos, my_pos, step):
        """分布式最小代价最大匹配，返回选中的目标索引或 None。

        每台机器人在"自己可见的机器人 × 自己可见的目标"上独立求解**同一个**
        分配问题，再执行判给自己的目标。取代原先的"最近者认领"——后者会让
        一台对所有目标都不是最近的机器人完全静止，把可达目标白白漏掉。

        一致性依据（可见性判定 `d <= sense_radius` 是对称的）：
          * 代价只依赖机器人与目标的**绝对位置**，而绝对位置对所有观察者都是同一真值；
          * 因此视角相同的机器人构造出**完全相同**的代价矩阵，匈牙利算法确定性，
            必然算出同一分配，不会争抢同一目标；
          * 我还能用绝对位置推断**同伴**对某目标的可见性与可达性（`d(peer, target)`），
            即使那个同伴此刻不追它，也不会被我误认为空闲人手。

        视角不完整时的冲突处理（`sense_radius=0.6`，两台都能看到同一目标的机器人
        彼此可能不可见，因为 `2 × 0.6 > 0.6`）：
          1. 代价加 `1e-6 × agent_index` 偏置。该量级高于观测 float32 量化噪声（约 2e-7），
             故净距离在噪声内相等时稳定地判给 index 较小者，且偏置本身也是确定性的，
             视角一致时结论仍完全一致；
          2. 只认领匹配判给**自己**的目标，绝不追判给别人的目标；
          3. 匹配把某台机器人分到填充列或不可行边时，该机器人保持静止——
             不动不得分也不扣分，还避免在目标点与同伴挤撞。
        残余风险：若目标 j 对机器人 r 可见、对我不可见，我无法把 j 纳入矩阵，
        可能误判 r 空闲。这类目标我本来就追不到（`d(self, j) > sense_radius`），
        损失上限是"少一台可用人手"，不会造成两台机器人追同一目标。
        """
        remaining = max((self.horizon - step) * self.dt, 0.0)

        # 行：自己 + 可见同伴，按 agent_index 升序（所有机器人同一规则，保证视角一致时结果一致）
        rob_idx = sorted([self.agent_index] + list(peer_ids))
        rob_pos = {self.agent_index: my_pos}
        for k, p in zip(peer_ids, peer_pos):
            rob_pos[k] = p
        # 列：自己可见的目标，按 target index 升序
        tgt_idx = [j for j in range(self.num_targets) if tgt_vis[j]]
        if not tgt_idx:
            return None

        nr, nc = len(rob_idx), len(tgt_idx)
        k = max(nr, nc)
        cost = np.full((k, k), _BIG_COST, dtype=np.float64)
        for a, r in enumerate(rob_idx):
            rp = rob_pos[r]
            for b, j in enumerate(tgt_idx):
                d_rj = float(np.hypot(tgt_pos[j][0] - rp[0], tgt_pos[j][1] - rp[1]))
                if d_rj > self.sense_radius:
                    continue          # 机器人 r 看不见目标 j，不会去追
                gap = d_rj - self.target_radius
                # 目标迎面靠近能折抵一部分需要走的距离；实测目标会以 5/6 概率
                # 在回合内随机转向（详见 LOG.md），故只按 CLOSING_CREDIT 打折计入。
                credit = 0.0
                if d_rj > 1e-9:
                    ux, uy = (tgt_pos[j][0] - rp[0]) / d_rj, (tgt_pos[j][1] - rp[1]) / d_rj
                    tv = self._tgt_vel[j]
                    closing = -float(tv[0] * ux + tv[1] * uy)
                    if closing > 0.0:
                        credit = self.CLOSING_CREDIT * closing * remaining
                net = gap - credit
                if net > self.REACH_BUDGET:
                    continue          # 超出位移预算，追不到
                cost[a, b] = net + _INDEX_EPS * r

        assign = _min_cost_assignment(cost)
        col = int(assign[rob_idx.index(self.agent_index)])
        if col < 0 or col >= nc or cost[rob_idx.index(self.agent_index), col] >= _BIG_COST:
            return None               # 分到填充列或不可行边：本步无可追目标
        return tgt_idx[col]

    def _displacement_curve(self, gain, v0, steps):
        """满油门沿固定方向推进的标量位移曲线 S[n]（n=1..steps）。

        与 MPE2 积分器逐位对齐：`pos += v_old*dt` 用**旧**速度，故第 1 步位移为
        当前速度 v0 贡献（v0=0 时第 1 步完全不位移）；速度递推
        `v_next = (1-damping)*v + (drive/mass)*act*dt`。`gain` 是无穷范数饱和
        带来的等效幅值增益（轴向 1.0、对角最多 √2，实测位移 0.249→0.352）。
        从静止起算的增量序列 0 / 0.010 / 0.0175 / 0.0231 … 与 LOG.md 实测一致。
        """
        accel = (self.drive_force / self.robot_mass) * self.dt * gain
        decay = 1.0 - self.damping
        s = np.zeros(steps + 1, dtype=np.float64)
        w = v0
        for n in range(1, steps + 1):
            s[n] = s[n - 1] + w * self.dt
            w = decay * w + accel
        return s

    def _encounter_steps(self, rel, tv, my_vel, remaining_steps):
        """在剩余步数预算内扫描最小可行相遇步数 n（含覆盖半径余量）。

        机器人沿 _displacement_curve 推进，目标按**当前速度匀速直线外推**
        `rel + tv*n*dt`，逐 n 检查位移能否盖住"外推后间距 - target_radius"。
        迭代修正"瞄准方向 → 饱和增益 → 相遇步数 → 新瞄准方向"至稳定
        （最多 _LEAD_ITERS 次，纯 NumPy 定点迭代）。实测目标以 5/6 概率在
        回合内随机转向，外推只用于估算相遇时间，不追求预判转向时刻。
        返回相遇步数 n；预算内无可行相遇时返回 None。
        """
        v0 = float(np.hypot(my_vel[0], my_vel[1]))
        aim = rel.copy()
        result = None
        for _ in range(self._LEAD_ITERS):
            eu = float(np.hypot(aim[0], aim[1]))
            if eu < 1e-12:
                return None
            u = aim / eu
            am = max(abs(u[0]), abs(u[1]))   # 单位方向无穷范数 ∈ [1/√2, 1]
            s = self._displacement_curve(1.0 / am, v0, remaining_steps)
            found = None
            for n in range(1, remaining_steps + 1):
                tgt = rel + tv * (n * self.dt)
                need = float(np.hypot(tgt[0], tgt[1])) - self.target_radius
                if s[n] >= need:
                    found = n
                    break
            result = found
            if found is None:
                return None
            new_aim = rel + tv * (found * self.dt)
            nm = float(np.hypot(new_aim[0], new_aim[1]))
            if nm < 1e-12:
                return found
            # 方向变化小于观测噪声量级（实测 float32 舍入约 5.8e-07）即认为收敛
            old_dir = aim / max(abs(aim[0]), abs(aim[1]))
            new_dir = new_aim / max(abs(new_aim[0]), abs(new_aim[1]))
            if float(np.max(np.abs(new_dir - old_dir))) < 1e-6:
                return found
            aim = new_aim
        return result

    def _pursue(self, j, tgt_pos, my_pos, my_vel, step):
        """返回 (基础动作, 是否处于跟随模式)。"""
        tp = tgt_pos[j]
        tv = self._tgt_vel[j]
        rel = tp - my_pos
        d = float(np.hypot(rel[0], rel[1]))

        if d <= self.target_radius * self.COVER_FRAC:
            # 已在覆盖圈内：速度跟随。期望速度 = 目标速度 + 位置回中修正，
            # 再按积分器 v_next = (1-damping)*v + (drive/mass)*act*dt 反解动作。
            desired_vel = tv + self.FOLLOW_KP * rel
            act = (desired_vel - (1.0 - self.damping) * my_vel) * self._inv_act_scale
            return act, True

        # 圈外：求相遇时间。优先用位移曲线 + 匀速外推的迭代解；
        # 预算内无可行相遇（或对角迭代未收敛）时，回退到固定 speed_est 估算。
        remaining = max((self.horizon - step) * self.dt, 0.0)
        remaining_steps = self.horizon - step
        n_meet = None
        if remaining_steps >= 1:
            n_meet = self._encounter_steps(rel, tv, my_vel, remaining_steps)
        if n_meet is not None:
            t_arrive = n_meet * self.dt
        else:
            gap = max(d - self.target_radius, 0.0)
            t_arrive = gap / self.SPEED_EST if self.SPEED_EST > 0.0 else 0.0
        lead = min(self.LEAD_GAIN * t_arrive, self.LEAD_CAP, remaining)
        aim = tp + tv * lead
        direction = aim - my_pos
        dn = float(np.hypot(direction[0], direction[1]))
        if dn < 1e-9:
            return np.zeros(2, dtype=np.float64), False
        return direction / dn, False

    def _avoidance(self, my_pos, peer_pos):
        """同伴进入安全距离时叠加斥力，权重随距离线性增大。"""
        out = np.zeros(2, dtype=np.float64)
        for p in peer_pos:
            delta = my_pos - p
            dist = float(np.hypot(delta[0], delta[1]))
            if 1e-9 < dist < self.SAFE_DIST:
                w = (self.SAFE_DIST - dist) / self.SAFE_DIST
                out[0] += (delta[0] / dist) * (w * self.AVOID_GAIN)
                out[1] += (delta[1] / dist) * (w * self.AVOID_GAIN)
        return out

    # ------------------------------------------------------------------
    # 协议入口
    # ------------------------------------------------------------------
    def act(self, observation):
        self_state = observation["self_state"]
        my_pos = np.array([float(self_state[0]), float(self_state[1])], dtype=np.float64)
        my_vel = np.array([float(self_state[2]), float(self_state[3])], dtype=np.float64)
        step = int(observation["step_index"])

        tgt_pos, tgt_vis = self._read_targets(observation, my_pos)
        self._estimate_target_velocity(tgt_pos, tgt_vis)
        peer_ids, peer_pos = self._read_peers(observation, my_pos)

        chosen = self._assign_target(tgt_pos, tgt_vis, peer_ids, peer_pos, my_pos, step)

        if chosen is None:
            # 看不到可达目标：保持静止。乱走只会增加碰撞风险，不得分也不扣分。
            base = np.zeros(2, dtype=np.float64)
            following = False
        else:
            base, following = self._pursue(chosen, tgt_pos, my_pos, my_vel, step)

        desired = base + self._avoidance(my_pos, peer_pos)
        if following:
            # 跟随模式保留精确幅值，饱和会破坏速度匹配的收敛
            action = np.clip(desired, -1.0, 1.0)
        else:
            action = _inf_saturate(desired)
        return np.ascontiguousarray(action, dtype=np.float32)


def build_policy(context):
    """协议入口：按 EpisodeContext 构建规则策略实例。"""
    return RuleCoveragePolicy(context)
