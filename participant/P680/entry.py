"""P680 规则策略：可达性过滤的目标认领 + 提前量拦截 + 覆盖后速度跟随 + 同伴避碰。

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


class RuleCoveragePolicy:
    """三段式规则策略。

    分配：先按位移预算过滤掉物理上追不到的目标，再在剩下的可见目标上做
      "最近机器人认领"，距离相同用 agent_index 小者优先。互相可见的同伴会算出
      同一个赢家，因此不会争抢同一目标；追不到的目标直接放弃，避免白跑并在
      目标点与同伴挤撞。
    接近：朝拦截点满油门推进。观测不提供目标速度，用相邻两帧绝对位置差分估计，
      再按到达时间做提前量。目标最大速度 0.2 低于机器人稳态 0.4，拦截优于追尾。
    保持：进入覆盖圈后切换为速度跟随，把机器人速度收敛到目标速度，让逐步覆盖率
      尽可能多地累计得分步。
    避碰：同伴进入安全距离时叠加斥力。接触力是驱动力的 100 倍、碰撞惩罚权重 0.2，
      宁可绕路也不贴身。
    """

    # ---- 可调增益 ----
    # 单一数据源：缺省值见模块级 _DEFAULT_PARAMS，实际取值由
    # artifacts/policy_params.json 覆盖，在 __init__ 中绑定为实例属性。
    # 调参只改配置文件，不改代码；消融记录见 LOG.md。

    def __init__(self, context):
        """context 是 BuildContext（spec / artifact_dir / device / limits / rng）。

        增益参数从 artifact_dir/policy_params.json 读取；物理参数（dt、damping、
        target_radius 等）在 reset() 时由 EpisodeContext.task 下发。
        """
        p = _load_params(context.artifact_dir)
        self.LEAD_GAIN = p["lead_gain"]          # 拦截提前量系数
        self.LEAD_CAP = p["lead_cap"]            # 提前时间上限（秒），等于整个回合长度
        self.SPEED_EST = p["speed_est"]          # 估计的自身平均速度，用于推算到达时间
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
        """在可达的可见目标上做最近者认领，返回选中的目标索引或 None。"""
        robot_ids = [self.agent_index] + list(peer_ids)
        robot_pos = [my_pos] + list(peer_pos)
        remaining = max((self.horizon - step) * self.dt, 0.0)

        chosen = None
        chosen_gap = float("inf")
        for j in range(self.num_targets):
            if not tgt_vis[j]:
                continue
            rel = tgt_pos[j] - my_pos
            d = float(np.hypot(rel[0], rel[1]))

            # 可达性过滤：目标迎面靠近能折抵一部分需要走的距离，
            # 但目标会随机转向，只按 CLOSING_CREDIT 打折计入。
            gap = d - self.target_radius
            credit = 0.0
            if d > 1e-9:
                ux, uy = rel[0] / d, rel[1] / d
                tv = self._tgt_vel[j]
                closing = -float(tv[0] * ux + tv[1] * uy)
                if closing > 0.0:
                    credit = self.CLOSING_CREDIT * closing * remaining
            if gap - credit > self.REACH_BUDGET:
                continue

            # 最近者认领，距离相同用 agent_index 小者优先，保证结论确定且一致
            winner = self.agent_index
            winner_d = d
            for k in range(1, len(robot_ids)):
                dk = float(np.hypot(tgt_pos[j, 0] - robot_pos[k][0],
                                    tgt_pos[j, 1] - robot_pos[k][1]))
                if dk < winner_d - 1e-9 or (abs(dk - winner_d) <= 1e-9 and robot_ids[k] < winner):
                    winner = robot_ids[k]
                    winner_d = dk
            if winner == self.agent_index and gap < chosen_gap:
                chosen_gap = gap
                chosen = j
        return chosen

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

        # 圈外：朝拦截点推进，提前时间取"到达时间"与剩余时间的较小值
        gap = max(d - self.target_radius, 0.0)
        t_arrive = gap / self.SPEED_EST if self.SPEED_EST > 0.0 else 0.0
        remaining = max((self.horizon - step) * self.dt, 0.0)
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
