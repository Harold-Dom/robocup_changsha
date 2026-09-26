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
