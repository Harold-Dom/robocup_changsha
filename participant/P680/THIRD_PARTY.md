# 第三方依赖与许可声明

本人提交为纯规则策略，推理仅依赖 NumPy（由官方评测镜像提供，见 `requirements-infer.lock`），
不含任何模型产物、权重文件或序列化载荷。

未引用任何第三方源码或预训练模型：策略逻辑（目标认领、拦截提前量、速度跟随、避碰斥力）
均为本人依据仓库公开的物理参数与评分口径自行推导实现，无外部代码改写或移植。

已移除参赛模板自带的 `train.py` 与 `artifacts/policy.npz`（SB3 PPO 示例产物），
因此 stable-baselines3、supersuit、torch 等训练侧依赖与本提交无关。
