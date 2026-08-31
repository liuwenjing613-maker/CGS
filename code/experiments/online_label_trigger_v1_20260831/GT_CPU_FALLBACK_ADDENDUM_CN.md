# GT 生成备选方案预注册补充（2026-08-31）

## 原因

ReplicaSSG 的 Habitat/EGL 渲染器在当前服务器账户下无法访问任何 EGL device。该失败发生在 holdout 结果生成和查看之前。为避免请求提权或重装环境，预注册一个只使用同一 ReplicaSSG `mesh_semantic.ply` 的 CPU 光线投射备选。

## 方法

- 相机轨迹、分辨率、水平视场角、帧范围与原 Habitat 生成器完全一致：`0:2000:5`、1200×680、HFOV=90°。
- 只为 mapper 本帧保留 observation 的 processed-mask 并集像素投射语义光线；并集之外写 0，因为评测不会读取这些位置。
- 每条光线同时返回语义三角形的 `object_id` 和深度；语义 ID 仍来自 ReplicaSSG，不进入 mapper 或触发分数。
- 深度与原 Replica RGB-D 在被评测像素上核对；GT 仅作为离线标签。

## 在 DEV 上冻结的可接受门槛

CPU 结果必须分别在 room0、office0 与已有 Habitat GT 对比，并且每个场景同时满足：

1. processed-mask 并集像素的精确 instance-ID 一致率 ≥ 0.98；
2. 其中 Habitat GT 为已标注实例的像素，精确 instance-ID 一致率 ≥ 0.98；
3. 单帧像素一致率中位数 ≥ 0.98，5% 分位数 ≥ 0.95；
4. observation 级 `mask_mixed` 二值标签一致率 ≥ 0.98，CPU 对 Habitat 阳性的召回率 ≥ 0.95；
5. observation 级严格 `mask_two_foreground` 二值标签一致率 ≥ 0.98，CPU 对 Habitat 阳性的召回率 ≥ 0.95；
6. 各帧有效深度像素中 5 cm 内比例的最小值 ≥ 0.99。

只有两个 DEV 场景全部过门槛，才允许给 room1、office1 生成 holdout 标签。Holdout manifest 必须绑定两个 DEV parity manifest 的 SHA-256。任何一项失败即停止，不查看 holdout 指标，并请求用户决定是否授权修复 EGL 环境。

## 不变项

- 触发分数、候选规则、阈值、DEV/HOLDOUT 划分和 GO/STOP 判据全部不变。
- 不允许根据 CPU/Habitat 差异修改目标定义或阈值。
- CPU GT 只解决离线评测标签生成，不改变在线 mapper，也不构成方法输入。
