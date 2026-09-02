# GT instance 逐帧参考查看器（2026-09-02）

## 目的

为 CloudCompare 中难以判断的 split/merge 提供逐帧 2D 参考。查看器并排显示：原始 RGB、完整 ReplicaSSG GT instance、ali-dev observation 按最终 3D owner 合并后的 2D mask。

## 数据与协议

- 场景：room1、office1。
- 每场景 400 个处理帧，对应 raw frame `0,5,...,1995`。
- 分辨率：1200×680；三联预览图为 1800×420。
- GT 来自冻结在线建图完成后的 ReplicaSSG 全帧 instance 渲染，仅用于事后参考，未进入 mapper 或 trigger。
- 精确归属检查：room1 4,473 条、office1 3,272 条 observation 全部唯一映射到最终 instance；缺失和重复 owner 均为 0。

## GT 对齐质量

| 场景 | 逐帧 5 cm 内深度一致率最低值 | 最大逐帧中位绝对误差 |
|---|---:|---:|
| room1 | 99.6305% | 2.54 mm |
| office1 | 99.7027% | 1.61 mm |

## 产物

- `gt_reference_viewer/index.html`：一次只加载当前图片的轻量查看器。
- 每场景 `frames/`：400 张 RGB/GT/owner 三联图。
- 每场景 `gt_instance_id_png16/`：400 张精确 16-bit GT instance ID PNG。
- `instance_to_gt_summary.csv`：每个 Ixx 的纯 observation→GT 分布、mixed-mask 比例和疑似 merge 标记。
- `potential_false_split_pairs.csv`：两个最终 instance 共享同一主 GT 的候选对。

## 自动候选规模

| 场景 | 疑似 false merge instance | 疑似 false split pair |
|---|---:|---:|
| room1 | 4 | 10 |
| office1 | 0 | 2 |

这些数字是审阅队列，不是最终错误结论。特别是 room1 中存在较多前端标签与 GT/RGB 明显不一致的 observation，必须看多帧轮廓和物理边界。

## 人工判断规则

- 一个 Ixx 在多个清晰视角反复覆盖两个不同前景 GT ID：支持 false merge。
- 两个 Ixx 在多个视角稳定对应同一个前景 GT ID：支持 false split。
- 单帧边界错位、遮挡、GT 背景 ID、mixed mask 或仅标签字符串不同：不足以决定拆分/合并，应标记 DEFER。
- GT 仍可能有 ReplicaSSG 标注粒度问题，例如 blanket/comforter 等可被标成独立实例；最终动作需同时符合 RGB 中的物理连续性。

## 校验

- 800/800 三联图和 800/800 GT PNG 生成完成。
- ZIP 完整性检查通过；SHA-256：`4be708e83ffdac8e72c96505ab098f51ae25341d3885c393bea96a6bdfb31e78`。
- 本地浏览器验证场景切换、滑条、方向键、帧号和图片加载均正常。
