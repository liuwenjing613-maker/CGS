# CloudCompare instance-observation 可视化交付（2026-09-01）

## 目的

把 online label trigger 两个 holdout 场景的最终在线 object map 导出为 CloudCompare 可交互实体，使 3D instance、其吸收的 observation 标签及次数能同时核查。

## 数据与时间协议

- 场景：Replica `room1`、`office1`。
- 输入：本次严格在线建图保存的最终 PCD map；未离线重建 object，未使用未来帧修改实例。
- 帧协议：raw frame `0..1995`，stride=5，共 400 个处理帧。
- observation 标签直接来自每个 object 保存的 `class_id` 与 `image_idx`，不使用最终 object 标签回填。

## 导出设计

- 每个 instance 单独导出一个 binary little-endian PLY，CloudCompare DB Tree 中一行对应一个 instance。
- 实体名格式：`FG/BG_InstanceID__stable-label__obs-总数__前三标签-次数`。
- 完整统计写入 `instance_summary.csv`、`instance_label_counts.csv` 和 `metadata.json`。
- 每个点保存 RGB 以及 5 个标量字段：`instance_index`、`observation_count`、`distinct_label_count`、`dominant_label_fraction`、`is_background`。
- 前景与背景目录分开；另提供 all / foreground / background 三个合并 PLY。

## 规模

| 场景 | instance | 前景/背景 | observation | 点数 | instance-label 行数 |
|---|---:|---:|---:|---:|---:|
| room1 | 54 | 50 / 4 | 4,473 | 493,330 | 130 |
| office1 | 25 | 25 / 0 | 3,272 | 205,879 | 128 |

## 严格校验

- 每个 instance 验证 `len(obs_uids) == 所有 observation label count 之和`；不一致即终止导出。
- 服务器 ZIP 完整性检查通过；本地与服务器 ZIP SHA-256 一致：`20dd56f0cdf5fb87ff946f8875921fc39ad9a9ee3fe97fdd1056bdff406dc823`。
- CloudCompare 2.13.2 命令行实际逐个载入全部 79 个 instance PLY 成功。
- 本地生成并回读验证：room1 54-entity BIN、room1 50-foreground-entity BIN、office1 25-entity BIN。

## 产物

- 服务器目录：`results/experiments/online_label_trigger_v1_20260831/cloudcompare_instance_observations/`
- 服务器压缩包：`results/experiments/online_label_trigger_v1_20260831/cloudcompare_instance_observations.zip`
- 本地短路径：`cc_obs_20260901/cloudcompare_instance_observations/`
- 导出器：`code/experiments/online_label_trigger_v1_20260831/export_cloudcompare_instances.py`

本地必须使用短路径。CloudCompare 在 Windows 上打开深层 artifacts 路径时可能触发路径长度限制；同一文件移到上述短路径后，全部实体均成功载入。

## 解释边界

- observation 数是最终 object 吸收的历史 observation 数，不是点数。
- 文件名只显示前三标签；CSV/JSON 保留全部标签。
- 颜色只区分 instance，不表示语义或错误类别。
- 该可视化用于定位和解释异常，不替代 trigger 的定量评测。
