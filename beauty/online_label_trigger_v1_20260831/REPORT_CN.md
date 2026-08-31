# 在线 observation-label consistency trigger V1：严格在线验证报告

日期：2026-08-31
正式判决：**MODIFY（保留为候选排序信号，不作为独立修复触发器）**

## 1. 最重要结论

本轮把 V0 中看起来较强的“object 内 observation 标签不一致”从冻结地图离线相关性，升级成了完整的因果在线实验：从 `frame=0` 空图开始重跑 mapper，每个时刻只读取当时已经进入当前 object 的 observation 标签历史；规则在 DEV 场景冻结后，原样应用到两个未见 HOLDOUT 场景。

最终结果是：

1. **信号真实存在，但场景稳定性不足。** HOLDOUT 的 office1 排序很强（AUROC 0.900，AP lift +0.183），room1 只有弱正向（AUROC 0.569，AP lift +0.057）。
2. **冻结阈值的 pooled endpoint precision/recall 尚可**：precision 0.862、recall 0.595；但 room1 的最终阴性触发率达到 0.667，不能安全地直接创建修复 ticket。
3. **首次触发时的真实在线 precision 只有 0.621。** endpoint precision 会把“现在还没错、以后才变成阳性”的对象算作正确，明显比真实首次触发更乐观。
4. **漏检具有明确机制。** 很多 mask 即使长期混入多个实例，检测器仍始终给出同一个 observation 类别；这时 label-conflict 分数接近 0，标签统计在信息论上无法看见错误。
5. 因此，这个方向**具备后续研究价值，但贡献范围必须收窄**：适合作为便宜的候选排序器或语义冲突分支，不适合作为兼顾 mask 与 association 错误的统一独立 trigger。

## 2. 实际实现是什么

### 2.1 在线生产侧只保留简单标签历史

每处理完一帧（`FRAME_CLOSED`），对当前 active object 读取：

- 当前成员 observation 的 `class_id`；
- 每个 observation 已发生的 frame；
- observation 数量。

生产 trigger 不读取：

- GT 或 Replica 实例图；
- 当前/历史 mask 像素；
- 深度、点云、体素；
- 最终 object membership；
- 未来帧；
- VLM。

因此它可以真实在线运行，不会泄露未来信息。

### 2.2 冻结分数 `repeated_alt_fraction`

设当前 object 有 `N` 个 observation：

1. `N < 5` 时不触发；
2. 找出当前出现次数最多的主标签；
3. 某个非主标签必须至少出现于 2 个 observation，且跨至少 2 个不同帧，才算“持续非主标签”；
4. 分数为：

```text
repeated_alt_fraction
= 持续非主标签 observation 数 / 当前 observation 总数
```

冻结规则：

```text
minimum observations = 5
score = repeated_alt_fraction
threshold = 0.25
```

直观例子：一个 object 当前有 20 次 observation，其中 14 次是 `chair`，6 次是跨多帧重复出现的 `table`，分数就是 `6/20=0.30`，超过 0.25，触发 review。

规则内部摘要：`19f275a58881f53c10b769fba75d8bf4380f670dfa220a7706259915f5d0f2fb`。冻结规则文件 SHA256：`0a76d13a4d23fa773eee9efad6a65c71c25a9aa8913cd6a76891db6be8b25d13`。

## 3. GT 如何独立评估

GT 只存在于离线 evaluator，绝不进入 mapper 或 trigger。

每个被 mapper 接受的 observation processed mask，与同帧官方 ReplicaSSG instance map 做像素相交：

- 单 observation 若 `top purity < 0.8`，或第二实例占比 `>= 0.1`，且主实例至少 25 像素，则记为 mixed mask；
- 当前 object 内 mixed observation 至少 2 个，且占可评 observation 至少 5%，则当前时刻为主目标 `repeated_mixed` 阳性；
- 严格目标 `repeated_two_foreground` 额外要求前两名是不同前景实例，排除 wall/floor/ceiling/unknown/undefined。

时间定义：

- `s_confirmable`：因果前缀第一次真正满足 repeated GT 的处理帧；
- `d_first`：冻结规则第一次触发的处理帧；
- `d_post`：不早于 `s_confirmable` 的第一次触发；
- 同时评估早触发、确认后召回和 `d_post-s_confirmable` 时延。

## 4. 严格实验协议

| 项目 | 设置 |
|---|---|
| DEV | room0、office0；只在这里选择候选分数和阈值 |
| HOLDOUT | room1、office1；冻结后一次性运行，不调阈值 |
| 在线范围 | 每场景 `start=0, end=2000, stride=5`，精确 400 个处理帧 |
| 初始状态 | 从空图开始，不读取已经建好的最终地图 |
| 候选分数 | 5 个固定简单分数；固定阈值网格 |
| Bootstrap | 排序指标报告 95% CI |
| HOLDOUT 后调参 | **未进行** |

DEV 没有任何规则同时达到“每场景 precision >= 0.75、每场景阴性触发率 <= 0.25、pooled recall >= 0.40”。因此按预注册协议冻结描述性最佳规则，并明确记录 `DEV gate failed`，没有把失败隐藏掉。

HOLDOUT 原有 legacy detection cache 存在文件名、schema 和类别表不兼容。正式 GT 指标生成前已写执行附录，放弃转换旧缓存，按冻结 source config 的 `force_detection=true` 从 frame=0 因果生成检测。该变更没有读取 HOLDOUT GT，也没有改变分数或阈值。

## 5. 数据与完整性

| split | scene | 处理帧 | GT 帧 | 保留 observation | final active object | 缺失最终成员 | GT 方法 | 深度 5 cm 内比例 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| DEV | room0 | 400/400 | 400 | 7,507 | 72 | 0 | Habitat EGL | 0.9944 |
| DEV | office0 | 400/400 | 400 | 3,106 | 35 | 0 | Habitat EGL | 0.9953 |
| HOLDOUT | room1 | 400/400 | 400 | 4,473 | 54 | 0 | Habitat EGL | 0.9963 |
| HOLDOUT | office1 | 400/400 | 400 | 3,272 | 25 | 0 | Habitat EGL | 0.9970 |

四个场景的完整性 gate 全部通过：帧序列精确、mapper 完成、GT 未进入 mapper manifest、processed mask hash 无失败、最终 membership 闭环无缺失。

曾尝试 CPU sparse GT fallback；几何深度可对齐，但 semantic ID 未通过官方 GT parity，因此该路线被明确废弃，**没有用于正式指标**。最终 HOLDOUT 使用官方 Habitat EGL 路线；room0 frame0 与既有官方 semantic GT 像素级完全一致。

HOLDOUT 在线建图主要耗时：room1 895.8 秒（约 14 分 56 秒），office1 747.0 秒（约 12 分 27 秒）。两场景检测均在线生成并同时保存审计证据。

## 6. DEV：规则如何冻结

主目标为 `repeated_mixed`。这里的 AP 必须和错误基准率一起看；因为 office0 阳性比例高达 0.913，单看 AP=0.977 会非常乐观。

| scene | 可评 object | 阳性 | 基准率 | AUROC（95% CI） | AP | AP lift（95% CI） |
|---|---:|---:|---:|---:|---:|---:|
| room0 | 41 | 26 | 0.634 | 0.618（0.430–0.784） | 0.735 | +0.101（-0.020–0.253） |
| office0 | 23 | 21 | 0.913 | 0.798（0.545–0.976） | 0.977 | +0.064（0.020–0.166） |

冻结阈值 0.25 的 DEV 操作指标：

| scene | endpoint precision | endpoint recall | 首次触发当前 precision | 阴性触发率 | post median / P90 |
|---|---:|---:|---:|---:|---:|
| room0 | 0.739 | 0.654 | 0.609 | 0.400 | 2 / 108.8 帧 |
| office0 | 0.944 | 0.810 | 0.667 | 0.500 | 0 / 8.4 帧 |
| pooled | 0.829 | 0.723 | 0.634 | 0.412 | 0 / 70.5 帧 |

这一步已经说明：规则有排序价值，但 DEV 操作门并未通过。

## 7. HOLDOUT 正式结果

### 7.1 主目标排序能力

| scene | 可评 object | 阳性 | 基准率 | AUROC（95% CI） | AP | AP lift（95% CI） | 预注册排序门 |
|---|---:|---:|---:|---:|---:|---:|---|
| room1 | 30 | 24 | 0.800 | **0.569**（0.250–0.857） | 0.857 | **+0.057**（-0.035–0.162） | 失败 |
| office1 | 23 | 18 | 0.783 | **0.900**（0.744–1.000） | 0.965 | **+0.183**（0.063–0.349） | 通过 |

room1 的置信区间很宽并跨过随机水平，AP lift 置信区间也跨 0；不能把 office1 的强结果外推为跨场景稳定结论。

### 7.2 严格双前景目标

| scene | 阳性 / 总数 | AUROC（95% CI） | AP lift |
|---|---:|---:|---:|
| room1 | 9 / 30 | 0.651（0.429–0.839） | +0.111 |
| office1 | 4 / 23 | 0.803（0.592–0.974） | +0.249 |

两个场景方向都大于 0.5，说明结果不是完全由背景边界案例支撑；但严格阳性样本很少，CI 仍宽。

### 7.3 冻结阈值 0.25 的真实在线操作指标

| scene | TP / FP / FN / TN | endpoint precision | endpoint recall | 首次触发当前 precision | 阴性触发率 | post recall | median / P90 delay |
|---|---|---:|---:|---:|---:|---:|---:|
| room1 | 12 / 4 / 12 / 2 | 0.750 | 0.500 | 0.625 | **0.667** | 0.500 | 0 / 62.8 |
| office1 | 13 / 0 / 5 / 5 | 1.000 | 0.722 | 0.615 | 0.000 | 0.667 | 2 / 46.4 |
| pooled | 25 / 4 / 17 / 7 | **0.862** | **0.595** | **0.621** | **0.364** | **0.571** | **0 / 59.9** |

`delay` 单位是处理帧；本实验 stride=5，因此 10 个处理帧对应 50 个原始帧。

### 7.4 预注册判决

| HOLDOUT gate | 结果 |
|---|---|
| 两场景主目标 AUROC >= 0.65 且 AP lift >= 0.10 | **失败**（room1） |
| pooled endpoint precision >= 0.75 | 通过 |
| pooled endpoint recall >= 0.40 | 通过 |
| pooled 首次触发当前 precision >= 0.60 | 通过（0.621，刚过门） |
| 每场景阴性触发率 <= 0.25 | **失败**（room1=0.667） |
| median post delay <= 10 | 通过 |
| 严格目标两场景方向均 > 0.5 | 通过 |

它没有触发 STOP：两个 HOLDOUT 的 AUROC 和 AP lift 都仍为正向，严格目标也未反转。但它没有达到 GO，所以正式判为 **MODIFY**。

## 8. 误报为什么发生

全部 4 个 FP 都出现在 room1：

| object | 首次触发 | 当时标签冲突 | 主要原因 |
|---|---:|---|---|
| mirror | raw frame 15，6 observations | picture ↔ mirror | 同义/粒度与早期小样本；真实 mixed 仅 2/138 |
| nightstand | raw frame 445，6 observations | nightstand ↔ end table | 早期小样本瞬时越阈，最终分数降到 0.057 |
| power outlet | raw frame 1570，79 observations | power outlet ↔ light switch | 相邻部件/类别混淆，但 mask GT 不 mixed |
| light switch | raw frame 680，8 observations | light switch ↔ box | 类别粒度/部件混淆；真实 mixed 为 0 |

所以分数检测到的首先是“类别意见不一致”，而不必然是“实例 mask 混合”。简单增加标签熵权重不会解决这个问题。

## 9. 漏报为什么发生

17 个 FN 中：

- 6 个对象的最终/历史最大标签冲突分数为 0；
- 8 个对象的真实 mixed observation 比例至少 50%；
- 4 个还是严格双前景阳性；
- 2 个对象从头到尾只有一个 observation label。

典型反例：

- room1 `lamp`：94.2% observation 为 mixed，但所有 observation 都叫 lamp，分数 0；
- office1 `towel`：94.3% mixed，分数 0；
- room1 `ceiling light`：48.0% mixed，只有一个标签，分数 0；
- room1 `bed`：95.5% mixed，但标签冲突分数仅 0.117。

这是机制上不可消除的盲区：如果多个真实实例被同一个 mask 覆盖，但 detector 每次都输出相同类别，任何只看 class histogram 的 trigger 都无法发现它。

另外，HOLDOUT 有 7 个最终阳性对象在 GT onset 之前就触发（room1 2 个、office1 5 个）。这些不能未经干预实验就称为“提前预测成功”；这正是 endpoint precision 0.862 而首次触发当前 precision 只有 0.621 的原因。

## 10. 与 V0 的 0.702 / 0.884 为什么不同

V0 的 0.702 / 0.884 是冻结 B0 最终地图上的**离线 label entropy 相关性**，可评对象分别为 69/33，阳性为 31/22。它回答“最终地图中标签熵和最终错误是否相关”，没有回答何时在线触发。

V1 做了三个关键升级：

1. 从 frame=0 重跑并逐帧计算，只使用当时成员；
2. 用 DEV 选择后冻结的分数是带跨帧 persistence 的 `repeated_alt_fraction`，不是 V0 的最终 label entropy；
3. 采用更严格的在线 membership/GT 可评门，DEV 最终进入排序的对象变为 41/23，阳性为 26/21。

因此 V1 的 DEV AUROC 0.618 / 0.798 不是对旧数字的复现失败，而是对更严格、不同问题的估计。它说明 V0 的高相关性包含了最终地图与离线选择带来的乐观成分，不能直接当作在线 trigger 性能。

## 11. 研究价值与下一步

### 可以保留的结论

- observation label persistence 是便宜、因果、无需 VLM 的在线信号；
- 对 office1 和严格双前景目标有明确正向排序价值；
- 适合用于候选优先级或语义冲突分支，减少昂贵 verifier/VLM 的检查量。

### 不能声称的结论

- 不能称其为高精度独立 trigger；
- 不能称其统一覆盖 mixed mask、false merge、false split 和 association；
- 不能在 HOLDOUT 上重新调阈值后报告改进；
- 自动 Replica GT 仍不等同于人工确认的 actionable repair truth。

### 建议的最小下一轮

保持体素内容简单，不增加复杂特征，改成两类候选分支加一个共同 verifier：

```text
分支 A：repeated label conflict
         → 找语义意见反复不一致的对象

分支 B：简单 voxel / mask co-coverage conflict
         → 找同标签但空间上可分、清晰视角反复分开的区域

两类候选
    → 简单空间 verifier（连通块、跨帧共覆盖/分开支持、最小支持与迟滞）
    → 只有验证通过才创建 repair ticket
```

下一轮必须换新的 DEV/HOLDOUT 场景并重新预注册；room1/office1 只能用于失败机制设计，不能再用于正式阈值选择。

## 12. 产物位置

服务器代码：

`/home/chenkejun/beauty/conceptgraphs/code/experiments/online_label_trigger_v1_20260831/`

服务器结果：

`/home/chenkejun/beauty/conceptgraphs/results/experiments/online_label_trigger_v1_20260831/`

关键文件：

- `PREREGISTRATION_CN.md`：运行 HOLDOUT 前冻结的协议；
- `HOLDOUT_DETECTION_EXECUTION_ADDENDUM_CN.md`：cache 不兼容后的预指标执行决策；
- `dev/selection/frozen_rule.json`：冻结规则；
- `holdout/evaluation/holdout_summary.json`：正式判决和全部门；
- `holdout/evaluation/failure_analysis.json`：逐对象 TP/FP/FN/TN 与因果首触发；
- `holdout/{room1,office1}/causal_trace.jsonl`：逐帧因果分数；
- `figures/`：四场景排序、HOLDOUT 操作指标、对象级诊断图；
- `logs/` 和 `mapping_records/`：成功与失败运行、配置和 timing 证据。

本报告只将本轮结果定义为**探索后的严格可行性验证**，不是跨数据集正式 SOTA 结论，也没有执行自动修复。
