# 在线 observation-label consistency trigger V1 预注册

冻结时间：2026-08-31；在 room1 / office1 holdout 运行前冻结。

## 1. 研究问题

从 frame=0 开始，只使用当前时刻已经进入当前 object 的 observation `class_id` 历史，能否及时、低误报地触发 repeated mixed-mask review？

生产分数严禁读取 GT、最终 object membership、未来 observation 或场景末百分位。ReplicaSSG GT 只在独立 evaluator 中事后生成目标标签。

## 2. 场景划分

- DEV：room0、office0。允许选择候选分数和阈值。
- HOLDOUT：room1、office1。冻结规则后各运行一次，不调规则、不换场景。
- 每个场景：start=0、end=2000、stride=5，共400个在线处理帧；使用同协议缓存检测，从空图开始重跑 mapper。

## 3. 在线可用输入

每个 `FRAME_CLOSED` 时刻，只允许使用当前 active object version 的：

- `class_histogram`；
- 当前 member observation UID 及其已发生 frame（仅用于重复标签跨帧判断）；
- 当前 observation 数量。

不使用 mask、深度、GT、最终成员、未来帧、体素或 VLM。

## 4. 候选分数（DEV内有限选择）

共同最小支持：当前 object 至少5个 observation。

1. `entropy_n5`：归一化标签熵。
2. `minority_n5`：`1 - dominant_label_ratio`。
3. `entropy_persistent`：只有至少一个非主标签在不少于2个 observation、且跨不少于2帧出现时，使用标签熵，否则为0。
4. `minority_persistent`：同一 persistence gate 下使用 minority ratio，否则为0。
5. `repeated_alt_fraction`：满足上述 persistence gate 的非主标签 observation 占比。

固定阈值网格：

- entropy：0.20、0.30、0.40、0.50、0.60；
- ratio：0.10、0.15、0.20、0.25、0.30。

不增加学习模型，不在 holdout 新增特征或搜索阈值。

## 5. GT目标（只供 evaluator）

每个在线 accepted observation 的 processed mask 与同帧 ReplicaSSG instance map 相交。

主目标 `repeated_mixed`：

- 单 observation：top GT purity < 0.8，或 second GT fraction >= 0.1；top GT 至少25像素；
- object：当前成员中 mixed observation >= 2，且占GT可评成员 >= 5%。

敏感性目标 `repeated_two_foreground`：

- second GT fraction >= 0.1、top/second 均至少25像素；
- top/second GT为不同实例，且标签均不属于 wall/floor/ceiling/unknown/undefined；
- object层仍要求 >=2 且占比 >=5%。

## 6. 时间定义

- `s_first`：最终阳性 object 中，构成第一次 repeated error 的首个 mixed observation frame。
- `s_confirmable`：该 object 的当前因果前缀首次满足 repeated target 的 frame。
- `d_first`：冻结触发规则第一次满足的 frame。
- `d_post`：不早于 `s_confirmable` 的第一次满足规则 frame。

同时报告早触发、确认后时延和漏检，不能只报告最有利的一种时延。

## 7. DEV规则选择

先要求候选分数在 room0、office0 的最终active objects上均满足 AUROC > 0.5 且 AP lift > 0。随后在固定阈值网格中筛选：

- 两个DEV场景的最终阳性 precision均 >= 0.75；
- 两个DEV场景的最终阴性触发率均 <= 0.25；
- pooled recall >= 0.40。

合格规则按以下顺序选择：最大化两场景中较小的 F0.5；再最大化较小的 recall；再选择更低复杂度（无persistence优先）和更高阈值。若没有规则合格，仍冻结DEV最佳描述性规则，但明确记为DEV gate失败。

## 8. HOLDOUT判据

`GO`（可进入下一轮系统集成）要求同时满足：

- room1、office1 主目标 AUROC均 >= 0.65，AP lift均 >= 0.10；
- pooled最终阳性 precision >= 0.75、recall >= 0.40；
- pooled首次触发时当前已经为阳性的 precision >= 0.60；
- 每个holdout场景最终阴性触发率 <= 0.25；
- 检出的阳性中，`d_post-s_confirmable` 中位数 <= 10个处理帧；
- 严格双前景目标方向不反转（两个场景 AUROC均 > 0.5）。

`MODIFY`：排序在两个holdout均为正向，但未通过一个或多个操作门。

`STOP`：任一holdout主目标 AUROC <= 0.5 或 AP lift <= 0，发生场景反转，或结果主要由背景/undefined边界案例支撑。

## 9. 必须报告

- endpoint AUROC、AP、AP lift、bootstrap 95% CI、Top/Bottom；
- first-admission precision、recall、F0.5、最终阴性触发率；
- early trigger、`d_post` recall、median/P90 delay；
- 主目标和严格目标；
- 每场景成功、误触发、漏检、真阴性案例；
- mapper/evidence完整性、final membership闭环、timing；
- 与冻结V0 endpoint结果差异及原因；
- 所有失败和限制，不以GT自动标签代替人工可行动真值。
