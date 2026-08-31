# HOLDOUT 检测执行补充预注册（2026-08-31）

## 触发原因

DEV 完成并冻结 `repeated_alt_fraction` 与阈值 `0.25` 后，检查 room1/office1 才发现：

1. 现存旧缓存是早期单文件格式，缺少当前 mapper 的记账字段；
2. 更关键的是，旧缓存使用每场景动态开放词表，而本实验冻结配置使用 `scannet200_classes.txt`。直接补字段后，mapper 在第 0 帧因 `vase` 不属于冻结词表而退出。

两次尝试均在第 0 帧建图形成前退出，未读取 HOLDOUT GT、未产生 HOLDOUT 分数，也未修改冻结规则。失败日志必须保留。

## 冻结处理

不重命名类别、不映射同义词、不使用旧缓存。room1/office1 改为执行各自冻结源配置原本声明的 `force_detection=true`：

- 从 frame=0 开始，按 `0:2000:5` 顺序逐帧运行检测与建图；
- 类别表、模型、阈值、association、merge/filter/denoise 参数保持冻结源配置；
- `make_edges=false`，不调用 VLM；
- 每帧检测结果写入新的实验专属缓存，仅用于复核和可重复回放；
- 严格证据仍保存处理后 mask 与完整 observation PCD；触发算法只读取当时 object 的 observation 类别历史，不读取 PCD、GT、未来帧或最终地图；
- `run_record` 记录源配置哈希、mapper 哈希、有效配置和检测执行模式。

这样与历史 aligned 主结果的执行方式一致，也避免把不同词表协议混入正式 HOLDOUT。

## 继续与停止条件

- 两场景都必须精确处理 400 帧，状态为 `MAP_COMPLETED_EVIDENCE_VALID`；
- 新缓存必须各有 400 个精确帧目录，且帧集合等于 `0,5,...,1995`；
- parity trace 和 evidence manifest 必须通过现有严格审计；
- 任一条件不满足，则不得解封 HOLDOUT 标签；
- 分数、阈值、标签定义、GO/STOP 门槛继续使用原预注册，不因本补充而改变。

## 探索性旧缓存适配器的地位

`adapt_legacy_detection_cache.py` 及其 smoke 产物仅用于定位兼容问题，不属于正式 HOLDOUT 输入。其 manifest 证明旧字段未变，但词表协议不一致，因此明确弃用，不能用于正式结论。
