# CGS — ConceptGraphs 服务器复现快照

本仓库保存 ConceptGraphs 在服务器上的复现代码、配置、实验日志和小型结果文件。上游 `ali-dev` 代码固定在提交：

```text
72f5962822b5e8678a446f367a06df1a977d2a4d
```

## 当前结果

- Replica `room0` 数据校验通过：RGB、Depth、Pose 各 2000 帧。
- ali-dev Smoke Test 完成 5/5 帧。
- `room0`、`stride=10` 完整映射完成 200/200 帧。
- 完整运行没有 CUDA OOM 或 traceback。
- 完整对象地图 JSON 含 72 个对象；`make_edges=false`，因此边 JSON 为空。

详细状态见 [`logs/SERVER_REPRO_STATUS.md`](logs/SERVER_REPRO_STATUS.md)，完整运行日志见 [`logs/ali_room0_stride10.log`](logs/ali_room0_stride10.log)。

## 目录

- `code/concept-graphs-ali/`：上游代码快照及本次兼容修复。
- `docs/`：服务器复现与本地可视化手册。
- `logs/`：环境、硬件、Smoke Test 和完整 room0 日志。
- `results/`：不含大点云的 JSON/config 结果。
- `outputs/smoke_vis/`：少量 Smoke Test 检测图。
- `scripts/server_env.sh`：当前服务器环境变量加载脚本。

## 本次兼容处理

- 默认使用 `CUDA_VISIBLE_DEVICES=0`。物理 GPU 3 在复现时无法初始化 CUDA；其余测试 GPU 可用。
- 固定 `numpy==1.24.3`、`supervision==0.18.0`。
- `make_edges=false` 时不初始化或调用 OpenAI API。
- 修正无边模式下 `merge_objects` 的返回值处理。

## 未上传的大文件

为避免 GitHub 大文件和仓库体积问题，以下内容保留在原服务器，不在本仓库中：

- Replica 数据集和逐帧检测缓存；
- Conda 环境、包缓存和 Python 二进制依赖；
- YOLO、SAM、OpenCLIP 等模型权重；
- 55 MiB 完整点云 `pcd_room0_mapping_stride10.pkl.gz`；
- Hugging Face、Torch、WandB 等运行缓存。

日志中的临时签名下载 URL 已在上传副本中替换为 `[REDACTED_SIGNED_DOWNLOAD_URL]`；原服务器日志未修改。

## 注意

手册参数使用 `use_rerun=false` 和 `save_rerun=true`。当前 ali-dev 提交只有在 `use_rerun && save_rerun` 时才保存 `.rrd`，因此本次服务器运行没有生成 `.rrd`。

按原要求，本次没有执行 Ubuntu 本地同步、Rerun/Open3D GUI 或本地可视化操作。
