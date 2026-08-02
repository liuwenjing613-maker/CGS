# ConceptGraphs 服务器复现状态

记录日期：2026-08-02（Asia/Shanghai）

## 已完成

- 服务器工作目录使用 `/home/chenkejun/beauty/conceptgraphs`（手册中的 `/root/autodl-tmp/conceptgraphs` 在当前非 root 环境不可用）。
- Replica `room0` 已完成下载、ZIP 完整性校验和解压：RGB 2000、Depth 2000、Pose 2000。
- `ali-dev` 固定在提交 `72f5962822b5e8678a446f367a06df1a977d2a4d`。
- `cg-ali` 环境、PyTorch 2.0.1 + CUDA 11.8、PyTorch3D 0.7.4 和项目依赖已安装。
- 为兼容当前 `ali-dev` 同时使用的 `ColorPalette.default()` 与 `ColorPalette.DEFAULT`，固定 `supervision==0.18.0`。
- 固定 `numpy==1.24.3`，Faiss 和关键 Python 导入已通过。
- YOLO-World、SAM 和 OpenCLIP 权重已完整下载：
  - `yolov8l-world.pt`: 95,667,042 bytes
  - `sam_l.pt`: 1,249,524,607 bytes
  - OpenCLIP ViT-H-14 safetensors: 3,944,517,836 bytes
- 手册中的 ali-dev Smoke Test 已按原参数完成 5/5 帧，并生成检测缓存、可视化图、地图 `.pkl.gz` 和 JSON。
- 已完成服务器 `room0` `stride=10` 完整映射（200/200 帧），无 OOM、无 traceback，最终对象地图已保存。
- 为使手册的 `make_edges=false` 路径与 ali-dev 当前代码一致，做了两处最小兼容修复：不创建 OpenAI 客户端/不汇总 caption；正确处理无边模式 `merge_objects` 的单返回值。原文件保留 `.original` 备份。
- 手册 19.1 的服务器运行记录已保存到 `logs/room0_stride10_baseline/`。

## 已定位并绕过的 CUDA 问题

最初 Smoke Test 在 `clip_model.to("cuda")` 处终止：

```text
RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.
```

提升权限后进一步核查发现，GPU 设备实际上可见，但物理 GPU 3 单独初始化失败；它被包含在默认的全卡可见集合时，会使 CUDA 全局初始化失败：

```text
torch 2.0.1
torch.version.cuda 11.8
cuda.is_available False
cuda.device_count 0
RuntimeError: No CUDA GPUs are available
```

```text
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7: torch.cuda.is_available() == True
CUDA_VISIBLE_DEVICES=3: torch.cuda.is_available() == False
```

GPU 0 已通过实际显存分配和 CUDA 运算测试。已在 `scripts/server_env.sh` 中设置默认 `CUDA_VISIBLE_DEVICES=0`，同时保留调用者显式选择其他 GPU 的能力。

## 已验收输出

- Smoke Test 检测缓存：`data/Replica/room0/exps/smoke_detections_stride20/`（5 帧、35 个 `.pkl.gz` 子文件、15 张可视化图）。
- Smoke Test 地图：`data/Replica/room0/exps/smoke_mapping_stride20_gpu0_cached/`。
- 完整 room0 地图：`data/Replica/room0/exps/room0_mapping_stride10/`，包含 55 MiB `pcd_*.pkl.gz`、72 个对象 JSON、对象/边配置 JSON。
- 因严格关闭实时 Rerun（`use_rerun=false`），当前代码不会生成 `.rrd`；`save_rerun=true` 不能单独触发保存，这是手册与该提交的行为差异。

## 尚未执行

- 未开始依赖 ali-dev 成功基线的 main、Grounded-SAM、三维地图、LLaVA、场景图和评估阶段。
- 按用户要求，未执行任何 Ubuntu 本地同步、Rerun/Open3D GUI 或本地可视化操作。
- 按用户要求，未执行任何 Ubuntu 本地同步、Rerun/Open3D GUI 或本地可视化操作。

## 续跑条件

每次服务器 shell 载入环境后先确认：

```bash
nvidia-smi
/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

默认 GPU 0 已验证正常；后续服务器任务继续使用该设置，不要把物理 GPU 3 加入 `CUDA_VISIBLE_DEVICES`。
