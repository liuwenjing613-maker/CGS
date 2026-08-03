# CGS 服务器端补充、Habitat 建图与论文对齐完整指南

> **适用仓库：** `https://github.com/liuwenjing613-maker/CGS.git`  
> **服务器现有工作区：** `/home/chenkejun/beauty/conceptgraphs`  
> **上游 ali-dev 固定提交：** `72f5962822b5e8678a446f367a06df1a977d2a4d`  
> **本地工作区约定：** `~/conceptgraphs`  
> **目标：** 保留现有 Replica room0 基线，在服务器接收 Ubuntu Habitat/MP3D 导出的 RGB-D+Pose 序列，生成对象地图、语义图和论文实验结果，再把轻量结果同步回本地执行查询与导航。  
> **更新时间：** 2026-08-02

---

# 0. 先明确当前服务器已经完成什么

根据当前公开仓库，已经完成：

- Replica `room0` 的 RGB、Depth、Pose 各 2000 帧校验；
- ali-dev Smoke Test 5/5 帧；
- `stride=10` 完整映射 200/200 帧；
- 无 CUDA OOM、无 traceback；
- 生成 72 个对象节点；
- 服务器保留约 55 MiB 完整点云；
- 固定 `numpy==1.24.3`、`supervision==0.18.0`；
- `make_edges=false` 时不再调用 OpenAI；
- 修复无边模式下 `merge_objects` 返回值兼容问题。

当前**尚未完成**：

- 本地同步与 Open3D 验收；
- Habitat/MP3D 输入接口；
- VLM 多视角节点描述；
- 场景关系边，当前 edge JSON 为 `{}`；
- Habitat 中从对象节点到可导航目标位姿；
- 导航执行、成功判定、SPL；
- main 分支论文指标严格复现；
- 面向论文的统一实验协议。

当前结果应准确称为：

> **YOLO-World + SAM + OpenCLIP 驱动的对象级三维地图基线。**

它还不是论文意义上包含自然语言 caption 和关系边的完整 ConceptGraphs。

---

# 1. 总体系统边界

## 1.1 服务器负责

```text
Ubuntu Habitat 导出的序列
        ↓ rsync
服务器输入验证
        ↓
YOLO-World + SAM + OpenCLIP
        ↓
多视角对象融合
        ↓
对象地图 pkl.gz + obj_json
        ↓
可选：VLM caption、几何关系、LLM关系
        ↓
map_bundle
        ↓ rsync
Ubuntu 查询、导航、评价
```

服务器负责：

- 大模型和视觉模型推理；
- 对象级三维地图；
- 节点描述和关系边；
- 批量建图；
- 批量查询实验；
- 地图层和检索层统计；
- 论文中的重计算消融。

## 1.2 服务器暂不负责

第一阶段不要在服务器运行 Habitat：

- 本地已装 Habitat 和部分 MP3D；
- 本地有桌面，调试传感器和坐标更方便；
- Habitat 与旧版 ConceptGraphs 依赖容易冲突；
- 服务器只接收统一格式数据，职责最清楚。

---

# 2. 本地与服务器唯一接口协议

后面所有脚本必须遵守这一协议。不要让服务器猜相机参数，也不要让本地猜地图坐标系。机器人研究已经有足够多的猜测了。

## 2.1 本地上传目录

本地导出：

```text
~/conceptgraphs/data/habitat_exports/<sequence_id>/
├── results/
│   ├── frame000000.jpg
│   ├── depth000000.png
│   ├── frame000001.jpg
│   └── depth000001.png
├── semantic/
│   ├── semantic000000.npy
│   └── semantic000001.npy
├── traj.txt
├── intrinsics.json
├── metadata.json
├── frames.jsonl
├── checksums.sha256
└── READY
```

上传到服务器：

```text
/home/chenkejun/beauty/conceptgraphs/data/HabitatMP3D/sequences/<sequence_id>/
```

`sequence_id` 建议：

```text
<MP3D场景ID>_<用途>_<版本>
```

例如：

```text
17DRP5sb8fy_smoke_v001
17DRP5sb8fy_map_v001
```

不要在 `sequence_id` 中包含 `/`、空格和中文。

## 2.2 RGB

```text
格式：JPEG
命名：frame%06d.jpg
色彩：RGB，写文件前可转为 OpenCV BGR
尺寸：第一版固定 640×480
```

## 2.3 Depth

```text
格式：uint16 PNG
单位：毫米
命名：depth%06d.png
无效值：0
加载到 ConceptGraphs 后除以 1000.0 得到米
```

服务器数据配置必须：

```yaml
png_depth_scale: 1000.0
```

禁止沿用 Replica 的 `6553.5`。

## 2.4 Pose

`traj.txt` 每行 16 个浮点数，按行展开 `4×4` 矩阵：

```text
T_habitat_world_from_opencv_camera
```

要求：

- camera-to-world；
- OpenCV 相机坐标：`+X` 右、`+Y` 下、`+Z` 前；
- Habitat 世界坐标：右手系、`+Y` 向上；
- 每帧 pose 对应真实 RGB/Depth 传感器，而不是只用 agent body pose。

Habitat 传感器姿态转换：

```python
T_habitat_world_from_habitat_sensor = ...
T_habitat_sensor_from_opencv_camera = np.diag([1.0, -1.0, -1.0, 1.0])

T_habitat_world_from_opencv_camera = (
    T_habitat_world_from_habitat_sensor
    @ T_habitat_sensor_from_opencv_camera
)
```

## 2.5 ConceptGraphs map frame

当前 `ReplicaDataset` 默认 `relative_pose=True`，会把第一帧变为单位阵：

```text
T_cg_map_from_camera_i
= inverse(T_world_from_camera_0) @ T_world_from_camera_i
```

因此对象地图坐标不是 Habitat world，而是第一帧 OpenCV 相机坐标系。

本地导出的 `metadata.json` 必须保存：

```json
{
  "T_habitat_world_from_cg_map": [
    [...],
    [...],
    [...],
    [...]
  ]
}
```

其值就是第一帧：

```text
T_habitat_world_from_opencv_camera_0
```

对象中心回到 Habitat 世界：

```python
p_habitat_world = (
    T_habitat_world_from_cg_map
    @ np.array([x_map, y_map, z_map, 1.0])
)[:3]
```

这条变换是整个导航闭环最关键的接口。丢失它以后，地图仍然漂亮，但机器人会去另一个平行宇宙寻找沙发。

## 2.6 metadata.json 必须字段

```json
{
  "format_version": "cgs-habitat-sequence-v1",
  "sequence_id": "17DRP5sb8fy_smoke_v001",
  "scene_id": "17DRP5sb8fy",
  "num_frames": 20,
  "rgb": {
    "width": 640,
    "height": 480,
    "pattern": "results/frame%06d.jpg"
  },
  "depth": {
    "pattern": "results/depth%06d.png",
    "storage_dtype": "uint16",
    "storage_unit": "millimeter",
    "png_depth_scale": 1000.0
  },
  "camera": {
    "hfov_deg": 90.0,
    "fx": 320.0,
    "fy": 320.0,
    "cx": 319.5,
    "cy": 239.5,
    "sensor_height_m": 1.25
  },
  "poses": {
    "file": "traj.txt",
    "type": "camera_to_world",
    "camera_frame": "opencv",
    "world_frame": "habitat"
  },
  "conceptgraphs": {
    "relative_pose": true,
    "map_frame": "first_opencv_camera"
  },
  "T_habitat_world_from_cg_map": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

## 2.7 READY 原则

本地只有在以下全部完成后才创建空文件 `READY`：

- RGB/Depth 数量一致；
- pose 数量一致；
- metadata 数量一致；
- checksums 生成；
- 本地校验脚本通过。

服务器只处理存在 `READY` 的序列。

---

# 3. 冻结现有 Replica 基线

先保护当前成果，不要边集成 Habitat 边改基线。

```bash
cd /home/chenkejun/beauty/conceptgraphs
source scripts/server_env.sh

git status
git rev-parse HEAD
```

创建基线标签：

```bash
git tag -a replica-ali-baseline-20260802 \
  -m "Replica room0 ali-dev baseline: 72 objects, stride 10"

git push origin replica-ali-baseline-20260802
```

创建新分支：

```bash
git switch -c habitat-integration
```

记录大文件哈希：

```bash
PCD=/home/chenkejun/beauty/conceptgraphs/data/Replica/room0/exps/room0_mapping_stride10/pcd_room0_mapping_stride10.pkl.gz

test -f "$PCD"
sha256sum "$PCD" | tee logs/replica_room0_pcd.sha256
ls -lh "$PCD"
```

如果实际文件名不同：

```bash
find data/Replica/room0/exps/room0_mapping_stride10 \
  -name '*.pkl.gz' -type f -ls
```

---

# 4. 服务器需要新增的目录

```bash
cd /home/chenkejun/beauty/conceptgraphs

mkdir -p \
  data/HabitatMP3D/sequences \
  results/HabitatMP3D \
  configs/habitat_mp3d \
  artifacts \
  scripts/habitat \
  habitat_bridge/server \
  tests/interface \
  logs/habitat
```

建议最终仓库：

```text
CGS/
├── code/concept-graphs-ali/
├── configs/habitat_mp3d/
├── habitat_bridge/
│   ├── server/
│   │   ├── validate_sequence.py
│   │   ├── generate_dataset_config.py
│   │   ├── build_map_bundle.py
│   │   ├── compute_geometry_edges.py
│   │   └── evaluate_map.py
│   └── interface_schema.json
├── scripts/habitat/
│   ├── verify_server.sh
│   ├── verify_sequence.sh
│   ├── run_smoke_mapping.sh
│   ├── run_full_mapping.sh
│   ├── package_map_bundle.sh
│   └── list_sequences.sh
├── artifacts/manifest.yaml
└── results/HabitatMP3D/
```

---

# 5. 扩展 server_env.sh

当前文件使用的实际工作区是：

```bash
/home/chenkejun/beauty/conceptgraphs
```

在 `scripts/server_env.sh` 末尾增加：

```bash
export CG_ALI_FOLDER="$CG_WORK/code/concept-graphs-ali"

export HABITAT_DATA_ROOT="$CG_WORK/data/HabitatMP3D"
export HABITAT_SEQUENCE_ROOT="$HABITAT_DATA_ROOT/sequences"
export HABITAT_RESULT_ROOT="$CG_WORK/results/HabitatMP3D"
export HABITAT_CONFIG_ROOT="$CG_WORK/configs/habitat_mp3d"

export CG_ALI_PYTHON="$CG_WORK/envs/cg-ali/bin/python"
```

不要删除现有：

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
```

因为仓库已记录物理 GPU 3 初始化失败。

验证：

```bash
source scripts/server_env.sh

echo "$CG_WORK"
echo "$CG_ALI_FOLDER"
echo "$HABITAT_SEQUENCE_ROOT"
echo "$CG_ALI_PYTHON"

test -x "$CG_ALI_PYTHON"
test -d "$CG_ALI_FOLDER"
```

---

# 6. 新增服务器环境自检脚本

创建：

```bash
nano scripts/habitat/verify_server.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/server_env.sh"

echo "CG_WORK=$CG_WORK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

"$CG_ALI_PYTHON" - <<'PY'
import torch
import numpy as np
import supervision
import open3d
import open_clip
import conceptgraph

print("torch:", torch.__version__)
print("numpy:", np.__version__)
print("supervision:", supervision.__version__)
print("cuda_available:", torch.cuda.is_available())
assert torch.cuda.is_available()
print("gpu:", torch.cuda.get_device_name(0))
print("server environment OK")
PY

test -f "$CG_ALI_FOLDER/conceptgraph/slam/rerun_realtime_mapping.py"
test -f "$CG_ALI_FOLDER/conceptgraph/scannet200_classes.txt"

echo "All server checks passed."
```

授权并运行：

```bash
chmod +x scripts/habitat/verify_server.sh
scripts/habitat/verify_server.sh
```

---

# 7. 服务器输入验证器要求

创建：

```text
habitat_bridge/server/validate_sequence.py
```

必须检查：

1. `READY` 存在；
2. `metadata.json` 的 `format_version`；
3. RGB/Depth 数量等于 `num_frames`；
4. `traj.txt` 行数等于 `num_frames`；
5. 每行恰好 16 个有限浮点数；
6. 每个旋转矩阵：
   - `det(R)` 接近 1；
   - `R.T @ R` 接近单位阵；
7. 深度：
   - dtype `uint16`；
   - 中位数在合理范围；
   - 非零比例不太低；
8. 第一帧 transform 与 metadata 的 `T_habitat_world_from_cg_map` 一致；
9. `checksums.sha256` 校验通过；
10. 生成 `VALIDATED`，失败则不生成。

运行接口：

```bash
"$CG_ALI_PYTHON" habitat_bridge/server/validate_sequence.py \
  --sequence-dir "$HABITAT_SEQUENCE_ROOT/17DRP5sb8fy_smoke_v001"
```

成功输出：

```text
sequence_id=...
num_frames=...
rgb_size=640x480
depth_scale=1000.0
pose_convention=camera_to_world/opencv
VALIDATION PASSED
```

---

# 8. 自动生成 Habitat 数据配置

创建：

```text
habitat_bridge/server/generate_dataset_config.py
```

输入：

```text
metadata.json
```

输出到序列目录：

```text
conceptgraphs_dataset.yaml
```

内容示例：

```yaml
dataset_name: replica

camera_params:
  image_height: 480
  image_width: 640
  fx: 320.0
  fy: 320.0
  cx: 319.5
  cy: 239.5
  png_depth_scale: 1000.0
  crop_edge: 0
```

这里暂时使用：

```yaml
dataset_name: replica
```

因为本地按 Replica 兼容目录导出，服务器复用现有 `ReplicaDataset`，避免第一天就引入新的 dataset class。

运行：

```bash
"$CG_ALI_PYTHON" habitat_bridge/server/generate_dataset_config.py \
  --sequence-dir "$HABITAT_SEQUENCE_ROOT/17DRP5sb8fy_smoke_v001"
```

检查：

```bash
cat "$HABITAT_SEQUENCE_ROOT/17DRP5sb8fy_smoke_v001/conceptgraphs_dataset.yaml"
```

---

# 9. 服务器端 20 帧 Smoke Test

前提：

```bash
SEQ=17DRP5sb8fy_smoke_v001
test -f "$HABITAT_SEQUENCE_ROOT/$SEQ/READY"
test -f "$HABITAT_SEQUENCE_ROOT/$SEQ/VALIDATED"
```

创建脚本：

```bash
nano scripts/habitat/run_smoke_mapping.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <sequence_id>"
  exit 2
fi

SEQ="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/server_env.sh"

SEQ_DIR="$HABITAT_SEQUENCE_ROOT/$SEQ"
CFG="$SEQ_DIR/conceptgraphs_dataset.yaml"

test -f "$SEQ_DIR/VALIDATED"
test -f "$CFG"

cd "$CG_ALI_FOLDER/conceptgraph"

"$CG_ALI_PYTHON" slam/rerun_realtime_mapping.py \
  dataset_root="$HABITAT_SEQUENCE_ROOT" \
  dataset_config="$CFG" \
  scene_id="$SEQ" \
  image_height=480 \
  image_width=640 \
  start=0 \
  end=20 \
  stride=1 \
  make_edges=false \
  use_rerun=false \
  save_rerun=false \
  force_detection=true \
  save_detections=true \
  detections_exp_suffix="${SEQ}_smoke_detections" \
  exp_suffix="${SEQ}_smoke_mapping" \
  save_video=false \
  save_objects_all_frames=false \
  obj_pcd_max_points=3000 \
  2>&1 | tee "$CG_WORK/logs/habitat/${SEQ}_smoke_mapping.log"
```

授权并运行：

```bash
chmod +x scripts/habitat/run_smoke_mapping.sh
scripts/habitat/run_smoke_mapping.sh "$SEQ"
```

## 9.1 Smoke Test 验收

```bash
find "$HABITAT_SEQUENCE_ROOT/$SEQ/exps" \
  -maxdepth 3 -type f | sort
```

必须得到：

- 检测可视化；
- 检测缓存；
- `obj_json`；
- `edge_json`，无边模式应为 `{}`；
- `.pkl.gz`；
- 配置 JSON；
- 完整日志。

检查 traceback：

```bash
grep -Ei 'traceback|cuda out of memory|nan|fatal|error' \
  "$CG_WORK/logs/habitat/${SEQ}_smoke_mapping.log"
```

无输出最好。

## 9.2 Smoke Test 的坐标验收

因为服务器没有 GUI，先生成低成本摘要：

- 所有对象 bbox center 范围；
- bbox extent 范围；
- 点云 min/max；
- 对象数量；
- NaN 数量；
- 距离原点最大值；
- 各类别计数。

创建：

```text
habitat_bridge/server/evaluate_map.py
```

至少输出：

```json
{
  "num_objects": 18,
  "num_nan_centers": 0,
  "bbox_center_min": [...],
  "bbox_center_max": [...],
  "max_radius_m": 12.4,
  "class_histogram": {...}
}
```

若 20 帧室内序列出现数百米尺度，优先检查：

- depth scale；
- pose 矩阵方向；
- Habitat/OpenCV 基变换。

不要调对象融合阈值。融合阈值无法把 600 米高的沙发劝回现实。

---

# 10. 全量 MP3D 建图

本地通过 Smoke Test 后，上传正式序列：

```text
17DRP5sb8fy_map_v001
```

建议初版：

- 300～500 帧；
- 640×480；
- 30°或45°旋转步长；
- 30～50 个导航采样位置；
- `stride=2` 或 `stride=3`；
- 先不开边。

创建：

```bash
nano scripts/habitat/run_full_mapping.sh
```

核心命令：

```bash
SEQ=17DRP5sb8fy_map_v001
CFG="$HABITAT_SEQUENCE_ROOT/$SEQ/conceptgraphs_dataset.yaml"

cd "$CG_ALI_FOLDER/conceptgraph"

"$CG_ALI_PYTHON" slam/rerun_realtime_mapping.py \
  dataset_root="$HABITAT_SEQUENCE_ROOT" \
  dataset_config="$CFG" \
  scene_id="$SEQ" \
  image_height=480 \
  image_width=640 \
  start=0 \
  end=-1 \
  stride=2 \
  make_edges=false \
  use_rerun=false \
  save_rerun=false \
  force_detection=true \
  save_detections=true \
  detections_exp_suffix="${SEQ}_detections_stride2" \
  exp_suffix="${SEQ}_mapping_stride2" \
  save_video=false \
  save_objects_all_frames=false \
  obj_pcd_max_points=5000 \
  2>&1 | tee "$CG_WORK/logs/habitat/${SEQ}_mapping_stride2.log"
```

后续只调整融合参数时：

```text
force_detection=false
detections_exp_suffix 保持完全一致
```

避免重复跑 YOLO/SAM/OpenCLIP。

---

# 11. 结果打包接口 map_bundle

服务器每次正式建图完成后，必须生成一个可同步目录：

```text
results/HabitatMP3D/<sequence_id>/<run_id>/map_bundle/
├── object_map.pkl.gz
├── objects.json
├── edges.json
├── mapping_config.json
├── sequence_metadata.json
├── map_manifest.json
├── map_statistics.json
├── query_classes.txt
├── checksums.sha256
└── COMPLETE
```

## 11.1 map_manifest.json

```json
{
  "format_version": "cgs-map-bundle-v1",
  "sequence_id": "17DRP5sb8fy_map_v001",
  "run_id": "mapping_stride2",
  "map_frame": "first_opencv_camera",
  "habitat_world_frame": "habitat",
  "T_habitat_world_from_cg_map": [
    [...],
    [...],
    [...],
    [...]
  ],
  "files": {
    "object_map": "object_map.pkl.gz",
    "objects": "objects.json",
    "edges": "edges.json"
  },
  "models": {
    "detector": "yolov8l-world.pt",
    "segmenter": "sam_l.pt",
    "image_encoder": "ViT-H-14/laion2b_s32b_b79k"
  }
}
```

`T_habitat_world_from_cg_map` 必须直接复制输入 `metadata.json`，不允许重新猜。

## 11.2 COMPLETE 原则

只有在以下完成后创建 `COMPLETE`：

- pkl 可加载；
- obj JSON 可解析；
- object 数量 > 0；
- map_statistics 无 NaN；
- map_manifest 完整；
- checksums 生成。

---

# 12. 可移植性修复

当前 `config_params.json` 中保存了服务器绝对路径，并残留上游作者路径。用于记录没问题，但不能作为跨机器接口。

必须新增模板配置：

```text
configs/habitat_mp3d/base_paths.template.yaml
```

内容：

```yaml
repo_root: ${oc.env:CG_ALI_FOLDER}
data_root: ${oc.env:HABITAT_SEQUENCE_ROOT}
```

新增：

```text
artifacts/manifest.yaml
```

记录未上传大文件：

```yaml
artifacts:
  - name: replica_room0_object_map
    server_path: /home/chenkejun/beauty/conceptgraphs/data/Replica/room0/exps/room0_mapping_stride10/pcd_room0_mapping_stride10.pkl.gz
    required_for:
      - local_replica_visualization
    sha256_file: logs/replica_room0_pcd.sha256

  - name: habitat_sequence_data
    server_path: /home/chenkejun/beauty/conceptgraphs/data/HabitatMP3D/sequences
    uploaded_to_git: false

  - name: model_caches
    server_path: /home/chenkejun/beauty/conceptgraphs/models
    uploaded_to_git: false
```

---

# 13. 节点描述与关系边的正确补充顺序

不要一开始就全量调用 VLM/LLM。

## 13.1 第一层：检测标签基线

当前已有：

```text
object_tag
bbox_center
bbox_extent
CLIP image feature
```

先完成：

- 直接类别查询；
- CLIP 文本查询；
- Habitat 导航闭环。

这是论文 Baseline B0。

## 13.2 第二层：几何关系边

服务器新增：

```text
habitat_bridge/server/compute_geometry_edges.py
```

只用三维几何生成：

- `near`
- `above`
- `below`
- `left_of`
- `right_of`
- `inside`
- `on`

输出：

```json
[
  {
    "source": 12,
    "target": 7,
    "relation": "near",
    "confidence": 0.91,
    "source_type": "geometry"
  }
]
```

几何边优先于 LLM 猜测，适合多实例消歧。

这是 Baseline B1。

## 13.3 第三层：VLM 多视角 caption

从 pkl 中提取每个对象最佳视角裁剪，使用冻结 VLM：

```text
对象图像证据
→ 多视角 caption
→ 结构化节点描述
```

建议先只对任务候选 Top-K 节点运行，不对整张图全部调用。

输出：

```json
{
  "object_id": 12,
  "object_tag": "chair",
  "caption": "a black dining chair next to a wooden table",
  "caption_confidence": 0.86,
  "evidence_views": [...]
}
```

这是 Baseline B2。

## 13.4 第四层：任务触发修复

最终论文方法：

- 查询候选分数接近；
- 多视角描述冲突；
- 到达后目标不可见；
- 图关系与当前观察冲突；

才触发 VLM：

```text
RELABEL
REJECT
MERGE
UPDATE_LOCATION
UPDATE_RELATION
```

这是 Proposed Method。

---

# 14. 服务器端论文实验矩阵

## 14.1 建图基线

| 编号 | 方法 |
|---|---|
| M0 | YOLO-World + SAM + OpenCLIP，当前 ali-dev |
| M1 | M0 + 几何关系 |
| M2 | M1 + VLM caption |
| M3 | M2 + 任务触发式修复 |

## 14.2 查询基线

| 编号 | 方法 |
|---|---|
| Q0 | detector tag exact/semantic match |
| Q1 | CLIP text-to-object similarity |
| Q2 | caption text embedding |
| Q3 | LLM/VLM 对候选子图重排 |
| Q4 | 任务触发式图修复后重排 |

## 14.3 消融

- 去掉当前视图；
- 去掉历史多视角；
- 去掉几何约束；
- 每次都调用 VLM；
- 从不调用 VLM；
- 只修标签；
- 标签+关系；
- 标签+关系+位置；
- Top-K = 3/5/8；
- 不确定性阈值变化。

---

# 15. 服务器应产出的论文指标

## 15.1 地图层

- 节点 Precision / Recall；
- 重复节点率；
- 目标实例召回率；
- 物体中心误差；
- 关系 F1；
- 节点 caption 正确率；
- 地图文件大小；
- 峰值显存；
- 每帧处理时间。

## 15.2 查询层

- Recall@1；
- Recall@3；
- 描述查询；
- affordance 查询；
- negation 查询；
- 多实例关系查询；
- 修复成功率；
- 错误修复率；
- VLM 调用次数。

导航的 Success/SPL 由本地 Habitat 端统计，服务器只接收最终 `episodes_result.jsonl` 做汇总。

---

# 16. main 分支严格论文复现放在什么时候

推荐顺序：

```text
先完成 Habitat 单场景闭环
→ 再补 main 分支 Replica 指标
```

原因：

- 目前研究目标是 Habitat 中完整机器人流程；
- main 分支依赖旧 Grounded-SAM、旧 LLaVA、历史 API；
- 它适合证明“原论文指标复现”，不负责证明新系统能导航。

在以下条件满足后再投入 main：

```text
[ ] Habitat 20 帧建图方向和尺度正确
[ ] Habitat 正式地图可查询
[ ] 对象中心能转为 Habitat world
[ ] 至少 10 个导航 episode 跑通
```

main 分支最终用于：

- Replica mAcc/F-mIoU；
- 原论文基线表；
- 证明对官方方法理解和复现充分。

---

# 17. 服务器阶段验收门

## S0：冻结 Replica 基线

```text
[ ] tag 已创建
[ ] 55 MiB pkl 哈希已记录
[ ] 当前日志和配置保留
```

## S1：Habitat 输入验证

```text
[ ] READY
[ ] checksums
[ ] metadata
[ ] pose 数量
[ ] depth scale
[ ] VALIDATED
```

## S2：20 帧 Smoke Test

```text
[ ] 无 traceback
[ ] object 数量 > 0
[ ] 无 NaN
[ ] 场景尺度合理
[ ] map_bundle 可生成
```

## S3：正式 MP3D 地图

```text
[ ] 300～500 帧输入
[ ] 检测缓存完成
[ ] 完整对象地图
[ ] map_statistics
[ ] COMPLETE
```

## S4：完整语义图

```text
[ ] 几何关系边
[ ] Top-K VLM caption
[ ] 查询 benchmark
```

## S5：论文方法

```text
[ ] 任务触发机制
[ ] VLM 修复动作
[ ] 几何约束
[ ] 消融与效率指标
```

---

# 18. 服务器常见故障

## 18.1 本地上传后服务器找不到帧

必须是：

```text
sequences/<sequence_id>/results/frame000000.jpg
sequences/<sequence_id>/results/depth000000.png
```

不要多出一层同名目录。

## 18.2 点云尺度异常

检查：

```text
depth PNG 是否毫米
png_depth_scale 是否 1000.0
```

## 18.3 点云上下颠倒或朝向错误

检查本地 pose：

```text
T_world_from_habitat_sensor @ diag(1,-1,-1,1)
```

不要在服务器侧偷偷再翻一次轴。

## 18.4 地图能看但导航目标错位

优先检查：

```text
T_habitat_world_from_cg_map
```

因为 ali-dev 默认相对第一帧。

## 18.5 没有 `.rrd`

当前仓库已记录：该提交只有 `use_rerun && save_rerun` 时保存 `.rrd`。现有 Replica 运行没有生成 `.rrd`。

第一阶段不要依赖 `.rrd`，同步 `.pkl.gz` 到本地 Open3D 查看。后面可以单独修复 Rerun 的“保存但不 spawn viewer”模式。

## 18.6 GPU 问题

保持：

```bash
CUDA_VISIBLE_DEVICES=0
```

除非先单独验证其他 GPU。

---

# 19. 服务器每日工作模板

```bash
cd /home/chenkejun/beauty/conceptgraphs
source scripts/server_env.sh
git status

scripts/habitat/verify_server.sh
scripts/habitat/list_sequences.sh

# 对新上传序列
"$CG_ALI_PYTHON" habitat_bridge/server/validate_sequence.py \
  --sequence-dir "$HABITAT_SEQUENCE_ROOT/$SEQ"

"$CG_ALI_PYTHON" habitat_bridge/server/generate_dataset_config.py \
  --sequence-dir "$HABITAT_SEQUENCE_ROOT/$SEQ"

scripts/habitat/run_smoke_mapping.sh "$SEQ"
```

实验结束：

```bash
git status
git add scripts habitat_bridge configs artifacts docs logs/*.md
git commit -m "Add Habitat sequence mapping pipeline"
git push origin habitat-integration
```

不要提交：

- MP3D；
- pkl 大文件；
- 模型；
- 检测缓存；
- API 密钥；
- 临时签名 URL。

---

# 20. 最终服务器交付标准

服务器端完成后，应能做到：

```bash
# 1. 本地上传一个 READY 序列
# 2. 服务器验证
python validate_sequence.py --sequence-dir ...

# 3. 一键建图
scripts/habitat/run_full_mapping.sh <sequence_id>

# 4. 一键打包
scripts/habitat/package_map_bundle.sh <sequence_id> <run_id>

# 5. 本地只需 rsync map_bundle
```

最终对论文的服务器角色是：

> **可靠、可复现地把 Habitat/MP3D 的 posed RGB-D 序列转换为开放词汇对象地图和任务相关场景图，并提供地图层、查询层和修复层的批量实验结果。**
