# CGS Ubuntu 本地：Habitat/MP3D 导出、地图查询、导航闭环与论文实验完整指南

> **本地固定根目录：** `~/conceptgraphs`  
> **本地现状：** 已完成 Habitat-Sim RGB-D/NavMesh 仿真、20 帧 v1 序列导出、服务器 ConceptGraphs 建图、map bundle 回传和本地 Open3D 可视化。  
> **服务器工作区：** `/home/chenkejun/beauty/conceptgraphs`  
> **目标：** 在 Ubuntu 本地完成 Habitat/MP3D 仿真、RGB-D+Pose 导出、地图同步、语言查询、对象目标位姿生成、导航执行和 Success/SPL 评价。  
> **更新时间：** 2026-08-03

> **当前验收边界：** 已跑通“Habitat RGB-D/Pose → 服务器视觉对象建图 → 本地可视化”。当前 `2azQ1b91cZZ` GLB 没有 MP3D 语义标注，因此基于语义 GT 的 ObjectNav Success/SPL 还不在本次验收范围内。

---

# 0. 本地最终承担的职责

本地分成两个独立运行环境：

## 0.1 `conceptgraph` 环境

负责：

- 克隆 CGS；
- 同步服务器地图；
- Open3D 可视化；
- 加载对象 pkl；
- CLIP 文本查询；
- 生成 `goal_request.json`；
- 查看节点和关系。

## 0.2 现有 Habitat 环境

负责：

- 加载 MP3D；
- RGB/Depth/Semantic 采集；
- NavMesh 和 PathFinder；
- 导出 Replica 兼容序列；
- 接收 `goal_request.json`；
- 采样目标观察位姿；
- 机器人导航；
- Success/SPL；
- 视频和轨迹记录。

不要强行把 Habitat 装进 `conceptgraph` 环境。两个历史依赖家族住进同一间房，通常不会促进跨学科交流，只会制造版本冲突。

---

# 1. 本地目录结构

创建：

```bash
mkdir -p ~/conceptgraphs/{CGS,code,data,results,logs,scripts,envs}
mkdir -p ~/conceptgraphs/data/{habitat_exports,mp3d_links}
mkdir -p ~/conceptgraphs/results/{replica,habitat}
mkdir -p ~/conceptgraphs/logs/{habitat,conceptgraph,episodes}
```

最终：

```text
~/conceptgraphs/
├── CGS/
├── data/
│   ├── habitat_exports/
│   └── mp3d_links/
├── results/
│   ├── replica/
│   └── habitat/
├── logs/
│   ├── habitat/
│   ├── conceptgraph/
│   └── episodes/
└── scripts/
```

---

# 2. 克隆 CGS 并建立集成分支

```bash
cd ~/conceptgraphs

if [[ ! -d CGS/.git ]]; then
  git clone https://github.com/liuwenjing613-maker/CGS.git CGS
fi

cd CGS
git fetch origin
git switch main
git pull --ff-only
git switch -c habitat-integration
```

检查：

```bash
git log --oneline -5
git status
```

公开仓库当前上游 ali-dev 固定提交应为：

```text
72f5962822b5e8678a446f367a06df1a977d2a4d
```

---

# 3. 配置服务器 SSH 接口

编辑：

```bash
nano ~/.ssh/config
```

示例：

```sshconfig
Host cg-server
    HostName 替换成服务器地址
    User 替换成服务器用户
    Port 替换成端口
    ServerAliveInterval 60
    ServerAliveCountMax 10
    Compression yes
```

测试：

```bash
ssh cg-server \
  'test -d /home/chenkejun/beauty/conceptgraphs && echo SERVER_OK'
```

定义本地变量：

```bash
cat > ~/conceptgraphs/scripts/local_env.sh <<'EOF'
#!/usr/bin/env bash

export CG_LOCAL="$HOME/conceptgraphs"
export CG_REPO="$CG_LOCAL/CGS"

export CG_SERVER_ALIAS="cg-server"
export CG_REMOTE="/home/chenkejun/beauty/conceptgraphs"

export HABITAT_EXPORT_ROOT="$CG_LOCAL/data/habitat_exports"
export LOCAL_RESULT_ROOT="$CG_LOCAL/results/habitat"
EOF

chmod +x ~/conceptgraphs/scripts/local_env.sh
source ~/conceptgraphs/scripts/local_env.sh
```

验证：

```bash
echo "$CG_LOCAL"
echo "$CG_REMOTE"
ssh "$CG_SERVER_ALIAS" "echo remote_ready"
```

---

# 4. 审计本地 `conceptgraph` 环境

先不要升级任何核心包。

```bash
conda env list
conda activate conceptgraph

which python
python --version

python - <<'PY'
packages = [
    "torch",
    "open3d",
    "open_clip",
    "numpy",
    "cv2",
]

for name in packages:
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "imported"))
    except Exception as exc:
        print(name, "FAILED:", repr(exc))
PY
```

保存：

```bash
conda env export --from-history \
  > ~/conceptgraphs/logs/conceptgraph/environment_history.yml

python -m pip freeze \
  > ~/conceptgraphs/logs/conceptgraph/pip_freeze_before.txt
```

## 4.1 安装本地可视化最低依赖

在 `conceptgraph` 环境：

```bash
python -m pip install --upgrade pip

python -m pip install \
  open3d \
  open_clip_torch \
  hydra-core \
  omegaconf \
  distinctipy \
  scikit-learn \
  scipy \
  pyquaternion \
  pillow \
  opencv-python \
  faiss-cpu
```

安装仓库代码：

```bash
cd ~/conceptgraphs/CGS/code/concept-graphs-ali
python -m pip install -e .
```

验证：

```bash
python - <<'PY'
import torch
import open3d
import open_clip
import conceptgraph

print("torch:", torch.__version__)
print("open3d:", open3d.__version__)
print("conceptgraph:", conceptgraph.__file__)
print("local ConceptGraphs visualization environment OK")
PY
```

---

# 5. 第一件事：验收服务器现有 Replica 地图

不要直接跳到 Habitat。先证明本地能正确读取现有服务器结果。

## 5.1 查找服务器 pkl

```bash
source ~/conceptgraphs/scripts/local_env.sh

ssh "$CG_SERVER_ALIAS" \
  "find '$CG_REMOTE/data/Replica/room0/exps/room0_mapping_stride10' \
   -name '*.pkl.gz' -type f -ls"
```

## 5.2 同步

```bash
mkdir -p ~/conceptgraphs/results/replica/room0

rsync -avP \
  "$CG_SERVER_ALIAS:$CG_REMOTE/data/Replica/room0/exps/room0_mapping_stride10/" \
  ~/conceptgraphs/results/replica/room0/ \
  --include='*.pkl.gz' \
  --include='*.json' \
  --exclude='*'
```

## 5.3 验证哈希

```bash
rsync -avP \
  "$CG_SERVER_ALIAS:$CG_REMOTE/logs/replica_room0_pcd.sha256" \
  ~/conceptgraphs/results/replica/room0/ || true
```

如果哈希文件暂未生成，先本地记录：

```bash
sha256sum ~/conceptgraphs/results/replica/room0/*.pkl.gz
```

## 5.4 Open3D 可视化

```bash
conda activate conceptgraph
cd ~/conceptgraphs/CGS/code/concept-graphs-ali

MAP=$(find ~/conceptgraphs/results/replica/room0 \
  -name '*.pkl.gz' | head -1)

python conceptgraph/scripts/visualize_cfslam_results.py \
  --result_path "$MAP"
```

按键：

- `r`：RGB；
- `i`：实例颜色；
- `f`：文本查询；
- `b`：背景；
- `c`：类别，若支持。

测试：

```text
sofa
chair
pillow
table
cabinet
```

## 5.5 Replica 本地验收门 U0

```text
[ ] pkl 成功同步
[ ] Open3D 窗口正常
[ ] 实例颜色合理
[ ] 地图没有严重重影
[ ] 至少 3 个文本查询返回合理对象
```

未通过 U0，不进入 Habitat。

---

# 6. 审计已有 Habitat 环境

先找环境：

```bash
conda env list
```

逐个尝试你曾使用的 Habitat 环境。假设名为：

```text
habitat
```

激活：

```bash
conda activate habitat
```

记录版本：

```bash
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError

for pkg in [
    "habitat-sim",
    "habitat-lab",
    "numpy",
    "torch",
    "numpy-quaternion",
]:
    try:
        print(pkg, version(pkg))
    except PackageNotFoundError:
        print(pkg, "NOT_INSTALLED")

import habitat_sim
print("habitat_sim:", habitat_sim.__file__)
PY
```

保存：

```bash
conda env export --from-history \
  > ~/conceptgraphs/logs/habitat/environment_history.yml

python -m pip freeze \
  > ~/conceptgraphs/logs/habitat/pip_freeze.txt
```

不要运行：

```bash
pip install -U habitat-sim
```

除非当前环境确认损坏。Habitat-Sim 和 Habitat-Lab 版本需要成对，随意升级是一种很高效的自我破坏。

---

# 7. 审计 MP3D 数据

查找 `.glb`：

```bash
find ~ \
  -type f \
  -path '*/scene_datasets/mp3d/*/*.glb' \
  2>/dev/null | head -50
```

查找导航和语义：

```bash
find ~ \
  -type f \
  \( -name '*.navmesh' \
     -o -name '*.house' \
     -o -name '*_semantic.ply' \
     -o -name '*.scene_dataset_config.json' \) \
  2>/dev/null | head -100
```

记录路径：

```bash
export MP3D_ROOT="/替换成实际路径/scene_datasets/mp3d"
export MP3D_SCENE_ID="17DRP5sb8fy"
export MP3D_SCENE="$MP3D_ROOT/$MP3D_SCENE_ID/$MP3D_SCENE_ID.glb"
```

检查：

```bash
test -f "$MP3D_SCENE"
find "$MP3D_ROOT/$MP3D_SCENE_ID" -maxdepth 1 -type f -ls
```

## 7.1 数据能力判断

只有 `.glb`：

- 可做 RGB；
- 可做 Depth；
- 可做 NavMesh；
- 可做对象地图和导航演示；
- 无法直接用语义真值评估目标实例。

同时有 semantic 数据：

- 可做目标实例可见性；
- 可做 ObjectNav 成功判定；
- 可做节点类别和位置真值评价。

第一阶段没有 semantic 也能跑通系统，但论文正式评价应补齐。

---

# 8. Habitat 最小启动测试

在 Habitat 环境创建：

```text
~/conceptgraphs/CGS/habitat_bridge/local/audit_habitat.py
```

要求：

1. 加载一个 MP3D `.glb`；
2. 加载 NavMesh；
3. 创建 RGB、Depth、Semantic 三个传感器；
4. 随机采样一个 navigable point；
5. 保存：
   - `rgb.png`
   - `depth.npy`
   - `semantic.npy`
6. 打印 agent pose 和 sensor pose；
7. 打印 PathFinder 是否 loaded。

运行：

```bash
conda activate habitat

python ~/conceptgraphs/CGS/habitat_bridge/local/audit_habitat.py \
  --scene "$MP3D_SCENE" \
  --output ~/conceptgraphs/logs/habitat/audit
```

验收：

```text
[ ] RGB 非全黑
[ ] Depth 有有限值
[ ] NavMesh loaded
[ ] sensor pose 可读取
[ ] semantic 可用则非空；不可用时明确记录
```

---

# 9. 本地到服务器的数据接口

本地导出：

```text
~/conceptgraphs/data/habitat_exports/<sequence_id>/
```

服务器目标：

```text
/home/chenkejun/beauty/conceptgraphs/data/HabitatMP3D/sequences/<sequence_id>/
```

## 9.1 文件布局

```text
<sequence_id>/
├── results/
│   ├── frame000000.jpg
│   └── depth000000.png
├── semantic/
│   └── semantic000000.npy
├── traj.txt
├── intrinsics.json
├── metadata.json
├── frames.jsonl
├── checksums.sha256
└── READY
```

## 9.2 RGB/Depth 参数

第一版固定：

```text
width = 640
height = 480
hfov = 90°
sensor height = 1.25m
depth unit = millimeter
depth dtype = uint16 PNG
```

内参：

```python
fx = width / (2 * tan(hfov / 2))
fy = fx
cx = (width - 1) / 2
cy = (height - 1) / 2
```

在 640×480、90° 下：

```text
fx = 320
fy = 320
cx = 319.5
cy = 239.5
```

---

# 10. 关键坐标转换

Habitat：

- 右手系；
- `+Y` 向上；
- 相机前方是 `-Z`。

OpenCV 相机：

- `+X` 右；
- `+Y` 下；
- `+Z` 前。

转换：

```python
T_hab_sensor_from_cv_camera = np.diag(
    [1.0, -1.0, -1.0, 1.0]
)

T_world_from_cv_camera = (
    T_world_from_hab_sensor
    @ T_hab_sensor_from_cv_camera
)
```

保存到 `traj.txt` 的是：

```text
T_world_from_cv_camera
```

注意：

- 使用 `agent_state.sensor_states["color_sensor"]`；
- 不要只使用 agent body pose；
- RGB、Depth、Semantic 传感器位置与旋转必须完全一致。

---

# 11. 导出脚本设计

创建目录：

```bash
cd ~/conceptgraphs/CGS

mkdir -p \
  habitat_bridge/local \
  habitat_bridge/configs \
  habitat_bridge/tests
```

需要实现：

```text
habitat_bridge/local/export_sequence.py
habitat_bridge/local/validate_export.py
habitat_bridge/local/generate_coverage_route.py
habitat_bridge/local/pose_utils.py
habitat_bridge/tests/test_pose_conversion.py
```

## 11.1 `pose_utils.py` 最小接口

```python
import numpy as np
import quaternion


HABITAT_SENSOR_FROM_OPENCV_CAMERA = np.diag(
    [1.0, -1.0, -1.0, 1.0]
)


def state_to_matrix(position, rotation) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion.as_rotation_matrix(rotation)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def habitat_sensor_pose_to_opencv_c2w(
    position,
    rotation,
) -> np.ndarray:
    T_world_from_hab_sensor = state_to_matrix(
        position,
        rotation,
    )
    return (
        T_world_from_hab_sensor
        @ HABITAT_SENSOR_FROM_OPENCV_CAMERA
    )


def transform_point(T, point):
    p = np.ones(4, dtype=np.float64)
    p[:3] = point
    return (T @ p)[:3]
```

## 11.2 必须做的单元测试

```python
def test_rotation_is_valid():
    T = ...
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_center_ray_points_forward():
    # OpenCV 中心射线为 +Z。
    p_cv = np.array([0.0, 0.0, 1.0, 1.0])
    p_world = T_world_from_cv @ p_cv

    # 应与 Habitat 相机前方 -Z 方向一致。
    ...
```

运行：

```bash
conda activate habitat
python -m pytest \
  ~/conceptgraphs/CGS/habitat_bridge/tests/test_pose_conversion.py -q
```

没有 `pytest`：

```bash
python -m pip install pytest
```

---

# 12. Smoke 序列导出

第一版不要设计复杂探索，先导出 20 帧：

- 一个可导航点；
- 每 18°转一次；
- 共 20 个方向；
- RGB/Depth/Semantic 同步；
- pose 使用 sensor state。

运行接口：

```bash
conda activate habitat

SEQ=17DRP5sb8fy_smoke_v001

python ~/conceptgraphs/CGS/habitat_bridge/local/export_sequence.py \
  --scene "$MP3D_SCENE" \
  --scene-id "$MP3D_SCENE_ID" \
  --sequence-id "$SEQ" \
  --output-root ~/conceptgraphs/data/habitat_exports \
  --mode panorama-smoke \
  --num-frames 20 \
  --width 640 \
  --height 480 \
  --hfov 90 \
  --sensor-height 1.25 \
  --seed 2027
```

导出时：

```python
depth_mm = np.clip(
    np.rint(depth_m * 1000.0),
    0,
    65535,
).astype(np.uint16)
```

RGB：

```python
rgb = observations["color_sensor"][..., :3]
cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
```

Semantic：

```python
np.save(path, observations["semantic_sensor"].astype(np.int32))
```

若场景没有 semantic，metadata 中写：

```json
"semantic_available": false
```

---

# 13. 本地导出验证

运行：

```bash
conda activate habitat

python ~/conceptgraphs/CGS/habitat_bridge/local/validate_export.py \
  --sequence-dir \
  ~/conceptgraphs/data/habitat_exports/17DRP5sb8fy_smoke_v001
```

必须检查：

```text
RGB=20
DEPTH=20
POSE=20
metadata num_frames=20
depth dtype=uint16
depth median 合理
pose determinant≈1
第一帧 T 与 metadata 一致
```

人工查看：

```bash
xdg-open \
  ~/conceptgraphs/data/habitat_exports/17DRP5sb8fy_smoke_v001/results/frame000000.jpg
```

深度预览：

```bash
python - <<'PY'
import cv2
import matplotlib.pyplot as plt

p = (
    "/home/" +
    __import__("getpass").getuser() +
    "/conceptgraphs/data/habitat_exports/"
    "17DRP5sb8fy_smoke_v001/results/depth000000.png"
)

d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
plt.imshow(d, cmap="viridis")
plt.colorbar(label="millimeter")
plt.show()
PY
```

生成 checksum：

```bash
cd ~/conceptgraphs/data/habitat_exports/17DRP5sb8fy_smoke_v001

find results semantic -type f -print0 | \
  sort -z | xargs -0 sha256sum > checksums.sha256

sha256sum traj.txt intrinsics.json metadata.json frames.jsonl \
  >> checksums.sha256
```

最后创建：

```bash
touch READY
```

---

# 14. 上传服务器

当前已部署的服务器全流程会依次执行：接口校验、数据集配置生成、ConceptGraphs 建图和 map bundle 打包。

```bash
source ~/conceptgraphs/scripts/local_env.sh

SEQ=2azQ1b91cZZ_smoke_v001

~/conceptgraphs/scripts/upload_sequence.sh "$SEQ"
ssh "$CG_SERVER_ALIAS" \
  "$CG_REMOTE/scripts/server_habitat_pipeline.sh" "$SEQ"
```

从本地导出一直跑到结果下载的推荐一键命令：

```bash
cd ~/conceptgraphs
./scripts/run_habitat_end_to_end.sh
```

默认复用已校验的本地 20 帧序列。若要从 Habitat 重新生成帧，使用 `REEXPORT=1 ./scripts/run_habitat_end_to_end.sh`。

---

# 15. 从服务器同步 map_bundle

服务器完成后（一键脚本会自动执行本步）：

```bash
source ~/conceptgraphs/scripts/local_env.sh

SEQ=2azQ1b91cZZ_smoke_v001
RUN=${SEQ}_smoke

~/conceptgraphs/scripts/download_map_bundle.sh "$SEQ" "$RUN"
```

验证：

```bash
cd "$LOCAL_RESULT_ROOT/$SEQ/$RUN/map_bundle"

sha256sum -c checksums.sha256
test -f COMPLETE
cat map_manifest.json
cat map_statistics.json
```

---

# 16. 本地查看 Habitat 对象地图

```bash
cd ~/conceptgraphs
./scripts/view_habitat_map.sh
```

该脚本会先校验 bundle 哈希，再使用 `conceptgraph` 环境打开 Open3D。快捷键：`R` RGB 颜色，`I` 实例颜色，`C` 类别颜色，`B` 背景对象，`G` 场景图边。

检查：

- 房间方向；
- 地面是否水平；
- 尺度；
- 相机周围对象；
- 实例碎片；
- 背景物体；
- 小物体；
- 重复节点。

## U1 验收门

```text
[ ] 20 帧 map_bundle 同步成功
[ ] Open3D 正立
[ ] 尺度合理
[ ] 对象节点 > 0
[ ] 没有严重 NaN/远距离爆点
```

---

# 17. 正式覆盖轨迹

Smoke 通过后，生成正式建图序列。

不要使用纯随机游走。推荐：

```text
1. PathFinder 随机采样大量可导航点
2. 过滤小 NavMesh island
3. 用最远点采样选择 30～50 个空间分散点
4. 每个点旋转 8 个方向
5. 共 240～400 帧
```

建议参数：

```text
num_locations = 40
yaw_views_per_location = 8
num_frames = 320
yaw_step = 45°
seed = 2027
```

运行：

```bash
SEQ=17DRP5sb8fy_map_v001

python ~/conceptgraphs/CGS/habitat_bridge/local/export_sequence.py \
  --scene "$MP3D_SCENE" \
  --scene-id "$MP3D_SCENE_ID" \
  --sequence-id "$SEQ" \
  --output-root ~/conceptgraphs/data/habitat_exports \
  --mode coverage-map \
  --num-locations 40 \
  --yaw-views 8 \
  --width 640 \
  --height 480 \
  --hfov 90 \
  --sensor-height 1.25 \
  --seed 2027
```

再验证、checksum、READY、rsync。

---

# 18. 对象语言查询

查询与 Habitat 分开进程执行，避免本地 8GB GPU 同时承担 OpenCLIP 和 Habitat 渲染。

## 18.1 查询程序

创建：

```text
habitat_bridge/local/query_object_map.py
```

输入：

```text
object_map.pkl.gz
map_manifest.json
自然语言 query
Top-K
```

输出：

```text
goal_request.json
```

示例：

```json
{
  "format_version": "cgs-goal-request-v1",
  "sequence_id": "17DRP5sb8fy_map_v001",
  "query": "find a comfortable place to sit",
  "candidates": [
    {
      "object_id": 12,
      "object_tag": "sofa",
      "score": 0.81,
      "center_cg_map": [1.2, -0.4, 3.5],
      "center_habitat_world": [4.1, 1.3, -2.2]
    }
  ],
  "selected_object_id": 12
}
```

## 18.2 坐标转换

```python
T_world_from_map = np.asarray(
    manifest["T_habitat_world_from_cg_map"],
    dtype=np.float64,
)

center_map_h = np.r_[center_map, 1.0]
center_world = (T_world_from_map @ center_map_h)[:3]
```

## 18.3 运行

```bash
conda activate conceptgraph

python ~/conceptgraphs/CGS/habitat_bridge/local/query_object_map.py \
  --map-bundle "$BUNDLE" \
  --query "find a chair" \
  --top-k 5 \
  --output ~/conceptgraphs/results/habitat/goal_request.json
```

退出 `conceptgraph` 环境后再启动 Habitat 导航。

---

# 19. 目标观察位姿采样

不能直接导航到对象 bbox 中心。

创建：

```text
habitat_bridge/local/goal_pose_sampler.py
```

对对象中心周围采样：

```text
半径：0.6、0.9、1.2、1.5m
每圈：16 个角度
```

每个候选：

1. `pathfinder.snap_point(candidate)`；
2. 检查返回不是 NaN；
3. snap 距离不超过阈值；
4. 与起点同 NavMesh island；
5. 最短路径存在；
6. 与障碍保持合理距离；
7. 相机朝向对象；
8. 可选：从该位姿渲染一帧，检查目标实例可见。

使用 PathFinder：

```python
path = habitat_sim.ShortestPath()
path.requested_start = agent_position
path.requested_end = snapped_goal

found = sim.pathfinder.find_path(path)
geodesic_distance = path.geodesic_distance
```

选择：

```text
可见且地理距离最短
```

若没有可见性真值，第一版选择：

```text
同层、可达、距离对象 0.8～1.5m、朝向对象
```

输出：

```json
{
  "object_id": 12,
  "object_center_world": [...],
  "goal_position_world": [...],
  "goal_yaw_rad": 1.57,
  "geodesic_distance": 6.4
}
```

---

# 20. Habitat 导航执行

第一版不训练策略，使用 Habitat 原生最短路径。

创建：

```text
habitat_bridge/local/run_objectnav.py
```

输入：

- scene；
- start pose；
- `goal_request.json`；
- goal pose；
- episode id。

执行两种模式：

## 20.1 Oracle Path 模式

直接使用 `ShortestPath.points`，沿路径移动或逐点设置 agent，用于先验证系统接口。

## 20.2 Discrete Follower 模式

使用：

- `move_forward`
- `turn_left`
- `turn_right`
- `stop`

记录动作和观测，用于更接近 ObjectNav。

每步写入：

```text
episodes/<episode_id>/trajectory.jsonl
episodes/<episode_id>/rgb/
episodes/<episode_id>/video.mp4
```

---

# 21. 成功判定

第一版：

```text
到目标观察位姿的地理距离 ≤ 0.5m
且执行 STOP
```

若 MP3D semantic 可用，再加：

```text
目标语义实例在当前视图或原地旋转视图中可见
```

目标可见率建议：

```text
目标像素数 / 图像总像素数 ≥ 0.001
```

具体阈值后续验证。

---

# 22. SPL 和核心指标

对 episode：

```python
success = int(task_succeeded)

spl = success * shortest_distance / max(
    shortest_distance,
    actual_path_length,
)
```

记录：

```json
{
  "episode_id": "ep_0001",
  "scene_id": "17DRP5sb8fy",
  "query": "find a chair",
  "selected_object_id": 12,
  "success": 1,
  "spl": 0.74,
  "shortest_distance": 6.1,
  "actual_path_length": 8.2,
  "target_visible": true,
  "wrong_target": false,
  "query_latency_ms": 820,
  "num_vlm_calls": 0
}
```

## 本地负责的论文指标

- Success；
- SPL；
- SoftSPL，可选；
- 实际路径长度；
- 错目标到达率；
- 最终目标可见率；
- 失败后恢复率；
- 导航时间；
- 重规划次数。

---

# 23. 第一批实验任务

## 23.1 直接对象类

至少 10 个：

```text
find a chair
find a sofa
find a table
find a bed
find a lamp
find a television
find a cabinet
find a plant
find a toilet
find a sink
```

## 23.2 功能查询

```text
find somewhere comfortable to sit
find somewhere to sleep
find something to watch
find a surface for putting books
find somewhere to store things
```

## 23.3 否定和关系查询

```text
find a seat that is not a sofa
find the chair near a table
find the bed closest to a lamp
find a cabinet next to a wall
```

关系查询需要服务器提供几何边或本地直接根据 bbox 计算。

---

# 24. 论文对齐实验阶段

## P0：Replica 现有基线

```text
服务器 room0 对象图
本地 Open3D 验收
CLIP 查询案例
```

## P1：Habitat-MP3D 建图基线

```text
oracle RGB-D pose
YOLO-World + SAM + OpenCLIP
对象图
```

## P2：开放词汇查询导航

```text
query → object node
node → Habitat world
world center → navigable observation pose
PathFinder → navigation
Success/SPL
```

## P3：几何场景图

加入：

```text
near
left/right
above/below
inside/on
```

测试多实例消歧。

## P4：VLM caption

任务候选 Top-K 多视角描述，支持功能和否定查询。

## P5：论文方法

任务触发条件：

- Top-1/Top-2 分数接近；
- 多视角 caption 冲突；
- 到达后目标不可见；
- 当前观察与图冲突。

修复动作：

```text
RELABEL
REJECT
MERGE
UPDATE_LOCATION
UPDATE_RELATION
```

闭环：

```text
查询
→ 候选图
→ 不确定性
→ VLM 修复
→ 导航
→ 视觉验证
→ 必要时重规划
```

---

# 25. 建议的本地一键脚本

## 25.1 上传序列

创建：

```bash
nano ~/conceptgraphs/scripts/upload_sequence.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

source "$HOME/conceptgraphs/scripts/local_env.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <sequence_id>"
  exit 2
fi

SEQ="$1"
SRC="$HABITAT_EXPORT_ROOT/$SEQ"
DST="$CG_SERVER_ALIAS:$CG_REMOTE/data/HabitatMP3D/sequences/$SEQ/"

test -f "$SRC/READY"

rsync -avP --delete-delay "$SRC/" "$DST"
```

授权：

```bash
chmod +x ~/conceptgraphs/scripts/upload_sequence.sh
```

## 25.2 下载 map_bundle

```bash
nano ~/conceptgraphs/scripts/download_map_bundle.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

source "$HOME/conceptgraphs/scripts/local_env.sh"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <sequence_id> <run_id>"
  exit 2
fi

SEQ="$1"
RUN="$2"

DST="$LOCAL_RESULT_ROOT/$SEQ/$RUN/map_bundle"
mkdir -p "$DST"

rsync -avP \
  "$CG_SERVER_ALIAS:$CG_REMOTE/results/HabitatMP3D/$SEQ/$RUN/map_bundle/" \
  "$DST/"
```

---

# 26. 本地阶段验收门

## U0：Replica 本地可视化

```text
[ ] 同步 pkl
[ ] Open3D
[ ] 文本查询
```

## U1：Habitat 审计

```text
[ ] 环境版本冻结
[ ] MP3D 场景加载
[ ] NavMesh loaded
[ ] RGB/Depth
[ ] Semantic 状态明确
```

## U2：接口 Smoke Test

```text
[ ] 20 帧
[ ] depth mm
[ ] OpenCV c2w
[ ] metadata
[ ] checksums
[ ] READY
```

## U3：Habitat 地图回传

```text
[ ] map_bundle
[ ] Open3D 正立
[ ] 坐标转换正确
```

## U4：单任务闭环

```text
[ ] query
[ ] object id
[ ] world center
[ ] goal pose
[ ] path
[ ] stop
[ ] success
```

## U5：批量导航

```text
[ ] 10+ 直接查询
[ ] Success/SPL
[ ] 视频
[ ] episodes_result.jsonl
```

## U6：论文实验

```text
[ ] 功能查询
[ ] 否定查询
[ ] 关系查询
[ ] VLM修复
[ ] 消融
```

---

# 27. 48 小时最优执行顺序

## 第一天上午

```text
1. clone CGS
2. 配 SSH
3. 审计 conceptgraph
4. 同步 Replica pkl
5. Open3D 验收
```

## 第一天下午

```text
6. 审计 Habitat
7. 找 MP3D 场景
8. RGB/Depth/NavMesh 测试
9. 保存环境版本
```

## 第一天晚上

```text
10. pose_utils.py
11. 坐标测试
12. 导出 20 帧
13. validate_export
14. READY
```

## 第二天上午

```text
15. rsync 服务器
16. 服务器 Smoke Mapping
17. 下载 map_bundle
18. Open3D 检查方向和尺度
```

## 第二天下午

```text
19. 修正接口问题
20. 正式导出 320 帧
21. 上传服务器完整建图
22. 本地实现 query_object_map
```

## 第二天晚上

```text
23. goal_pose_sampler
24. oracle shortest path
25. 跑通第一个导航 episode
```

---

# 28. 常见问题

## 28.1 Habitat 导出的深度地图全黑

检查：

- Depth sensor type；
- min/max depth；
- 保存前是否乘 1000；
- `cv2.imwrite` 是否接收 uint16。

## 28.2 点云颠倒

只在本地 exporter 修复基变换，不要服务器和本地各翻一次。

## 28.3 地图与 Habitat 目标错位

检查：

```text
T_habitat_world_from_cg_map
```

它必须是第一帧 OpenCV camera-to-world。

## 28.4 目标点不可达

对象中心不是导航点。必须环形采样、snap 到 NavMesh、检查同 island 和 path。

## 28.5 本地 GPU OOM

查询和 Habitat 分两个进程：

```text
conceptgraph 环境查询并退出
→ Habitat 环境导航
```

不要同时加载 OpenCLIP ViT-H 和 Habitat。

## 28.6 MP3D 没有 semantic

先用几何成功条件跑通；正式评价再补 semantic 资源或使用 ObjectNav episode 数据。

---

# 29. 本地最终交付标准

执行链应变成：

```bash
# 1. Habitat 导出
python export_sequence.py ...

# 2. 验证
python validate_export.py ...

# 3. 上传
upload_sequence.sh <sequence_id>

# 4. 服务器完成后下载
download_map_bundle.sh <sequence_id> <run_id>

# 5. 查询
python query_object_map.py ...

# 6. 导航
python run_objectnav.py ...

# 7. 汇总
python summarize_episodes.py ...
```

最终演示应能清楚展示：

```text
自然语言目标
→ ConceptGraphs 选中三维对象
→ 对象坐标转换到 Habitat world
→ 采样可导航观察点
→ 机器人规划并移动
→ 到达后验证
→ 输出 Success/SPL
```

这才是从“服务器上成功生成 JSON”升级为“机器人完整执行任务”的闭环。
