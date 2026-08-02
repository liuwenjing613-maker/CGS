# ConceptGraphs 完整复现手册
## 服务器计算 + 本地 Ubuntu 可视化

> **本地固定目录：** `~/conceptgraphs`  
> **服务器推荐目录：** `/root/autodl-tmp/conceptgraphs`  
> **推荐路线：** 先跑通官方 `ali-dev` 分支，再用 `main` 分支复现论文实验。  
> **更新时间：** 2026-07-31

---

# 0. 总体原则

这套流程把任务分成两部分：

| 位置 | 负责内容 |
|---|---|
| 服务器 | Replica 数据、模型、Conda 环境、检测/分割、三维建图、场景图、评估 |
| 本地 Ubuntu | 代码、少量结果、Rerun/Open3D 可视化、后续代码修改 |

本地不下载 Replica，不保存 Grounded-SAM 和 LLaVA 权重，只同步：

```text
.pkl.gz   三维对象地图
.json     节点、边和配置
.rrd      Rerun 可视化记录
.log      运行日志
少量检测示例图
```

推荐服务器至少准备：

- 4090 24GB 或同级 GPU；
- 64GB 内存更稳；
- 150GB 可用磁盘；
- Ubuntu 20.04/22.04；
- Conda。

---

# 1. 最终目录结构

## 1.1 本地 Ubuntu

```text
~/conceptgraphs/
├── code/
│   ├── concept-graphs-ali/
│   └── concept-graphs-main/
├── envs/
│   └── cg-vis/
├── results/
│   ├── ali-dev/
│   └── main/
├── logs/
├── scripts/
└── README_REPRODUCE.md
```

创建：

```bash
# 【本地】
mkdir -p ~/conceptgraphs/{code,envs,results/ali-dev,results/main,logs,scripts}
cd ~/conceptgraphs
```

## 1.2 服务器

```text
/root/autodl-tmp/conceptgraphs/
├── code/
├── data/
│   ├── Replica/
│   └── ReplicaSemanticGT/
├── models/
├── envs/
│   ├── cg-ali/
│   └── cg-main/
├── logs/
├── outputs/
└── scripts/
```

---

# 2. 本地配置 SSH

以下均在 **本地 Ubuntu** 执行。

编辑：

```bash
nano ~/.ssh/config
```

加入：

```sshconfig
Host cg-server
    HostName 替换成服务器地址
    User root
    Port 替换成SSH端口
    ServerAliveInterval 60
    ServerAliveCountMax 10
    TCPKeepAlive yes
    Compression yes
```

设置权限并测试：

```bash
chmod 600 ~/.ssh/config
ssh cg-server
```

可选免密：

```bash
ssh-keygen -t ed25519 -C "conceptgraphs"
ssh-copy-id cg-server
```

---

# 3. 服务器初始化

连接：

```bash
# 【本地】
ssh cg-server
```

检查：

```bash
# 【服务器】
nvidia-smi
df -h
df -h /root/autodl-tmp
free -h
conda --version
```

创建目录：

```bash
# 【服务器】
export CG_WORK=/root/autodl-tmp/conceptgraphs

mkdir -p "$CG_WORK"/{code,data,models,envs,logs,outputs,scripts}
```

写入 `~/.bashrc`：

```bash
cat >> ~/.bashrc <<'EOF'

export CG_WORK=/root/autodl-tmp/conceptgraphs
EOF

source ~/.bashrc
```

安装系统包：

```bash
apt update

apt install -y \
  git git-lfs wget curl unzip rsync tmux aria2 \
  build-essential gcc g++ cmake ninja-build \
  ffmpeg \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libegl1 libglfw3 libglfw3-dev \
  libxkbcommon-x11-0 libxcb-xinerama0

git lfs install
```

启动 tmux：

```bash
tmux new -s cg
```

离开但保持运行：

```text
Ctrl+B，然后按 D
```

恢复：

```bash
tmux attach -t cg
```

---

# 4. 服务器下载 Replica

```bash
# 【服务器】
cd "$CG_WORK/data"

aria2c -x 8 -s 8 -c \
  https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip \
  -o Replica.zip
```

或：

```bash
wget -c \
  https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
```

验证并解压：

```bash
ls -lh Replica.zip
unzip -t Replica.zip | tail -20
unzip Replica.zip
```

检查：

```bash
find "$CG_WORK/data/Replica/room0" -maxdepth 2 -type f | head -20
```

应看到：

```text
Replica/room0/traj.txt
Replica/room0/results/frame000000.jpg
Replica/room0/results/depth000000.png
```

检查帧数：

```bash
RGB_COUNT=$(find "$CG_WORK/data/Replica/room0/results" -name 'frame*.jpg' | wc -l)
DEPTH_COUNT=$(find "$CG_WORK/data/Replica/room0/results" -name 'depth*.png' | wc -l)
POSE_COUNT=$(wc -l < "$CG_WORK/data/Replica/room0/traj.txt")

echo "RGB=$RGB_COUNT"
echo "DEPTH=$DEPTH_COUNT"
echo "POSE=$POSE_COUNT"
```

查看空间：

```bash
du -sh "$CG_WORK/data/Replica"
du -sh "$CG_WORK/data/Replica"/*
df -h "$CG_WORK"
```

确认正常后删除压缩包：

```bash
rm "$CG_WORK/data/Replica.zip"
```

---

# 5. 阶段 A：安装 ali-dev 环境

`ali-dev` 是官方重构后的快速版本，适合先跑通建图和可视化。

## 5.1 创建环境

```bash
# 【服务器】
conda create \
  -p "$CG_WORK/envs/cg-ali" \
  python=3.10 -y

conda activate "$CG_WORK/envs/cg-ali"
```

检查：

```bash
which python
python --version
```

## 5.2 安装 PyTorch

官方测试组合：

```bash
conda install -y \
  pytorch==2.0.1 \
  torchvision==0.15.2 \
  torchaudio==2.0.2 \
  pytorch-cuda=11.8 \
  -c pytorch -c nvidia
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

## 5.3 安装 Faiss 和 PyTorch3D

```bash
conda install -y \
  -c pytorch \
  faiss-cpu=1.7.4 \
  mkl=2021 \
  blas=1.0=mkl

conda install -y \
  https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
```

验证：

```bash
python - <<'PY'
import faiss
import pytorch3d
print("faiss:", faiss.__version__)
print("pytorch3d:", pytorch3d.__version__)
PY
```

## 5.4 安装依赖

```bash
python -m pip install --upgrade pip setuptools wheel ninja

python -m pip install \
  tyro open_clip_torch wandb h5py \
  "openai>=1,<2" \
  hydra-core omegaconf distinctipy \
  ultralytics dill supervision open3d \
  imageio imageio-ffmpeg natsort kornia \
  rerun-sdk pyliblzfse pypng pyquaternion \
  scikit-learn scipy opencv-python

python -m pip install \
  git+https://github.com/ultralytics/CLIP.git
```

## 5.5 克隆 ali-dev

```bash
cd "$CG_WORK/code"

git clone https://github.com/concept-graphs/concept-graphs.git \
  concept-graphs-ali

cd concept-graphs-ali
git checkout ali-dev

git rev-parse HEAD | tee "$CG_WORK/logs/ali_dev_commit.txt"

python -m pip install -e .
```

验证：

```bash
python - <<'PY'
import torch
import open3d
import rerun
import conceptgraph
print("ali-dev import OK")
print("torch:", torch.__version__)
print("open3d:", open3d.__version__)
PY
```

保存环境：

```bash
conda env export --from-history \
  > "$CG_WORK/logs/cg_ali_history.yml"

python -m pip freeze \
  > "$CG_WORK/logs/cg_ali_pip_freeze.txt"
```

---

# 6. 配置 ali-dev

## 6.1 设置路径

```bash
# 【服务器】
cd "$CG_WORK/code/concept-graphs-ali/conceptgraph/hydra_configs"

cp base_paths.yaml base_paths.yaml.original
cp rerun_realtime_mapping.yaml rerun_realtime_mapping.yaml.original
```

写入：

```bash
cat > base_paths.yaml <<EOF
repo_root: $CG_WORK/code/concept-graphs-ali
data_root: $CG_WORK/data
EOF
```

检查：

```bash
cat base_paths.yaml
```

## 6.2 确保使用 Replica

编辑：

```bash
nano rerun_realtime_mapping.yaml
```

开头应为：

```yaml
defaults:
  - base
  - base_mapping
  - replica
  - sam
  - classes
  - logging_level
  - _self_
```

若出现 `record3d`，改成 `replica`。

## 6.3 Smoke Test 配置建议

```yaml
detections_exp_suffix: smoke_detections_stride20
force_detection: true
save_detections: true

make_edges: false

start: 0
end: 100
stride: 20

exp_suffix: smoke_mapping_stride20

save_video: false
save_objects_all_frames: false

use_rerun: false
save_rerun: true

downsample_voxel_size: 0.025
obj_pcd_max_points: 3000

debug_render: false
vis_render: false

denoise_interval: 10
filter_interval: 10
merge_interval: 10

run_denoise_final_frame: true
run_filter_final_frame: true
run_merge_final_frame: true

dbscan_remove_noise: true
dbscan_eps: 0.1
dbscan_min_points: 10

obj_min_points: 0
obj_min_detections: 1
```

服务器上关闭实时 GUI：

```text
use_rerun: false
```

但保存 Rerun 文件：

```text
save_rerun: true
```

---

# 7. 服务器 Smoke Test

开一个额外终端监控：

```bash
# 【本地另一个终端】
ssh cg-server
watch -n 1 nvidia-smi
```

正式运行：

```bash
# 【服务器 tmux】
conda activate "$CG_WORK/envs/cg-ali"

cd "$CG_WORK/code/concept-graphs-ali/conceptgraph"

export HYDRA_FULL_ERROR=1

python slam/rerun_realtime_mapping.py \
  end=100 \
  stride=20 \
  make_edges=false \
  use_rerun=false \
  save_rerun=true \
  force_detection=true \
  detections_exp_suffix=smoke_detections_stride20 \
  exp_suffix=smoke_mapping_stride20 \
  2>&1 | tee "$CG_WORK/logs/ali_smoke_room0.log"
```

第一次可能自动下载：

- YOLO-World；
- SAM；
- OpenCLIP。

检查输出：

```bash
find "$CG_WORK/data/Replica/room0/exps" \
  -maxdepth 3 -type f | sort | head -100
```

应出现类似：

```text
room0/exps/smoke_detections_stride20/
room0/exps/smoke_mapping_stride20/
```

检查关键文件：

```bash
find "$CG_WORK/data/Replica/room0/exps" \
  \( -name '*.pkl.gz' -o -name '*.rrd' -o -name '*.json' \) \
  -type f | sort
```

复制少量检测图用于本地查看：

```bash
mkdir -p "$CG_WORK/outputs/smoke_vis"

find "$CG_WORK/data/Replica/room0/exps/smoke_detections_stride20/vis" \
  -type f | head -5 | \
  xargs -I{} cp "{}" "$CG_WORK/outputs/smoke_vis/"
```

Smoke Test 成功标准：

```text
[ ] 数据读取成功
[ ] 检测/分割生成
[ ] 没有持续 OOM
[ ] 生成 pkl.gz
[ ] 配置 JSON 保存
[ ] 程序正常结束
```

---

# 8. 服务器完整运行 room0

先使用 `stride=10`：

```bash
# 【服务器】
conda activate "$CG_WORK/envs/cg-ali"
cd "$CG_WORK/code/concept-graphs-ali/conceptgraph"

python slam/rerun_realtime_mapping.py \
  start=0 \
  end=-1 \
  stride=10 \
  make_edges=false \
  force_detection=true \
  save_detections=true \
  detections_exp_suffix=room0_detections_stride10 \
  exp_suffix=room0_mapping_stride10 \
  use_rerun=false \
  save_rerun=true \
  save_video=false \
  save_objects_all_frames=false \
  obj_pcd_max_points=5000 \
  2>&1 | tee "$CG_WORK/logs/ali_room0_stride10.log"
```

## 8.1 复用检测缓存

后续改融合参数时：

```bash
python slam/rerun_realtime_mapping.py \
  start=0 \
  end=-1 \
  stride=10 \
  make_edges=false \
  force_detection=false \
  detections_exp_suffix=room0_detections_stride10 \
  exp_suffix=room0_mapping_stride10_retry \
  use_rerun=false \
  save_rerun=true \
  2>&1 | tee "$CG_WORK/logs/ali_room0_stride10_retry.log"
```

`detections_exp_suffix` 必须与已有检测目录一致。

## 8.2 小规模测试关系边

设置 API：

```bash
export OPENAI_API_KEY="替换成你的密钥"
```

运行：

```bash
python slam/rerun_realtime_mapping.py \
  end=100 \
  stride=20 \
  make_edges=true \
  force_detection=false \
  detections_exp_suffix=smoke_detections_stride20 \
  exp_suffix=smoke_mapping_edges \
  use_rerun=false \
  save_rerun=true \
  2>&1 | tee "$CG_WORK/logs/ali_smoke_edges.log"
```

注意：历史代码中的模型名称可能失效。需要替换成当前账户可用的视觉模型，并记录实际模型、日期和 Prompt。更换 API 后不能声称是严格同模型复现。

---

# 9. 安全停止和结果检查

ali-dev 支持安全提前退出：

```bash
# 【服务器】
cd "$CG_WORK/code/concept-graphs-ali/conceptgraph/hydra_configs"
nano early_exit.json
```

设置：

```json
{
  "exit_early": true
}
```

保存后，程序会完成当前迭代并保存结果。

下次运行前恢复：

```json
{
  "exit_early": false
}
```

查找最新地图：

```bash
find "$CG_WORK/data/Replica/room0/exps" \
  -name '*.pkl.gz' -type f -printf '%T@ %p\n' | \
  sort -nr | head -20
```

查找 Rerun：

```bash
find "$CG_WORK/data/Replica/room0/exps" \
  -name '*.rrd' -type f -printf '%T@ %p\n' | \
  sort -nr | head -20
```

---

# 10. 同步结果到本地

以下在 **本地 Ubuntu** 执行。

同步指定实验：

```bash
mkdir -p ~/conceptgraphs/results/ali-dev/room0

rsync -avP \
  cg-server:/root/autodl-tmp/conceptgraphs/data/Replica/room0/exps/room0_mapping_stride10/ \
  ~/conceptgraphs/results/ali-dev/room0/room0_mapping_stride10/
```

只同步关键文件：

```bash
mkdir -p ~/conceptgraphs/results/ali-dev/room0/key_files

rsync -avP \
  --include='*.pkl.gz' \
  --include='*.json' \
  --include='*.rrd' \
  --include='*.mp4' \
  --exclude='*' \
  cg-server:/root/autodl-tmp/conceptgraphs/data/Replica/room0/exps/room0_mapping_stride10/ \
  ~/conceptgraphs/results/ali-dev/room0/key_files/
```

同步检测样例：

```bash
mkdir -p ~/conceptgraphs/results/ali-dev/room0/smoke_vis

rsync -avP \
  cg-server:/root/autodl-tmp/conceptgraphs/outputs/smoke_vis/ \
  ~/conceptgraphs/results/ali-dev/room0/smoke_vis/
```

同步日志：

```bash
mkdir -p ~/conceptgraphs/logs/server

rsync -avP \
  cg-server:/root/autodl-tmp/conceptgraphs/logs/ \
  ~/conceptgraphs/logs/server/
```

---

# 11. 本地 Rerun 可视化

这是本地最轻量方案。

```bash
# 【本地】
conda create \
  -p ~/conceptgraphs/envs/cg-vis \
  python=3.10 -y

conda activate ~/conceptgraphs/envs/cg-vis

python -m pip install --upgrade pip
python -m pip install rerun-sdk
```

查找文件：

```bash
find ~/conceptgraphs/results/ali-dev \
  -name '*.rrd' -type f
```

打开：

```bash
rerun \
  ~/conceptgraphs/results/ali-dev/room0/key_files/替换成实际文件.rrd
```

可查看：

- RGB；
- 深度；
- 相机轨迹；
- 三维对象；
- 包围盒；
- 时间序列；
- 已生成的关系边。

---

# 12. 本地 Open3D 可视化和文本查询

## 12.1 本地克隆 ali-dev

```bash
# 【本地】
cd ~/conceptgraphs/code

git clone https://github.com/concept-graphs/concept-graphs.git \
  concept-graphs-ali

cd concept-graphs-ali
git checkout ali-dev
```

最好与服务器使用同一 commit：

```bash
SERVER_COMMIT=$(ssh cg-server \
  "cat /root/autodl-tmp/conceptgraphs/logs/ali_dev_commit.txt")

git checkout "$SERVER_COMMIT"
```

## 12.2 安装本地依赖

```bash
conda activate ~/conceptgraphs/envs/cg-vis

python -m pip install \
  open3d open_clip_torch \
  hydra-core omegaconf distinctipy \
  scikit-learn scipy pyquaternion \
  pillow opencv-python faiss-cpu
```

有 NVIDIA GPU：

```bash
python -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

仅 CPU：

```bash
python -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cpu
```

安装代码：

```bash
cd ~/conceptgraphs/code/concept-graphs-ali
python -m pip install -e .
```

## 12.3 打开三维对象地图

```bash
find ~/conceptgraphs/results/ali-dev \
  -name '*.pkl.gz' -type f
```

运行：

```bash
cd ~/conceptgraphs/code/concept-graphs-ali

python conceptgraph/scripts/visualize_cfslam_results.py \
  --result_path \
  ~/conceptgraphs/results/ali-dev/room0/key_files/替换成实际文件.pkl.gz
```

常用按键：

- `r`：RGB；
- `i`：实例 ID；
- `f`：文本查询；
- `b`：背景开关，部分结果支持；
- `c`：类别颜色，检测器版本支持。

按 `f` 后在终端输入：

```text
cabinet
```

继续测试：

```text
sofa
pillow
table
chair
lamp
```

本地 GUI 报错时：

```bash
export DISPLAY=:0
export XKB_CONFIG_ROOT=/usr/share/X11/xkb
```

Qt `xcb` 冲突：

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install --force-reinstall opencv-python
```

---

# 13. 阶段 B：main 分支论文复现环境

完成 ali-dev 后再开始，避免同时和两个时代的依赖搏斗。

## 13.1 创建 main 环境

```bash
# 【服务器】
conda create \
  -p "$CG_WORK/envs/cg-main" \
  python=3.10 -y

conda activate "$CG_WORK/envs/cg-main"
```

安装官方测试组合：

```bash
conda install -y \
  pytorch==2.0.1 \
  torchvision==0.15.2 \
  torchaudio==2.0.2 \
  pytorch-cuda=11.8 \
  -c pytorch -c nvidia

conda install -y \
  -c pytorch \
  faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

conda install -y \
  https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
```

依赖：

```bash
python -m pip install --upgrade pip setuptools wheel ninja

python -m pip install \
  tyro open_clip_torch wandb h5py \
  "openai>=1,<2" \
  hydra-core omegaconf distinctipy \
  ultralytics dill supervision open3d \
  imageio imageio-ffmpeg natsort kornia \
  pyliblzfse pypng pyquaternion \
  scikit-learn scipy opencv-python
```

## 13.2 准备 CUDA 编译工具

检查：

```bash
which nvcc
nvcc --version
ls -l /usr/local/cuda || true
```

系统有 CUDA：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

没有 `nvcc` 时按官方旧方案：

```bash
conda install -y -c conda-forge cudatoolkit-dev

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

## 13.3 ChamferDist

```bash
cd "$CG_WORK/code"

git clone https://github.com/krrish94/chamferdist.git
cd chamferdist

export MAX_JOBS=4
python -m pip install . --no-build-isolation
```

验证：

```bash
python -c "import chamferdist; print('chamferdist OK')"
```

## 13.4 GradSLAM

```bash
cd "$CG_WORK/code"

git clone https://github.com/gradslam/gradslam.git
cd gradslam
git checkout conceptfusion

python -m pip install -e .
```

验证：

```bash
python -c "import gradslam; print('gradslam OK')"
```

## 13.5 main 代码

```bash
cd "$CG_WORK/code"

git clone https://github.com/concept-graphs/concept-graphs.git \
  concept-graphs-main

cd concept-graphs-main
git checkout main

git rev-parse HEAD | tee "$CG_WORK/logs/main_commit.txt"

python -m pip install -e .
```

---

# 14. Grounded-SAM 和权重

```bash
# 【服务器】
cd "$CG_WORK/code"

git clone \
  https://github.com/IDEA-Research/Grounded-Segment-Anything.git

cd Grounded-Segment-Anything

git checkout a4d76a2b55e348943cba4cd57d7553c354296223
```

按照该提交 README 安装 GroundingDINO、SAM、RAM 等依赖。

设置：

```bash
export GSA_PATH="$CG_WORK/code/Grounded-Segment-Anything"
export PYTHONPATH="$GSA_PATH:${PYTHONPATH:-}"
```

需要的权重：

```text
ram_swin_large_14m.pth
groundingdino_swint_ogc.pth
sam_vit_h_4b8939.pth
```

可选：

```text
tag2text_swin_14m.pth
```

权重下载以该提交 README 为准。

设置 main 变量：

```bash
export REPLICA_ROOT="$CG_WORK/data/Replica"
export CG_FOLDER="$CG_WORK/code/concept-graphs-main"
export REPLICA_CONFIG_PATH="$CG_FOLDER/conceptgraph/dataset/dataconfigs/replica/replica.yaml"
```

检查：

```bash
echo "$REPLICA_ROOT"
echo "$CG_FOLDER"
echo "$REPLICA_CONFIG_PATH"
echo "$GSA_PATH"
```

---

# 15. main：二维分割和三维对象地图

进入：

```bash
# 【服务器】
conda activate "$CG_WORK/envs/cg-main"

cd "$CG_FOLDER/conceptgraph"

SCENE_NAME=room0
```

## 15.1 几何 Sanity Check

服务器无 GUI，不使用 `--visualize`：

```bash
python scripts/run_slam_rgb.py \
  --dataset_root "$REPLICA_ROOT" \
  --dataset_config "$REPLICA_CONFIG_PATH" \
  --scene_id "$SCENE_NAME" \
  --image_height 480 \
  --image_width 640 \
  --stride 20 \
  2>&1 | tee "$CG_WORK/logs/main_room0_rgb_sanity.log"
```

## 15.2 原始 ConceptGraphs 二维掩码

先快速测试：

```bash
python scripts/generate_gsa_results.py \
  --dataset_root "$REPLICA_ROOT" \
  --dataset_config "$REPLICA_CONFIG_PATH" \
  --scene_id "$SCENE_NAME" \
  --class_set none \
  --stride 20 \
  2>&1 | tee "$CG_WORK/logs/main_room0_gsa_none_stride20.log"
```

正式论文参数：

```bash
python scripts/generate_gsa_results.py \
  --dataset_root "$REPLICA_ROOT" \
  --dataset_config "$REPLICA_CONFIG_PATH" \
  --scene_id "$SCENE_NAME" \
  --class_set none \
  --stride 5 \
  2>&1 | tee "$CG_WORK/logs/main_room0_gsa_none_stride5.log"
```

输出：

```text
$REPLICA_ROOT/room0/gsa_vis_none
$REPLICA_ROOT/room0/gsa_detections_none
```

## 15.3 ConceptGraphs-Detect

```bash
CLASS_SET=ram

python scripts/generate_gsa_results.py \
  --dataset_root "$REPLICA_ROOT" \
  --dataset_config "$REPLICA_CONFIG_PATH" \
  --scene_id "$SCENE_NAME" \
  --class_set "$CLASS_SET" \
  --box_threshold 0.2 \
  --text_threshold 0.2 \
  --stride 5 \
  --add_bg_classes \
  --accumu_classes \
  --exp_suffix withbg_allclasses \
  2>&1 | tee "$CG_WORK/logs/main_room0_gsa_ram.log"
```

## 15.4 原始 ConceptGraphs 三维建图

```bash
THRESHOLD=1.2

python slam/cfslam_pipeline_batch.py \
  dataset_root="$REPLICA_ROOT" \
  dataset_config="$REPLICA_CONFIG_PATH" \
  stride=5 \
  scene_id="$SCENE_NAME" \
  spatial_sim_type=overlap \
  mask_conf_threshold=0.95 \
  match_method=sim_sum \
  sim_threshold="$THRESHOLD" \
  dbscan_eps=0.1 \
  gsa_variant=none \
  class_agnostic=True \
  skip_bg=True \
  max_bbox_area_ratio=0.5 \
  save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
  merge_interval=20 \
  merge_visual_sim_thresh=0.8 \
  merge_text_sim_thresh=0.8 \
  2>&1 | tee "$CG_WORK/logs/main_room0_mapping_cg.log"
```

## 15.5 ConceptGraphs-Detect 三维建图

```bash
THRESHOLD=1.2

python slam/cfslam_pipeline_batch.py \
  dataset_root="$REPLICA_ROOT" \
  dataset_config="$REPLICA_CONFIG_PATH" \
  stride=5 \
  scene_id="$SCENE_NAME" \
  spatial_sim_type=overlap \
  mask_conf_threshold=0.25 \
  match_method=sim_sum \
  sim_threshold="$THRESHOLD" \
  dbscan_eps=0.1 \
  gsa_variant=ram_withbg_allclasses \
  skip_bg=False \
  max_bbox_area_ratio=0.5 \
  save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1 \
  2>&1 | tee "$CG_WORK/logs/main_room0_mapping_cgd.log"
```

查看：

```bash
find "$REPLICA_ROOT/room0/pcd_saves" \
  -type f -name '*.pkl.gz' \
  -printf '%T@ %p\n' | sort -nr
```

优先使用 `_post` 结果。

---

# 16. LLaVA 节点描述和场景图

## 16.1 安装论文对应 LLaVA

```bash
# 【服务器】
cd "$CG_WORK/code"

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

git checkout 8fc54a09a6be74b2abd913c468fb3d42ae826194

conda activate "$CG_WORK/envs/cg-main"
python -m pip install -e .
```

准备模型目录：

```bash
mkdir -p "$CG_WORK/models/llava/LLaVA-7B-v0"
```

设置：

```bash
export LLAVA_PYTHON_PATH="$CG_WORK/code/LLaVA"
export LLAVA_CKPT_PATH="$CG_WORK/models/llava/LLaVA-7B-v0"
export PYTHONPATH="$LLAVA_PYTHON_PATH:${PYTHONPATH:-}"
```

按照旧 LLaVA README 下载 `LLaVA-7B-v0`。

## 16.2 旧版兼容修复

备份：

```bash
cp "$CG_WORK/code/LLaVA/llava/mm_utils.py" \
  "$CG_WORK/code/LLaVA/llava/mm_utils.py.original"
```

把：

```python
if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
    return True
```

改为：

```python
if torch.equal(
    output_ids[0, -keyword_id.shape[0]:],
    keyword_id
):
    return True
```

## 16.3 生成场景图

设置 API：

```bash
export OPENAI_API_KEY="替换成你的密钥"
```

选择实际 `_post` 文件：

```bash
SCENE_NAME=room0
PKL_FILENAME="替换成实际_post文件名.pkl.gz"
```

确认：

```bash
test -f "$REPLICA_ROOT/$SCENE_NAME/pcd_saves/$PKL_FILENAME" \
  && echo "map file OK"
```

提取节点描述：

```bash
cd "$CG_FOLDER/conceptgraph"

python scenegraph/build_scenegraph_cfslam.py \
  --mode extract-node-captions \
  --cachedir "$REPLICA_ROOT/$SCENE_NAME/sg_cache" \
  --mapfile "$REPLICA_ROOT/$SCENE_NAME/pcd_saves/$PKL_FILENAME" \
  2>&1 | tee "$CG_WORK/logs/main_room0_extract_captions.log"
```

汇总：

```bash
python scenegraph/build_scenegraph_cfslam.py \
  --mode refine-node-captions \
  --cachedir "$REPLICA_ROOT/$SCENE_NAME/sg_cache" \
  --mapfile "$REPLICA_ROOT/$SCENE_NAME/pcd_saves/$PKL_FILENAME" \
  2>&1 | tee "$CG_WORK/logs/main_room0_refine_captions.log"
```

生成关系边：

```bash
python scenegraph/build_scenegraph_cfslam.py \
  --mode build-scenegraph \
  --cachedir "$REPLICA_ROOT/$SCENE_NAME/sg_cache" \
  --mapfile "$REPLICA_ROOT/$SCENE_NAME/pcd_saves/$PKL_FILENAME" \
  2>&1 | tee "$CG_WORK/logs/main_room0_build_scenegraph.log"
```

检查：

```bash
find "$REPLICA_ROOT/$SCENE_NAME/sg_cache" \
  -maxdepth 3 -type f | sort
```

重点：

```text
sg_cache/map/scene_map_cfslam_pruned.pkl.gz
sg_cache/cfslam_object_relations.json
```

---

# 17. Replica 语义分割评估

下载官方 README 指向的 Replica 语义点云真值，解压到：

```text
/root/autodl-tmp/conceptgraphs/data/ReplicaSemanticGT
```

设置：

```bash
# 【服务器】
export REPLICA_SEMANTIC_ROOT="$CG_WORK/data/ReplicaSemanticGT"
```

评估 ConceptGraphs：

```bash
cd "$CG_FOLDER/conceptgraph"

python scripts/eval_replica_semseg.py \
  --replica_root "$REPLICA_ROOT" \
  --replica_semantic_root "$REPLICA_SEMANTIC_ROOT" \
  --n_exclude 6 \
  --pred_exp_name \
  none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub \
  2>&1 | tee "$CG_WORK/logs/main_eval_cg.log"
```

评估 CG-D：

```bash
python scripts/eval_replica_semseg.py \
  --replica_root "$REPLICA_ROOT" \
  --replica_semantic_root "$REPLICA_SEMANTIC_ROOT" \
  --n_exclude 6 \
  --pred_exp_name \
  ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_masksub \
  2>&1 | tee "$CG_WORK/logs/main_eval_cgd.log"
```

对应：

```text
mrecall = mAcc
fmiou   = F-mIoU
```

论文参考值：

| 方法 | mAcc | F-mIoU |
|---|---:|---:|
| ConceptGraphs | 40.63 | 35.95 |
| ConceptGraphs-Detector | 38.72 | 35.82 |

---

# 18. main 结果同步和本地可视化

## 18.1 同步对象地图

```bash
# 【本地】
mkdir -p ~/conceptgraphs/results/main/room0/pcd_saves

rsync -avP \
  --include='*_post*.pkl.gz' \
  --include='*post.pkl.gz' \
  --exclude='*' \
  cg-server:/root/autodl-tmp/conceptgraphs/data/Replica/room0/pcd_saves/ \
  ~/conceptgraphs/results/main/room0/pcd_saves/
```

## 18.2 同步场景图

```bash
mkdir -p ~/conceptgraphs/results/main/room0/sg_cache

rsync -avP \
  cg-server:/root/autodl-tmp/conceptgraphs/data/Replica/room0/sg_cache/ \
  ~/conceptgraphs/results/main/room0/sg_cache/
```

## 18.3 本地克隆 main

```bash
cd ~/conceptgraphs/code

git clone https://github.com/concept-graphs/concept-graphs.git \
  concept-graphs-main

cd concept-graphs-main
git checkout main
```

切到服务器相同 commit：

```bash
SERVER_COMMIT=$(ssh cg-server \
  "cat /root/autodl-tmp/conceptgraphs/logs/main_commit.txt")

git checkout "$SERVER_COMMIT"
```

安装：

```bash
conda activate ~/conceptgraphs/envs/cg-vis

cd ~/conceptgraphs/code/concept-graphs-main
python -m pip install -e .
```

打开对象地图：

```bash
cd ~/conceptgraphs/code/concept-graphs-main/conceptgraph

python scripts/visualize_cfslam_results.py \
  --result_path \
  ~/conceptgraphs/results/main/room0/pcd_saves/替换成实际_post文件.pkl.gz
```

打开场景图：

```bash
python scripts/visualize_cfslam_results.py \
  --result_path \
  ~/conceptgraphs/results/main/room0/sg_cache/map/scene_map_cfslam_pruned.pkl.gz \
  --edge_file \
  ~/conceptgraphs/results/main/room0/sg_cache/cfslam_object_relations.json
```

按：

```text
g 显示关系图
r RGB
i 实例颜色
f 文本查询
+/- 调整点大小
```

---

# 19. 实验记录和同步脚本

## 19.1 服务器运行记录

```bash
# 【服务器】
RUN_NAME="room0_stride10_baseline"
RUN_DIR="$CG_WORK/logs/$RUN_NAME"

mkdir -p "$RUN_DIR"

date > "$RUN_DIR/date.txt"
nvidia-smi > "$RUN_DIR/nvidia_smi.txt"
df -h > "$RUN_DIR/disk.txt"
free -h > "$RUN_DIR/memory.txt"

git -C "$CG_WORK/code/concept-graphs-ali" \
  rev-parse HEAD > "$RUN_DIR/git_commit.txt"

python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
```

## 19.2 本地自动同步脚本

```bash
# 【本地】
nano ~/conceptgraphs/scripts/sync_ali_results.sh
```

写入：

```bash
#!/usr/bin/env bash

set -euo pipefail

REMOTE_ROOT=/root/autodl-tmp/conceptgraphs
LOCAL_ROOT="$HOME/conceptgraphs"

mkdir -p "$LOCAL_ROOT/results/ali-dev/room0"
mkdir -p "$LOCAL_ROOT/logs/server"

rsync -avP \
  --include='*/' \
  --include='*.pkl.gz' \
  --include='*.json' \
  --include='*.rrd' \
  --include='*.log' \
  --exclude='*' \
  "cg-server:$REMOTE_ROOT/data/Replica/room0/exps/" \
  "$LOCAL_ROOT/results/ali-dev/room0/"

rsync -avP \
  "cg-server:$REMOTE_ROOT/logs/" \
  "$LOCAL_ROOT/logs/server/"
```

授权：

```bash
chmod +x ~/conceptgraphs/scripts/sync_ali_results.sh
```

运行：

```bash
~/conceptgraphs/scripts/sync_ali_results.sh
```

---

# 20. 常见错误

## 20.1 CUDA OOM

依次处理：

```text
1. end=100、stride=20
2. use_rerun=false
3. save_objects_all_frames=false
4. obj_pcd_max_points=2000
5. 使用 MobileSAM
6. 检查残留 Python 进程
```

查看：

```bash
nvidia-smi
ps aux | grep python
```

## 20.2 `ModuleNotFoundError: conceptgraph`

```bash
cd "$CG_WORK/code/concept-graphs-ali"
python -m pip install -e .
```

main 同理。

## 20.3 PyTorch 看不到 GPU

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

检查是否误装 CPU 版本。

## 20.4 PyTorch3D `undefined symbol`

目标组合：

```text
PyTorch 2.0.1
CUDA 11.8
PyTorch3D 0.7.4 py310 cu118 pyt201
```

## 20.5 `nvcc not found`

系统 CUDA：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

Conda CUDA：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
```

## 20.6 ChamferDist 编译失败

```bash
cd "$CG_WORK/code/chamferdist"

rm -rf build dist *.egg-info

export MAX_JOBS=2

python -m pip install . --no-build-isolation -v
```

## 20.7 数据集长度为 0

检查：

```bash
ls "$CG_WORK/data/Replica/room0/results"/frame*.jpg | head
ls "$CG_WORK/data/Replica/room0/results"/depth*.png | head
ls "$CG_WORK/data/Replica/room0/traj.txt"
```

警惕多解压一层：

```text
Replica/Replica/room0
```

## 20.8 找不到检测缓存

`force_detection=false` 时：

```text
detections_exp_suffix
```

必须与已有检测目录完全相同。

## 20.9 API 导入错误

新接口：

```python
from openai import OpenAI
```

需要：

```bash
python -m pip uninstall -y openai
python -m pip install "openai>=1,<2"
```

旧 main 代码可能使用旧 SDK。新旧 API 冲突时保持两个独立环境，不要继续把依赖搅成考古遗址。

## 20.10 三维重影

优先检查：

```text
深度
位姿
相机内参
坐标系
帧顺序
```

不要先调 CLIP 阈值。语义模型没有义务修复坐标系。

---

# 21. 空间清理

服务器查看大目录：

```bash
du -h --max-depth=3 "$CG_WORK" | sort -h | tail -40
```

清理缓存：

```bash
conda clean -a -y
python -m pip cache purge
```

可以删除：

- 已解压的 ZIP；
- 明确废弃的实验目录；
- 不需要的视频；
- 不需要的 `saved_obj_all_frames`；
- 重复检测缓存。

不要删除：

- 最终 `.pkl.gz`；
- 场景图 JSON；
- 配置；
- 日志；
- 尚未同步的实验；
- 论文表格对应结果。

---

# 22. 验收清单

## ali-dev

```text
[ ] 服务器 Replica 正常
[ ] cg-ali 安装成功
[ ] CUDA 正常
[ ] Smoke Test 有检测图
[ ] Smoke Test 有 pkl.gz
[ ] room0 stride=10 完成
[ ] 生成 .rrd
[ ] 本地 Rerun 可播放
[ ] 本地 Open3D 可显示
[ ] 文本查询 cabinet/sofa 有响应
```

## main

```text
[ ] cg-main 环境成功
[ ] ChamferDist 成功
[ ] GradSLAM 成功
[ ] Grounded-SAM 成功
[ ] SAM/RAM/GroundingDINO 权重齐全
[ ] gsa_vis_none 正常
[ ] gsa_vis_ram_withbg_allclasses 正常
[ ] pcd_saves 生成
[ ] _post 结果可视化
[ ] LLaVA 节点描述生成
[ ] GPT 汇总成功
[ ] 场景图 JSON 生成
[ ] 本地按 g 可显示关系
[ ] Replica mrecall/fmiou 输出
```

---

# 23. 最推荐执行顺序

严格按照：

```text
1. 本地配置 SSH
2. 服务器创建目录
3. 服务器下载 Replica
4. 服务器创建 cg-ali
5. ali-dev Smoke Test
6. 同步 .rrd 和 .pkl.gz
7. 本地 Rerun
8. 本地 Open3D
9. 服务器完整 room0
10. 冻结 ali-dev baseline
11. 新建 cg-main
12. Grounded-SAM
13. 二维分割
14. 三维对象地图
15. LLaVA caption
16. 场景图
17. 语义分割评估
18. 冻结 main baseline
19. 再开始 VLM 驱动改进
```

核心原则：

> **服务器存数据和计算，本地只拉结果；先验证二维感知，再验证三维融合，最后才处理 VLM、LLM 和场景图。**

---

# 参考来源

- ConceptGraphs 官方仓库：`https://github.com/concept-graphs/concept-graphs`
- `ali-dev` 分支：`https://github.com/concept-graphs/concept-graphs/tree/ali-dev`
- 项目主页：`https://concept-graphs.github.io/`
- Replica RGB-D：`https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip`
- 论文：`https://arxiv.org/abs/2309.16650`
