# CGS Habitat 全流程已完成内容总结

> 记录日期：2026-08-03  
> 本地工作区：`/home/abc/conceptgraphs`  
> 服务器工作区：`/home/chenkejun/beauty/conceptgraphs`  
> 已验收主线：Habitat-Sim 本地仿真 → RGB-D/Pose 导出 → 服务器 ConceptGraphs 建图 → map bundle 下载 → 服务器 OpenCLIP 文本查询 → Ubuntu 坐标转换/NavMesh 观察位姿 → 自动导航 → predicted-object Success/SPL

## 1. 结论

目前已经真实跑通一条完整的 smoke 链路，不是只完成环境安装或脚本编写。

```text
Ubuntu 本地 Habitat-Sim
    │  生成 20 帧 RGB + Depth + Camera Pose
    ▼
CGS Habitat v1 序列（READY）
    │  rsync/SSH
    ▼
ConceptGraphs 服务器
    │  接口校验（VALIDATED）
    │  YOLO-World + SAM + OpenCLIP 对象建图
    ▼
map_bundle（COMPLETE，26 个对象）
    │  rsync + SHA-256 校验
    ▼
Ubuntu 本地 Open3D 可视化
    │  服务器 OpenCLIP Top-K 查询
    ▼
Habitat world 目标、NavMesh 可见观察位姿与最短路径
    │  自动执行动作、停止和 Depth 可见性检查
    ▼
Success=1、SPL=1.0、视频、轨迹和确定性签名
```

已验收的关键数据：

| 项目 | 结果 |
|---|---|
| Habitat 场景 | `2azQ1b91cZZ` |
| 序列 ID | `2azQ1b91cZZ_smoke_v001` |
| 图像分辨率 | 640 × 480 |
| RGB / Depth / Pose | 20 / 20 / 20 |
| Depth 格式 | `uint16` PNG，单位毫米，scale=1000 |
| Depth 中位数 | 1385 mm |
| 有效 Depth 平均比例 | 0.999861 |
| 序列 checksum | 64 项通过 |
| Pose 转换单测 | 3/3 通过 |
| 服务器建图帧数 | 20/20 |
| 最终对象数 | 26 |
| NaN 对象中心 | 0 |
| 最大对象中心半径 | 约 10.568 m |
| map bundle checksum | 8 个负载文件全部通过 |
| 本地 pkl 反序列化 | 成功，26 个对象 |
| Habitat 交互窗口 | 已创建并测试 |
| Open3D 对象图窗口 | 已实际开窗并保持运行 |
| `sofa` OpenCLIP Top-1 | `sofa chair`，相似度 0.219635 |
| 自动导航 | 56 步，0 碰撞 |
| predicted-object Success / SPL | 1 / 1.0 |
| 可重复性 | 两次独立运行签名相同 |

## 2. 本次已完成的内容

### 2.1 Ubuntu SSH 服务器连接

最初现象是 Windows 能连接 FRP 服务器，Ubuntu 不能连接。Ubuntu 端已完成以下处理：

- 将 `server-3048-out` 和指南别名 `cg-server` 指向已解析的 FRP 端点 `116.162.205.41:64906`；
- 使用 `BindInterface wlp0s20f3` 绕过 Clash TUN/全局模式对该 SSH 连接的影响；
- 保留 `ServerAliveInterval 60`、`ServerAliveCountMax 10` 和压缩；
- 已完成 SSH 公钥登录，`ssh cg-server` 可免密登录；
- 远端实际身份已确认为 `chenkejun@ubun`。

当前关键 SSH 配置位于 `/home/abc/.ssh/config`。总结文档不保存任何密码或私钥内容。

> 注意：配置中现在使用的是 `frp-van.com` 在本次处理时解析到的 IP。如果 FRP 公网 IP 以后改变，需要更新两个别名的 `HostName`。

### 2.2 本地系统和双 Conda 环境

本地已核对的基础环境：

| 项目 | 当前值 |
|---|---|
| 系统 | Ubuntu 22.04，kernel `6.8.0-136-generic` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| NVIDIA 驱动 | 580.173.02 |
| Habitat 环境 | Conda `habitat`，Python 3.9.23 |
| Habitat-Sim | 0.3.3，headless + Bullet 构建 |
| Habitat-Lab | 0.3.3 系列 |
| ConceptGraphs 环境 | Conda `conceptgraph`，Python 3.10.19 |
| PyTorch | 2.0.1 |
| Open3D | 0.17.0 |
| supervision | 0.14.0 |

Habitat 和 ConceptGraphs 保持在两个独立环境中，避免两套历史依赖相互污染。所有 Habitat 入口脚本都会清理外部 `LD_LIBRARY_PATH` 和 `PYTHONPATH`，避免 ROS/其他 SDK 的动态库导致 EGL 崩溃。

### 2.3 Habitat 场景审计和仿真

本地当前可见的 3 个 MP3D GLB：

- `data/scenes/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb`；
- `data/scenes/mp3d/8194nk5LbLH/8194nk5LbLH.glb`；
- `data/scenes/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb`。

已验收使用的场景为 `2azQ1b91cZZ.glb`，大小 89,064,932 bytes，SHA-256：

```text
982d3028b6fa82ca63930122065597171c8a8e84b17fe4926f67d9489a321444
```

已确认：

- Habitat-Sim 0.3.3 可在 RTX 4060 上通过 EGL 渲染 RGB 和 Depth；
- 场景没有预置 NavMesh 时，可在运行时重新生成 NavMesh；
- PathFinder 可随机采样可导航点；
- 交互查看器可展示左侧 RGB、右侧伪彩色 Depth；
- 支持 W/S/A/D、重置、自动行走、保存 RGB-D 和退出。

当前 GLB 没有 Habitat/MP3D 语义注释资源，因此 `semantic_available=false`。生成的 semantic NPY 只是为保持接口目录结构，不应当作真实语义 GT。

### 2.4 CGS Habitat v1 接口和数据导出

接口主 schema：

```text
/home/abc/conceptgraphs/docs/CGS_Habitat_Interface_v1.schema.json
```

已实现的本地 bridge：

| 文件 | 用途 |
|---|---|
| `CGS/habitat_bridge/local/audit_habitat.py` | 审计场景、RGB/Depth、NavMesh 和 PathFinder |
| `CGS/habitat_bridge/local/pose_utils.py` | Habitat/OpenGL 相机到 OpenCV c2w 坐标变换 |
| `CGS/habitat_bridge/local/export_sequence.py` | 导出 RGB、Depth、Semantic 占位、Pose 和元数据 |
| `CGS/habitat_bridge/local/validate_export.py` | Schema、帧数、Depth、Pose、内参和 checksum 校验 |
| `CGS/habitat_bridge/local/view_habitat.py` | 本地 RGB + Depth 交互可视化 |
| `CGS/habitat_bridge/tests/test_pose_conversion.py` | Pose 转换单元测试 |

已导出序列：

```text
/home/abc/conceptgraphs/data/habitat_exports/2azQ1b91cZZ_smoke_v001
```

序列约定：

- `format_version`: `cgs-habitat-sequence-v1`；
- RGB: `results/frame%06d.jpg`；
- Depth: `results/depth%06d.png`，`uint16`，毫米；
- Pose: `traj.txt`，每帧一个 4×4 row-major camera-to-world 矩阵；
- 相机坐标系: OpenCV；
- 世界坐标系: Habitat；
- ConceptGraphs map frame: 第一帧 OpenCV 相机；
- `T_habitat_world_from_cg_map` 保存在 metadata 和最终 bundle 中，可将对象图坐标转回 Habitat world。

序列通过 schema、RGB/Depth shape、Depth 合理性、Pose 正交性、坐标系首帧一致性、内参一致性和 64 项 SHA-256 校验，然后创建 `READY`。

### 2.5 服务器 ConceptGraphs 管线

已在服务器 `/home/chenkejun/beauty/conceptgraphs` 部署：

| 文件 | 用途 |
|---|---|
| `habitat_bridge/server/validate_sequence.py` | 复用 v1 validator 检查上传数据，成功创建 `VALIDATED` |
| `habitat_bridge/server/generate_dataset_config.py` | 根据 metadata 生成 ConceptGraphs/Replica-compatible 相机配置 |
| `habitat_bridge/server/package_map_bundle.py` | 检查对象图、统计、打包和生成 checksum/`COMPLETE` |
| `scripts/server_env.sh` | 服务器路径、Python、CUDA 和模型缓存环境 |
| `scripts/server_habitat_smoke.sh` | 按 `metadata.json` 中的帧数和分辨率运行 ConceptGraphs smoke mapping 并打包；本次验收序列为 20 帧 |
| `scripts/server_habitat_pipeline.sh` | 串联校验、配置、建图和打包 |

服务器已完成的处理顺序：

1. 接收本地 `READY` 序列；
2. 重新校验 v1 schema 和所有 payload checksum；
3. 创建 `VALIDATED`；
4. 生成 `conceptgraphs_dataset.yaml`；
5. 调用 `slam/rerun_realtime_mapping.py`；
6. 对 20 帧强制重新运行检测并保存 detection；
7. 生成 pkl、对象 JSON 和边 JSON；
8. 检查对象数、NaN 中心和空间半径；
9. 打包 `map_bundle`、生成 checksum 和 `COMPLETE`。

服务器本次映射运行使用 CUDA，涉及 YOLO-World 检测路径、SAM ViT-H 分割和 OpenCLIP ViT-H-14（LAION2B-s32B-b79K）视觉特征。`make_edges=false`，因此本次 `edges.json` 为空数组，这是预期配置，不是丢失数据。

### 2.6 map bundle 和本地回传

服务器标准产物路径：

```text
/home/chenkejun/beauty/conceptgraphs/results/HabitatMP3D/
  2azQ1b91cZZ_smoke_v001/
  2azQ1b91cZZ_smoke_v001_smoke/
  map_bundle/
```

本地下载路径：

```text
/home/abc/conceptgraphs/results/habitat/
  2azQ1b91cZZ_smoke_v001/
  2azQ1b91cZZ_smoke_v001_smoke/
  map_bundle/
```

bundle 文件：

| 文件 | 含义 |
|---|---|
| `object_map.pkl.gz` | ConceptGraphs 可序列化对象点云、bbox、特征和类别 |
| `objects.json` | 轻量对象标签和属性 |
| `edges.json` | 场景图边；本次未启用边生成，所以为 `[]` |
| `sequence_metadata.json` | 原始序列 metadata 副本 |
| `mapping_config.json` | 实际建图参数 |
| `map_manifest.json` | bundle 版本、序列、运行 ID 和坐标变换 |
| `map_statistics.json` | 对象数、NaN、空间范围和类别直方图 |
| `query_classes.txt` | 可查看的去重类别文本 |
| `checksums.sha256` | bundle 负载完整性校验 |
| `COMPLETE` | 打包和质量检查已完成的标记 |

26 个对象的类别统计：

```text
bench 1, cabinet 1, closet door 1, coffee kettle 1, cushion 1,
desk 1, folded chair 1, microwave 1, mirror 2, oven 1,
paper bag 1, piano 1, picture 3, plate 1, power outlet 1,
radiator 2, sofa chair 1, window 5
```

### 2.7 本地可视化和兼容修复

已增加本地对象图可视化入口 `scripts/view_habitat_map.sh`。它会：

1. 检查 `COMPLETE`；
2. 运行 `sha256sum -c checksums.sha256`；
3. 使用 `conceptgraph` Conda 环境；
4. 使用 Open3D 加载 `object_map.pkl.gz`；
5. 默认使用 `--no_clip`，避免本地只看图时加载大型 CLIP 模型。

本地 `supervision==0.14.0` 与项目可视化代码的旧 API 不一致，已在两份本地 ConceptGraphs `utils/vis.py` 中做了最小兼容处理：

- `ColorPalette.DEFAULT` → `ColorPalette.default()`；
- `Color.BLACK` → `Color.black()`。

修复后 Open3D 窗口已实际创建，12 秒验收期间程序一直保持运行，最后由测试超时器关闭。

### 2.8 文本查询到 Habitat 自动导航闭环

已新增服务器 OpenCLIP 对象查询、ConceptGraphs map frame 到 Habitat world frame 转换、NavMesh 可见观察位姿采样、最短路径自动执行、Depth 几何可见性检查、Success/SPL、视频和轨迹归档。

固定 `sofa` Episode 已独立运行两次，两次均为 56 步、0 碰撞、Success=1、SPL=1.0，确定性签名相同：

```text
525e5b003926058f5971e9754fdf659150e1727164880011dff9eff43296d21b
```

当前评价协议为 `predicted_object_visibility_v1`，表示导航到 ConceptGraphs 预测对象的可见观察位姿，不是 Habitat 官方语义 GT ObjectNav。

## 3. 新增的运行脚本

| 脚本 | 运行位置 | 作用 |
|---|---|---|
| `scripts/local_env.sh` | 本地 | 定义本地、服务器、场景和结果路径 |
| `scripts/run_habitat_viewer.sh` | 本地 | 打开 Habitat RGB + Depth 交互窗口 |
| `scripts/run_habitat_smoke.sh` | 本地 | 导出并验证 20 帧 smoke 序列 |
| `scripts/upload_sequence.sh` | 本地 | 通过 rsync 上传 `READY` 序列 |
| `scripts/download_map_bundle.sh` | 本地 | 下载 bundle，自动校验 checksum 和 `COMPLETE` |
| `scripts/run_habitat_end_to_end.sh` | 本地 | 导出/复用、上传、远程建图、下载和 pkl 验证 |
| `scripts/view_habitat_map.sh` | 本地 | 本地 Open3D 查看对象图 |
| `scripts/run_cgs_objectnav.sh` | 本地 | 一条命令完成服务器查询、目标确认、本地导航和最终验收 |
| `scripts/server_env.sh` | 服务器 | 服务器 ConceptGraphs/CUDA 运行环境 |
| `scripts/server_habitat_smoke.sh` | 服务器 | 按序列 metadata 动态配置的 GPU 建图和 bundle 打包；本次验收序列为 20 帧 |
| `scripts/server_habitat_pipeline.sh` | 服务器 | 服务器全流程入口 |
| `scripts/server_query_habitat_map.sh` | 服务器 | 在 RTX 5880 上执行 OpenCLIP Top-K 对象查询 |

## 4. 完整运行指令

### 4.1 立即查看 Habitat 仿真

在 Ubuntu 桌面终端中运行：

```bash
cd /home/abc/conceptgraphs
./scripts/run_habitat_viewer.sh
```

控制键：

- `W` / `S`：前进 / 后退；
- `A` / `D`：左转 / 右转；
- `R`：重新采样一个可导航位置；
- `Space`：开关自动行走；
- `P`：保存当前 RGB 和 Depth；
- `Q` / `Esc`：退出。

启动后立即自动行走：

```bash
cd /home/abc/conceptgraphs
source scripts/local_env.sh
./scripts/run_habitat_viewer.sh "$MP3D_SCENE" walk
```

### 4.2 查看已生成的 26 对象图

```bash
cd /home/abc/conceptgraphs
./scripts/view_habitat_map.sh
```

Open3D 快捷键：

- `R`：RGB 颜色；
- `I`：实例颜色；
- `C`：类别颜色；
- `B`：显示/隐藏背景对象；
- `G`：显示/隐藏场景图边（本次边为空）。

### 4.3 复用已验证序列，重跑服务器全流程

```bash
cd /home/abc/conceptgraphs
./scripts/run_habitat_end_to_end.sh
```

该命令会重新验证本地序列、上传、在服务器运行 20 帧建图、打包、下载并校验。服务器会使用 GPU，且会加载大型模型。

### 4.4 从 Habitat 重新生成帧再跑全流程

```bash
cd /home/abc/conceptgraphs
REEXPORT=1 ./scripts/run_habitat_end_to_end.sh
```

### 4.5 分步手动运行

```bash
cd /home/abc/conceptgraphs
source scripts/local_env.sh

SEQ=2azQ1b91cZZ_smoke_v001
RUN=${SEQ}_smoke

OVERWRITE=1 ./scripts/run_habitat_smoke.sh "$SEQ"
./scripts/upload_sequence.sh "$SEQ"
ssh "$CG_SERVER_ALIAS" \
  "$CG_REMOTE/scripts/server_habitat_pipeline.sh" "$SEQ"
./scripts/download_map_bundle.sh "$SEQ" "$RUN"
./scripts/view_habitat_map.sh "$SEQ" "$RUN"
```

如果使用一个新的 sequence ID，尚无同名导出目录，则可以不加 `OVERWRITE=1`：

```bash
./scripts/run_habitat_smoke.sh "$SEQ"
```

## 5. 验收和排错命令

### 5.1 SSH

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 cg-server \
  'hostname; whoami; pwd'
```

期望至少看到 `ubun` 和 `chenkejun`。

### 5.2 本地序列

```bash
cd /home/abc/conceptgraphs
env -u LD_LIBRARY_PATH -u PYTHONPATH \
  conda run --no-capture-output -n habitat \
  python CGS/habitat_bridge/local/validate_export.py \
  --sequence-dir data/habitat_exports/2azQ1b91cZZ_smoke_v001 \
  --schema docs/CGS_Habitat_Interface_v1.schema.json \
  --mark-ready
```

期望结尾为 `VALIDATION PASSED`。

### 5.3 本地 map bundle

```bash
BUNDLE=/home/abc/conceptgraphs/results/habitat/2azQ1b91cZZ_smoke_v001/2azQ1b91cZZ_smoke_v001_smoke/map_bundle

(cd "$BUNDLE" && sha256sum -c checksums.sha256)
test -f "$BUNDLE/COMPLETE"
cat "$BUNDLE/map_statistics.json"
```

### 5.4 服务器状态

```bash
ssh cg-server '
  source /home/chenkejun/beauty/conceptgraphs/scripts/server_env.sh
  test -f "$HABITAT_SEQUENCE_ROOT/2azQ1b91cZZ_smoke_v001/READY"
  test -f "$HABITAT_SEQUENCE_ROOT/2azQ1b91cZZ_smoke_v001/VALIDATED"
  test -f "$HABITAT_RESULT_ROOT/2azQ1b91cZZ_smoke_v001/2azQ1b91cZZ_smoke_v001_smoke/map_bundle/COMPLETE"
  echo SERVER_STATE_OK
'
```

### 5.5 日志

服务器建图日志：

```text
/home/chenkejun/beauty/conceptgraphs/logs/habitat/
  2azQ1b91cZZ_smoke_v001_smoke_mapping.log
```

查看：

```bash
ssh cg-server \
  'tail -100 /home/chenkejun/beauty/conceptgraphs/logs/habitat/2azQ1b91cZZ_smoke_v001_smoke_mapping.log'
```

## 6. 当前边界和未完成内容

以下内容不应对外声称为已完成：

1. **语义 GT 未具备**  
   当前 `2azQ1b91cZZ.glb` 没有 MP3D semantic scene/instance 注释，所以不能基于 Habitat 语义 GT 做严格 ObjectNav 评价。

2. **Habitat 官方语义 GT ObjectNav 未完成**  
   已跑通 predicted-object 文本查询→目标位姿→路径执行→Success/SPL；但因当前场景没有语义注释，还不是官方 GT evaluator。

3. **当前只是 smoke 轨迹**  
   20 帧是同一可导航点的 360° panorama smoke，用于验证接口和建图管线，不代表全房间覆盖。

4. **映射模式仍是 smoke 配置**  
   服务器已改为从 metadata 动态读取帧数和分辨率，但脚本仍使用 smoke 命名和配置，正式长轨迹需配套全局覆盖 exporter。

5. **本次未建立场景图关系边**  
   建图配置为 `make_edges=false`，因此 `edges.json=[]`。

6. **Open3D 快速查看仍不加载 CLIP**  
   `view_habitat_map.sh` 仍使用 `--no_clip`；真正文本查询已迁移到服务器 `query_object_map.py`，由 `run_cgs_objectnav.sh` 调用。

7. **识别结果是视觉模型输出，不是语义 GT**  
   26 个对象可用于验证管线和可视化，类别准确率还需要人工检查或与正式 GT 比较。

## 7. 后续建议顺序

1. 补齐带 Habitat semantic annotations 的 MP3D 完整场景资源；
2. 将本地导出从 panorama smoke 升级为 NavMesh 全局覆盖轨迹；
3. 补充 Habitat 语义 evaluator，把 predicted-object 协议升级为官方 ObjectNav Success/SPL；
4. 在多场景、多目标、多起点上执行批量验收；
5. 增加场景图关系边和复杂语言目标推理。

## 8. 相关文档

- 完整指南：`docs/CGS_Ubuntu端_Habitat完整仿真与导航指南.md`；
- v1 接口：`docs/CGS_Habitat_Interface_v1.schema.json`；
- 服务器专项总结：`docs/CGS_Habitat_服务器部署与运行总结_2026-08-03.md`。
- 一条命令导航复现：`docs/CGS_ObjectNav_一条命令复现与最终验收指南.md`。
