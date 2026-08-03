# CGS Habitat 服务器部署与运行总结

> 记录日期：2026-08-03  
> 服务器：`chenkejun@ubun`  
> 工作区：`/home/chenkejun/beauty/conceptgraphs`  
> 当前状态：Habitat v1 序列验证、ConceptGraphs GPU 建图、map bundle 打包和回传已跑通

## 1. 服务器结论

服务器端已从“只有 ConceptGraphs/Replica 基线”补齐为可直接接收 Ubuntu Habitat 序列的映射管线。已完成一次真实 GPU 运行：

```text
2azQ1b91cZZ_smoke_v001
  → READY 检查
  → v1 schema + 64 checksums 验证
  → VALIDATED
  → ConceptGraphs 数据集配置
  → 20/20 帧 GPU 对象建图
  → 26 个对象，0 个 NaN 中心
  → map_bundle + checksums.sha256 + COMPLETE
```

在此基础上，服务器已增加 OpenCLIP 文本查询入口。`sofa` 查询的 Top-1 为 `sofa chair`，查询结果已被 Ubuntu 用于完成固定起点至可见对象位姿的自动导航。

## 2. 系统和 GPU 环境

| 项目 | 当前值 |
|---|---|
| Hostname | `ubun` |
| 系统 | Ubuntu 22.04 |
| Kernel | `6.8.0-110-generic` |
| GPU | 8 × NVIDIA RTX 5880 Ada Generation |
| 单卡显存 | 49,140 MiB |
| NVIDIA 驱动 | 580.126.09 |
| ConceptGraphs Python | `/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python` |
| Python | 3.10.20 |
| PyTorch | 2.0.1 |
| CUDA runtime | 11.8 |
| Open3D | 0.19.0 |

`scripts/server_env.sh` 默认设置 `CUDA_VISIBLE_DEVICES=0`，因此当前单次 smoke mapping 只使用一张卡。可在调用前显式指定其他可用卡，例如：

```bash
CUDA_VISIBLE_DEVICES=1 \
  /home/chenkejun/beauty/conceptgraphs/scripts/server_habitat_pipeline.sh \
  2azQ1b91cZZ_smoke_v001
```

## 3. 服务器目录布局

```text
/home/chenkejun/beauty/conceptgraphs/
├── code/
│   └── concept-graphs-ali/             # ConceptGraphs 运行代码
├── envs/
│   └── cg-ali/                        # 服务器 Python 环境
├── habitat_bridge/
│   ├── interface_schema.json          # 服务器 v1 schema 副本
│   ├── local/                         # validator 复用的公共实现
│   └── server/
│       ├── validate_sequence.py
│       ├── generate_dataset_config.py
│       ├── package_map_bundle.py
│       └── query_object_map.py
├── scripts/
│   ├── server_env.sh
│   ├── server_habitat_pipeline.sh
│   ├── server_habitat_smoke.sh
│   └── server_query_habitat_map.sh
├── data/HabitatMP3D/sequences/
│   └── 2azQ1b91cZZ_smoke_v001/
├── results/HabitatMP3D/
│   └── 2azQ1b91cZZ_smoke_v001/
│       └── 2azQ1b91cZZ_smoke_v001_smoke/map_bundle/
├── logs/habitat/
├── results/ObjectNavAcceptance/             # Ubuntu 导航验收证据归档
└── models/                           # 模型缓存，当前约 3.7G
```

## 4. 环境入口 `server_env.sh`

标准启动方式：

```bash
source /home/chenkejun/beauty/conceptgraphs/scripts/server_env.sh
```

关键环境变量：

| 变量 | 值/作用 |
|---|---|
| `CG_WORK` | `/home/chenkejun/beauty/conceptgraphs` |
| `CG_ALI_FOLDER` | `$CG_WORK/code/concept-graphs-ali` |
| `CG_ALI_PYTHON` | `$CG_WORK/envs/cg-ali/bin/python` |
| `HABITAT_SEQUENCE_ROOT` | `$CG_WORK/data/HabitatMP3D/sequences` |
| `HABITAT_RESULT_ROOT` | `$CG_WORK/results/HabitatMP3D` |
| `CUDA_VISIBLE_DEVICES` | 默认 `0`，允许调用者覆盖 |
| `HF_HOME` | `$CG_WORK/models/huggingface` |
| `TORCH_HOME` | `$CG_WORK/models/torch` |
| `XDG_CACHE_HOME` | `$CG_WORK/.cache` |
| `YOLO_CONFIG_DIR` | `$CG_WORK/.config/Ultralytics` |
| `WANDB_DIR` | `$CG_WORK/logs/wandb` |

模型和配置缓存被统一放在工作区，不依赖不可写的系统目录。

## 5. 已部署的服务器程序

### 5.1 `validate_sequence.py`

作用：

- 要求上传目录已有 `READY`；
- 验证 `metadata.json` 符合 CGS Habitat v1 schema；
- 检查 RGB/Depth/语义占位文件数量；
- 检查 Depth 为 16-bit 毫米 PNG；
- 检查 4×4 Pose 正交性和坐标系元数据；
- 检查内参和 `frames.jsonl`；
- 重新计算序列 checksum；
- 全部通过后创建 `VALIDATED`。

### 5.2 `generate_dataset_config.py`

从序列 metadata 读取分辨率、fx/fy/cx/cy 和 `png_depth_scale`，生成 ConceptGraphs 可直接读取的 Replica-compatible YAML：

```text
<sequence>/conceptgraphs_dataset.yaml
```

### 5.3 `server_habitat_smoke.sh`

作用：

- 要求 `READY`、`VALIDATED` 和 dataset YAML 已存在；
- 运行 ConceptGraphs `rerun_realtime_mapping.py`；
- 从 `metadata.json` 动态读取帧数，本次为第 0–19 帧，`stride=1`；
- 从 metadata 动态读取分辨率，本次为 640×480；
- 强制重新检测并保存 detection；
- 关闭 Rerun、W&B 和视频，降低 smoke 运行的额外开销；
- 将单对象点数上限设为 3000；
- 建图后自动调用 packager。

### 5.4 `package_map_bundle.py`

作用：

- 寻找 mapping pkl、object JSON、edge JSON 和序列 metadata；
- 将产物标准化命名为 `object_map.pkl.gz`等；
- 反序列化 pkl 并统计对象中心；
- 对象数为 0 或出现 NaN 中心时拒绝创建合格 bundle；
- 生成 `map_manifest.json`、`map_statistics.json` 和 `query_classes.txt`；
- 生成 SHA-256 清单；
- 成功后创建 `COMPLETE`。

### 5.5 `server_habitat_pipeline.sh`

这是服务器推荐入口，按顺序执行 validator、dataset config generator 和 smoke mapping/packager。

### 5.6 `query_object_map.py` 与 `server_query_habitat_map.sh`

查询入口会先校验 bundle，然后在 CUDA 上使用 OpenCLIP ViT-H-14 对文本编码，与所有 ConceptGraphs 对象 `clip_ft` 计算余弦相似度。输出 Top-K 的类别、UUID、中心、bbox、检测次数和对象表面采样点，供 Ubuntu 做坐标转换和可见性导航。

## 6. 已完成的数据和映射运行

### 6.1 输入序列

```text
/home/chenkejun/beauty/conceptgraphs/data/HabitatMP3D/sequences/
  2azQ1b91cZZ_smoke_v001/
```

目录大小约 61 MiB。根目录已有：

- `READY`；
- `VALIDATED`；
- `metadata.json`；
- `intrinsics.json`；
- `frames.jsonl`；
- `traj.txt`；
- `checksums.sha256`；
- `conceptgraphs_dataset.yaml`；
- `results/`、`semantic/` 和 `exps/`。

输入验收：20 RGB、20 Depth、20 Pose，64 项 checksum，Depth 中位数 1385 mm，平均有效比例 0.999861。

### 6.2 建图运行

| 项目 | 值 |
|---|---|
| Run ID | `2azQ1b91cZZ_smoke_v001_smoke` |
| Mapping suffix | `2azQ1b91cZZ_smoke_v001_smoke_mapping` |
| Detection suffix | `2azQ1b91cZZ_smoke_v001_smoke_detections` |
| 帧范围 | `[0, 20)` |
| Stride | 1 |
| Device | CUDA |
| Detector | YOLO-World 路径 |
| Segmenter | SAM，ViT-H |
| Visual encoder | OpenCLIP ViT-H-14，LAION2B-s32B-b79K |
| 强制检测 | true |
| 保存 detections | true |
| 场景图边 | false |
| 单对象最大点数 | 3000 |
| 映射帧完成 | 20/20 |

建图日志：

```text
/home/chenkejun/beauty/conceptgraphs/logs/habitat/
  2azQ1b91cZZ_smoke_v001_smoke_mapping.log
```

日志最终确认 pkl、object JSON 和 edge JSON 均已写入。

### 6.3 标准 map bundle

请使用这个标准路径：

```text
/home/chenkejun/beauty/conceptgraphs/results/HabitatMP3D/
  2azQ1b91cZZ_smoke_v001/
  2azQ1b91cZZ_smoke_v001_smoke/
  map_bundle/
```

目录大小约 5.7 MiB，`COMPLETE` 已存在，8 个 bundle 负载文件 checksum 全部通过。

统计：

| 指标 | 值 |
|---|---|
| 对象数 | 26 |
| NaN 中心 | 0 |
| bbox center min | `[-2.1991, 0.3203, 0.2316]` |
| bbox center max | `[3.8590, 2.0882, 10.0860]` |
| 最大中心半径 | 10.5678 m |

对象类别：

```text
bench 1, cabinet 1, closet door 1, coffee kettle 1, cushion 1,
desk 1, folded chair 1, microwave 1, mirror 2, oven 1,
paper bag 1, piano 1, picture 3, plate 1, power outlet 1,
radiator 2, sofa chair 1, window 5
```

`make_edges=false`，所以 `edges.json=[]` 是预期结果。

> 早期首次打包曾把 bundle 放到少一层 sequence ID 的旧路径 `results/HabitatMP3D/2azQ1b91cZZ_smoke_v001_smoke/map_bundle`。脚本已修正，后续不要再使用该旧路径。

## 7. 服务器运行命令

### 7.1 对已上传序列执行完整管线

```bash
cd /home/chenkejun/beauty/conceptgraphs
./scripts/server_habitat_pipeline.sh 2azQ1b91cZZ_smoke_v001
```

或：

```bash
source /home/chenkejun/beauty/conceptgraphs/scripts/server_env.sh
"$CG_WORK/scripts/server_habitat_pipeline.sh" \
  2azQ1b91cZZ_smoke_v001
```

### 7.2 只重新验证序列

```bash
source /home/chenkejun/beauty/conceptgraphs/scripts/server_env.sh

SEQ=2azQ1b91cZZ_smoke_v001

"$CG_ALI_PYTHON" \
  "$CG_WORK/habitat_bridge/server/validate_sequence.py" \
  --sequence-dir "$HABITAT_SEQUENCE_ROOT/$SEQ"
```

### 7.3 只重新生成数据集配置

```bash
source /home/chenkejun/beauty/conceptgraphs/scripts/server_env.sh

SEQ=2azQ1b91cZZ_smoke_v001

"$CG_ALI_PYTHON" \
  "$CG_WORK/habitat_bridge/server/generate_dataset_config.py" \
  --sequence-dir "$HABITAT_SEQUENCE_ROOT/$SEQ"
```

### 7.4 只运行 smoke mapping 和打包

```bash
cd /home/chenkejun/beauty/conceptgraphs
./scripts/server_habitat_smoke.sh 2azQ1b91cZZ_smoke_v001
```

### 7.5 校验已生成 bundle

```bash
source /home/chenkejun/beauty/conceptgraphs/scripts/server_env.sh

SEQ=2azQ1b91cZZ_smoke_v001
RUN=${SEQ}_smoke
BUNDLE="$HABITAT_RESULT_ROOT/$SEQ/$RUN/map_bundle"

(cd "$BUNDLE" && sha256sum -c checksums.sha256)
test -f "$BUNDLE/COMPLETE"
cat "$BUNDLE/map_manifest.json"
cat "$BUNDLE/map_statistics.json"
```

### 7.6 监控日志和 GPU

```bash
tail -f /home/chenkejun/beauty/conceptgraphs/logs/habitat/2azQ1b91cZZ_smoke_v001_smoke_mapping.log
```

GPU：

```bash
watch -n 1 nvidia-smi
```

## 8. 从 Ubuntu 本地调用服务器

本地入口会自动上传、调用本文档的服务器管线、下载并校验：

```bash
cd /home/abc/conceptgraphs
./scripts/run_habitat_end_to_end.sh
```

完全重新导出 Habitat 帧：

```bash
cd /home/abc/conceptgraphs
REEXPORT=1 ./scripts/run_habitat_end_to_end.sh
```

SSH 别名为 `cg-server`，已配置公钥免密登录。服务器文档不记录用户密码或任何私钥内容。

## 9. 排错

### 9.1 `READY` 缺失

说明本地序列未通过导出校验，不应在服务器绕过。返回 Ubuntu 本地运行 validator 并重新上传。

### 9.2 `VALIDATED` 缺失

先运行 `validate_sequence.py`。不要手工 `touch VALIDATED`，否则会跳过接口和 checksum 保护。

### 9.3 CUDA 不可用

先确保运行了：

```bash
source /home/chenkejun/beauty/conceptgraphs/scripts/server_env.sh
```

再检查：

```bash
"$CG_ALI_PYTHON" -c '
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
'
```

### 9.4 结果下载不到

必须使用包含 sequence ID 和 run ID 的标准层级：

```text
results/HabitatMP3D/<SEQ>/<RUN>/map_bundle
```

不要使用早期少一层 `<SEQ>` 的旧产物路径。

### 9.5 模型重复下载

确保使用 `server_env.sh`。`HF_HOME`、`TORCH_HOME` 和其他缓存都已指向 `$CG_WORK/models` 或 `$CG_WORK/.cache`。

## 10. 当前限制和后续工作

1. 当前是 20 帧 panorama smoke，不是正式全屋覆盖建图。
2. `server_habitat_smoke.sh` 已从 metadata 动态读取帧数和分辨率，但仍是 smoke 命名/参数集。
3. 当前输入 GLB 没有语义标注，无法完成严格的语义 ObjectNav Success/SPL 验收。
4. 当前 `make_edges=false`，场景图关系边未生成。
5. 对象类别是视觉模型预测，尚未与正式语义 GT 计算准确率。
6. 服务器语言查询和 Ubuntu predicted-object 可见导航已完成；下一阶段是带语义 GT 的官方 Habitat ObjectNav evaluator。

## 11. 验收标记

当前 3 个标记均已存在：

```text
data/HabitatMP3D/sequences/2azQ1b91cZZ_smoke_v001/READY
data/HabitatMP3D/sequences/2azQ1b91cZZ_smoke_v001/VALIDATED
results/HabitatMP3D/2azQ1b91cZZ_smoke_v001/
  2azQ1b91cZZ_smoke_v001_smoke/map_bundle/COMPLETE
```

`READY` 代表本地导出已验证，`VALIDATED` 代表服务器已独立复核，`COMPLETE` 代表对象图质量检查和打包已完成。

服务器还已归档第二次导航验收：

```text
/home/chenkejun/beauty/conceptgraphs/results/ObjectNavAcceptance/
  2azQ1b91cZZ/sofa_acceptance_run2/
```

该归档包含视频、轨迹、目标位姿、Success/SPL、确定性签名和 SHA-256 清单；服务器重新校验全部通过。
