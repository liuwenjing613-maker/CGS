# CGS ObjectNav 一条命令复现与最终验收指南

> 最后更新：2026-08-03  
> 固定场景：`2azQ1b91cZZ`  
> 固定序列：`2azQ1b91cZZ_smoke_v001`  
> 固定 seed：`2027`  
> 默认查询：`sofa`  
> 当前评价协议：`predicted_object_visibility_v1`

## 1. 唯一主入口

以后完整复现不再分别以“窗口能打开”、“pkl 能加载”或“服务器不报错”为最终验收。

唯一入口是：

```bash
cd /home/abc/conceptgraphs

./scripts/run_cgs_objectnav.sh \
  --query sofa \
  --expected-class sofa \
  --episode-id sofa_user_v001 \
  --seed 2027 \
  --compare-signature \
    /home/abc/conceptgraphs/results/objectnav/2azQ1b91cZZ/sofa_acceptance_run1/reproducibility_signature.json
```

这是交互模式，你会参与目标对象确认，并在 Ubuntu 桌面看到 Habitat RGB/Depth 自动导航窗口。
最后还会把本次签名与已验收的同输入基线比较，因此这一条命令可以覆盖最终清单中的可重复性检查。

第一次使用该 `episode-id` 时不需要 `--overwrite`。如果需要覆盖同名实验，必须显式加：

```bash
--overwrite
```

## 2. 你会实际经历的 6 个阶段

### Stage 1：固定输入和地图校验

输入：

- 场景 `2azQ1b91cZZ.glb`；
- seed `2027`；
- 已下载 `map_bundle`；
- `COMPLETE` 和 `checksums.sha256`。

程序会重新计算 map bundle 中所有负载文件的 SHA-256。任何文件被替换或损坏，都会立即终止，不会带着错误地图继续导航。

效果：

```text
Map bundle checksum passed
Object-map SHA-256: 32bf541c...
```

### Stage 2：服务器 OpenCLIP 文本查询

输入：

```text
query = sofa
```

本地通过 SSH 请求服务器。服务器会：

1. 先重新校验服务器 map bundle；
2. 在 RTX 5880 上加载 OpenCLIP ViT-H-14；
3. 对文本 `sofa` 生成特征；
4. 与 26 个 ConceptGraphs 对象特征计算余弦相似度；
5. 返回 Top-5。

已验收的 Top-5：

```text
rank 1  sofa chair     0.219635
rank 2  folded chair   0.214453
rank 3  plate          0.212819
rank 4  cushion        0.211598
rank 5  power outlet   0.206562
```

生成：

```text
results/objectnav/_queries/q_sofa_<map-hash>.json
```

文件包含：

- 对象 UUID；
- 类别；
- CLIP 相似度；
- 检测次数；
- ConceptGraphs map frame 中心和 bbox；
- 128 个对象表面点；
- 服务器对象图 SHA-256。

### Stage 3：你参与确认目标

终端会显示 Top-5，然后询问：

```text
Select candidate rank [1]:
```

你需要：

- 输入 `1` 并回车；
- 或直接回车，默认选择第 1 名。

为了防止误操作，本地还会检查选中的 `class_name` 是否包含 `--expected-class sofa`。如果你选择 folded chair，这次 sofa 验收将失败并终止。

自动化验收时可以不询问：

```bash
--yes
```

### Stage 4：坐标转换、NavMesh 目标采样和自动导航

程序会使用 map bundle 中的：

```text
T_habitat_world_from_cg_map
```

将对象中心从 ConceptGraphs map frame 转到 Habitat world frame。

已验收 sofa 中心：

```text
CG map:        [-0.283854, 0.445237, 3.974496]
Habitat world: [ 2.466944, 0.931872, 5.096223]
```

程序不会把 bbox 中心直接当作导航点，而是：

1. 在目标周围 0.8–2.0 m 采样观察点；
2. 每个半径采样 36 个角度；
3. 投影到 NavMesh；
4. 检查从固定起点是否可达；
5. 计算最短路径；
6. 将相机朝向目标；
7. 将对象表面点投影到相机；
8. 与 Habitat Depth 比较，确认该物体从候选点真的可见；
9. 从可见候选中选出最佳观察位姿。

已验收数据：

```text
固定起点: [2.750798, 0.127109, 9.070719]
可达候选: 163
目标点:     [3.500606, 0.127109, 4.640620]
最短路径: 4.885038 m
目标可见点: 46/128
```

然后 Habitat `GreedyGeodesicFollower` 会自动产生：

- `move_forward`；
- `turn_left`；
- `turn_right`；
- `stop`。

你会在桌面窗口中看到左侧 RGB、右侧 Depth；窗口按与输出视频一致的 12 FPS 节奏播放，便于看清每个动作。`Q` 或 `Esc` 可中断，但中断的 Episode 不会通过验收。

### Stage 5：停止、可见性和 Success/SPL

停止后，程序会通过转向动作让相机面向目标，然后再次投影 128 个对象表面点。

最终图像中：

- 绿点：投影深度与 Habitat Depth 一致，可见；
- 红点：在画面中，但被其他几何遮挡或 Depth 不一致。

当前 Success 条件：

```text
文本选中 expected class
AND 目标点在 NavMesh
AND 路径存在
AND follower 自动停止
AND 最终 geodesic distance <= 0.35 m
AND 至少 3 个对象表面点通过 Depth 可见性检查
```

已验收结果：

```text
动作步数:       56
实际移动距离:   4.765165 m
碰撞:             0
终点 geodesic:    0.095495 m
停止可见点:      46/128
success:          true
SPL:              1.0
```

### Stage 6：最终验收清单

程序会逐项打印：

```text
[PASS] One command launched the workflow
[PASS] Fixed scene, seed and start
[PASS] Map bundle checksum passed
[PASS] Text query selected the expected class
[PASS] Object coordinates transformed to Habitat world
[PASS] Observation goal is on NavMesh
[PASS] A path exists from start to goal
[PASS] The agent automatically executed navigation
[PASS] The target is visible at stop
[PASS] success = 1
[PASS] SPL is finite and calculable
[PASS] Video, trajectory, config and metrics are saved
[PASS] Same input produced the same signature
```

## 3. Episode 会生成什么

```text
results/objectnav/2azQ1b91cZZ/<episode-id>/
├── COMPLETE
├── checksums.sha256
├── episode_spec.json
├── environment.json
├── map_bundle_ref.json
├── query_result.json
├── goal_candidates.json
├── goal_pose.json
├── shortest_path.json
├── trajectory.jsonl
├── metrics.json
├── reproducibility_signature.json
├── navigation.mp4
├── final_rgb.jpg
├── final_depth.png
└── final_visibility.jpg
```

| 文件 | 作用 |
|---|---|
| `episode_spec.json` | 场景、seed、query、固定起点和评价协议 |
| `map_bundle_ref.json` | 本次导航使用的地图 SHA-256 |
| `query_result.json` | 服务器 OpenCLIP Top-K 完整结果 |
| `goal_candidates.json` | 所有通过 NavMesh/路径检查的候选观察点 |
| `goal_pose.json` | 选中对象、坐标变换、起点和最终目标 |
| `shortest_path.json` | 最短路径长度和路径点 |
| `trajectory.jsonl` | 每个动作、位置、姿态、碰撞和剩余距离 |
| `metrics.json` | Success、SPL、路径、步数、碰撞和可见点 |
| `navigation.mp4` | RGB + Depth 完整导航过程 |
| `final_visibility.jpg` | 带红/绿表面点的最终可见性画面 |
| `reproducibility_signature.json` | 对场景、地图、seed、起终点、动作和指标的确定性签名 |
| `checksums.sha256` | Episode 全部证据完整性校验 |
| `COMPLETE` | 只在 Success 为 true 且所有证据已保存后创建 |

## 4. 查看结果

视频：

```bash
xdg-open /home/abc/conceptgraphs/results/objectnav/2azQ1b91cZZ/sofa_user_v001/navigation.mp4
```

最终可见画面：

```bash
xdg-open /home/abc/conceptgraphs/results/objectnav/2azQ1b91cZZ/sofa_user_v001/final_visibility.jpg
```

指标：

```bash
jq . /home/abc/conceptgraphs/results/objectnav/2azQ1b91cZZ/sofa_user_v001/metrics.json
```

完整性：

```bash
cd /home/abc/conceptgraphs/results/objectnav/2azQ1b91cZZ/sofa_user_v001
sha256sum -c checksums.sha256
test -f COMPLETE
```

## 5. 可重复性验收

第一次的签名：

```text
results/objectnav/2azQ1b91cZZ/sofa_user_v001/
  reproducibility_signature.json
```

第二次使用同场景、同 seed、同 query，并比较签名：

```bash
cd /home/abc/conceptgraphs

./scripts/run_cgs_objectnav.sh \
  --query sofa \
  --expected-class sofa \
  --episode-id sofa_user_v002 \
  --seed 2027 \
  --yes \
  --compare-signature \
    /home/abc/conceptgraphs/results/objectnav/2azQ1b91cZZ/sofa_user_v001/reproducibility_signature.json
```

完全一致时会看到：

```text
[PASS] Same input produced the same signature
```

已经实际完成两次独立验收：

```text
sofa_acceptance_run1
sofa_acceptance_run2
```

两次签名完全一致：

```text
525e5b003926058f5971e9754fdf659150e1727164880011dff9eff43296d21b
```

## 6. 自动模式

不显示窗口、自动选第 1 名：

```bash
cd /home/abc/conceptgraphs

./scripts/run_cgs_objectnav.sh \
  --query sofa \
  --expected-class sofa \
  --episode-id sofa_auto_v001 \
  --seed 2027 \
  --yes \
  --no-visualize
```

复用本地已缓存的查询 JSON，不重新请求服务器：

```bash
--no-server-query
```

仅当 query 和 object-map SHA 对应的缓存文件已存在时才能使用。

## 7. Ubuntu 和服务器的分工

| 阶段 | Ubuntu 本地 | ConceptGraphs 服务器 |
|---|---|---|
| map bundle checksum | 执行 | 执行 |
| OpenCLIP 文本编码 |  | RTX 5880 |
| 对象 Top-K | 下载和显示 | 计算和保存 |
| CG map → Habitat world | 执行 |  |
| NavMesh 候选位姿 | 执行 |  |
| 可见性/Depth 一致性 | 执行 |  |
| 最短路径和动作 | 执行 |  |
| Success/SPL | 执行 |  |
| 视频和轨迹 | 执行 | 验收后归档 |

## 8. 严格的评价边界

当前场景没有 Habitat semantic annotations，所以：

- `success=true` 表示导航到 ConceptGraphs 预测对象的可见观察位姿；
- `target_visible=true` 表示对象图表面点与 Habitat Depth 几何一致；
- `query_selected_expected_class=true` 表示 ConceptGraphs 的对象类别符合查询；
- 不代表已用 Habitat 语义 GT 证明这个几何实例必然是 sofa。

`metrics.json` 已明确写入：

```json
{
  "evaluation_protocol": "predicted_object_visibility_v1",
  "official_habitat_objectnav_gt": false
}
```

当补齐语义 MP3D 资源后，应保留整条管线，另外增加 Habitat 官方 semantic ObjectNav evaluator。在此之前，不应将当前指标写成官方 benchmark Success/SPL。

## 9. 常见问题

### SSH 失败

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 cg-server 'hostname; whoami'
```

期望输出 `ubun` 和 `chenkejun`。

### `DISPLAY is missing`

请在 Ubuntu 桌面终端运行，不要在无图形会话的 SSH 终端运行。如果只做自动验收，使用 `--no-visualize`。

### 选错 Top-K

程序会根据 `--expected-class` 失败关闭。重新运行并选择正确排名。

### Episode 目录已存在

使用新的 `--episode-id`，或在明确需要覆盖时加 `--overwrite`。

### 视频能看但没有 `COMPLETE`

说明 Success、可见性或证据文件中至少一项未通过。先查看：

```bash
jq . <episode-dir>/metrics.json
```
