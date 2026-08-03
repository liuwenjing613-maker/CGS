# 本地与服务器执行状态（2026-08-03）

当前已跑通：

```text
Ubuntu Habitat RGB-D/Pose 导出
  → CGS Habitat v1 校验
  → SSH/rsync 上传
  → 服务器 ConceptGraphs 20 帧 GPU 建图
  → 26 对象 map bundle
  → checksum 校验和回传
  → Ubuntu Open3D 可视化
  → 服务器 OpenCLIP 文本查询
  → CG map 到 Habitat world 坐标转换
  → NavMesh 可见观察位姿生成
  → Habitat 最短路径自动导航
  → predicted-object Success/SPL、视频和轨迹
```

已验收：

- SSH `cg-server` 公钥免密登录；
- Habitat-Sim 0.3.3，RTX 4060 EGL RGB/Depth 和 NavMesh；
- `2azQ1b91cZZ_smoke_v001`：20 RGB、20 Depth、20 Pose、64 checksum；
- Pose 单测 3/3 通过；
- 服务器 20/20 帧处理成功；
- map bundle：26 个对象，0 NaN，`COMPLETE` 和 checksum 通过；
- Habitat 交互窗口和 Open3D 对象图窗口已测试。
- `sofa` OpenCLIP Top-1 为 `sofa chair`；
- 固定起点到可见目标位姿自动导航通过：56 步、0 碰撞、Success=1、SPL=1.0；
- 两次独立运行确定性签名相同：`525e5b003926058f5971e9754fdf659150e1727164880011dff9eff43296d21b`。

尚未完成：

- 当前 GLB 没有语义标注；
- 正式全屋覆盖轨迹；
- Habitat 语义 GT 的官方 ObjectNav Success/SPL；当前已完成 predicted-object 可见导航协议；
- 场景图关系边（当前 `make_edges=false`）。

详细记录：

- `docs/CGS_Habitat_全流程已完成内容总结_2026-08-03.md`；
- `docs/CGS_Habitat_服务器部署与运行总结_2026-08-03.md`；
- `docs/CGS_ObjectNav_一条命令复现与最终验收指南.md`；
- `docs/CGS_Ubuntu端_Habitat完整仿真与导航指南.md`。
