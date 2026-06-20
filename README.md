# OriginCar 医院赛道自主车 — 交接说明

> 这是一台基于地平线 **RDK X5** 的 ROS2 自主小车，参加"医院赛道"比赛。
> 本文档是接手这个项目的**第一份要读的东西**。读完你应该清楚：项目要干嘛、
> 现在做到哪了、怎么跑起来、当前唯一卡点是什么、还差什么能完赛。
>
> 维护历史：原作者独立完成（摄像头/IMU/嵌入式/状态机整套），现转交。
> 最后一次外场实跑 = 2026-06-17。比赛目标是**完赛（跑完全程），不是夺冠**。

---

## 0. 30 秒看懂任务

```
出P点 → 扫二维码(定C环顺/逆) → 走黄色通道 → C环绕满一圈(途中拍图文牌交LLM描述) → 回P点停车
```

纯导航 + 感知，**没有机械臂、没有取放动作**。
完赛硬指标四项，缺一不算（详见 `docs/01_比赛规则.md`）：
1. 扫到二维码　2. C 环绕满一圈且方向正确　3. 读出 ≥1 块图文牌（LLM 描述图片）　4. 回 P 点停车

限时 180s，巡航目标 ~0.3 m/s。碰绿区/网格墙/撞墙 = 罚时。

**这是阿克曼底盘：不能原地转。** 所有转向都得连续小角度 + 前进/后退配合。这条贯穿全部设计。

---

## 1. 现在能干什么 / 不能干什么（诚实状态）

最后一次外场（6-17）的真实情况：

### ✅ 已经稳的（实车红线轨迹验证过）
- **A 区前 2/3：`P → 沿黑虚线 → 接近二维码 → 停车扫码 → 解析方向`** —— 整段可靠。这是感知最难的部分，搞定了。
- **避障**：YOLO（板载 BPU，yolo11n.bin）检蓝桶 + 区域感知绕障（正前方打方向、侧方加速冲过）。用户认可"避障可以了"。
- **二维码解码**：用 inchworm（走一段→停→静止解码，避免运动模糊）。`ClockWise→右环 / AntiClockWise→左环`。
- **行驶卡顿**：已修（QR 解码加时间预算，不再饿死控制环）。

### ⚠️ 当前唯一卡点（卡在这，没过）—— 见第 4 节
**扫码后倒车进中间通道**。扫完码车顶死在码牌前 ~0.3m，需要倒车 + 转头朝**中间黄色通道**（两侧是黄白网格 = 禁区）。
现在倒车会走，但**车头转向方向是反的**（往两侧网格去了，不是往中间）。
根因：阿克曼**倒车时**前轮转向产生的车头偏转方向和前进时**相反**，符号无法靠脑补确定。
最后已加了 yaw 朝向日志（commit `9c970b7`，未上板），就差**开机跑一次读数据**定符号。

### ❌ 还没碰过实车（都在卡点下游，需要先过通道）
- 通道内对准 1m gap 不蹭网格墙（代码留了 `lane_mask_erode_px` 腐蚀，默认关，待现场调核）
- C 环 0.5m 窄道绕圈不碰绿
- 环上侧拍图文牌 + qwen-vl 读图（代码就绪，未实跑验证）
- 返航认 P 点停车（`RETURN_HOME` 现在是轨迹回放/兜底，未实车闭环）

---

## 2. 仓库结构

```
rdk_origincar_handoff/
├── README.md                         ← 你在这
├── docs/
│   ├── 01_比赛规则.md                  ★ 唯一事实来源：赛道/任务/完赛判定/二维码语义
│   ├── 02_项目总览.md                  整体怎么运转 + 怎么上手跑 + 排查表（必读）
│   ├── 03_当前架构_2026-06-10.md        状态机/各模块/数据流
│   ├── 04_重构方案_2026-06-14.md        从"地图航点"转向"视觉循迹"的重构说明（当前架构由来）
│   ├── 05_运动标定参数.md               底盘速度/转向标定曲线
│   ├── 06~09_*.md                      外场测试手册/变更记录/执行手册/离线核验
│   ├── 90_符号索引_ARCHITECTURE.md      旧版详细符号索引（部分流程已过期，查代码位置用）
│   ├── 板上配置快照_legacy_2026-06-15.json   板上实际跑的 legacy 配置（注意是 6-15 的，非最终，见第5节）
│   ├── route_overlay/*.png             map2d 路线叠加图（map 模式已弃用，仅作场地俯视参考）
│   └── field_reference_images/         ★ 车载相机真实帧 + INDEX.md（调 HSV/理解视角必看）
└── origin_competition_auto/           ← ROS2 包（要部署到板子的代码）
    ├── package.xml / setup.py / setup.cfg / resource/
    ├── launch/rdk_yolo_detection.launch.py
    ├── config/
    │   ├── mission_defaults.json            ★ 几乎所有可调参数都在这个文件
    │   └── mission_defaults.annotated.jsonc 带注释版（看每个键什么意思）
    ├── yolo11n.pt                       YOLO11n 基座权重（训练起点；板上跑的是量化后的 .bin，见第6节）
    └── origin_competition_auto/*.py     源码（核心 = auto_mission.py 状态机）
```

**核心代码地图：**
| 文件 | 作用 |
|---|---|
| `auto_mission.py` | **主状态机**（~3500 行）。整个任务流程都在这。改行为基本只改它。 |
| `vision_detector.py` | HSV 颜色检测（蓝桶/黄道/橙牌/绿区）+ 合并 YOLO 检测，输出避障决策 |
| `lane_follow.py` | 黄线/黑线循迹（HSV 质心 + PD 控制） |
| `motion_state.py` | 里程计 + IMU 融合（算走了多远、转了多少度、绝对航向） |
| `yolo_detector.py` | 板载 BPU YOLO 推理封装（.bin 模型） |
| `decision_parser.py` | 二维码内容 → left/right 解析（顺/逆时针） |
| `llm_client.py` | 调 qwen-vl 读图文牌（OpenAI 兼容接口） |
| `route_map.py` | map2d 航点表（**已弃用**，legacy 模式不走这里） |

---

## 3. 怎么跑起来

### 连板子
- 板子是 RDK X5，跑 ROS2 Humble。账号 root（**密码不在仓库**，问原作者）。
- **同一局域网直连**：`ssh root@<板子IP>`（IP 每换网会变）。
- 板子上所有命令先进环境：
  ```bash
  source /opt/ros/humble/setup.bash && source /root/dev_ws/install/setup.bash
  ```
- 代码在板子 `/root/dev_ws/src/origin_competition_auto/`。改完 `.py` 用 symlink 安装即时生效；改 `package.xml`/`setup.py` 要重新编译：
  ```bash
  cd /root/dev_ws && colcon build --symlink-install --packages-select origin_competition_auto
  ```
> 原作者外场时板子在现场热点、台式机够不着，走的是 SSH 反向隧道远程部署。新人若和板子同网，直接 scp/ssh 即可，不用管隧道那套。

### 四级测试（由安全到危险）
本机（无 ROS）也能跑前两步的 dry-run：
```bash
# 1) 纸上推演：打印完整状态流，不碰硬件，最安全
ros2 run origin_competition_auto auto_mission --config <cfg> --dry-run --mock-qr-content '{"direction":"left"}'

# 2) 静态自检：车不动，验相机/解码/AI（屏蔽轮子指令）
ros2 run origin_competition_auto auto_mission --config <cfg> --no-motion --start-state SCAN_QR

# 3) 架空低速：车架起来轮子空转，验电机方向/循迹打方向
# 4) 地面慢速：真跑（保守速度）
ros2 run origin_competition_auto auto_mission --config <cfg>
```
`--start-state` 可从任意状态开始测，调下游不用每次从头跑。

部署后想确认"传上去的真是这版代码"，比 md5：本机 `md5sum auto_mission.py` 对比板上运行时 `python -c "import ...; print(md5)"`。

---

## 4. ★ 当前卡点：扫码后倒车朝向 —— 开机第一件事

代码位置：`auto_mission.py` 的 `_bridge_backup()`（约 1757 行）和 `_bridge_qr_to_corridor()`。

**问题**：扫完码后要倒车 + 转头进中间通道。倒车按里程退固定距离（`bridge_backup_distance_m=0.35`）这步 OK，
但倒车走弧线时给的 `bridge_backup_angular=0.30`（>0 本意是车头左转朝中间），
**实跑车头却转向两侧网格（错方向）**。

**为什么没当场修好**：阿克曼倒车时转向符号和前进相反（自行车模型 linear<0 翻转），
`+0.30` 到底把车头转左还是右，脑补不可靠、板子已关机当时测不了。
所以没有盲赌符号，而是**加了 yaw 朝向日志**（commit `9c970b7`，本地有、**还没上板**）：
`_bridge_backup` 记录倒车前后 `absolute_yaw`，打印
`BRIDGE_BACKUP: heading turned ±Xdeg (LEFT/RIGHT)`。

**开机第一件事（把"赌"变成"读数"）：**
1. 把本地最新代码（含 yaw 日志，commit `9c970b7`）部署上板。
2. 跑一次到扫码完成，读日志里 `BRIDGE_BACKUP: heading turned ...`：
   - 若车头**转向中间通道** → 符号对，只需微调角度/距离大小。
   - 若车头**转向两侧网格** → 把 config（或代码默认）里 `bridge_backup_angular` 从 `+0.30` 翻成 `-0.30`（一个字符）。
3. 定了符号后，再调倒车距离 `bridge_backup_distance_m` / 角速度大小，让车头正对 1m 通道口。

> 注：板上最后跑的是 commit `a46729a`（弧线倒车，无 yaw 日志）；本地领先一个 commit `9c970b7`（加了 yaw 日志）。

---

## 5. 配置：一个文件 + 板上实际跑的关键改动

本项目设计：**几乎所有能调的都在 `config/mission_defaults.json`**，场地日只改数字不碰代码。
带注释版 `mission_defaults.annotated.jsonc` 解释每个键。

> ⚠️ `auto_mission` **必须带 `--config` 指到这个文件**才加载，否则用代码内置默认（那套默认 LLM 密钥变量名是给 OpenAI 的，不是 DashScope）。

### ‼️ 仓库里的 config 不等于板上实际跑的 config
板上是外场手改过的，最关键的差异（新人务必对齐，否则丢掉全部现场调参）：

| 键 | 仓库 json 里 | **板上实跑（要对齐成这个）** | 为什么 |
|---|---|---|---|
| `route_mode` | `map2d` | **`legacy`** | map2d/fixed2d 依赖 odom 航点，实车 odom 会瞬跳→撞墙，**已弃用**。只有 legacy 走视觉循迹分支。 |
| `obstacle_danger_area_ratio` | `0.035` | **`0.05`** | 6-17 实测蓝桶"刚好该避"距离占屏 ~9.6%，取 0.05 留余量 |
| `use_imu_yaw` | （代码默认已是 true） | **`true`** | /imu/data 20Hz 健康；之前被某次热补丁关成 false 导致航向不对 |

`bridge_backup_*` 和 `use_imu_yaw=true` 在**代码 dataclass 默认里已经是对的**（见 `auto_mission.py` MissionConfig）。
`docs/板上配置快照_legacy_2026-06-15.json` 是 6-15 的板上 legacy 配置快照（注意：**早于** 6-17 最终调参，仅作起点参考，按上表对齐）。

### 最常调的几个旋钮
| 想调什么 | 改哪个键 |
|---|---|
| 车开多快 | `cruise_linear` / `lane_base_linear`（默认很慢，安全优先） |
| 黄/黑线认不准 | `color_profiles.yellow` / `color_profiles.black` 的 HSV（家里光线≠场地，常要收紧） |
| 循迹蛇行/迟钝 | `lane_kp`（比例）/ `lane_kd`（阻尼） |
| 二维码停太远/近 | `qr_min_area_ratio`（越大停越近） |
| 避障太早/太晚 | `obstacle_danger_area_ratio`（见上表） |
| 绕环不到一圈就停 | `loop_complete_fraction`（占 360° 比例） |
| 通道蹭网格墙 | `lane_mask_erode_px`（>0 腐蚀掉细网格线，只留实心黄道；默认 0=关，现场调） |

---

## 6. 完赛还差什么（按优先级）

1. **【最高】扫码后倒车朝向定符号**（第 4 节）—— 不过这关，下游全空跑。
2. **通道对准 1m gap**：进通道后用 `lane_mask_erode_px` 区分实心黄道 vs 两侧黄白网格墙，正对 gap 穿过不蹭墙。
3. **C 环 0.5m 窄道绕一圈不碰绿**：实车低速验证阿克曼转弯半径；调 `ring_bias` 贴内侧。
4. **拍牌 + qwen-vl 读图**：代码就绪，需实跑验证拍到清晰牌 + LLM 返回描述（注意 `target_card_seen` 在环里可能误触发拍假牌）。
5. **返航认 P 停车**：`RETURN_HOME` 现为轨迹回放兜底，未实车闭环；P 点识别（深蓝方块+白框+P字）认停未实现。

每改一处的流程：本机 `python -m py_compile` + dry-run（legacy 模式）→ 上板 → 实车验证 → 单独 commit。

---

## 7. 关键经验教训（别重踩这些坑）

- **日志不能全信**：原作者撞车后会手动重新摆车再继续，一次 run 的日志只信到"第一次撞/介入"之前。判断实车行为以**肉眼/手画轨迹**为准，不是日志下游状态。
- **阿克曼倒车转向符号翻转**：见第 4 节，这是当前卡点的根因，别凭直觉赌。
- **控制环里绝不能放可能跑 ~1.8s 的重函数**：早期 `qr_decoder.detect` 在行进中被橙色地面字误触发重路径，饿死循迹环把车开飞。已加时间预算修复。
- **odom 会瞬跳**：所以放弃了 map2d 航点导航，全程视觉循迹。别想着"用定位走航点"，这条路试过了，不行。
- **底盘串口会因 USB 抖动崩**：`/cmd_vel` 没反应时 `systemctl restart origincar-base` 修复。开测前先验底盘节点在 ROS 图里、`/cmd_vel` 有订阅。
- **YOLO 那条 `ai_msgs unavailable` WARN 不用管**：那只是冗余的话题版 YOLO 没起；进程内 BPU YOLO（yolo11n.bin）是正常在跑的，避障靠它。
- **二维码只能静止解码**：运动模糊下解不出，靠 inchworm（走停走停）。
- **LLM 用 `qwen-vl-plus`**（`qwen-vl-max-latest` 当前密钥没权限会 403）。

---

## 8. 不在这个仓库里的东西（怎么拿）

为了能公开上传 GitHub + 控制体积，以下**故意没放进来**，向原作者索取：

| 东西 | 在哪 | 说明 |
|---|---|---|
| **板子 root 密码** | 问原作者 | 没放进版本库（安全） |
| **DashScope 密钥** | 板子 `/root/dev_ws/.dashscope_key`（权限 600） | 代码只读环境变量 `DASHSCOPE_API_KEY`，跑前 `export DASHSCOPE_API_KEY=$(cat /root/dev_ws/.dashscope_key)`。新人要自己的百炼 key 也行。 |
| **训练用数据集（~235MB）** | 原作者本地 `field_captures/`（唯一副本） | 400+ 张实拍 + 标注，重训 YOLO / 重调 HSV 用。需单独拷贝（网盘/U盘），别走 git。 |
| **板上量化 YOLO 模型 `.bin`** | 板子上 | 仓库只带基座 `yolo11n.pt`。板上跑的是量化后 .bin，量化流程见 `yolo_pipeline.py` 和 `bpu_quant/`（原作者本地）。 |

---

## 9. 一句话给接手的人

**先把第 4 节的倒车朝向符号读出来定掉**（开机部署 `9c970b7` → 跑一次 → 看 `BRIDGE_BACKUP: heading turned`），
这是当前唯一卡死完赛的点；过了它，下游就是按第 6 节清单一段段调参的活，代码框架都已就位。
有疑问先查 `docs/01_比赛规则.md`（规则）和 `docs/02_项目总览.md`（怎么跑 + 排查表）。
