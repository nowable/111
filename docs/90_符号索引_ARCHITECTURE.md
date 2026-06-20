# OriginCar 医院赛道 — 架构索引文档

> **用途**：先读本文档即可精确定位需要修改的文件、类、函数与配置键，无需大海捞针。
>
> **锚点约定**：本文档以**符号名**（类 / 函数 / 方法 / 配置键 / 状态名）作为定位锚点，**不写代码行号**。定位时直接 `Grep` 符号名即可命中——行号会随每次编辑失效，符号名只在增删/重命名时才变。维护规约：**只有增删、重命名类/函数/配置键/状态时才需要更新本文档**；往函数体内加几行代码无需改动。
>
> **配套文档**：面向人的总览见同目录 `README_OVERVIEW.md`（或项目根 `全过程计划书.md`）；逐阶段实施决策见项目根 `IMPLEMENTATION_PLAN.md`；中文操作手册见 `../origin_competition_project/docs/`。
>
> **最后更新**：2026-06-02（七阶段 A–G 全部完成：深蓝障碍颜色重定义 / motion_state odom+IMU 融合 / DRIVE_TO_QR 接近至可见 / lane_follow 黄色循迹 / 分区过渡 zone 标签 + fail-soft / reverse_plan 轨迹重放返航 / DashScope qwen-vl-plus 接入 + 图像缩放 + 结果发布）

---

## 一、项目概览

OriginCar 是基于地平线 **RDK X5** 的 ROS2（Humble）自主小车，参加**医院主题**场地赛。完整任务：P 点（A 区浅蓝大厅）发车 → 避**深蓝色三角路障** → 右侧**扫二维码**（顺/逆时针 = 右/左环）→ 走 B 区 **1m 黄色通道** → C 区沿 **0.5m 黄色环道**绕中央绿色诊疗室（禁入）一圈，拐角识别**橙色图文牌**送多模态 LLM → 返回 P 点。导航主靠**地面黄色循迹** + odom/IMU 兜底。

**技术栈**：ROS2 Humble + rclpy；OpenCV（颜色/形状/QR）；可选板载 YOLOv8（地平线 `dnn_node_example`）；阿里百炼 DashScope（OpenAI 兼容视觉 LLM）。单 ament_python 包 `origin_competition_auto`。

**设计哲学**：① 全参数化——所有阈值/速度/超时集中在 `MissionConfig` + `mission_defaults.json`，场地日只调数不改码；② 纯函数可离线单测——`lane_follow` / `motion_state` helper / `decision_parser` / `vision_detector` 不依赖 rclpy，ROS 仅在板上注入；③ 安全默认——`llm_mode=placeholder`、`--no-motion`、速度硬钳制、`--dry-run` 全流程推演。

**板子访问**：真板子是 ssh-manager 的 `rdkx5`，命令用 `bash -lc '...'` 包裹并 `source /opt/ros/humble/setup.bash && source /root/dev_ws/install/setup.bash`。源码本地 `E:\AI\rdk program\origin_competition_auto\`，板子镜像 `/root/dev_ws/src/origin_competition_auto/`。

---

## 二、目录结构总览

```
origin_competition_auto/                 ament_python 包
├── package.xml              依赖：rclpy geometry_msgs sensor_msgs std_msgs ai_msgs numpy opencv
├── setup.py                 20 个 console_scripts 入口 + config/launch 安装
├── setup.cfg
├── ARCHITECTURE.md          ★ 本文档（AI 索引）
├── config/
│   ├── mission_defaults.json     ★ 全部任务参数（MissionConfig 的默认值来源）
│   ├── run_profiles.json         运行档：safe_static / bench_motion / field_low_speed
│   └── rdk_yolov8workconfig.json 板载 YOLOv8 DNN 推理配置（COCO 80 类）
├── launch/
│   └── rdk_yolo_detection.launch.py   hobot_shm + hobot_codec + dnn_node_example
└── origin_competition_auto/      Python 源码
    ├── auto_mission.py           ★★ 主状态机（13 状态，~1700 行）
    ├── motion_state.py           odom+IMU 融合（yaw/距离/轨迹 + reverse_plan）
    ├── lane_follow.py            黄色循迹 PD 控制器（纯函数）
    ├── vision_detector.py        OpenCV 颜色/形状 + YOLO 融合检测
    ├── llm_client.py             OpenAI 兼容视觉 LLM（DashScope）
    ├── decision_parser.py        QR 文本 → 左右转决策
    ├── qr_capture.py             单帧 QR 抓取调试工具
    ├── competition_run.py        一键运行编排（self-check→YOLO→mission→replay）
    ├── system_check.py           赛前只读自检
    ├── motion_calibration.py     限幅运动标定
    ├── dataset_capture.py        数据集采集（ROS 帧 / 本地图，可 OpenCV 粗标）
    ├── dataset_audit.py          数据集完整性审计 + dataset.yaml
    ├── vision_tune.py            视觉阈值网格搜索调优
    ├── mission_replay.py         离线主流程回放（无车无 ROS）
    ├── field_session.py          场地采集会话（清单/进度/复盘）
    ├── field_data_review.py      复盘流水线（audit+replay+tune→补丁）
    ├── apply_review_recommendations.py  安全应用复盘补丁到 config
    ├── yolo_pipeline.py          YOLO 训练/导出/部署产物固化
    ├── solution_audit.py         方案就绪度验收矩阵
    └── handoff_bundle.py         赛前交付包打包
```

---
<!-- PLACEHOLDER_SECTIONS -->

## 三、核心任务流水线（auto_mission 及其依赖）

> 下表「锚点」列即可直接 `Grep` 的符号名。

### 3.1 `auto_mission.py` — 主状态机

模块级常量：`PACKAGE_NAME`、`STATE_SEQUENCE`（13 项含末尾 `DONE`；`--start-state` 取 `[:-1]`）、`STATE_ZONES`（状态→赛区 `A_HALL`/`B_CORRIDOR`/`C_RING`/`RETURN`，每个 EVENT 自动带 `zone` 字段）。

#### 状态流转（7 阶段）

| 状态 | 处理方法 | 职责 / 下一状态 |
|------|---------|----------------|
| `IDLE` | `_state_idle` | → `PREFLIGHT` |
| `PREFLIGHT` | `_state_preflight` | 等 `/cmd_vel` 订阅者 + 新鲜图像 + 电压 + odom 样本；失败 → `STOP`，否则 → `DRIVE_TO_QR` |
| `DRIVE_TO_QR` | `_state_drive_to_qr` | 低速接近至 QR 板可见；odom 距离封顶 + 深蓝避障 + 久未见搜寻；→ `SCAN_QR` |
| `SCAN_QR` | `_state_scan_qr` | 停车解码 QR（支持 `--mock-qr-content`）；→ `DECIDE_DIRECTION`，超时 → `STOP` |
| `DECIDE_DIRECTION` | `_state_decide_direction` | `parse_qr_instruction`/`--force-direction` 得 left/right；A→B 过渡 → `CORRIDOR_FOLLOW` |
| `CORRIDOR_FOLLOW` | `_state_corridor_follow` | Zone B 直走廊黄线跟随，距离/绿地/超时退出；B→C 过渡 → `LOOP_LEFT`/`LOOP_RIGHT` |
| `LOOP_LEFT`/`LOOP_RIGHT` | `_state_loop(direction)` | Zone C 黄环跟随 ~360°，内边贴边 + 沿途抓橙标；→ `CAPTURE_TARGET_IMAGE` |
| `CAPTURE_TARGET_IMAGE` | `_state_capture_target_image` | 优先环中最佳橙标，否则现场检测裁剪目标卡；→ `CALL_LLM_API` |
| `CALL_LLM_API` | `_state_call_llm_api` | `LlmClient.analyze` 并发布 `/competition/llm_result`；→ `RETURN_HOME` |
| `RETURN_HOME` | `_state_return_home` | trajectory 模式 `reverse_plan` 回放，否则定时倒车；→ `STOP` |
| `STOP` | `_state_stop` | → `DONE` |

#### `MissionConfig`（dataclass，全部参数集中处）

字段分组：ROS 话题 / odom 航向开关 / 基础运行（速度限幅）/ Phase C 循迹 / Phase F 返航 / 各状态超时 / YOLO / 避障 / HSV 颜色角色 / 目标卡裁剪 / LLM。完整键见 §五配置清单。
方法：`from_mapping(data)`、`apply_args(args)`、`validate()`（区间/枚举校验）、`detector_config()`→`DetectorConfig`、`llm_config()`→`LlmConfig`、`lane_follow_config(bias, side_mode)`→`LaneFollowConfig`。

#### Helper 类（auto_mission 内）

| 锚点 | 职责 |
|------|------|
| `QrResult` / `QrDecoder` | 多策略 QR 解码（与 `qr_capture.py` 同源算法）。`detect(image)` 主入口；`_qr_crops_from_points`/`_qr_warps_from_points`/`_order_points` 角点裁剪+透视矫正+6x 放大+Otsu |
| `VisionBuffer` | 订阅 `CompressedImage` 帧缓冲：`get_latest(max_age)` / `has_fresh_image` / `save_image` |
| `MotionCommander` | `/cmd_vel` 发布 + 速度钳制 + `--no-motion` 抑制：`publish` / `stop_once` / `subscriber_count` / `_clamp` |
| `SafetyGuard` | 运行时长 + 电压看门狗：`runtime_ok` / `runtime_remaining` / `has_voltage` |
| `MissionAbort(RuntimeError)` | 致命中断异常 |

#### `AutoMissionNode(Node)` 关键方法（非 `_state_*`）

- 主循环/分发：`run_mission()`、`transition(state)`、`_event(event, **fields)`、`_run_state(state)`（handler 分发表）。
- YOLO：`_yolo_callback(msg)`、`_fresh_yolo_detections()`。
- DRIVE_TO_QR 辅助：`_handle_drive_obstacle`、`_drive_probe_qr`、`_qr_points_center_area`。
- 走廊/环：`_green_floor_ratio`、`_lane_lost_step`、`_state_loop_timed`（回退）、`_run_ring_follow`、`_maybe_capture_marker`。
- 返航（Phase F）：`_execute_return_segments`、`_rotate_to_heading`（yaw 闭环，用 `angle_diff`）、`_drive_distance`（odom 闭环）。
- 通用：`_timed_motion`、`_stop_with_spin`、`_settle_transition`（区间短停）、`_spin_sleep`、`_wait_until`、`_save_debug_image`、`_print_summary`。

模块级函数：`parse_direction`、`default_config_path`、`load_config`、`run_dry_run`、`flow_from_start`、`build_parser`、`run`、`main`。

### 3.2 `motion_state.py` — odom+IMU 融合（Phase B/F）

纯数学 helper 不依赖 rclpy，可离线单测。`/odom` + `/imu/data` → 融合航向 + 行驶距离 + 位姿轨迹。

模块级：`_HAS_ROS`（离线降级标志）、`yaw_from_quaternion(x,y,z,w)`、`angle_diff(a,b)`（最小有符号角差，wrap 到 (-π,π]）、`Segment`（dataclass：`target_heading`+`distance`）、**`reverse_plan(trajectory, waypoint_stride, min_segment_dist)`**→`List[Segment]`（外出轨迹降采样反向，构造逐段转向+直行回放计划，纯函数）。

`MotionState` 类公开 API：`absolute_yaw()`、`integrated_yaw()`（自 reset 起展开累计，环判完用）、`traveled_distance()`、`pose()`、`has_odom()`/`has_imu()`、`reset()`（清零积分+轨迹）、**`reset_yaw()`**（仅清积分 yaw，**保留轨迹/距离**——绕圈用，避免破坏返航轨迹）、`distance_marker()`（快照距离）、`start_recording()`/`stop_recording()`/`trajectory()`。内部：`_yaw_source()`（优先新鲜 IMU）、`_on_update()`、`_maybe_record(yaw)`。`main(argv)` CLI 2Hz 打印调试。

> **关键约定**：IMU madgwick 静止时 yaw 缓漂 ~0.5°/8s，故环完成判定用 `integrated_yaw` 相对累计 + `loop_timeout` 兜底，不用绝对 yaw。

### 3.3 `lane_follow.py` — 黄色循迹 PD 控制器（Phase C）

纯「图像进/命令出」，无 ROS 依赖。用于 CORRIDOR_FOLLOW 和 LOOP_*。

- `LaneFollowConfig`（dataclass）：`roi_y_ratio` `roi_height_ratio` `kp` `kd` `base_linear` `max_angular` `min_mask_area_ratio` `bias`（>0 贴右 <0 贴左）`side_mode`（center|left_edge|right_edge）。
- `LaneCommand`（dataclass）：`linear` `angular` `lane_found` `offset_norm` `mask_area_ratio`。
- `LaneFollower` 类：`compute(image)`→`LaneCommand`（底 ROI 黄掩膜，面积不足判丢线，否则 PD `-(kp*offset+kd*d_offset)+bias`）；`_lane_offset(mask, width)`（center 列加权质心，left/right_edge 取最左/右黄列贴内边）；`reset()`；复用 `VisionDetector.color_mask` 保证阈值一致。`main(argv)` CLI。

### 3.4 `vision_detector.py` — OpenCV/YOLO 融合检测（Phase A）

整条流水线共享的 HSV 阈值与检测，被 `lane_follow`（`color_mask`）、`dataset_capture`、`vision_tune`、`mission_replay` 复用。

- `ColorProfile`（dataclass）：HSV 阈值带（含 `wrap_hue` 处理跨 0 的红/橙，`mask` 取两段 OR）。`default_color_profiles()`→`dark_blue`/`yellow`/`orange`/`green`/`black`。
- `Detection`（`label`/`confidence`/`bbox`/`source`，属性 `area`/`center_x`/`center_y`）、`VisionDecision`（`obstacle_danger`/`obstacle_zone`/`obstacle`/`target_card`/`detections`）、`DetectorConfig`（障碍 ROI/面积/危险带、`black_*`、目标卡、`yolo_min_confidence`、`color_profiles`、四颜色角色、三角门控、三组 YOLO 标签白名单）。
- `VisionDetector` 类：`analyze(image, yolo_detections)`→`VisionDecision`（合并 OpenCV+YOLO 选最佳）、**`color_mask(image, profile_name, roi_y_ratio)`**（被 LaneFollower 共享）、`detect_color_regions`（轮廓+可选三角门）、`detect_obstacles`/`detect_target_cards`/`detect_black_obstacles`（遗留别名）、`crop_detection`、`draw_debug`、`_best_obstacle`/`_best_target_card`/`_zone_for_bbox`。
- `parse_ai_targets_msg(msg, min_confidence)`（`ai_msgs/PerceptionTargets`→`Detection`）、`load_detector_config(path)`、`main(argv)` CLI。

### 3.5 `llm_client.py` — 视觉 LLM 客户端（Phase G）

OpenAI 兼容，DashScope compatible-mode 直接适配。

- `LlmConfig`（dataclass）：`mode` `api_url` `model` `api_key_env` `timeout` `prompt` `max_tokens` `temperature` `max_image_bytes` `max_image_dim` `jpeg_quality` `downscale_enabled`。
- `LlmClient` 类：`analyze(image_path)`（按 mode 分派 disabled/placeholder/openai-compatible）、`_analyze_openai_compatible`、`_image_data_url`（读图→可选降采样→base64）、**`_downscale_jpeg(data)`**（cv2 缩到 `max_image_dim` 内重编码 JPEG，cv2 缺失返回 None 回退原图）、`_request_payload`（text+image_url）、`_post_json`、`_extract_text`（兼容 `choices[].message.content` 字符串/列表、`output_text`）。`main(argv)` CLI（`LLM_RESULT:`/`LLM_ERROR` 返回码 2）。

### 3.6 `decision_parser.py` 与 `qr_capture.py`

- **`decision_parser.py`**：QR 文本（JSON / URL query / `key=value` / 自由中英文）→ 左右转。常量 `LEFT_VALUES`/`RIGHT_VALUES`/`LEFT_PHRASES`/`RIGHT_PHRASES`/`PRIORITY_KEYS`。`QrInstruction`（dataclass）。`parse_qr_instruction(text)`（多来源候选打分：字段 0.95/0.75 > 文本 0.55 > 令牌 0.45，冲突判 ambiguous）、`parse_direction(text)`（便捷）。`main(argv)` CLI 打印 `QR_PARSE`。
- **`qr_capture.py`**：从 `/image` 抓帧或读本地图解码 QR 的独立调试工具。`QrCaptureNode(Node)`，多策略解码与 `auto_mission` 的 `QrDecoder` 同源（原图/灰度/放大→透视矫正→裁剪 6x→Otsu）。

---

## 四、支持工具链（数据/调优/回放/交付）

所有工具均为独立 CLI（`ros2 run origin_competition_auto <tool>`），统一 `build_parser()` + `main(argv)` 入口。绝对路径与 console_scripts 名见 §六。

| 工具 | 用途 | 关键符号 / 子命令 | 读写 |
|------|------|------------------|------|
| `competition_run.py` | 一键运行编排：`system_check → (可选)YOLO → auto_mission → mission_replay → 后置 check` | `DEFAULT_PROFILES`、`load_profiles`、`build_auto_mission_command`、`start_yolo`/`wait_for_yolo`、**`parse_mission_output`**（解析 `STATE:`/`SAVED_IMAGE:`/`MISSION_SUMMARY`/`EVENT {json}`）、`run` | 读 `run_profiles.json`；写 `runs/<ts>_<profile>/`（command.txt/system_check.json/mission_output.log/run_summary.json） |
| `system_check.py` | 只读赛前自检：服务/话题图/相机帧/电压 | `CheckResult`、`SystemCheckNode`、`service_active`、`run_checks`（`origincar-base`/`camera.service`/`/image`/`/cmd_vel` 无外部 pub/`/hobot_dnn_detection`/电压） | `--json`，无文件写。被 competition_run/solution_audit 调用 |
| `motion_calibration.py` | 限幅 `/cmd_vel` 运动标定 | `MotionStep`、`MotionCalibrationNode`、`validate_args`（linear≤0.12/angular≤0.8 硬限）、`sequence_for`（stop/forward/reverse/left/right/all） | 仅发布，`--dry-run` 只打印；无订阅者则中止 |
| `dataset_capture.py` | 采集帧→YOLO 目录，可 OpenCV 粗标 | `CameraFrameBuffer`、`DEFAULT_CLASSES`、`auto_label_image`（用 VisionDetector 生成框）、`save_sample`、`--auto-label-mode`/`--split`/`--count` | 写 datasets 目录 `images/`/`labels/`/`debug/`/`classes.txt`，追加 `metadata.jsonl` |
| `dataset_audit.py` | 数据集完整性审计 + dataset.yaml | `DatasetReport`、**`audit_dataset`**（被复用）、`write_yolo_yaml`、`draw_label_overlay`、`--strict` | 读 dataset；写 dataset.yaml/report-json/preview |
| `vision_tune.py` | 视觉阈值网格搜索（标签作 GT 评分） | `make_grid`（9 维）、`evaluate_config`、`metrics`（P/R/F1）、`write_best_config` | 写 `vision_tune_report.json` + `best_detector_config.json` |
| `mission_replay.py` | 离线主流程回放（无车无 ROS） | **`replay_image`**（QR→指令→视觉→裁剪→可选 LLM）、`infer_next_state`、`summarize`、`--skip-llm`/`--save-overlays` | 读图+mission config；写 overlays/crops/report |
| `field_session.py` | 场地采集会话（清单/进度/复盘） | `DEFAULT_TASKS`（6 类场景）、`create_session`、`build_status`（按 scene_label 统计）、`review_session`；子命令 `create`/`status`/`review` | 写会话目录 manifest/status/采集清单.md/commands.sh |
| `field_data_review.py` | 复盘流水线：audit+replay+tune→补丁 | `run_dataset_audit`/`run_mission_replay`/`run_vision_tune`、`replay_rates`、**`make_recommendations`**（对照 min-rate 门禁出 readiness） | 写 `field_data_review_report.json` + **`recommended_mission_config_patch.json`** + **`recommended_run_profiles_patch.json`** |
| `apply_review_recommendations.py` | 安全应用复盘补丁到 config | `MISSION_ALLOWED_KEYS`/`RUN_PROFILE_ALLOWED_KEYS`（白名单）、`validate_mission_patch`、`validate_run_profiles_patch`（cruise_linear≤0.08）、默认 dry-run + `.bak` 备份 | 读 patch + config；`--apply` 写回 + 备份 |
| `yolo_pipeline.py` | YOLO 训练/导出/部署产物固化（不实训） | 子命令 `prepare`（跑 audit+写 workconfig/脚本/manifest）/`validate`/`promote`（固化 pt/onnx/rdk_bin）/`workconfig` | 写 models 目录全套产物 |
| `solution_audit.py` | 方案就绪度验收矩阵 | `EXPECTED_TOOLS`/`EXPECTED_DOCS`、各 `check_*`（验 safe_static 禁动+限速≤0.08、找成功 run、验 rdk_bin）、`build_matrix`、`remaining_gaps` | 读全项目证据；写 `--output-json`/`--output-md`，fail 非零退出 |
| `handoff_bundle.py` | 赛前交付包打包 | `build_bundle`、`copy_docs`/`copy_configs`/`copy_runs`/`copy_models`、`write_readme`、`--zip` | 写 `handoff_<ts>/`（docs/config/launch/runs/models/bundle_index.json） |

**协作流程**：采集（`field_session create`→`dataset_capture`）→ 复盘（`field_data_review` 编排 `dataset_audit`+`mission_replay`+`vision_tune`，出就绪度+补丁）→ 固化（`apply_review_recommendations` 白名单校验后写回 config）→ YOLO 分支（`yolo_pipeline`）→ 运行回归（`competition_run` 按 profile 实跑）→ 验收交付（`solution_audit`+`handoff_bundle`，只消费产物不反向依赖）。

---

## 五、配置键清单（`config/mission_defaults.json`）

> 这是 `MissionConfig` 默认值的来源。**改任何阈值/速度/超时都在这里改**。`auto_mission` 必须带 `--config <path>` 才加载本文件，否则用 dataclass 内置默认（其中 `llm_api_key_env` 内置默认是 `OPENAI_API_KEY`，本文件改为 `DASHSCOPE_API_KEY`）。

| 分组 | 键（默认值） |
|------|-------------|
| 话题/IO | `cmd_vel_topic`(/cmd_vel) `image_topic`(/image) `voltage_topic`(/PowerVoltage) `state_topic`(/competition/state) `odom_topic` `imu_topic` `debug_dir` `rate_hz`(10) `stop_repeat`(5) |
| odom 航向 | `motion_state_enabled` `use_imu_yaw` `waypoint_min_dist` `waypoint_min_yaw` `odom_stale_timeout` |
| 运动限幅 | `max_linear`(0.08) `max_angular`(0.5) `cruise_linear`(0.03) `turn_angular`(0.2) `loop_linear`(0.03) `loop_angular`(0.2) |
| 循迹(Phase C) | `lane_follow_enabled`(true) `lane_roi_y_ratio`(0.6) `lane_roi_height_ratio`(0.35) `lane_kp`(0.6) `lane_kd`(0.1) `lane_base_linear`(0.03) `lane_min_mask_area_ratio`(0.01) `lane_lost_behavior`(creep) |
| 环/走廊 | `ring_bias`(0.25) `corridor_length_m`(2.0) `green_exit_ratio`(0.15) `loop_complete_fraction`(0.92) `loop_timeout`(40) `corridor_timeout`(30) `corridor_require_green_exit`(false) |
| 过渡/采集 | `transition_settle_s`(0.3) `marker_capture_min_area`(0.02) |
| 返航(Phase F) | `return_mode`(trajectory) `return_waypoint_stride`(3) `return_min_segment_dist`(0.05) `return_drive_linear`(0.04) `return_turn_angular`(0.3) `return_heading_tol`(0.10) `return_dist_tol`(0.05) `return_segment_timeout`(12) `return_total_timeout`(60) `return_linear`(-0.03) `return_angular`(0) `return_duration`(1.0) |
| QR 接近 | `preflight_timeout`(5) `drive_to_qr_timeout`(20) `drive_to_qr_max_distance`(2.5) `qr_min_area_ratio`(0.01) `qr_center_band`(0.4) `qr_seek_after_s`(4) `qr_seek_angular`(0.15) `scan_qr_timeout`(8) |
| 运行时 | `loop_duration`(2) `capture_timeout`(3) `max_runtime`(120) `image_stale_timeout`(3) |
| YOLO | `use_yolo_detections`(true) `yolo_topic`(/hobot_dnn_detection) `yolo_stale_timeout`(1) `yolo_min_confidence`(0.45) |
| 避障 | `obstacle_avoid_enabled`(true) `obstacle_turn_angular`(0.18) `obstacle_roi_y_ratio`(0.45) `obstacle_min_area_ratio`(0.015) `obstacle_danger_area_ratio`(0.035) `obstacle_center_band_ratio`(0.36) |
| 颜色 | `black_v_max`(85) `black_s_min`(20) `obstacle_color`(dark_blue) `lane_color`(yellow) `marker_color`(orange) `green_color`(green) `require_triangle`(false) `color_profiles`(dark_blue/yellow/orange/green/black 各 HSV min/max) |
| 目标裁剪 | `target_crop_enabled`(true) `target_min_area_ratio`(0.025) `target_white_s_max`(90) `target_white_v_min`(150) |
| LLM(Phase G) | `llm_mode`(placeholder) `llm_api_url`(dashscope compatible-mode) `llm_model`(qwen-vl-plus) `llm_api_key_env`(DASHSCOPE_API_KEY) `llm_timeout`(20) `llm_prompt`(中文医院巡检) `llm_max_image_dim`(1024) `llm_jpeg_quality`(85) `llm_downscale_enabled`(true) `llm_result_topic`(/competition/llm_result) |

**`run_profiles.json`** 三档：`safe_static`（纯静态，`motion_capable=false` 禁动，limit≤0.08）、`bench_motion`（架空低速冒烟，`motion_capable=true`）、`field_low_speed`（保守地面测试，max_runtime 90s）。三档共同 `require_yolo=false` `llm_mode=placeholder` `cruise_linear=0.03`。

---

## 六、入口与命令速查（setup.py console_scripts）

20 个可执行（`ros2 run origin_competition_auto <名>`）：核心 `auto_mission`(主状态机) `competition_run`(编排) `motion_state_debug`(motion_state:main) `lane_debug`(lane_follow) `vision_debug`(vision_detector) `llm_debug`(llm_client) `qr_parse_debug`(decision_parser) `qr_capture` `motion_calibration`；工具链 `mission_replay` `dataset_capture` `dataset_audit` `vision_tune` `system_check` `field_data_review` `apply_review_recommendations` `field_session` `yolo_pipeline` `solution_audit` `handoff_bundle`。

`data_files` 安装：`config/*.json` → `share/origin_competition_auto/config`；`launch/*.launch.py` → `share/.../launch`。

**常用调用**（板子，需先 source）：
```bash
# 全流程推演（不碰硬件）
ros2 run origin_competition_auto auto_mission --config <cfg> --dry-run --mock-qr-content '{"direction":"left"}'
# 静态自检不动车
ros2 run origin_competition_auto auto_mission --config <cfg> --no-motion --start-state <STATE>
# LLM 真实调用（需 export DASHSCOPE_API_KEY）
ros2 run origin_competition_auto auto_mission --config <cfg> --no-motion --start-state CAPTURE_TARGET_IMAGE --llm-mode openai-compatible
```

> **LLM key**：板子存 `/root/dev_ws/.dashscope_key`（chmod 600，不进 git），运行时 `export DASHSCOPE_API_KEY=$(cat /root/dev_ws/.dashscope_key)`。可用模型 `qwen-vl-plus`（`qwen-vl-max-latest` 该 key 无权限 403）。

---

## 七、常见修改场景速查

> 「去哪改」列均为可直接 `Grep` 的符号名 / 配置键 / 状态名。

| 想做什么 | 去哪改 |
|----------|--------|
| 改任何速度/阈值/超时 | `config/mission_defaults.json` 对应键（见 §五）；**不要改代码默认值** |
| 改黄色/橙色/深蓝/绿色 HSV | `mission_defaults.json` `color_profiles.<名>`；用 `dataset_capture` 采正/负样本，用 `lane_debug --color <名> --config <cfg>` 看 `mask_area_ratio`，深蓝/橙色再用 `vision_debug` 看叠加图 |
| 改循迹手感（蛇行/迟钝） | `lane_kp` / `lane_kd` / `lane_base_linear`；逻辑在 `lane_follow.py` `LaneFollower.compute` |
| 改环贴边方向 | `ring_bias` + `_state_loop` 里的 `side_mode`（left ring→right_edge） |
| 改环完成判定 | `loop_complete_fraction`（×360°）/ `loop_timeout`；逻辑 `_run_ring_follow` 用 `integrated_yaw` |
| 改返航策略 | `return_mode`（trajectory/timed）；轨迹逻辑 `_state_return_home`/`_execute_return_segments`/`reverse_plan`，调 `return_waypoint_stride`/`return_heading_tol` |
| 改 QR 接近停车点 | `qr_min_area_ratio`(面积越大停越近) / `qr_center_band`；逻辑 `_drive_probe_qr` |
| 改避障灵敏度 | `obstacle_danger_area_ratio` / `obstacle_center_band_ratio`；逻辑 `_handle_drive_obstacle` |
| 改 QR 指令文本格式 | `decision_parser.py` `LEFT_VALUES`/`RIGHT_VALUES`/`LEFT_PHRASES`/`PRIORITY_KEYS` |
| 改 LLM 模型/提示词 | `llm_model` / `llm_prompt`；逻辑 `llm_client.py` |
| 改 LLM 图像缩放 | `llm_max_image_dim` / `llm_jpeg_quality`；逻辑 `_downscale_jpeg` |
| 加新状态 | `STATE_SEQUENCE` + `STATE_ZONES` + `_run_state` 分发表 + 新 `_state_xxx` 方法 + `flow_from_start` |
| 加分区过渡短停 | 在状态返回前调 `_settle_transition('X->Y')`；时长 `transition_settle_s` |
| 加运行档 | `run_profiles.json` 新 profile（需含 `PROFILE_REQUIRED_KEYS`）；`competition_run` 自动识别 |
| 改赛前自检项 | `system_check.py` `run_checks` |
| 改验收门禁 | `solution_audit.py` `EXPECTED_TOOLS`/`check_*` |

---

## 八、关键约定与坑（务必先读）

1. **配置加载**：`auto_mission` 不带 `--config` 时用 dataclass 内置默认（`llm_api_key_env=OPENAI_API_KEY`、`api_url=openai`），**必须显式 `--config mission_defaults.json`** 才走 DashScope。`--llm-mode` 等 CLI 只 override 单项。
2. **轨迹连续性**：CORRIDOR/LOOP 阶段只能调 `reset_yaw()`（重置积分 yaw 计圈），**绝不能调 `reset()`**——后者清空全程轨迹会破坏返航。轨迹从 `DRIVE_TO_QR` 入口 `reset()`+`start_recording()` 起贯穿到返航。
3. **IMU 缓漂**：静止时 yaw ~0.5°/8s 漂移，环判定用 `integrated_yaw` 相对累计 + timeout，不用绝对 yaw。
4. **速度硬限**：`MotionCommander._clamp` + `MissionConfig.validate` 双重钳制；`motion_calibration` 另有 linear≤0.12/angular≤0.8 硬限。`--no-motion` 抑制所有 `/cmd_vel`。
5. **家里测试图黄色 HSV 偏松**会误触发循迹；当前 `vision_tune` 不扫描四颜色 `color_profiles`，场地日要用 `lane_debug` 对黄线正/负样本看 `mask_area_ratio` 后手动收紧。
6. **底盘串口节点**会因 USB 抖动崩溃，现象是 `/cmd_vel` 无响应；修复 `restart origincar-base.service`。
7. **EVENT 日志契约**：`parse_mission_output`（competition_run）依赖 `STATE: `/`SAVED_IMAGE: `/`EVENT {json}`/`MISSION_SUMMARY` 行前缀。新增日志行不要用这些前缀，否则污染解析。dry-run 的 `ZONE:` 行已验证不破坏解析。
8. **纯函数模块离线可测**：`lane_follow`/`motion_state`（helper）/`decision_parser`/`vision_detector` 不依赖 rclpy，改这些优先写离线断言测试再上板。
9. **文件编辑**：单次 Write/Edit ≤50 行，超出分多次（项目约定）。



