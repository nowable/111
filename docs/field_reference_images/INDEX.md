# 现场真实参考帧（车载相机实拍）

这些是 2026-06-14 / 06-17 外场实跑时从板子相机拉下来的真实帧。
新人调 HSV 阈值 / 理解每个区域相机到底看到什么，先看这些图，比凭空想象快得多。
（图来自板上 `/root/dev_ws/debug/`，已是唯一副本之外的拷贝。）

| 文件 | 是什么 | 用途 |
|---|---|---|
| `scene_qr.jpg` | 大厅看二维码立牌的视角 | QR 接近 / `qr_min_area_ratio` 调停车距离 |
| `scene_0617.jpg` | 6-17 大厅全景 | 浅蓝地面 + 黑虚线 + 蓝桶整体观感 |
| `obs_idealpos_0617.jpg` | **车正对蓝桶、处于"刚好该避障"的距离** | 标定 `obstacle_danger_area_ratio`：用户说这个距离开始触发避障最合适，实测桶占屏 ~9.6% |
| `corridor_first.jpg` | 刚进黄色通道 | 黄道循迹 / 区分实心黄道 vs 两侧黄白网格墙 |
| `corridor_last.jpg` | 通道末端接近 C 环 | green_floor 退出判定参考 |
| `mouth.jpg` | 通道口（1m gap，两侧网格墙） | bridge 对准通道口的难点现场 |
| `lane_true.jpg` | 黑虚线被正确识别的帧 | 黑线 HSV (`black` profile) 正样本 |
| `lane_false.jpg` | 黑线误识别 / 丢失的帧 | 黑线 HSV 负样本 |
| `b1_blackdetect.jpg` | 黑线检测中间结果 | 同上 |
| `qrfail.jpg` | 二维码解码失败的帧 | 运动模糊 / 距离导致解码失败的样子 |
| `end_0617.jpg` | 6-17 一次 run 结束时的视角 | 复盘 |
| `lost_0617.jpg` | 丢线时的视角 | 丢线恢复逻辑调参参考 |
