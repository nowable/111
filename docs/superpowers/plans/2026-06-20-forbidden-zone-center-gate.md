# Forbidden Zone Yellow/Black Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bridge entrance-guessing logic with a three-stage flow: fixed 3-second forward push, right-turn search until yellow appears, then left-turn travel along the forbidden zone until black line appears and triggers immediate right-turn entry.

**Architecture:** Keep QR backup, then split the bridge into three explicit phases. Yellow is only used as a binary “forbidden zone has appeared” signal, and black line is the only signal that triggers the final right turn into the yellow safety zone. Once yellow appears, the robot should not go straight; it should turn left and run along the forbidden zone edge until black line is reacquired, then immediately right-turn into the yellow area at maximum angular speed.

**Tech Stack:** Python, ROS2, OpenCV, existing `origin_competition_auto` mission code.

---

### Task 1: Add Three-Stage Bridge Configuration

**Files:**
- Modify: `origin_competition_auto/config/mission_defaults.json`
- Modify: `origin_competition_auto/origin_competition_auto/auto_mission.py`

- [ ] **Step 1: Add black-line reentry config keys to defaults**

Add the needed bridge phase keys near the other `bridge_*` keys in `origin_competition_auto/config/mission_defaults.json`:

```json
"bridge_phase1_forward_s": 3.0,
"bridge_yellow_detect_min_ratio": 0.03,
"bridge_yellow_search_linear": 0.08,
"bridge_yellow_search_angular": 0.35,
"bridge_yellow_follow_linear": 0.08,
"bridge_yellow_follow_angular": 0.30,
"bridge_black_search_linear": 0.06,
"bridge_black_reentry_enabled": true,
"bridge_black_reentry_roi_y_ratio": 0.45,
"bridge_black_reentry_confirm_frames": 2,
"bridge_black_reentry_turn_angle_deg": 90.0,
"bridge_black_reentry_turn_angular": 0.5,
"bridge_black_reentry_forward_s": 1.2,
```

- [ ] **Step 2: Add the dataclass fields**

Add matching fields near the other bridge config fields in `origin_competition_auto/origin_competition_auto/auto_mission.py`.

```python
bridge_phase1_forward_s: float = 3.0
bridge_yellow_detect_min_ratio: float = 0.03
bridge_yellow_search_linear: float = 0.08
bridge_yellow_search_angular: float = 0.35
bridge_yellow_follow_linear: float = 0.08
bridge_yellow_follow_angular: float = 0.30
bridge_black_search_linear: float = 0.06
bridge_black_reentry_enabled: bool = True
bridge_black_reentry_roi_y_ratio: float = 0.45
bridge_black_reentry_confirm_frames: int = 2
bridge_black_reentry_turn_angle_deg: float = 90.0
bridge_black_reentry_turn_angular: float = 0.5
bridge_black_reentry_forward_s: float = 1.2
```

- [ ] **Step 3: Keep the field in exported config mappings**

Ensure the new field is included anywhere nearby bridge config is serialized through the mission config mapping block.

- [ ] **Step 4: Verify syntax**

Run:

```bash
python -m py_compile origin_competition_auto/origin_competition_auto/auto_mission.py
```

Expected: no output.

### Task 2: Replace Bridge Search With Three Explicit Phases

**Files:**
- Modify: `origin_competition_auto/origin_competition_auto/auto_mission.py`

- [ ] **Step 1: Add a short fixed forward phase after backup**

After backup and post-backup pulse, push forward for a fixed 3 seconds before any searching.

```python
deadline = time.monotonic() + self.config.bridge_phase1_forward_s
while time.monotonic() < deadline and self.guard.runtime_ok():
    self.motion.publish(self.config.bridge_align_linear, 0.0)
    self._spin_sleep(interval)
```

- [ ] **Step 2: Add yellow-as-forbidden-zone detection**

Reuse `VisionDetector.color_mask(..., lane_color, ...)` to compute a yellow area ratio, but use it only as a binary signal that the forbidden zone has appeared.

```python
yellow_mask, _ = self.vision_detector.color_mask(image, self.config.lane_color, roi_y_ratio=self.config.bridge_target_roi_y_ratio)
yellow_ratio = float((yellow_mask > 0).sum()) / float(max(1, yellow_mask.size))
yellow_seen = yellow_ratio >= self.config.bridge_yellow_detect_min_ratio
```

- [ ] **Step 3: Before yellow appears, walk forward while turning right**

If `yellow_seen` is false, keep searching with forward motion plus right turn.

```python
self.motion.publish(
    self.config.bridge_yellow_search_linear,
    -abs(self.config.bridge_yellow_search_angular),
)
```

- [ ] **Step 4: After yellow appears, switch to left-turn travel along the forbidden zone**

Once `yellow_seen` becomes true, stop the right-turn search and instead turn left while moving forward so the robot runs along the forbidden zone edge while checking for black line.

```python
self.motion.publish(
    self.config.bridge_yellow_follow_linear,
    abs(self.config.bridge_yellow_follow_angular),
)
```

- [ ] **Step 5: Reuse black line follower during straight search**

Create a black `LaneFollower` search helper for the entrance region and count stable hits only during the straight-search phase.

```python
black_reentry_follower = LaneFollower(
    self.vision_detector,
    'black',
    self.config.lane_follow_config(
        bias=0.0,
        side_mode='center',
        roi_y_ratio=self.config.bridge_black_reentry_roi_y_ratio,
        roi_height_ratio=1.0 - self.config.bridge_black_reentry_roi_y_ratio,
    ),
)
```

- [ ] **Step 6: Trigger the fixed right turn when black line is reacquired**

When `black_reentry_hits` reaches the configured confirmation threshold, execute the right-turn helper immediately and return bridge success.

```python
if black_reentry_hits >= self.config.bridge_black_reentry_confirm_frames:
    self._bridge_black_reentry_turn(interval)
    return True
```

- [ ] **Step 7: Keep the existing right-turn helper and short forward settle**

After the turn completes, push forward briefly into the yellow area.

```python
deadline = time.monotonic() + self.config.bridge_black_reentry_forward_s
while time.monotonic() < deadline and self.guard.runtime_ok():
    self.motion.publish(self.config.bridge_align_linear, 0.0)
    self._spin_sleep(interval)
```

- [ ] **Step 8: Verify syntax again**

Run:

```bash
python -m py_compile origin_competition_auto/origin_competition_auto/auto_mission.py
```

Expected: no output.

### Task 3: Deploy And Verify On Board

**Files:**
- Copy: local `origin_competition_auto/origin_competition_auto/auto_mission.py` to board source tree
- Copy: local `origin_competition_auto/config/mission_defaults.json` to board source tree, or patch the single new key/value in place

- [ ] **Step 1: Upload the updated mission code**

Run:

```bash
scp <local auto_mission.py> root@192.168.43.155:/root/dev_ws/src/origin_competition_auto/origin_competition_auto/auto_mission.py
```

- [ ] **Step 2: Upload or patch the updated config**

Ensure board config contains the new black-line reentry keys.

- [ ] **Step 3: Verify board file syntax and config values**

Run:

```bash
ssh root@192.168.43.155 "python3 -m py_compile /root/dev_ws/src/origin_competition_auto/origin_competition_auto/auto_mission.py && python3 - <<'PY'
import json
from pathlib import Path
p = Path('/root/dev_ws/src/origin_competition_auto/config/mission_defaults.json')
obj = json.loads(p.read_text())
for key in [
    'bridge_black_reentry_enabled',
    'bridge_black_reentry_roi_y_ratio',
    'bridge_black_reentry_confirm_frames',
    'bridge_black_reentry_turn_angle_deg',
    'bridge_black_reentry_turn_angular',
    'bridge_black_reentry_forward_s',
]:
    print(key, obj.get(key))
PY"
```

- [ ] **Step 4: Field verification command**

Run the existing bridge test path:

```bash
source /opt/ros/humble/setup.bash
source /opt/tros/humble/setup.bash
source /root/dev_ws/install/setup.bash
export DASHSCOPE_API_KEY=$(cat /root/dev_ws/.dashscope_key)
CFG=/root/dev_ws/src/origin_competition_auto/config/mission_defaults.json
ros2 run origin_competition_auto auto_mission --config $CFG --start-state SCAN_QR
```

Expected field behavior:
- after backup it should first push forward for 3 seconds
- before yellow appears, it should move while turning right
- once yellow appears, it should switch to left-turn travel along the forbidden zone
- once black line is reacquired, it should immediately execute one right turn and enter the yellow area
