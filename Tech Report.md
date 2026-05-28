# RoboSub Autonomous Submarine — Technical Report

**Version 3.0 — May 2026**

---

## 1. Mission Overview

The simulation models a full out-and-back competition run. The submarine executes the following ten-task mission plan in order:

| # | Task | Description |
|---|------|-------------|
| 1 | `GateTask` | Locate and pass through the start gate |
| 2 | `StabilizeTask` (3 s) | Arrest residual velocity after gate passage |
| 3 | `VictoryDanceTask` | 360° yaw spin followed by 360° roll spin |
| 4 | `StabilizeTask` (5 s) | Recover level flight after the roll maneuver |
| 5 | `SlalomTask` | Navigate the outbound slalom course |
| 6 | `StabilizeTask` (3 s) | Arrest residual velocity before return |
| 7 | `SlalomTask` (reversed) | Navigate the return slalom on the opposite side |
| 8 | `GateTask` | Re-acquire and pass through the start gate on return |
| 9 | `TimedHoverTask` (10 s) | Hold station at mission depth |
| 10 | `SurfaceTask` | Ascend to the surface to end the run |

The two `SlalomTask` instances share information: the forward pass records which side of the red pole the sub used, and the reversed pass automatically selects the opposite side.

---

## 2. Architecture

### 2.1 File Tree

```
RoboSim/
├── main.py               — mission assembly and entry point
├── simulator.py          — physics engine, rendering, sensor generation
├── world.py              — 3-D obstacle and gate geometry
├── config.py             — simulation constants, HSV colour ranges
├── data_structures.py    — SensorSuite, ThrusterCommands, VisionData
├── utils.py              — angle_diff and other math helpers
└── ai/
    ├── submarine.py      — Submarine class: mission execution, shared control primitives
    ├── vision.py         — find_blobs_hsv colour-blob detector
    └── tasks/
        ├── __init__.py
        ├── task_base.py          — Task ABC and TaskStatus enum
        ├── gate_task.py          — GateTask
        ├── victory_dance_task.py — VictoryDanceTask
        ├── slalom_task.py        — SlalomTask
        ├── stabilize_task.py     — StabilizeTask
        ├── hover_task.py         — HoverTask (base class)
        ├── timed_hover_task.py   — TimedHoverTask
        └── surface_task.py       — SurfaceTask
```

### 2.2 Task Interface

Every task inherits from `Task` and implements three methods:

```python
def process_vision(self, sub, camera_image) -> VisionData: ...
def execute(self, sub, dt, sensors, vision_data, config) -> (TaskStatus, ThrusterCommands): ...
def reset(self): ...
```

An optional `on_start(sub, sensors)` hook is called by `Submarine.update()` immediately after the previous task completes. `TaskStatus` is either `RUNNING` or `COMPLETED`.

### 2.3 Hardware Model

**Horizontal thrusters** — four thrusters in an X-drive configuration:

```
h_port_bow      = surge + sway + yaw
h_starboard_bow = surge - sway - yaw
h_port_aft      = surge - sway + yaw
h_starboard_aft = surge + sway - yaw
```

**Vertical thrusters** — two thrusters mounted on the port and starboard sides of the hull. Because of this placement they produce **roll torque** about the fore-aft axis, not pitch:

```
v_port      = heave + roll
v_starboard = heave - roll
```

All six commands are normalized together so the largest magnitude in any channel does not exceed 1.0.

**Sensor Suite** (`SensorSuite` dataclass):

| Field | Description |
|-------|-------------|
| `camera_image` | Forward-facing camera as a `np.ndarray` (H×W×3, BGR) |
| `depth` | Pressure-derived depth (m) |
| `heading` | Compass heading (°, 0–360) |
| `pitch` | Nose-up pitch angle (°) |
| `roll` | Port-up roll angle (°, wraps ±180°) |
| `imu.gyro_z` | Yaw rate (rad/s) |
| `angular_velocity_y` | Pitch rate (rad/s) |
| `angular_velocity_x` | Roll rate (rad/s) |
| `x, y` | World-frame position (m) |
| `velocity_x, velocity_y` | World-frame velocity (m/s) |
| `velocity_z` | Vertical velocity (m/s, positive = descending) |

**Roll physics** — roll is integrated with modular wrap to allow full 360° rotation:

```python
self.subPhysics.roll = ((self.subPhysics.roll + degrees(angular_velocity_x * dt) + 180) % 360) - 180
```

**Rendering** — the simulator uses OpenCV (`cv2`) exclusively. The main window and the camera view are plain `np.ndarray` images drawn with `cv2.line`, `cv2.rectangle`, `cv2.fillPoly`, `cv2.putText`, and displayed with `cv2.imshow`. The camera feed is rotated by `-roll` degrees using `cv2.warpAffine` so the image horizon tilts with the submarine; vision always runs on the unrotated array so detection coordinates are unaffected. All colour constants in `config.py` are in BGR order to match OpenCV's channel convention.

### 2.4 Control Primitives (Submarine class)

`Submarine` owns all reusable PID/damping helpers. Tasks call these instead of computing thruster commands directly.

| Method | Purpose |
|--------|---------|
| `_mix_and_normalize_commands(surge, sway, heave, yaw, pitch, roll)` | Thruster mixing + normalization |
| `_get_damping_commands(sensors, target_depth)` | Full-stop with depth + roll recovery |
| `_get_pid_hover_commands(sensors, dt, tx, ty, td, ...)` | XY PID position hold |
| `get_go_to_visual_target_commands(sensors, nav_x, nav_y, surge)` | Visual servoing — steer toward pixel target |
| `get_search_commands(sensors, target_depth, ...)` | Slow yaw sweep at depth |
| `get_depth_change_commands(sensors, depth, heading, surge)` | Depth change with heading hold |
| `get_spin_damping_commands(sensors)` | Damp translational velocity after a spin |

**Roll recovery PD** — used in `_get_damping_commands` and any task that must maintain level flight:

```python
roll = clip(-sensors.roll * ROLL_RECOVERY_P_GAIN
            - sensors.angular_velocity_x * ROLL_RECOVERY_D_GAIN, -1.0, 1.0)
```

Default gains: `ROLL_RECOVERY_P_GAIN = 0.03`, `ROLL_RECOVERY_D_GAIN = 2.5`.

### 2.5 ROS2 Portability

The task architecture maps cleanly onto ROS2 actions. Each `Task` subclass corresponds to a ROS2 action server; `Submarine.update()` is the action client dispatcher. `SensorSuite.camera_image` is a `np.ndarray`, which `cv_bridge` converts directly to and from `sensor_msgs/Image` with a single call — there is no longer a simulator-specific type in the interface. All control logic inside tasks and `Submarine` is pure Python with no simulator dependencies.

---

## 3. Task Descriptions

### 3.1 GateTask

**States:** `SEARCHING → ALIGNING → APPROACHING → CLEARING_GATE`

**Vision** — detects gate poles by pairing red blobs (pole top) with black blobs (pole body) using proximity thresholds, then selecting the pair with the smallest height difference. From the matched pair the task computes:
- `left_passage_center_x` / `right_passage_center_x` — the ¼ and ¾ horizontal positions across the gate opening
- `gate_center_y` — one-third up from the gate bottom (to clear the lower crossbar)
- `apparent_height` — used to judge distance

**SEARCHING** — yaws at a constant rate while holding depth. Tries two search depths (`0.8 m` and `1.5 m`) before giving up. Transitions to `ALIGNING` as soon as the gate is visible.

**ALIGNING** — visual servo: yaw P+D to center the gate, heave P+D to drive `gate_center_y` to the camera horizontal midline, translational damping to hold position. Completes when the gate is centered within 10 px, yaw rate < 0.05 rad/s, and depth is settled — or after an 8-second timeout.

**APPROACHING** — drives forward using `get_go_to_visual_target_commands`. Passage selection follows this priority:
1. If `forced_passage` is set (return run), always target that specific side.
2. Otherwise, lock to whichever passage is nearest to the camera centre on first sighting and hold that choice for the rest of the approach.

On completion, the chosen side is written to `sub.gate_passage_side`.

**CLEARING_GATE** — surges forward at full speed for 3.5 s to guarantee complete passage, then signals `COMPLETED`.

**Return-gate coordination** — `GateTask.on_start` checks `sub.gate_passage_side`. If it is already set (meaning an earlier `GateTask` cleared the gate), `forced_passage` is set to the opposite side so the sub re-enters through the same physical opening it exited.

### 3.2 VictoryDanceTask

A four-state celebration maneuver performed between gate passage and the slalom.

**States (indexed as `dance_step`):**

| Step | Name | Behaviour |
|------|------|-----------|
| 0 | `SPIN_YAW` | Apply constant yaw command (0.5) while holding depth. Track accumulated heading change; advance when ≥ 355°. |
| 1 | `RECOVER_YAW` | PID hover back to `dance_center_heading`. Complete when heading error < 5° and yaw rate < 0.15 rad/s. |
| 2 | `SPIN_ROLL` | Apply constant roll command (0.6). Heave is scaled by `sign / max(0.3, |cos(roll)|)` to maintain depth authority when inverted. Track accumulated roll with wraparound; advance when ≥ 355°. |
| 3 | `RECOVER_ROLL` | PD roll recovery + sign-aware heave scale + PID hover. Complete when `|roll| < 5°` and roll rate < 0.15 rad/s. |

**Accumulated angle tracking** uses `angle_diff` for yaw and a wraparound delta for roll:

```python
delta = sensors.roll - self.last_roll
if delta > 180:  delta -= 360
if delta < -180: delta += 360
self.roll_accumulated += delta
```

**Depth authority when inverted** — when the sub is past 90° roll the vertical thrusters point downward in the world frame. The heave command is inverted and scaled up to compensate:

```python
cos_roll = math.cos(math.radians(sensors.roll))
sign = 1.0 if cos_roll >= 0 else -1.0
heave_scale = sign / max(0.3, abs(cos_roll))   # clamped to avoid division blow-up near 90°
```

### 3.3 StabilizeTask

Brings the submarine to a complete stop. On first execution it latches the current heading and depth as targets. While the sub is still coasting (speed > 0.1 m/s) it slides `target_x/y` with the submarine so the position P-term stays near zero and only D-gain braking acts — this prevents a backward lurch from the I-term. Completes when the elapsed time exceeds `duration` **and** horizontal speed drops below `speed_threshold` (default 0.05 m/s).

Roll recovery is applied every tick:

```python
roll_cmd = clip(-sensors.roll * ROLL_RECOVERY_P_GAIN
                - sensors.angular_velocity_x * ROLL_RECOVERY_D_GAIN, -1.0, 1.0)
return sub._get_pid_hover_commands(..., roll=roll_cmd)
```

The 5-second `StabilizeTask` placed after `VictoryDanceTask` is specifically sized to allow the roll angle to converge before the slalom begins.

### 3.4 SlalomTask

Navigates a lane of alternating red and white poles by making repeated gate passes (gatelets).

**States:** `DIVING → SEARCHING → ALIGNING → APPROACH → CLEAR → REALIGNING`

**Pass-side determination** — on the first pass the task determines the side by observing which side of the camera the red pole is on. If the red pole is in the right half of the frame the sub passes to the left of it (red is the port boundary), and vice-versa. The result is stored in `sub.slalom_pass_side` for use by the reversed return task.

**`DIVING`** — holds course heading while descending to `target_depth`.

**`SEARCHING`** — identifies the closest valid gatelet (a group of poles containing at least one red and one white) by sorting blobs by `max_y` (largest = closest). If no poles are visible, transitions to `REALIGNING`.

**`ALIGNING`** — two-phase: first `YAW_ALIGN` squares the sub to `course_axis_heading` (tolerance 3°), then `SWAY_ALIGN` uses a sway P+D controller to center the gatelet horizontally. Sway is computed in the body frame from pixel error; a forward damping term prevents the sub from drifting through the gate during lateral adjustment.

**`APPROACH`** — drives forward with `get_go_to_visual_target_commands` toward the gatelet center. Transitions to `CLEAR` 0.75 s after the gatelet disappears (pole cleared).

**`CLEAR`** — surges forward for 1.5 s to ensure complete passage, then returns to `SEARCHING`.

**`REALIGNING`** — re-establishes course heading. Signals `COMPLETED` when heading error < 5°. This path is also taken if the reversed task runs out of poles, ending the return leg.

**Reversed return pass** — `SlalomTask(reversed=True)` sets `course_axis_heading` to `(sub.course_heading + 180) % 360` (opposite direction). `Submarine.update()` automatically sets `forced_pass_side` to the opposite of the forward pass before the reversed task begins.

### 3.5 TimedHoverTask

Holds station at `target_depth` for a fixed `duration` using the full XY PID hover controller. Like `StabilizeTask`, it slides the position target while the sub is still coasting to avoid integral wind-up on arrival.

### 3.6 SurfaceTask

Applies full upward heave (`heave = -1.0`) until `depth < 0.1 m`, then signals `COMPLETED`. No vision processing.

---

## 4. Vision Pipeline

All vision runs on the unrotated (world-frame) camera array. `find_blobs_hsv(camera_image, hsv_ranges, min_pixels)` in `ai/vision.py` converts the BGR array to HSV with `cv2.cvtColor`, applies colour masks with `cv2.inRange`, and returns a list of blob dictionaries with `center_x`, `center_y`, `min_x`, `max_x`, `min_y`, `max_y`, `width`, `height`.

**Colour ranges** defined in `config.py`:
- `RED_HSV_RANGES` — two hue ranges to capture both sides of the red wrap-around (0–10° and 170–180°)
- `BLACK_HSV_RANGE` — low-value mask for pole bodies
- `WHITE_HSV_RANGE` — high-value, low-saturation mask for white slalom poles

---

## 5. Key Gains Summary

| Gain | Value | Used in |
|------|-------|---------|
| `ALIGN_YAW_P_GAIN` | 0.6 | Gate/slalom yaw alignment |
| `YAW_D_GAIN` | 2.0 | Yaw rate damping |
| `HEAVE_P_GAIN` | 1.5 | Visual depth servoing |
| `HEAVE_D_GAIN` | 1.5 | Vertical velocity damping |
| `HOVER_DEPTH_P_GAIN` | 1.2 | Depth hold |
| `HOVER_DEPTH_D_GAIN` | 0.8 | Depth rate damping |
| `HOVER_PITCH_P_GAIN` | 0.9 | Pitch levelling |
| `HOVER_PITCH_D_GAIN` | 0.7 | Pitch rate damping |
| `HOVER_YAW_P_GAIN` | 0.3 | Heading hold |
| `HOVER_XY_P_GAIN` | 2.0 | XY position P |
| `HOVER_XY_I_GAIN` | 1.2 | XY position I |
| `HOVER_XY_D_GAIN` | 1.8 | XY position D |
| `ROLL_RECOVERY_P_GAIN` | 0.03 | Roll angle correction |
| `ROLL_RECOVERY_D_GAIN` | 2.5 | Roll rate damping |
| `ALIGN_DAMPING_GAIN` | 0.5 | Translational braking during alignment |
| `MANEUVER_DAMPING_GAIN` | 3.0 | Sway damping during visual approach |
| `SLALOM_SWAY_D_GAIN` | 3.0 | Sway rate damping during gatelet centering |
| `SURGE_MAX_SPEED` | 0.8 | Maximum forward thrust |
