#!/usr/bin/env python3
"""
Contains the main SubmarineSimulator class.
Handles cv2 rendering, physics, and the main game loop.
"""
import math
import random
import time
from typing import Tuple, Optional

import cv2
import numpy as np

from config import *
from world import Gate, PathMarker, SlalomPole, SubmarinePhysicsState
from data_structures import ThrusterCommands, MPU6050Readings, SensorSuite
from ai.submarine import Submarine
from ai.tasks import GateTask, SlalomTask
from ai.tasks.slalom_task import SlalomTaskState

WINDOW_TITLE = "Autonomous Submarine Simulator"
CAM_W, CAM_H = 320, 240
FONT = cv2.FONT_HERSHEY_SIMPLEX


class SubmarineSimulator:
    def __init__(self, submarine_ai: Submarine, width=1200, height=800):
        self.width, self.height = width, height
        self.config = SimulationConfig()
        self.scaleX = width * 0.7 / self.config.worldWidth
        self.scaleY = height * 0.8 / self.config.worldHeight
        self.cameraSurface = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

        bg = cv2.imread("background.jpg")
        if bg is not None:
            bg_h_target = 480
            bg_w_target = int(bg.shape[1] * (bg_h_target / bg.shape[0]))
            self.camera_background = cv2.resize(bg, (bg_w_target, bg_h_target))
            self.camera_background_pano = np.hstack([self.camera_background, self.camera_background])
        else:
            self.camera_background = None
            self.camera_background_pano = None

        self.subMass, self.subInertia = 4.0, 0.15
        self.netBuoyancyForce, self.thrusterMaxForce = -0.4, 0.8
        self.linearDragCoeff, self.angularDragCoeff = 1.5, 0.1

        self.submarineAI = submarine_ai

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        self.resetSimulation()

    def resetSimulation(self):
        self.gate = Gate(x=15.0, center_y=7.5, z=random.uniform(0.6, 1.8))
        self.path_marker = PathMarker(x=self.gate.x+2.0, y=self.gate.center_y, z=self.config.worldDepth-0.2, heading=0)
        self.slalom_poles = []
        ssx, ssy, sst = self.gate.x+8.0, 1.524, 1.524*0.25
        last_y = self.config.worldHeight / 2
        sz = self.config.worldDepth - (0.9 + random.uniform(0.3, 0.6))
        for i in range(3):
            sx = ssx + i * 4.0
            y = last_y + random.uniform(-sst, sst) if i > 0 else last_y
            y = float(np.clip(y, sst*2, self.config.worldHeight-(sst*2)))
            last_y = y
            self.slalom_poles += [
                SlalomPole(x=sx, y=y-ssy, z=sz, color=WHITE),
                SlalomPole(x=sx, y=y,     z=sz, color=RED),
                SlalomPole(x=sx, y=y+ssy, z=sz, color=WHITE),
            ]
        start_heading = random.choice([0, 90, 270])
        self.subPhysics = SubmarinePhysicsState(
            x=self.gate.x-9.144, y=self.gate.center_y+random.uniform(-0.9144, 0.9144),
            z=0.5, heading=start_heading)
        self.submarineAI.reset()
        self.startTime = time.time()
        self.running, self.paused = True, False
        self.lastThrusterCommands = ThrusterCommands()
        self.last_imu_readings = MPU6050Readings()

    def skip_to_slalom(self):
        self.resetSimulation()
        task = next((t for t in self.submarineAI.mission_plan if isinstance(t, SlalomTask)), None)
        idx  = next((i for i, t in enumerate(self.submarineAI.mission_plan) if isinstance(t, SlalomTask)), None)
        if task and idx is not None:
            self.submarineAI.current_task_index = idx
            self.submarineAI.gateCompleted = True
            self.subPhysics.x = self.gate.x + 4.0
            self.subPhysics.y = self.config.worldHeight / 2
            self.subPhysics.z = task.target_depth
            self.subPhysics.heading = 0.0
            task.current_state = SlalomTaskState.SEARCHING

    def worldToScreen(self, x, y) -> Tuple[int, int]:
        return int(x*self.scaleX+50), int((self.config.worldHeight-y)*self.scaleY+50)

    def handleInput(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.running = False
        elif key == ord(' '):
            self.paused = not self.paused
        elif key == ord('r'):
            self.resetSimulation()
        elif key == ord('s'):
            self.skip_to_slalom()
        try:
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
        except cv2.error:
            self.running = False

    def applyPhysics(self, dt, commands):
        f = [c * self.thrusterMaxForce for c in [
            commands.h_port_bow, commands.h_starboard_bow,
            commands.h_port_aft, commands.h_starboard_aft,
            commands.v_port, commands.v_starboard]]
        ca, sa = math.cos(math.radians(45)), math.sin(math.radians(45))
        surge = (f[0]+f[1]+f[2]+f[3]) * ca
        sway  = (f[0]-f[1]-f[2]+f[3]) * sa
        heave = f[4] + f[5]
        yaw_t  = (f[0]-f[1]+f[2]-f[3]) * ca * (self.config.submarineWidth/2)
        roll_t = (f[4]-f[5]) * (self.config.submarineWidth/2)
        r = math.radians(self.subPhysics.roll)
        cr, sr = math.cos(r), math.sin(r)
        heave_r = sway*sr + heave*cr
        sway_r  = sway*cr - heave*sr
        h, p = math.radians(self.subPhysics.heading), math.radians(self.subPhysics.pitch)
        ch, sh, cp, sp = math.cos(h), math.sin(h), math.cos(p), math.sin(p)
        fsh = surge*cp + heave_r*sp
        fwv = heave_r*cp - surge*sp
        fx = fsh*ch - sway_r*sh
        fy = fsh*sh + sway_r*ch
        fz = fwv + self.netBuoyancyForce
        speed = math.hypot(self.subPhysics.velocity_x, self.subPhysics.velocity_y)
        dm = self.linearDragCoeff * speed * speed
        dx = -dm*(self.subPhysics.velocity_x/speed) if speed > 0 else 0
        dy = -dm*(self.subPhysics.velocity_y/speed) if speed > 0 else 0
        dz = -self.linearDragCoeff*2*self.subPhysics.velocity_z*abs(self.subPhysics.velocity_z)
        d_yaw  = -self.angularDragCoeff * self.subPhysics.angular_velocity_z**2 * np.sign(self.subPhysics.angular_velocity_z)
        d_roll = -self.angularDragCoeff * self.subPhysics.angular_velocity_x**2 * np.sign(self.subPhysics.angular_velocity_x)
        ax, ay, az = (fx+dx)/self.subMass, (fy+dy)/self.subMass, (fz+dz)/self.subMass
        self.last_imu_readings = MPU6050Readings(
            accel_x=-ax*sh+ay*ch, accel_y=ax*ch+ay*sh, accel_z=az,
            gyro_z=self.subPhysics.angular_velocity_z)
        self.subPhysics.velocity_x += ax*dt
        self.subPhysics.velocity_y += ay*dt
        self.subPhysics.velocity_z += az*dt
        self.subPhysics.angular_velocity_z += ((yaw_t  + d_yaw)  / self.subInertia) * dt
        self.subPhysics.angular_velocity_x += ((roll_t + d_roll) / self.subInertia) * dt
        self.subPhysics.x += self.subPhysics.velocity_x * dt
        self.subPhysics.y += self.subPhysics.velocity_y * dt
        self.subPhysics.z += self.subPhysics.velocity_z * dt
        self.subPhysics.heading = (self.subPhysics.heading + math.degrees(self.subPhysics.angular_velocity_z * dt)) % 360
        self.subPhysics.roll = ((self.subPhysics.roll + math.degrees(self.subPhysics.angular_velocity_x * dt) + 180) % 360) - 180
        margin = 0.5
        self.subPhysics.x = float(np.clip(self.subPhysics.x, margin, self.config.worldWidth  - margin))
        self.subPhysics.y = float(np.clip(self.subPhysics.y, margin, self.config.worldHeight - margin))
        self.subPhysics.z = float(np.clip(self.subPhysics.z, 0.0,    self.config.worldDepth  - margin))

    def project3D(self, world_pos: Tuple[float, float, float]) -> Optional[Tuple[int, int, float]]:
        dx = world_pos[0] - self.subPhysics.x
        dy = world_pos[1] - self.subPhysics.y
        dz = world_pos[2] - self.subPhysics.z
        h, p = math.radians(-self.subPhysics.heading), math.radians(-self.subPhysics.pitch)
        ch, sh, cp, sp = math.cos(h), math.sin(h), math.cos(p), math.sin(p)
        x_yaw = dx*ch - dy*sh
        y_yaw = dx*sh + dy*ch
        cz = x_yaw*cp + dz*sp
        cy = x_yaw*sp - dz*cp
        cx = y_yaw
        if cz < 0.2:
            return None
        f = CAM_W / (2 * math.tan(math.radians(self.config.cameraFov / 2)))
        return int(CAM_W/2 - f*(cx/cz)), int(CAM_H/2 - f*(cy/cz)), math.hypot(dx, dy, dz)

    def generateCameraView(self):
        if self.camera_background_pano is not None:
            bg_h, bg_w = self.camera_background.shape[:2]
            x_off = int((self.subPhysics.heading / 360) * bg_w)
            y_off = int(np.clip((bg_h - CAM_H) / 2 - self.subPhysics.pitch * 2.0, 0, bg_h - CAM_H))
            self.cameraSurface[:] = self.camera_background_pano[y_off:y_off+CAM_H, x_off:x_off+CAM_W]
        else:
            self.cameraSurface[:] = WATER_COLOR
            hp = self.project3D((self.subPhysics.x+20, self.subPhysics.y, self.config.worldDepth))
            if hp:
                y_floor = max(0, min(hp[1], CAM_H))
                if y_floor < CAM_H:
                    self.cameraSurface[y_floor:, :] = POOL_FLOOR_COLOR

        drawable = []
        dw = self.gate.dividerWidth / 2
        ph = 0.305
        pz = self.gate.z + 0.1 + ph/2
        gp = {
            "tpt": (self.gate.x, self.gate.topPoleY,    self.gate.z),
            "bpt": (self.gate.x, self.gate.bottomPoleY, self.gate.z),
            "tpb": (self.gate.x, self.gate.topPoleY,    self.gate.z + self.gate.poleHeight),
            "bpb": (self.gate.x, self.gate.bottomPoleY, self.gate.z + self.gate.poleHeight),
            "dtl": (self.gate.x, self.gate.center_y - dw, self.gate.z),
            "dtr": (self.gate.x, self.gate.center_y + dw, self.gate.z),
            "dbl": (self.gate.x, self.gate.center_y - dw, self.gate.z + self.gate.dividerHeight),
            "dbr": (self.gate.x, self.gate.center_y + dw, self.gate.z + self.gate.dividerHeight),
            "sp":  (self.gate.x, (self.gate.center_y + self.gate.topPoleY)    / 2, pz),
            "fp":  (self.gate.x, (self.gate.center_y + self.gate.bottomPoleY) / 2, pz),
        }
        proj = {k: self.project3D(v) for k, v in gp.items()}

        if proj["tpt"] and proj["bpt"]:
            drawable.append((proj["tpt"][2], 'line', GRAY, proj["tpt"][:2], proj["bpt"][:2], 5))
        if proj["bpt"] and proj["bpb"]:
            p1, p2 = proj["bpt"][:2], proj["bpb"][:2]
            my = int((p1[1]+p2[1])/2)
            drawable.extend([
                (proj["bpt"][2], 'line', RED,   p1,           (p1[0], my), 8),
                (proj["bpt"][2], 'line', BLACK, (p1[0], my),  p2,          8),
            ])
        if proj["tpt"] and proj["tpb"]:
            p1, p2 = proj["tpt"][:2], proj["tpb"][:2]
            my = int((p1[1]+p2[1])/2)
            drawable.extend([
                (proj["tpt"][2], 'line', BLACK, p1,           (p1[0], my), 8),
                (proj["tpt"][2], 'line', RED,   (p1[0], my),  p2,          8),
            ])
        d_keys = ["dtl", "dtr", "dbr", "dbl"]
        if all(proj.get(k) for k in d_keys):
            drawable.append((
                sum(proj[k][2] for k in d_keys) / 4,
                'polygon', RED, [proj[k][:2] for k in d_keys]))
        for key, color in [("sp", SHARK_BLUE), ("fp", SAWFISH_GREEN)]:
            if proj[key]:
                x, y, d = proj[key]
                s = min(120, max(10, int(800 / (d+1) * ph)))
                drawable.append((d, 'rect', color, (x-s//2, y-s//2, s, s)))

        m_pts = [self.project3D((self.path_marker.x+c[0], self.path_marker.y+c[1], self.path_marker.z+c[2]))
                 for c in [(-0.6,-0.075,0),(0.6,-0.075,0),(0.6,0.075,0),(-0.6,0.075,0)]]
        if all(m_pts):
            drawable.append((sum(p[2] for p in m_pts)/4, 'polygon', self.path_marker.color, [p[:2] for p in m_pts]))

        for pole in self.slalom_poles:
            tp = self.project3D((pole.x, pole.y, pole.z))
            bp = self.project3D((pole.x, pole.y, pole.z + pole.height))
            if tp and bp:
                drawable.append(((tp[2]+bp[2])/2, 'line', pole.color, tp[:2], bp[:2], 8))

        drawable.sort(key=lambda d: d[0], reverse=True)
        for d in drawable:
            if d[1] == 'polygon':
                pts = np.array(d[3], dtype=np.int32)
                cv2.fillPoly(self.cameraSurface, [pts], d[2])
            elif d[1] == 'line':
                cv2.line(self.cameraSurface, d[3], d[4], d[2], d[5])
            elif d[1] == 'rect':
                x, y, w, h = d[3]
                cv2.rectangle(self.cameraSurface, (x, y), (x+w, y+h), d[2], -1)

    def render(self, screen: np.ndarray):
        screen[:] = LIGHT_BLUE

        # World border
        bx2 = int(self.config.worldWidth  * self.scaleX + 60)
        by2 = int(self.config.worldHeight * self.scaleY + 60)
        cv2.rectangle(screen, (40, 40), (bx2, by2), BLACK, 2)

        # Gate
        gate_color = GREEN if self.submarineAI.gateCompleted else GRAY
        tp = self.worldToScreen(self.gate.x, self.gate.topPoleY)
        bp = self.worldToScreen(self.gate.x, self.gate.bottomPoleY)
        cp = self.worldToScreen(self.gate.x, self.gate.center_y)
        cv2.line(screen, tp, bp, gate_color, 2)
        # Top pole marker: black top half, red bottom half
        cv2.ellipse(screen, tp, (8, 8), 0, 180, 360, BLACK, 2)
        cv2.ellipse(screen, tp, (8, 8), 0,   0, 180, RED,   2)
        # Bottom pole marker: red top half, black bottom half
        cv2.ellipse(screen, bp, (8, 8), 0, 180, 360, RED,   2)
        cv2.ellipse(screen, bp, (8, 8), 0,   0, 180, BLACK, 2)
        cv2.circle(screen, cp, 6, RED, -1)

        # Path marker
        mx, my = self.worldToScreen(self.path_marker.x, self.path_marker.y)
        mw = max(1, int(self.path_marker.length * self.scaleX / 2))
        mh = max(1, int(self.path_marker.width  * self.scaleY / 2))
        cv2.rectangle(screen, (mx-mw, my-mh), (mx+mw, my+mh), self.path_marker.color, -1)

        # Slalom poles
        for pole in self.slalom_poles:
            pos = self.worldToScreen(pole.x, pole.y)
            cv2.circle(screen, pos, 5, pole.color, -1)
            cv2.circle(screen, pos, 5, BLACK, 1)

        # Submarine outline
        sub_pos = self.worldToScreen(self.subPhysics.x, self.subPhysics.y)
        cos_h = math.cos(math.radians(self.subPhysics.heading))
        sin_h = math.sin(math.radians(self.subPhysics.heading))
        pvc_s = int(self.config.submarineWidth * self.scaleX / 2)
        outline = np.array([
            _rot(sub_pos, dx, dy, cos_h, sin_h)
            for dx, dy in [(-pvc_s,-pvc_s),(pvc_s,-pvc_s),(pvc_s,pvc_s),(-pvc_s,pvc_s)]
        ], np.int32)
        cv2.polylines(screen, [outline], True, YELLOW, 4)

        box_w = max(1, int(0.127 * self.scaleY / 2))
        box_l = max(1, int(self.config.submarineLength * self.scaleX / 2))
        body = np.array([
            _rot(sub_pos, dx, dy, cos_h, sin_h)
            for dx, dy in [(-box_l,-box_w),(box_l,-box_w),(box_l,box_w),(-box_l,box_w)]
        ], np.int32)
        cv2.fillPoly(screen, [body], CONTROL_BOX_GRAY)

        arrow_tip_x = int(box_l + 0.2 * self.scaleX)
        arrow = np.array([
            _rot(sub_pos, dx, dy, cos_h, sin_h)
            for dx, dy in [(box_l,-box_w),(box_l,box_w),(arrow_tip_x,0)]
        ], np.int32)
        cv2.fillPoly(screen, [arrow], YELLOW)

        self._renderUi(screen)

        # Camera view with roll rotation
        roll = self.subPhysics.roll
        if abs(roll) > 0.5:
            pad = 400
            padded = np.zeros((pad, pad, 3), dtype=np.uint8)
            py, px = (pad - CAM_H) // 2, (pad - CAM_W) // 2
            padded[py:py+CAM_H, px:px+CAM_W] = self.cameraSurface
            M = cv2.getRotationMatrix2D((pad/2, pad/2), -roll, 1.0)
            rotated = cv2.warpAffine(padded, M, (pad, pad))
            ry, rx = (pad - CAM_H) // 2, (pad - CAM_W) // 2
            display_cam = rotated[ry:ry+CAM_H, rx:rx+CAM_W]
        else:
            display_cam = self.cameraSurface

        scaled = cv2.resize(display_cam, (400, 300))
        cx1, cy1 = self.width - 420, 20
        screen[cy1:cy1+300, cx1:cx1+400] = scaled
        cv2.rectangle(screen, (cx1, cy1), (cx1+400, cy1+300), BLACK, 2)

    def _drawThrusterBar(self, screen: np.ndarray, x: int, y: int, label: str, value: float):
        bw, bh = 25, 80
        mid_y = y + bh // 2
        cv2.rectangle(screen, (x, y), (x+bw, y+bh), GRAY, 2)
        if value > 0:
            fh = int(value * bh / 2)
            cv2.rectangle(screen, (x+1, mid_y-fh), (x+bw-1, mid_y), GREEN, -1)
        elif value < 0:
            fh = int(abs(value) * bh / 2)
            cv2.rectangle(screen, (x+1, mid_y), (x+bw-1, mid_y+fh), RED, -1)
        cv2.line(screen, (x, mid_y), (x+bw, mid_y), BLACK, 1)
        cv2.putText(screen, label, (x, y+bh+14), FONT, 0.35, BLACK, 1, cv2.LINE_AA)

    def _renderUi(self, screen: np.ndarray):
        y = 30
        cv2.putText(screen, "Autonomous Submarine Simulator", (20, y), FONT, 0.7, BLACK, 2, cv2.LINE_AA)
        y += 35
        speed = math.hypot(self.subPhysics.velocity_x, self.subPhysics.velocity_y)
        for text in [
            f"Time: {time.time()-self.startTime:.1f}s",
            f"Task:  {self.submarineAI.get_current_task_name()}",
            f"State: {self.submarineAI.get_current_state_name()}",
            f"Speed:   {speed:.2f} m/s",
            f"Heading: {self.subPhysics.heading:.1f} deg",
            f"Pitch:   {self.subPhysics.pitch:.1f} deg",
            f"Roll:    {self.subPhysics.roll:.1f} deg",
            f"Depth:   {self.subPhysics.z:.2f} m",
        ]:
            cv2.putText(screen, text, (20, y), FONT, 0.5, BLACK, 1, cv2.LINE_AA)
            y += 20

        y += 6
        imu = self.last_imu_readings
        for text in [
            "IMU:",
            f"  Surge accel: {imu.accel_y:+.2f} m/s2",
            f"  Sway  accel: {imu.accel_x:+.2f} m/s2",
            f"  Heave accel: {imu.accel_z:+.2f} m/s2",
            f"  Yaw rate:    {math.degrees(imu.gyro_z):+.1f} deg/s",
        ]:
            cv2.putText(screen, text, (20, y), FONT, 0.45, BLACK, 1, cv2.LINE_AA)
            y += 18

        cv2.putText(screen, "R=Reset  SPACE=Pause  S=Skip  ESC=Quit",
                    (20, self.height - 25), FONT, 0.45, BLACK, 1, cv2.LINE_AA)

        tx, ty = self.width - 420, 350
        cv2.putText(screen, "Thruster Output:", (tx, ty), FONT, 0.5, BLACK, 1, cv2.LINE_AA)
        ty += 25
        tc = self.lastThrusterCommands
        for i, (label, val) in enumerate([
            ("H-PB", tc.h_port_bow), ("H-SB", tc.h_starboard_bow),
            ("H-PA", tc.h_port_aft), ("H-SA", tc.h_starboard_aft),
        ]):
            self._drawThrusterBar(screen, tx + i*50, ty, label, val)
        ty += 110
        for i, (label, val) in enumerate([("V-P", tc.v_port), ("V-S", tc.v_starboard)]):
            self._drawThrusterBar(screen, tx + i*50, ty, label, val)

    def run(self):
        screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        last_time = time.perf_counter()

        while self.running:
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0.1:
                dt = 0.1

            self.handleInput()
            if not self.running:
                break

            if self.paused:
                self.render(screen)
                cv2.imshow(WINDOW_TITLE, screen)
                continue

            self.generateCameraView()
            sensors = SensorSuite(
                camera_image=self.cameraSurface,
                depth=self.subPhysics.z,
                heading=self.subPhysics.heading,
                pitch=self.subPhysics.pitch,
                roll=self.subPhysics.roll,
                imu=self.last_imu_readings,
                x=self.subPhysics.x,
                y=self.subPhysics.y,
                velocity_x=self.subPhysics.velocity_x,
                velocity_y=self.subPhysics.velocity_y,
                angular_velocity_y=self.subPhysics.angular_velocity_y,
                angular_velocity_x=self.subPhysics.angular_velocity_x,
                velocity_z=self.subPhysics.velocity_z,
            )

            thrusterCommands, vision_data = self.submarineAI.update(dt, sensors)

            if thrusterCommands.pause_simulation:
                self.paused = True

            self.lastThrusterCommands = thrusterCommands

            # Debug overlays drawn onto cameraSurface after AI has processed it
            for pole in vision_data.potential_poles:
                cv2.rectangle(self.cameraSurface,
                              (pole['min_x'], pole['min_y']),
                              (pole['max_x'], pole['max_y']), ORANGE, 1)
            if vision_data.gate_is_visible:
                cv2.rectangle(self.cameraSurface,
                              (vision_data.min_x, vision_data.min_y),
                              (vision_data.max_x, vision_data.max_y), YELLOW, 1)
            for pole in vision_data.visible_poles:
                color = GREEN if pole.get('color') == 'white' else YELLOW
                cv2.rectangle(self.cameraSurface,
                              (pole['min_x'], pole['min_y']),
                              (pole['max_x'], pole['max_y']), color, 1)
            for pole in vision_data.selected_slalom_poles:
                cv2.rectangle(self.cameraSurface,
                              (pole['min_x'], pole['min_y']),
                              (pole['max_x'], pole['max_y']), MAGENTA, 3)
            for pole in vision_data.avoidance_poles:
                cv2.rectangle(self.cameraSurface,
                              (pole['min_x'], pole['min_y']),
                              (pole['max_x'], pole['max_y']), ORANGE, 3)

            self.applyPhysics(dt, thrusterCommands)
            self.render(screen)
            cv2.imshow(WINDOW_TITLE, screen)

        cv2.destroyAllWindows()


def _rot(origin: Tuple[int, int], dx: int, dy: int, cos_h: float, sin_h: float) -> Tuple[int, int]:
    return (int(origin[0] + dx*cos_h - dy*sin_h),
            int(origin[1] - (dx*sin_h + dy*cos_h)))
