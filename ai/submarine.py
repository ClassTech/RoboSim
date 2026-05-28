#!/usr/bin/env python3
"""
Contains the "brain" of the sub: the Submarine class that manages the mission plan and AI logic.
"""
import math
import numpy as np
from typing import Tuple, List, Optional

from data_structures import ThrusterCommands, SensorSuite, VisionData
from config import SimulationConfig
from utils import angle_diff
from .tasks import TaskStatus
from .tasks.task_base import Task
from .tasks.hover_task import HoverTask
from .tasks.slalom_task import SlalomTask


class Submarine:
    def __init__(self, mission_plan: List[Task]):
        # PID and Control Gains
        self.ALIGN_YAW_P_GAIN = 0.6
        self.ALIGN_SWAY_P_GAIN = 1.8
        self.ALIGN_DAMPING_GAIN = 0.5
        self.YAW_D_GAIN = 2.0
        self.HEAVE_P_GAIN = 1.5
        self.HEAVE_D_GAIN = 1.5
        self.HOVER_DEPTH_P_GAIN = 1.2
        self.HOVER_DEPTH_D_GAIN = 0.8
        self.HOVER_PITCH_P_GAIN = 0.9
        self.HOVER_PITCH_D_GAIN = 0.7
        self.HOVER_YAW_P_GAIN = 0.3
        self.HOVER_XY_P_GAIN = 2.0
        self.HOVER_XY_I_GAIN = 1.2
        self.HOVER_XY_D_GAIN = 1.8
        self.DAMPING_GAIN = 1.0
        self.ROLL_RECOVERY_P_GAIN = 0.03   # per degree — drives angle back to 0
        self.ROLL_RECOVERY_D_GAIN = 2.5    # per rad/s — damps oscillation
        self.MANEUVER_DAMPING_GAIN = 3.0
        self.SURGE_MIN_SPEED = 0.3
        self.SURGE_MAX_SPEED = 0.8
        self.SEARCH_TURN_COMMAND = 0.25
        self.SCAN_TURN_COMMAND = 0.2
        self.SLALOM_NAV_YAW_GAIN = 1.5
        self.SLALOM_SWAY_D_GAIN = 2.0 
        self.ALIGN_POLE_HEIGHT_P_GAIN = 0.02
        self.ALIGN_HEADING_TOLERANCE = 3.0
        self.ALIGN_POS_TOLERANCE = 0.08
        self.ALIGN_SPEED_TOLERANCE_SQ = 0.0025
        self.MIN_PIXELS_FOR_DETECTION = 20
        
        self.mission_plan = mission_plan
        self.config = SimulationConfig()
        self.reset()

    def reset(self):
        self.current_task_index = 0
        for task in self.mission_plan:
            task.reset()
        self.gateCompleted = False
        self.lastApparentSize = 0.0
        self.last_error_x = 0.0
        self.dance_center_heading = 0.0
        self.gate_passage_side = 'left'
        self.target_x, self.target_y, self.target_heading, self.target_pitch, self.target_roll = 0.0, 0.0, 0.0, 0.0, 0.0
        self.integral_x_err, self.integral_y_err, self.integral_clamp, self.approach_heading = 0.0, 0.0, 2.0, 0.0
        self.pass_start_pos, self.pass_end_pos = None, None
        self.course_heading = 0.0
        self.slalom_pass_side = None

    def update(self, dt: float, sensors: SensorSuite) -> Tuple[ThrusterCommands, VisionData]:
        if self.current_task_index >= len(self.mission_plan):
            return ThrusterCommands(), VisionData()

        current_task = self.mission_plan[self.current_task_index]
        # CORRECTED: The process_vision method expects the camera_image, not the entire sensor suite.
        vision_data = current_task.process_vision(self, sensors.camera_image)
        if vision_data.gate_is_visible:
            self.lastApparentSize = vision_data.apparent_height / sensors.camera_image.get_height()
        
        status, commands = current_task.execute(self, dt, sensors, vision_data, self.config)
        
        if status == TaskStatus.COMPLETED and self.current_task_index < len(self.mission_plan) - 1:
            if isinstance(current_task, SlalomTask) and not current_task.reversed:
                for task in self.mission_plan[self.current_task_index + 1:]:
                    if isinstance(task, SlalomTask) and task.reversed:
                        if self.slalom_pass_side:
                            task.forced_pass_side = 'right' if self.slalom_pass_side == 'left' else 'left'
                            task.reset()
                        break
            
            self.current_task_index += 1
            next_task = self.mission_plan[self.current_task_index]
            # CORRECTED: The on_start method expects the submarine object (self) as the first argument.
            if hasattr(next_task, 'on_start'):
                next_task.on_start(self, sensors)
            
            if isinstance(next_task, HoverTask):
                self.target_x, self.target_y = sensors.x, sensors.y
                self.integral_x_err, self.integral_y_err = 0.0, 0.0
        
        return commands, vision_data

    def get_current_task_name(self) -> str:
        if self.current_task_index < len(self.mission_plan):
            return self.mission_plan[self.current_task_index].__class__.__name__
        return "MISSION_COMPLETE"
    
    def get_current_state_name(self) -> str:
        if self.current_task_index < len(self.mission_plan):
            return self.mission_plan[self.current_task_index].state_name
        return ""

    def _get_navigation_target(self, vision_data: VisionData, cam_w: int) -> Tuple[float | None, str | None]:
        l, r = vision_data.left_passage_center_x, vision_data.right_passage_center_x
        if l and r:
            return (l, 'left') if abs(l - cam_w/2) < abs(r - cam_w/2) else (r, 'right')
        return (l, 'left') if l else ((r, 'right') if r else (None, None))

    def get_search_commands(self, sensors, target_depth, target_heading=None, surge_power=0.0):
        yaw = self.SEARCH_TURN_COMMAND
        if target_heading:
            yaw = np.clip(angle_diff(target_heading, sensors.heading) * self.HOVER_YAW_P_GAIN, -self.SCAN_TURN_COMMAND, self.SCAN_TURN_COMMAND)
        pitch = (0 - sensors.pitch) * self.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * self.HOVER_PITCH_D_GAIN
        heave = (target_depth - sensors.depth) * self.HOVER_DEPTH_P_GAIN - sensors.velocity_z * self.HOVER_DEPTH_D_GAIN
        return self._mix_and_normalize_commands(surge_power, 0, heave, yaw, pitch)

    def get_spin_damping_commands(self, sensors: SensorSuite):
        h_rad, cos_h, sin_h = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        wd_x, wd_y = -sensors.velocity_x * self.ALIGN_DAMPING_GAIN, -sensors.velocity_y * self.ALIGN_DAMPING_GAIN
        surge, sway = wd_x * cos_h + wd_y * sin_h, -wd_x * sin_h + wd_y * cos_h
        yaw = -sensors.imu.gyro_z * self.YAW_D_GAIN
        heave = (sensors.depth - sensors.depth) * self.HOVER_DEPTH_P_GAIN - sensors.velocity_z * self.HOVER_DEPTH_D_GAIN
        pitch = (0 - sensors.pitch) * self.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * self.HOVER_PITCH_D_GAIN
        return self._mix_and_normalize_commands(surge, sway, heave, yaw, pitch)
    
    def get_depth_change_commands(self, sensors, depth, heading, surge_power=0.0):
        heave = (depth - sensors.depth) * self.HOVER_DEPTH_P_GAIN - sensors.velocity_z * self.HOVER_DEPTH_D_GAIN
        yaw = angle_diff(heading, sensors.heading) * self.HOVER_YAW_P_GAIN
        pitch = (0 - sensors.pitch) * self.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * self.HOVER_PITCH_D_GAIN
        return self._mix_and_normalize_commands(surge_power, 0, heave, yaw, pitch)

    def _get_pid_hover_commands(self, sensors: SensorSuite, dt: float, tx: float, ty: float, td: float,
                                yaw_p_gain_override: Optional[float] = None,
                                pitch_p_gain_override: Optional[float] = None,
                                roll: float = 0.0,
                                heave_scale: float = 1.0) -> ThrusterCommands:
        
        yaw_p_gain = yaw_p_gain_override if yaw_p_gain_override is not None else self.HOVER_YAW_P_GAIN
        pitch_p_gain = pitch_p_gain_override if pitch_p_gain_override is not None else self.HOVER_PITCH_P_GAIN

        ex, ey = tx - sensors.x, ty - sensors.y
        self.integral_x_err = np.clip(self.integral_x_err + ex * dt, -self.integral_clamp, self.integral_clamp)
        self.integral_y_err = np.clip(self.integral_y_err + ey * dt, -self.integral_clamp, self.integral_clamp)
        
        wtx = (ex * self.HOVER_XY_P_GAIN) + (self.integral_x_err * self.HOVER_XY_I_GAIN) - (sensors.velocity_x * self.HOVER_XY_D_GAIN)
        wty = (ey * self.HOVER_XY_P_GAIN) + (self.integral_y_err * self.HOVER_XY_I_GAIN) - (sensors.velocity_y * self.HOVER_XY_D_GAIN)
        
        h_rad, c, s = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        fsh, sway = wtx * c + wty * s, -wtx * s + wty * c
        
        pitch_cmd = (self.target_pitch - sensors.pitch) * pitch_p_gain - sensors.angular_velocity_y * self.HOVER_PITCH_D_GAIN
        yaw_cmd = np.clip(angle_diff(self.target_heading, sensors.heading) * yaw_p_gain, -1.0, 1.0)
        fwv = (((td - sensors.depth) * self.HOVER_DEPTH_P_GAIN) - (sensors.velocity_z * self.HOVER_DEPTH_D_GAIN)) * heave_scale

        p_rad = math.radians(sensors.pitch)
        cp, sp = math.cos(p_rad), math.sin(p_rad)
        surge, heave = fsh * cp - fwv * sp, fsh * sp + fwv * cp
        
        return self._mix_and_normalize_commands(surge, sway, heave, yaw_cmd, pitch_cmd, roll=roll)

    def _get_damping_commands(self, sensors: SensorSuite, target_depth) -> ThrusterCommands:
        wdx, wdy = -sensors.velocity_x * self.DAMPING_GAIN, -sensors.velocity_y * self.DAMPING_GAIN
        h_rad = math.radians(sensors.heading)
        c, s = math.cos(h_rad), math.sin(h_rad)
        fsh, sway = wdx*c+wdy*s, -wdx*s+wdy*c
        p_rad = math.radians(sensors.pitch)
        cp, sp = math.cos(p_rad), math.sin(p_rad)
        pitch = (0 - sensors.pitch) * self.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * self.HOVER_PITCH_D_GAIN
        yaw = angle_diff(self.target_heading, sensors.heading) * self.HOVER_YAW_P_GAIN
        fwv = ((target_depth - sensors.depth) * self.HOVER_DEPTH_P_GAIN) - (sensors.velocity_z * self.HOVER_DEPTH_D_GAIN)
        surge, heave = fsh*cp-fwv*sp, fsh*sp+fwv*cp
        roll = np.clip(-sensors.roll * self.ROLL_RECOVERY_P_GAIN - sensors.angular_velocity_x * self.ROLL_RECOVERY_D_GAIN, -1.0, 1.0)
        return self._mix_and_normalize_commands(surge, sway, heave, yaw, pitch, roll=roll)
        
    def _mix_and_normalize_commands(self, surge, sway, heave, yaw, pitch, roll=0.0) -> ThrusterCommands:
        commands = ThrusterCommands()
        commands.v_port, commands.v_starboard = heave + roll, heave - roll
        commands.h_port_bow, commands.h_starboard_bow = surge+sway+yaw, surge-sway-yaw
        commands.h_port_aft, commands.h_starboard_aft = surge-sway+yaw, surge+sway-yaw
        max_abs = max(1.0, abs(commands.v_port), abs(commands.v_starboard), abs(commands.h_port_bow),
                      abs(commands.h_starboard_bow), abs(commands.h_port_aft), abs(commands.h_starboard_aft))
        if max_abs > 1.0:
            commands.v_port/=max_abs; commands.v_starboard/=max_abs; commands.h_port_bow/=max_abs
            commands.h_starboard_bow/=max_abs; commands.h_port_aft/=max_abs; commands.h_starboard_aft/=max_abs
        return commands
    def get_go_to_visual_target_commands(self, sensors: SensorSuite, nav_target_x: float, vertical_target_y: float, surge_power: float):
        """
        A robust, generic 'point-and-drive' controller for approaching any visual target.
        Uses yaw to steer and sway to damp.
        """
        cam_w, cam_h = sensors.camera_image.get_size()

        # 1. Yaw control steers the submarine toward the horizontal target
        pixel_error_x = nav_target_x - (cam_w / 2)
        yaw_p = -(pixel_error_x / (cam_w / 2)) * self.ALIGN_YAW_P_GAIN
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

        # 2. Sway control damps sideways motion
        h_rad = math.radians(sensors.heading)
        cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
        sway = (sensors.velocity_x * sin_h - sensors.velocity_y * cos_h) * self.MANEUVER_DAMPING_GAIN
        
        # 3. Heave and Pitch control
        heave_p = ((vertical_target_y - cam_h/2) / (cam_h/2)) * self.HEAVE_P_GAIN
        heave_d = -sensors.velocity_z * self.HEAVE_D_GAIN
        heave = np.clip(heave_p + heave_d, -1.0, 1.0)
        pitch = (0 - sensors.pitch) * self.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * self.HOVER_PITCH_D_GAIN
        
        return self._mix_and_normalize_commands(surge_power, sway, heave, yaw, pitch)