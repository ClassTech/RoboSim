#!/usr/bin/env python3
"""
Contains the specific implementation for the Slalom task using a sequential strategy.
"""
import math
from enum import Enum, auto
from typing import Tuple, List, Dict, Optional
import pygame
import numpy as np

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig, RED_HSV_RANGES, WHITE_HSV_RANGE
from ai.vision import find_blobs_hsv
from utils import angle_diff


class SlalomTaskState(Enum):
    DIVING = auto()
    SEARCHING = auto()
    ALIGNING = auto()
    APPROACH = auto()
    CLEAR = auto()
    REALIGNING = auto()

class SlalomTask(Task):
    def __init__(self, target_depth: float, forced_pass_side: Optional[str] = None, reversed: bool = False, num_passes_to_complete: int = 3):
        self.target_depth = target_depth
        self.forced_pass_side = forced_pass_side
        self.reversed = reversed
        self.num_passes_to_complete = num_passes_to_complete
        
        self.CLEAR_DURATION = 1.5
        self.SEARCH_SWEEP_ANGLE = 80.0
        self.ALIGN_SWAY_P_GAIN = 1.0
        
        self.reset()

    def reset(self):
        self.current_state = SlalomTaskState.DIVING
        self.state_timer = 0.0
        self.align_phase = 'YAW_ALIGN'
        self.current_pass_side = self.forced_pass_side
        self.task_is_initialized = False
        self.course_axis_heading = None
        self.time_since_poles_lost = 0.0
        self.passes_completed = 0

    def on_start(self, sub: 'Submarine', sensors: SensorSuite):
        """Called by the submarine controller when the task begins."""
        if self.reversed:
            self.course_axis_heading = (sub.course_heading + 180) % 360
        else:
            self.course_axis_heading = sensors.heading
            sub.course_heading = self.course_axis_heading
        
        self.task_is_initialized = True

    @property
    def state_name(self) -> str:
        if self.current_state == SlalomTaskState.ALIGNING:
            return f"ALIGNING ({self.align_phase})"
        return self.current_state.name
        
    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        vision_data = VisionData()
        reds = find_blobs_hsv(camera_image, RED_HSV_RANGES, sub.MIN_PIXELS_FOR_DETECTION)
        whites = find_blobs_hsv(camera_image, WHITE_HSV_RANGE, sub.MIN_PIXELS_FOR_DETECTION)
        for r in reds: r['color'] = 'red'
        for w in whites: w['color'] = 'white'
        all_blobs = reds + whites
        vision_data.visible_poles = [b for b in all_blobs if b['height'] > b['width'] * 1.2]
        return vision_data
    
    def _get_closest_pole_set(self, poles: List[Dict]) -> List[Dict]:
        if not poles: return []
        poles.sort(key=lambda p: p['max_y'], reverse=True)
        if not poles: return []
        
        groups = []
        current_group = [poles[0]]
        
        for i in range(1, len(poles)):
            if abs(poles[i]['max_y'] - current_group[-1]['max_y']) < poles[i]['height'] * 0.5:
                current_group.append(poles[i])
            else:
                groups.append(current_group)
                current_group = [poles[i]]
        groups.append(current_group)

        valid_gatelets = []
        for group in groups:
            has_red = any(p.get('color') == 'red' for p in group)
            has_white = any(p.get('color') == 'white' for p in group)
            if has_red and has_white:
                valid_gatelets.append(group)
        
        if not valid_gatelets:
            return []

        return max(valid_gatelets, key=lambda g: np.mean([p['max_y'] for p in g]))

    def _get_target_gatelet(self, sub: 'Submarine', poles: List[Dict]) -> Optional[Tuple[Dict, Dict]]:
        closest_poles = self._get_closest_pole_set(poles)
        red_pole = next((p for p in closest_poles if p.get('color') == 'red'), None)
        white_poles = [p for p in closest_poles if p.get('color') == 'white']

        if not red_pole or not white_poles: return None

        correct_white_pole = None
        if self.current_pass_side == 'left':
            poles_to_left = [p for p in white_poles if p['center_x'] < red_pole['center_x']]
            if not poles_to_left: return None
            poles_to_left.sort(key=lambda p: red_pole['center_x'] - p['center_x'])
            correct_white_pole = poles_to_left[0]
        else: # 'right' side
            poles_to_right = [p for p in white_poles if p['center_x'] > red_pole['center_x']]
            if not poles_to_right: return None
            poles_to_right.sort(key=lambda p: p['center_x'] - red_pole['center_x'])
            correct_white_pole = poles_to_right[0]
        
        if correct_white_pole: return (red_pole, correct_white_pole)
        return None

    def _get_pd_heading_commands(self, sub: 'Submarine', sensors: SensorSuite, target_heading: float, surge_power: float = 0.0) -> ThrusterCommands:
        yaw_p = angle_diff(target_heading, sensors.heading) * sub.ALIGN_YAW_P_GAIN
        yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        
        heave = (self.target_depth - sensors.depth) * sub.HOVER_DEPTH_P_GAIN - sensors.velocity_z * sub.HOVER_DEPTH_D_GAIN
        pitch = (0 - sensors.pitch) * sub.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * sub.HOVER_PITCH_D_GAIN
        
        roll = np.clip(-sensors.roll * sub.ROLL_RECOVERY_P_GAIN - sensors.angular_velocity_x * sub.ROLL_RECOVERY_D_GAIN, -1.0, 1.0)
        return sub._mix_and_normalize_commands(surge_power, 0, heave, yaw, pitch, roll=roll)

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        if not self.task_is_initialized:
            self.on_start(sub, sensors)

        if self.current_state == SlalomTaskState.DIVING:
            if abs(self.target_depth - sensors.depth) < 0.2:
                self.current_state = SlalomTaskState.SEARCHING
            return TaskStatus.RUNNING, self._get_pd_heading_commands(sub, sensors, self.course_axis_heading)

        elif self.current_state == SlalomTaskState.SEARCHING:
            visible_poles = self._get_closest_pole_set(vision_data.visible_poles)
            if visible_poles:
                if self.current_pass_side is None:
                    red_pole = next((p for p in visible_poles if p.get('color') == 'red'), None)
                    if red_pole:
                        cam_w = sensors.camera_image.get_width()
                        side = 'left' if red_pole['center_x'] > cam_w / 2 else 'right'
                        self.current_pass_side = side
                        if not self.reversed:
                            sub.slalom_pass_side = side
                
                self.current_state = SlalomTaskState.ALIGNING
                self.align_phase = 'YAW_ALIGN'
                return TaskStatus.RUNNING, ThrusterCommands()
            else:
                self.current_state = SlalomTaskState.REALIGNING
                return TaskStatus.RUNNING, self._get_pd_heading_commands(sub, sensors, self.course_axis_heading)

        elif self.current_state == SlalomTaskState.ALIGNING:
            if self.align_phase == 'YAW_ALIGN':
                is_square = abs(angle_diff(sensors.heading, self.course_axis_heading)) < sub.ALIGN_HEADING_TOLERANCE
                if is_square: self.align_phase = 'SWAY_ALIGN'
                return TaskStatus.RUNNING, self._get_pd_heading_commands(sub, sensors, self.course_axis_heading)
            
            elif self.align_phase == 'SWAY_ALIGN':
                gatelet = self._get_target_gatelet(sub, vision_data.visible_poles)
                if not gatelet:
                    self.time_since_poles_lost += dt
                    if self.time_since_poles_lost > 2.0:
                        self.current_state = SlalomTaskState.SEARCHING
                        self.time_since_poles_lost = 0.0
                    return TaskStatus.RUNNING, sub._get_damping_commands(sensors, self.target_depth)

                self.time_since_poles_lost = 0.0
                vision_data.selected_slalom_poles = list(gatelet)
                cam_w, _ = sensors.camera_image.get_size()
                gatelet_center_x = (gatelet[0]['center_x'] + gatelet[1]['center_x']) / 2.0
                pixel_error = gatelet_center_x - cam_w / 2

                is_centered = abs(pixel_error) < (cam_w * sub.ALIGN_POS_TOLERANCE)
                if is_centered:
                    self.current_state = SlalomTaskState.APPROACH
                    return TaskStatus.RUNNING, sub._get_damping_commands(sensors, self.target_depth)

                sway_p = -(pixel_error / (cam_w / 2)) * self.ALIGN_SWAY_P_GAIN
                h_rad = math.radians(sensors.heading)
                sideways_velocity = -sensors.velocity_x * math.sin(h_rad) + sensors.velocity_y * math.cos(h_rad)
                sway_d = -sideways_velocity * sub.SLALOM_SWAY_D_GAIN
                sway = np.clip(sway_p + sway_d, -1.0, 1.0)

                cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
                surge = -(sensors.velocity_x * cos_h + sensors.velocity_y * sin_h) * sub.ALIGN_DAMPING_GAIN
                yaw_p = angle_diff(self.course_axis_heading, sensors.heading) * sub.ALIGN_YAW_P_GAIN
                yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
                yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
                heave = (self.target_depth - sensors.depth) * sub.HOVER_DEPTH_P_GAIN - sensors.velocity_z * sub.HOVER_DEPTH_D_GAIN
                pitch = (0 - sensors.pitch) * sub.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * sub.HOVER_PITCH_D_GAIN

                roll = np.clip(-sensors.roll * sub.ROLL_RECOVERY_P_GAIN - sensors.angular_velocity_x * sub.ROLL_RECOVERY_D_GAIN, -1.0, 1.0)
                return TaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, sway, heave, yaw, pitch, roll=roll)

        elif self.current_state == SlalomTaskState.APPROACH:
            gatelet = self._get_target_gatelet(sub, vision_data.visible_poles)
            
            if gatelet:
                self.time_since_poles_lost = 0.0
                vision_data.selected_slalom_poles = list(gatelet)
            else:
                self.time_since_poles_lost += dt

            if self.time_since_poles_lost > 0.75:
                self.current_state = SlalomTaskState.CLEAR
                self.state_timer = 0.0
                self.time_since_poles_lost = 0.0
                return TaskStatus.RUNNING, self._get_pd_heading_commands(sub, sensors, self.course_axis_heading, sub.SURGE_MAX_SPEED * 0.4)

            if not gatelet:
                return TaskStatus.RUNNING, self._get_pd_heading_commands(sub, sensors, self.course_axis_heading, sub.SURGE_MAX_SPEED * 0.4)

            cam_w, cam_h = sensors.camera_image.get_size()
            nav_target_x = (gatelet[0]['center_x'] + gatelet[1]['center_x']) / 2.0
            vertical_target = np.mean([p['center_y'] for p in gatelet])

            return TaskStatus.RUNNING, sub.get_go_to_visual_target_commands(sensors, nav_target_x, vertical_target, sub.SURGE_MAX_SPEED * 0.4)

        elif self.current_state == SlalomTaskState.CLEAR:
            self.state_timer += dt
            if self.state_timer > self.CLEAR_DURATION:
                self.current_state = SlalomTaskState.SEARCHING
                
            surge_power = sub.SURGE_MAX_SPEED * 0.4
            yaw_p = angle_diff(self.course_axis_heading, sensors.heading) * sub.ALIGN_YAW_P_GAIN
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
            heave = -sensors.velocity_z * sub.HOVER_DEPTH_D_GAIN
            pitch = (0 - sensors.pitch) * sub.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * sub.HOVER_PITCH_D_GAIN
            roll = np.clip(-sensors.roll * sub.ROLL_RECOVERY_P_GAIN - sensors.angular_velocity_x * sub.ROLL_RECOVERY_D_GAIN, -1.0, 1.0)
            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(surge_power, 0, heave, yaw, pitch, roll=roll)

        elif self.current_state == SlalomTaskState.REALIGNING:
            if abs(angle_diff(self.course_axis_heading, sensors.heading)) < 5.0:
                return TaskStatus.COMPLETED, ThrusterCommands()
            
            return TaskStatus.RUNNING, self._get_pd_heading_commands(sub, sensors, self.course_axis_heading)
        
        return TaskStatus.RUNNING, ThrusterCommands()