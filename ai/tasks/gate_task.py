#!/usr/bin/env python3
"""
Contains the specific implementation for the Gate task.
"""
import math
from enum import Enum, auto
from typing import Tuple
import pygame
import numpy as np

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig, RED_HSV_RANGES, BLACK_HSV_RANGE
from ai.vision import find_blobs_hsv
from utils import angle_diff


class GateTaskState(Enum):
    SEARCHING = auto()
    ALIGNING = auto()
    APPROACHING = auto()
    CLEARING_GATE = auto()

class GateTask(Task):
    def __init__(self, target_depth: float):
        self.target_depth = target_depth
        self.ALIGN_CENTER_TOLERANCE_PX = 10 
        self.ALIGN_SQUARE_TOLERANCE_PX = 10 
        self.ALIGN_YAW_RATE_TOLERANCE_RPS = 0.05 
        self.CLEAR_GATE_DURATION = 3.5
        self.search_depths = [0.8, 1.5]
        self.reset()

    def reset(self):
        self.current_state = GateTaskState.SEARCHING
        self.state_timer, self.search_depth_index = 0.0, 0
        self.search_start_heading, self.has_completed_spin = None, False
        self.time_since_gate_lost = 0.0
        self.clearing_depth = None

    @property
    def state_name(self) -> str:
        return self.current_state.name

    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        vision_data = VisionData()
        red_blobs = find_blobs_hsv(camera_image, RED_HSV_RANGES, sub.MIN_PIXELS_FOR_DETECTION)
        black_blobs = find_blobs_hsv(camera_image, BLACK_HSV_RANGE, sub.MIN_PIXELS_FOR_DETECTION)
        if not red_blobs or not black_blobs:
            return vision_data

        potential_poles = []
        used_red_blobs = set()
        for b_blob in black_blobs:
            best_match, smallest_y_diff = None, float('inf')
            for r_blob in red_blobs:
                if id(r_blob) in used_red_blobs:
                    continue
                x_diff = abs(r_blob['center_x'] - b_blob['center_x'])
                y_diff = min(abs(r_blob['max_y'] - b_blob['min_y']), abs(b_blob['max_y'] - r_blob['min_y']))
                if x_diff < (r_blob['width'] + b_blob['width']) * 1.5 and y_diff < 15:
                    if y_diff < smallest_y_diff:
                        smallest_y_diff, best_match = y_diff, r_blob
            
            if best_match:
                bounds = {'min_x': min(best_match['min_x'],b_blob['min_x']),'max_x': max(best_match['max_x'],b_blob['max_x']),
                          'min_y': min(best_match['min_y'],b_blob['min_y']),'max_y': max(best_match['max_y'],b_blob['max_y'])}
                pole_dict = {**bounds, 'height': bounds['max_y']-bounds['min_y'], 'center_x': (bounds['min_x']+bounds['max_x'])/2}
                potential_poles.append(pole_dict)
                used_red_blobs.add(id(best_match))
        
        vision_data.potential_poles = potential_poles
        if len(potential_poles) < 2:
            return vision_data

        best_pair, min_height_diff = None, float('inf')
        for i in range(len(potential_poles)):
            for j in range(i + 1, len(potential_poles)):
                p1, p2 = potential_poles[i], potential_poles[j]
                height_diff = abs(p1['height'] - p2['height'])
                if height_diff < min_height_diff:
                    min_height_diff, best_pair = height_diff, (p1, p2)
        
        if best_pair is None or min_height_diff > 75:
            return vision_data
        
        left_pole, right_pole = sorted(best_pair, key=lambda p: p['center_x'])
        
        vision_data.gate_is_visible = True
        vision_data.min_x, vision_data.max_x = left_pole['min_x'], right_pole['max_x']
        vision_data.min_y, vision_data.max_y = min(left_pole['min_y'], right_pole['min_y']), max(left_pole['max_y'], right_pole['max_y'])
        
        gate_width = right_pole['center_x'] - left_pole['center_x']
        vision_data.left_passage_center_x = left_pole['center_x'] + (gate_width * 0.25)
        vision_data.right_passage_center_x = left_pole['center_x'] + (gate_width * 0.75)

        gate_height = vision_data.max_y - vision_data.min_y
        vision_data.gate_center_y = vision_data.max_y - (gate_height / 3.0)
        vision_data.apparent_height, vision_data.left_pole_height, vision_data.right_pole_height = gate_height, left_pole['height'], right_pole['height']
        return vision_data

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        if self.current_state == GateTaskState.SEARCHING:
            if vision_data.gate_is_visible:
                self.current_state = GateTaskState.ALIGNING
                self.state_timer = 0.0
                return TaskStatus.RUNNING, ThrusterCommands()
            else:
                if self.search_start_heading is None: 
                    self.search_start_heading, self.has_completed_spin = sensors.heading, False
                if not self.has_completed_spin and abs(angle_diff(sensors.heading, self.search_start_heading)) < 15.0 and self.state_timer > 2.0: 
                    self.has_completed_spin = True
                if self.has_completed_spin and self.search_depth_index < 1:
                    self.search_depth_index += 1
                    self.search_start_heading, self.state_timer = None, 0.0
                self.state_timer += dt
                return TaskStatus.RUNNING, sub.get_search_commands(sensors, self.search_depths[self.search_depth_index])
        
        if self.current_state == GateTaskState.ALIGNING:
            if not vision_data.gate_is_visible:
                self.time_since_gate_lost += dt
                if self.time_since_gate_lost > 2.0:
                    self.current_state = GateTaskState.SEARCHING
                    return TaskStatus.RUNNING, ThrusterCommands()
                else:
                    return TaskStatus.RUNNING, sub.get_spin_damping_commands(sensors)
            
            self.time_since_gate_lost = 0.0
            self.state_timer += dt
            cam_w, cam_h = sensors.camera_image.get_size()
            gate_center_x = (vision_data.min_x + vision_data.max_x) / 2

            # 1. Yaw control to center the gate
            pixel_error_x = gate_center_x - (cam_w / 2)
            yaw_p = -(pixel_error_x / (cam_w / 2)) * sub.ALIGN_YAW_P_GAIN
            yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
            yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

            # 2. Damping for surge and sway to bring the sub to a stop
            h_rad = math.radians(sensors.heading)
            cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)
            surge = -(sensors.velocity_x * cos_h + sensors.velocity_y * sin_h) * sub.ALIGN_DAMPING_GAIN
            sway = -(-sensors.velocity_x * sin_h + sensors.velocity_y * cos_h) * sub.ALIGN_DAMPING_GAIN

            # 3. Visual servo for depth: drive gate_center_y to camera center
            vertical_target_y = vision_data.gate_center_y if vision_data.gate_center_y is not None else cam_h / 2
            heave_p = ((vertical_target_y - cam_h/2) / (cam_h/2)) * sub.HEAVE_P_GAIN
            heave_d = -sensors.velocity_z * sub.HEAVE_D_GAIN
            heave = np.clip(heave_p + heave_d, -1.0, 1.0)
            pitch = (0 - sensors.pitch) * sub.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * sub.HOVER_PITCH_D_GAIN

            # 4. Completion: horizontal + yaw aligned, AND depth servo settled (or 8s timeout)
            is_centered = abs(pixel_error_x) < self.ALIGN_CENTER_TOLERANCE_PX
            is_stable = abs(sensors.imu.gyro_z) < self.ALIGN_YAW_RATE_TOLERANCE_RPS
            is_depth_aligned = abs(vertical_target_y - cam_h / 2) < 10

            if is_centered and is_stable and (is_depth_aligned or self.state_timer >= 8.0):
                self.current_state = GateTaskState.APPROACHING
                self.state_timer = 0.0
                return TaskStatus.RUNNING, sub._get_damping_commands(sensors, self.target_depth)

            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, sway, heave, yaw, pitch)
        
        if self.current_state == GateTaskState.APPROACHING:
            if vision_data.gate_is_visible:
                self.time_since_gate_lost = 0.0
            else:
                self.time_since_gate_lost += dt

            if self.time_since_gate_lost > 0.75:
                self.current_state = GateTaskState.CLEARING_GATE
                self.state_timer = self.CLEAR_GATE_DURATION
                self.clearing_depth = sensors.depth
                return TaskStatus.RUNNING, ThrusterCommands()

            nav_target_x, _ = sub._get_navigation_target(vision_data, sensors.camera_image.get_width())
            if nav_target_x is None:
                nav_target_x = sensors.camera_image.get_width() / 2
            
            vertical_target_y = vision_data.gate_center_y if vision_data.gate_center_y is not None else sensors.camera_image.get_height() / 2
            
            return TaskStatus.RUNNING, sub.get_go_to_visual_target_commands(sensors, nav_target_x, vertical_target_y, sub.SURGE_MAX_SPEED)

        if self.current_state == GateTaskState.CLEARING_GATE:
            self.state_timer -= dt
            clear_depth = max(self.clearing_depth or self.target_depth, self.target_depth)
            if self.state_timer <= 0:
                sub.gateCompleted = True
                sub.target_x, sub.target_y = sensors.x, sensors.y
                sub.target_heading = sensors.heading
                return TaskStatus.COMPLETED, sub._get_damping_commands(sensors, clear_depth)

            return TaskStatus.RUNNING, sub.get_depth_change_commands(sensors, clear_depth, sensors.heading, sub.SURGE_MAX_SPEED)
        
        return TaskStatus.RUNNING, ThrusterCommands()