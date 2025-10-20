#!/usr/bin/env python3
"""
A task to drive straight toward the main gate, with reactive avoidance for slalom poles.
"""
import math
from typing import Tuple
import pygame
import numpy as np

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig, RED_HSV_RANGES, BLACK_HSV_RANGE, WHITE_HSV_RANGE
from ai.vision import find_blobs_hsv
from utils import angle_diff

class ReturnTransitTask(Task):
    def __init__(self, target_depth: float):
        self.target_depth = target_depth
        self.return_heading = None
        self.AVOIDANCE_HEIGHT_THRESHOLD = 0.15
        self.reset()

    def reset(self):
        self.return_heading = None

    @property
    def state_name(self) -> str:
        return "RETURNING_TO_GATE"
    
    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        vision_data = VisionData()
        
        red_blobs = find_blobs_hsv(camera_image, RED_HSV_RANGES, sub.MIN_PIXELS_FOR_DETECTION)
        black_blobs = find_blobs_hsv(camera_image, BLACK_HSV_RANGE, sub.MIN_PIXELS_FOR_DETECTION)
        white_blobs = find_blobs_hsv(camera_image, WHITE_HSV_RANGE, sub.MIN_PIXELS_FOR_DETECTION)

        gate_red_blob_ids = set()

        if red_blobs and black_blobs:
            potential_poles = []
            used_red_blobs = set()
            for b_blob in black_blobs:
                best_match, smallest_y_diff = None, float('inf')
                for r_blob in red_blobs:
                    if id(r_blob) in used_red_blobs: continue
                    x_diff = abs(r_blob['center_x'] - b_blob['center_x'])
                    y_diff = min(abs(r_blob['max_y'] - b_blob['min_y']), abs(b_blob['max_y'] - r_blob['min_y']))
                    if x_diff < (r_blob['width'] + b_blob['width']) * 1.5 and y_diff < 15:
                        if y_diff < smallest_y_diff:
                            smallest_y_diff, best_match = y_diff, r_blob
                if best_match:
                    bounds = {'min_x': min(best_match['min_x'],b_blob['min_x']),'max_x': max(best_match['max_x'],b_blob['max_x']),
                              'min_y': min(best_match['min_y'],b_blob['min_y']),'max_y': max(best_match['max_y'],b_blob['max_y'])}
                    pole_dict = {**bounds, 'height': bounds['max_y']-bounds['min_y'], 'red_blob': best_match}
                    potential_poles.append(pole_dict)
                    used_red_blobs.add(id(best_match))

            if len(potential_poles) >= 2:
                best_pair, min_height_diff = None, float('inf')
                for i in range(len(potential_poles)):
                    for j in range(i + 1, len(potential_poles)):
                        p1, p2 = potential_poles[i], potential_poles[j]
                        height_diff = abs(p1['height'] - p2['height'])
                        if height_diff < min_height_diff:
                            min_height_diff, best_pair = height_diff, (p1, p2)
                
                if best_pair is not None and min_height_diff < 75:
                    vision_data.gate_is_visible = True
                    gate_red_blob_ids.add(id(best_pair[0]['red_blob']))
                    gate_red_blob_ids.add(id(best_pair[1]['red_blob']))

        non_gate_red_blobs = [b for b in red_blobs if id(b) not in gate_red_blob_ids]
        slalom_blobs = non_gate_red_blobs + white_blobs
        vision_data.visible_poles = [b for b in slalom_blobs if b['height'] > b['width'] * 1.2]
        
        return vision_data

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: 'SimulationConfig') -> Tuple[TaskStatus, ThrusterCommands]:
        if self.return_heading is None:
            self.return_heading = (sub.course_heading + 180) % 360

        if vision_data.gate_is_visible and not vision_data.visible_poles:
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors, self.target_depth)
        
        # --- Yaw Control: Always target the final return heading ---
        yaw_p = angle_diff(self.return_heading, sensors.heading) * sub.ALIGN_YAW_P_GAIN
        yaw_d = -sensors.imu.gyro_z * sub.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)

        # --- Sway Control: Default to damping, but add proportional nudge to dodge ---
        # 1. Damping (Derivative term) is always active to prevent drift.
        h_rad = math.radians(sensors.heading)
        sway_velocity = -sensors.velocity_x * math.sin(h_rad) + sensors.velocity_y * math.cos(h_rad)
        sway_d = -sway_velocity * sub.SLALOM_SWAY_D_GAIN

        # 2. Proportional term is added ONLY when an obstacle is detected.
        sway_p = 0.0
        cam_w, cam_h = sensors.camera_image.get_size()
        imminent_poles = [p for p in vision_data.visible_poles if p['height'] / cam_h > self.AVOIDANCE_HEIGHT_THRESHOLD]
        vision_data.avoidance_poles = [] # Clear the list each frame
        
        if imminent_poles:
            closest_pole_to_center = min(imminent_poles, key=lambda p: abs(p['center_x'] - cam_w/2))
            if abs(closest_pole_to_center['center_x'] - cam_w/2) < cam_w * 0.4:
                vision_data.avoidance_poles.append(closest_pole_to_center)
                pixel_error = closest_pole_to_center['center_x'] - cam_w/2
                # Proportional command to push away from the pole
                sway_p = -(pixel_error / (cam_w / 2)) * 1.5

        # 3. Combine P and D terms for the final sway command.
        sway_command = np.clip(sway_p + sway_d, -1.0, 1.0)
        
        # --- Heave and Pitch Control ---
        heave = (self.target_depth - sensors.depth) * sub.HOVER_DEPTH_P_GAIN - sensors.velocity_z * sub.HOVER_DEPTH_D_GAIN
        pitch = (0 - sensors.pitch) * sub.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * sub.HOVER_PITCH_D_GAIN
        
        return TaskStatus.RUNNING, sub._mix_and_normalize_commands(sub.SURGE_MAX_SPEED, sway_command, heave, yaw, pitch)