#!/usr/bin/env python3
"""
Contains the specific implementation for the Victory Dance task.
"""
import math
import numpy as np
from typing import Tuple

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig
from utils import angle_diff

class VictoryDanceTask(Task):
    def __init__(self, target_depth: float):
        self.target_depth = target_depth

        self.DANCE_YAW_SPIN_COMMAND = 0.5
        self.DANCE_ROLL_SPIN_COMMAND = 0.6
        self.DANCE_SETTLE_ANGLE = 5.0    # degrees
        self.DANCE_RATE_SETTLE = 0.15    # rad/s
        self.reset()

    def reset(self):
        self.dance_step = 0
        self.yaw_accumulated = 0.0
        self.roll_accumulated = 0.0
        self.last_heading = None
        self.last_roll = None

    def on_start(self, sub: 'Submarine', sensors: SensorSuite):
        sub.dance_center_heading = sensors.heading
        sub.target_heading = sensors.heading
        sub.target_pitch = 0.0
        sub.target_roll = 0.0
        sub.target_x, sub.target_y = sensors.x, sensors.y
        sub.integral_x_err, sub.integral_y_err = 0.0, 0.0

    @property
    def state_name(self) -> str:
        steps = ["SPIN_YAW", "RECOVER_YAW", "SPIN_ROLL", "RECOVER_ROLL", "FINISH"]
        return steps[min(self.dance_step, 4)]

    def process_vision(self, sub: 'Submarine', camera_image) -> VisionData:
        return VisionData()

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:

        if self.dance_step == 0:  # SPIN_YAW
            if self.last_heading is None:
                self.last_heading = sensors.heading
            delta = angle_diff(sensors.heading, self.last_heading)
            self.yaw_accumulated += delta
            self.last_heading = sensors.heading
            if abs(self.yaw_accumulated) >= 355:
                self.dance_step += 1

            heave = (self.target_depth - sensors.depth) * sub.HOVER_DEPTH_P_GAIN - sensors.velocity_z * sub.HOVER_DEPTH_D_GAIN
            pitch = (0 - sensors.pitch) * sub.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * sub.HOVER_PITCH_D_GAIN
            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(0, 0, heave, self.DANCE_YAW_SPIN_COMMAND, pitch)

        elif self.dance_step == 1:  # RECOVER_YAW
            sub.target_heading = sub.dance_center_heading
            h_err = abs(angle_diff(sensors.heading, sub.dance_center_heading))
            if h_err < self.DANCE_SETTLE_ANGLE and abs(sensors.imu.gyro_z) < self.DANCE_RATE_SETTLE:
                self.dance_step += 1
            return TaskStatus.RUNNING, sub._get_pid_hover_commands(
                sensors, dt, sub.target_x, sub.target_y, self.target_depth,
                yaw_p_gain_override=0.5)

        elif self.dance_step == 2:  # SPIN_ROLL
            if self.last_roll is None:
                self.last_roll = sensors.roll
            delta = sensors.roll - self.last_roll
            if delta > 180: delta -= 360
            if delta < -180: delta += 360
            self.roll_accumulated += delta
            self.last_roll = sensors.roll
            if abs(self.roll_accumulated) >= 355:
                self.dance_step += 1

            # Invert heave command when upside-down so vertical thrusters still push up in world frame
            cos_roll = math.cos(math.radians(sensors.roll))
            sign = 1.0 if cos_roll >= 0 else -1.0
            heave_scale = sign / max(0.3, abs(cos_roll))
            heave = ((self.target_depth - sensors.depth) * sub.HOVER_DEPTH_P_GAIN - sensors.velocity_z * sub.HOVER_DEPTH_D_GAIN) * heave_scale
            yaw = np.clip(angle_diff(sub.dance_center_heading, sensors.heading) * sub.HOVER_YAW_P_GAIN, -1.0, 1.0)
            pitch = (0 - sensors.pitch) * sub.HOVER_PITCH_P_GAIN - sensors.angular_velocity_y * sub.HOVER_PITCH_D_GAIN
            return TaskStatus.RUNNING, sub._mix_and_normalize_commands(0, 0, heave, yaw, pitch, roll=self.DANCE_ROLL_SPIN_COMMAND)

        elif self.dance_step == 3:  # RECOVER_ROLL
            if abs(sensors.roll) < self.DANCE_SETTLE_ANGLE and abs(sensors.angular_velocity_x) < self.DANCE_RATE_SETTLE:
                self.dance_step += 1
            cos_roll = math.cos(math.radians(sensors.roll))
            sign = 1.0 if cos_roll >= 0 else -1.0
            heave_scale = sign / max(0.3, abs(cos_roll))
            roll_cmd = np.clip(-sensors.roll * sub.ROLL_RECOVERY_P_GAIN - sensors.angular_velocity_x * sub.ROLL_RECOVERY_D_GAIN, -1.0, 1.0)
            return TaskStatus.RUNNING, sub._get_pid_hover_commands(
                sensors, dt, sub.target_x, sub.target_y, self.target_depth,
                yaw_p_gain_override=0.5, roll=roll_cmd, heave_scale=heave_scale)

        return TaskStatus.COMPLETED, ThrusterCommands()
