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
        # Accept and store the target depth for this specific task instance
        self.target_depth = target_depth
        
        # Task-specific parameters
        self.DANCE_ROLL_ANGLE = 45.0
        self.DANCE_COMPLETION_TOLERANCE = 5.0
        self.DANCE_YAW_P_GAIN = 0.20
        self.DANCE_ROLL_P_GAIN = 0.008   # per degree — keeps command proportional
        self.DANCE_ROLL_D_GAIN = 0.15
        self.DANCE_ROLL_MAX = 0.35       # cap so heave always dominates
        self.DANCE_ROLL_RATE_SETTLE = 0.15  # rad/s — roll rate must be below this to advance
        self.reset()

    def reset(self):
        self.dance_step = 0

    def on_start(self, sub: 'Submarine', sensors: SensorSuite):
        sub.dance_center_heading = sensors.heading
        sub.target_heading = sensors.heading
        sub.target_pitch = 0.0
        sub.target_roll = 0.0
        sub.target_x, sub.target_y = sensors.x, sensors.y
        sub.integral_x_err, sub.integral_y_err = 0.0, 0.0

    @property
    def state_name(self) -> str:
        return f"DANCE_STEP_{self.dance_step}"

    def process_vision(self, sub: 'Submarine', camera_image: 'pygame.Surface') -> VisionData:
        return VisionData()

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        moves = {
            0: "SET_TURN_LEFT", 1: "WAIT_TURN_LEFT", 2: "SET_TURN_CENTER", 3: "WAIT_TURN_CENTER",
            4: "SET_TURN_RIGHT", 5: "WAIT_TURN_RIGHT", 6: "SET_TURN_CENTER", 7: "WAIT_TURN_CENTER",
            8: "SET_ROLL_RIGHT", 9: "WAIT_ROLL_RIGHT", 10: "SET_ROLL_CENTER", 11: "WAIT_ROLL_CENTER",
            12: "SET_ROLL_LEFT", 13: "WAIT_ROLL_LEFT", 14: "SET_ROLL_CENTER", 15: "WAIT_ROLL_CENTER",
            16: "SET_TURN_CENTER", 17: "WAIT_TURN_CENTER",
            18: "FINISH"
        }
        move = moves.get(self.dance_step, "FINISH")

        roll_err = sub.target_roll - sensors.roll
        roll_cmd = np.clip(
            roll_err * self.DANCE_ROLL_P_GAIN - sensors.angular_velocity_x * self.DANCE_ROLL_D_GAIN,
            -self.DANCE_ROLL_MAX, self.DANCE_ROLL_MAX
        )

        # Scale heave target depth error by 1/cos(roll) so depth authority is
        # maintained as the vertical thrusters tilt with the sub.
        cos_roll = max(0.35, math.cos(math.radians(sensors.roll)))
        heave_scale = 1.0 / cos_roll

        commands = sub._get_pid_hover_commands(
            sensors, dt, sub.target_x, sub.target_y, self.target_depth,
            yaw_p_gain_override=self.DANCE_YAW_P_GAIN,
            roll=roll_cmd,
            heave_scale=heave_scale
        )

        if "SET_" in move:
            if move == "SET_TURN_LEFT": sub.target_heading = (sub.dance_center_heading - 90) % 360
            elif move == "SET_TURN_RIGHT": sub.target_heading = (sub.dance_center_heading + 90) % 360
            elif move == "SET_TURN_CENTER": sub.target_heading = sub.dance_center_heading
            elif move == "SET_ROLL_RIGHT": sub.target_roll = self.DANCE_ROLL_ANGLE
            elif move == "SET_ROLL_LEFT": sub.target_roll = -self.DANCE_ROLL_ANGLE
            elif move == "SET_ROLL_CENTER": sub.target_roll = 0.0
            self.dance_step += 1
        elif "WAIT_" in move:
            h_err = abs(angle_diff(sensors.heading, sub.target_heading))
            r_err = abs(sensors.roll - sub.target_roll)
            roll_settled = abs(sensors.angular_velocity_x) < self.DANCE_ROLL_RATE_SETTLE
            if ("TURN" in move and h_err < self.DANCE_COMPLETION_TOLERANCE) or \
               ("ROLL" in move and r_err < self.DANCE_COMPLETION_TOLERANCE and roll_settled):
                self.dance_step += 1
        elif move == "FINISH":
            return TaskStatus.COMPLETED, commands
            
        return TaskStatus.RUNNING, commands