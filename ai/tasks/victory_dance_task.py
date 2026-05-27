#!/usr/bin/env python3
"""
Contains the specific implementation for the Victory Dance task.
"""
import math
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
        self.DANCE_PITCH_ANGLE = 90.0
        self.DANCE_COMPLETION_TOLERANCE = 5.0
        self.DANCE_YAW_P_GAIN = 0.20
        self.DANCE_PITCH_P_GAIN = 0.25
        self.reset()

    def reset(self):
        self.dance_step = 0

    def on_start(self, sub: 'Submarine', sensors: SensorSuite):
        sub.dance_center_heading = sensors.heading
        sub.target_heading = sensors.heading
        sub.target_pitch = 0.0

    @property
    def state_name(self) -> str:
        return f"DANCE_STEP_{self.dance_step}"

    def process_vision(self, sub: 'Submarine', camera_image: 'pygame.Surface') -> VisionData:
        return VisionData()

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        moves = {
            0: "SET_TURN_LEFT", 1: "WAIT_TURN_LEFT", 2: "SET_TURN_CENTER", 3: "WAIT_TURN_CENTER",
            4: "SET_TURN_RIGHT", 5: "WAIT_TURN_RIGHT", 6: "SET_TURN_CENTER", 7: "WAIT_TURN_CENTER",
            8: "SET_PITCH_DOWN", 9: "WAIT_PITCH_DOWN", 10: "SET_PITCH_CENTER", 11: "WAIT_PITCH_CENTER",
            12: "SET_PITCH_UP", 13: "WAIT_PITCH_UP", 14: "SET_PITCH_CENTER", 15: "WAIT_PITCH_CENTER",
            16: "SET_TURN_CENTER", 17: "WAIT_TURN_CENTER",
            18: "FINISH"
        }
        move = moves.get(self.dance_step, "FINISH")

        # Call the hover command using the task's own stored target_depth
        commands = sub._get_pid_hover_commands(
            sensors, dt, sub.target_x, sub.target_y, self.target_depth,
            yaw_p_gain_override=self.DANCE_YAW_P_GAIN,
            pitch_p_gain_override=self.DANCE_PITCH_P_GAIN
        )

        if "SET_" in move:
            if move == "SET_TURN_LEFT": sub.target_heading = (sub.dance_center_heading - 90) % 360
            elif move == "SET_TURN_RIGHT": sub.target_heading = (sub.dance_center_heading + 90) % 360
            elif move == "SET_TURN_CENTER": sub.target_heading = sub.dance_center_heading
            elif move == "SET_PITCH_DOWN": sub.target_pitch = -self.DANCE_PITCH_ANGLE
            elif move == "SET_PITCH_UP": sub.target_pitch = self.DANCE_PITCH_ANGLE
            elif move == "SET_PITCH_CENTER": sub.target_pitch = 0.0
            self.dance_step += 1
        elif "WAIT_" in move:
            h_err = abs(angle_diff(sensors.heading, sub.target_heading))
            p_err = abs(sensors.pitch - sub.target_pitch)
            if ("TURN" in move and h_err < self.DANCE_COMPLETION_TOLERANCE) or \
               ("PITCH" in move and p_err < self.DANCE_COMPLETION_TOLERANCE):
                self.dance_step += 1
        elif move == "FINISH":
            return TaskStatus.COMPLETED, commands
            
        return TaskStatus.RUNNING, commands