#!/usr/bin/env python3
"""
A simple task to turn the submarine by a relative number of degrees.
"""
from typing import Tuple
import pygame

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from utils import angle_diff

class TurnTask(Task):
    def __init__(self, turn_degrees: float, target_depth: float):
        self.turn_degrees = turn_degrees
        self.target_depth = target_depth
        self.TURN_COMPLETION_TOLERANCE = 5.0 # Degrees
        self.target_heading = None

    def reset(self):
        self.target_heading = None

    def on_start(self, sensors: SensorSuite):
        """Called by the submarine controller when the task begins."""
        self.target_heading = (sensors.heading + self.turn_degrees) % 360

    @property
    def state_name(self) -> str:
        if self.target_heading is None:
            return "TURNING"
        return f"TURNING_TO_{self.target_heading:.0f}"

    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        return VisionData()

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: 'SimulationConfig') -> Tuple[TaskStatus, ThrusterCommands]:
        if self.target_heading is None:
            self.on_start(sensors)

        if abs(angle_diff(self.target_heading, sensors.heading)) < self.TURN_COMPLETION_TOLERANCE:
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors, self.target_depth)

        return TaskStatus.RUNNING, sub.get_depth_change_commands(sensors, self.target_depth, self.target_heading)