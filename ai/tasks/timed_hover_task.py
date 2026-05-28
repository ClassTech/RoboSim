#!/usr/bin/env python3
"""
A task to hover in place for a specified duration.
"""
import math
from typing import Tuple
import numpy as np

from ai.tasks.hover_task import HoverTask

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands

SETTLE_SPEED = 0.08  # m/s — track the sub until it's this slow, then lock position

class TimedHoverTask(HoverTask):
    def __init__(self, duration: float, target_depth: float):
        self.duration = duration
        self.target_depth = target_depth
        self.timer = 0.0

    def reset(self):
        self.timer = 0.0

    @property
    def state_name(self) -> str:
        return f"HOVERING ({self.timer:.1f}s / {self.duration:.1f}s)"

    def process_vision(self, sub: 'Submarine', camera_image: np.ndarray) -> VisionData:
        return VisionData()

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: 'SimulationConfig') -> Tuple[TaskStatus, ThrusterCommands]:
        self.timer += dt

        speed = math.hypot(sensors.velocity_x, sensors.velocity_y)
        if speed > SETTLE_SPEED:
            # Sub still coasting — slide the hold target along with it so the
            # P term stays zero and only D-gain braking is applied.
            sub.target_x, sub.target_y = sensors.x, sensors.y
            sub.integral_x_err, sub.integral_y_err = 0.0, 0.0

        if self.timer >= self.duration:
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors, sensors.depth)

        return TaskStatus.RUNNING, sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y, self.target_depth)