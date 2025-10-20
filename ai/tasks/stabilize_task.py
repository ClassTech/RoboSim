#!/usr/bin/env python3
"""
A simple task to bring the submarine to a full stop.
"""
import math
from typing import Tuple

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig

class StabilizeTask(Task):
    def __init__(self, duration: float = 3.0, speed_threshold: float = 0.05):
        self.STABILIZE_DURATION = duration
        self.SPEED_THRESHOLD = speed_threshold
        self.reset()

    def reset(self):
        self.state_timer = 0.0
        self.target_set = False
        self.target_depth = 0.0

    @property
    def state_name(self) -> str:
        speed = getattr(self, '_current_speed', 0)
        return f"STABILIZING (Speed: {speed:.2f} m/s)"

    def process_vision(self, sub: 'Submarine', camera_image: 'pygame.Surface') -> VisionData:
        return VisionData()

    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        # On the first execution, lock the current position and RESET THE PID INTEGRALS.
        if not self.target_set:
            sub.target_x, sub.target_y = sensors.x, sensors.y
            sub.target_heading = sensors.heading
            sub.target_pitch = 0.0
            # CORRECTED: Resetting the integral terms prevents the sub from backing up
            # due to accumulated error from the previous task.
            sub.integral_x_err, sub.integral_y_err = 0.0, 0.0
            self.target_depth = sensors.depth
            self.target_set = True

        self.state_timer += dt
        speed = math.hypot(sensors.velocity_x, sensors.velocity_y)
        self._current_speed = speed

        # Task is complete if the timer has run down AND we are slow enough
        if self.state_timer > self.STABILIZE_DURATION and speed < self.SPEED_THRESHOLD:
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors, self.target_depth)

        # Actively hover at the locked position to ensure a full stop
        return TaskStatus.RUNNING, sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y, self.target_depth)