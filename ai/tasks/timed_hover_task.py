#!/usr/bin/env python3
"""
A task to hover in place for a specified duration.
"""
from typing import Tuple
import pygame

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands

class TimedHoverTask(Task):
    def __init__(self, duration: float, target_depth: float):
        self.duration = duration
        self.target_depth = target_depth
        self.timer = 0.0

    def reset(self):
        self.timer = 0.0

    @property
    def state_name(self) -> str:
        return f"HOVERING ({self.timer:.1f}s / {self.duration:.1f}s)"
    
    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        return VisionData()
    
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: 'SimulationConfig') -> Tuple[TaskStatus, ThrusterCommands]:
        self.timer += dt
        if self.timer >= self.duration:
            return TaskStatus.COMPLETED, sub._get_damping_commands(sensors, sensors.depth)
        
        return TaskStatus.RUNNING, sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y, self.target_depth)