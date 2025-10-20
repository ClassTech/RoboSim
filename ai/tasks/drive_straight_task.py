#!/usr/bin/env python3
"""
A simple task to drive straight for a set duration.
"""
from typing import Tuple
import pygame

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands

class DriveStraightTask(Task):
    def __init__(self, duration: float, target_depth: float, speed: float = 0.5):
        self.duration = duration
        self.target_depth = target_depth
        self.speed = speed
        self.timer = 0.0
        self.heading_to_hold = None

    def reset(self):
        self.timer = 0.0
        self.heading_to_hold = None

    @property
    def state_name(self) -> str:
        return f"DRIVING_STRAIGHT ({self.timer:.1f}s)"
    
    def process_vision(self, sub: 'Submarine', camera_image: pygame.Surface) -> VisionData:
        return VisionData()
    
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: 'SimulationConfig') -> Tuple[TaskStatus, ThrusterCommands]:
        if self.heading_to_hold is None:
            self.heading_to_hold = sensors.heading
        
        self.timer += dt
        if self.timer >= self.duration:
            return TaskStatus.COMPLETED, ThrusterCommands()
        
        return TaskStatus.RUNNING, sub.get_depth_change_commands(sensors, self.target_depth, self.heading_to_hold, surge_power=self.speed)