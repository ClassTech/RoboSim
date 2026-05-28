#!/usr/bin/env python3
"""
Contains the specific implementation for the Hover task.
"""
from typing import Tuple
import numpy as np

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig



class HoverTask(Task):
    def __init__(self, target_depth: float):
        self.target_depth = target_depth

    @property
    def state_name(self) -> str:
        return "HOLDING_POSITION"
    
    def reset(self):
        pass
    
    def process_vision(self, sub: 'Submarine', camera_image: np.ndarray) -> VisionData:
        return VisionData()
    
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        return TaskStatus.RUNNING, sub._get_pid_hover_commands(sensors, dt, sub.target_x, sub.target_y, self.target_depth)