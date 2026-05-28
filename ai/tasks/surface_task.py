#!/usr/bin/env python3
"""
A simple task to bring the submarine to the surface.
"""
from typing import Tuple
import numpy as np

from .task_base import Task, TaskStatus
from data_structures import SensorSuite, VisionData, ThrusterCommands

class SurfaceTask(Task):
    def __init__(self, target_depth: float = 0.0):
        pass

    def reset(self):
        pass

    @property
    def state_name(self) -> str:
        return "SURFACING"
    
    def process_vision(self, sub: 'Submarine', camera_image: np.ndarray) -> VisionData:
        return VisionData()
    
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: 'SimulationConfig') -> Tuple[TaskStatus, ThrusterCommands]:
        if sensors.depth < 0.1:
            return TaskStatus.COMPLETED, ThrusterCommands()
        
        surge, sway, yaw, pitch = 0, 0, 0, 0
        # CORRECTED: Heave must be negative to generate upward force
        heave = -1.0
        
        return TaskStatus.RUNNING, sub._mix_and_normalize_commands(surge, sway, heave, yaw, pitch)