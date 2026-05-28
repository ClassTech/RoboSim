#!/usr/bin/env python3
"""
Contains the foundation for all tasks: the Task abstract base class.
"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Tuple
import numpy as np

from data_structures import SensorSuite, VisionData, ThrusterCommands
from config import SimulationConfig


class TaskStatus(Enum):
    RUNNING = auto()
    COMPLETED = auto()

class Task(ABC):
    @abstractmethod
    def execute(self, sub: 'Submarine', dt: float, sensors: SensorSuite, vision_data: VisionData, config: SimulationConfig) -> Tuple[TaskStatus, ThrusterCommands]:
        """The main logic loop for the task."""
        pass
    
    @abstractmethod
    def process_vision(self, sub: 'Submarine', camera_image: np.ndarray) -> VisionData:
        """Processes the camera image to extract relevant data for the task."""
        pass

    @abstractmethod
    def reset(self):
        """Resets the internal state of the task."""
        pass

    @property
    @abstractmethod
    def state_name(self) -> str:
        """Returns the name of the current internal state for UI display."""
        pass