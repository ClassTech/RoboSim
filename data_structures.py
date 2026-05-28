#!/usr/bin/env python3
"""
Defines simple data classes used for passing information between modules.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

@dataclass
class MPU6050Readings:
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_z: float = 0.0

@dataclass
class SensorSuite:
    camera_image: np.ndarray
    depth: float
    heading: float
    pitch: float
    imu: MPU6050Readings
    x: float = 0.0
    y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    angular_velocity_y: float = 0.0
    angular_velocity_x: float = 0.0
    velocity_z: float = 0.0
    roll: float = 0.0

@dataclass
class VisionData:
    gate_is_visible: bool = False
    min_x: int = 0
    max_x: int = 0
    min_y: int = 0
    max_y: int = 0
    left_passage_center_x: Optional[float] = None
    right_passage_center_x: Optional[float] = None
    gate_center_y: Optional[float] = None
    left_pole_height: int = 0
    right_pole_height: int = 0
    apparent_height: float = 0.0
    visible_poles: List[Dict] = field(default_factory=list)
    potential_poles: List[Dict] = field(default_factory=list)
    selected_slalom_poles: List[Dict] = field(default_factory=list)
    avoidance_poles: List[Dict] = field(default_factory=list)

@dataclass
class ThrusterCommands:
    h_port_bow: float = 0.0
    h_starboard_bow: float = 0.0
    h_port_aft: float = 0.0
    h_starboard_aft: float = 0.0
    v_port: float = 0.0
    v_starboard: float = 0.0
    pause_simulation: bool = False