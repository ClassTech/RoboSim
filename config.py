#!/usr/bin/env python3
"""
Contains all constants, tuning parameters, and configuration classes for the simulation.
"""
from dataclasses import dataclass

# --- Constants and Configuration ---
# Color constants for drawing
WHITE, BLACK, BLUE, LIGHT_BLUE = (255, 255, 255), (0, 0, 0), (120, 50, 20), (200, 150, 100)
RED, GREEN, YELLOW, GRAY = (0, 0, 255), (50, 200, 50), (0, 255, 255), (128, 128, 128)
ORANGE = (0, 165, 255)
CONTROL_BOX_GRAY = (80, 80, 80)
SHARK_BLUE, SAWFISH_GREEN = (150, 100, 70), (100, 140, 100)
POOL_FLOOR_COLOR, WATER_COLOR = (100, 60, 40), (120, 50, 20)
MAGENTA = (255, 0, 255)

# HSV Color Ranges for vision (H: 0-360, S: 0-100, V: 0-100)
RED_HSV_RANGES = [((0, 40, 40), (15, 100, 100)), ((340, 40, 40), (360, 100, 100))]
BLACK_HSV_RANGE = [((0, 0, 0), (360, 100, 30))]
WHITE_HSV_RANGE = [((0, 0, 70), (360, 25, 100))]

@dataclass
class SimulationConfig:
    worldWidth: float = 40.0
    worldHeight: float = 15.0
    worldDepth: float = 2.1 # 7ft
    cameraFov: float = 70.0
    submarineWidth: float = 0.46
    submarineLength: float = 0.457