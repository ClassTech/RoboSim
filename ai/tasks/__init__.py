#!/usr/bin/env python3
"""
Makes the 'tasks' directory a package and simplifies imports.
"""
from .task_base import Task, TaskStatus
from .gate_task import GateTask
from .victory_dance_task import VictoryDanceTask
from .stabilize_task import StabilizeTask
from .slalom_task import SlalomTask
from .hover_task import HoverTask
from .timed_hover_task import TimedHoverTask
from .surface_task import SurfaceTask