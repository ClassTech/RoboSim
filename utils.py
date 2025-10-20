#!/usr/bin/env python3
"""
Contains general-purpose utility functions.
"""

def angle_diff(a1, a2):
    """Calculates the shortest difference between two angles in degrees."""
    diff = a1 - a2
    while diff <= -180: diff += 360
    while diff > 180: diff -= 360
    return diff