#!/usr/bin/env python3
"""
Main entry point for the Autonomous Submarine Simulator.
Assembles and starts the simulation.
"""
from simulator import SubmarineSimulator
from ai.submarine import Submarine
from ai.tasks import (GateTask, SlalomTask, StabilizeTask, 
                      TimedHoverTask, SurfaceTask)

if __name__ == "__main__":
    # Define mission-specific parameters in one place
    MISSION_DEPTH = 1.2

    # 1. Define the mission plan using the smarter, reusable SlalomTask.
    mission = [
        # --- Outbound Journey ---
        GateTask(target_depth=MISSION_DEPTH),
        StabilizeTask(duration=3.0),
        SlalomTask(target_depth=MISSION_DEPTH),

        # --- Return Journey ---
        StabilizeTask(duration=3.0),
        # This SlalomTask will be configured on the fly by the submarine AI
        # to run in reverse and on the opposite side.
        SlalomTask(target_depth=MISSION_DEPTH, reversed=True),
        GateTask(target_depth=MISSION_DEPTH),
        TimedHoverTask(duration=10.0, target_depth=MISSION_DEPTH),
        SurfaceTask(target_depth=0.0)
    ]

    # 2. Create the Submarine AI engine and give it the mission plan.
    submarine_ai = Submarine(mission_plan=mission)

    # 3. Create the simulator and inject the fully-configured AI.
    sim = SubmarineSimulator(submarine_ai=submarine_ai)
    
    # 4. Run the simulation.
    sim.run()