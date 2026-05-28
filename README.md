# RoboSub Simulator

A Python/OpenCV simulation of an autonomous submarine competing in a RoboSub-style course.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy
python main.py
```

## Mission

The submarine runs a full out-and-back course autonomously:

1. Pass through the start gate
2. Perform a victory dance (360° yaw + 360° roll)
3. Navigate the slalom outbound
4. Navigate the slalom return (reversed, opposite side)
5. Pass back through the gate
6. Hover, then surface

## Structure

```
main.py          — mission assembly
simulator.py     — physics, rendering, sensor generation
world.py         — 3-D course geometry
config.py        — constants and HSV colour ranges
data_structures.py
ai/
  submarine.py   — mission execution and control primitives
  vision.py      — HSV colour-blob detector
  tasks/         — one file per task
```

See `Tech Report.md` for full architecture and algorithm details.
