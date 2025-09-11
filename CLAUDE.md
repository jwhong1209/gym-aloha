# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gym-aloha is a Gymnasium environment for the ALOHA robotic manipulation tasks. It provides two main environments:
- `AlohaInsertion-v0`: Insertion task where both arms manipulate socket and peg for mid-air insertion
- `AlohaTransferCube-v0`: Transfer task where one arm picks up a cube and transfers it to the other arm

The codebase uses MuJoCo for physics simulation via dm-control, with optional GPU rendering support through EGL.

## Development Commands

### Environment Setup
```bash
# Install with development dependencies
poetry install --all-extras

# Install pre-commit hooks for code quality
pre-commit install
```

### Code Quality
```bash
# Run linting and formatting checks
ruff check
ruff format

# Run pre-commit hooks manually
pre-commit

# Check poetry configuration
poetry check
```

### Testing
```bash
# Run all tests with coverage
pytest tests -v --cov=./gym_aloha --durations=0

# Run specific test
pytest tests/test_env.py -v

# For GPU rendering tests, ensure EGL is available:
# sudo apt-get update && sudo apt-get install -y libegl1-mesa-dev
# Set MUJOCO_GL=egl environment variable
```

### Package Management
```bash
# Add new dependency
poetry add <package>

# Add development dependency  
poetry add --group dev <package>

# Update lock file
poetry lock --no-update
```

## Architecture

### Core Components
- `gym_aloha/env.py`: Main `AlohaEnv` class implementing Gymnasium interface
- `gym_aloha/tasks/sim.py`: Task definitions (`InsertionTask`, `TransferCubeTask`)
- `gym_aloha/tasks/sim_end_effector.py`: End-effector specific task variants
- `gym_aloha/constants.py`: Environment constants (actions, joints, assets)
- `gym_aloha/utils.py`: Utility functions for pose sampling

### Environment Registration
Environments are registered in `gym_aloha/__init__.py` using Gymnasium's registration system with:
- 300 max episode steps
- `nondeterministic=True` due to rendering variations
- Task-specific kwargs (`task`: "insertion" or "transfer_cube")

### Observation Types
- `pixels`: RGB camera feeds from multiple angles
- `pixels_agent_pos`: Pixels plus agent position information  
- `state`: Joint positions/velocities (not implemented)

### Action Space
14-dimensional continuous vector:
- 6 joint positions per arm (12 total)
- 1 gripper position per arm (2 total), normalized 0-1

## Code Style

- Python 3.10+ required
- Line length: 110 characters
- Uses Ruff for linting and formatting
- Pre-commit hooks enforce code quality
- Poetry for dependency management
- Excludes `example.py` from style checks