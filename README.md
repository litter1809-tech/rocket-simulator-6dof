# Rocket Simulator

A lightweight atmospheric model for rocket and flight simulations. It includes:

- U.S. Standard Atmosphere 1976 profile from 0 to 20,000 m
- Temperature, pressure, density, and speed-of-sound calculations
- A simple wind model with background drift, jet stream bump, and vertical oscillation
- A built-in self-check and demo output

## Run it

```bash
python main.py
```

Or directly:

```bash
python simulator.py
```

## Dependencies

- Python 3.10+
- NumPy

## Project metadata

The project is set up as a simple Python package with a console entry point:

```bash
rocket-sim
```
