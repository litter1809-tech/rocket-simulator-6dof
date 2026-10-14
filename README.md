[README.md](https://github.com/user-attachments/files/33411123/README.md)
# 6-DOF Rocket Trajectory Simulation

A Python simulation for modeling 6-DOF rocket flight dynamics. I built this to move away from messy single-script prototypes and structure it as a clean, tested engineering package. It handles altitude-dependent atmospheric physics, variable mass depletion during motor burn, and adaptive numerical integration.

## Features

- **Modular Layout:** Code is split into a dedicated `src/` directory separating the environment, vehicle specs, equations of motion, and plotting tools.
- **Atmospheric Model:** Uses the 1976 US Standard Atmosphere to calculate temperature, pressure, density, and speed of sound at varying altitudes.
- **Variable Mass & Thrust:** Tracks propellant burnout over time, switching cleanly from powered ascent to a coasting flight phase.
- **Numerical Integration:** Uses `scipy.integrate.solve_ivp` with RK45 to step through the trajectory equations.
- **Testing:** Includes basic unit tests via `pytest` to make sure the atmosphere calculations match reference values.
- **Telemetry Output:** Automatically plots and saves a performance dashboard (`rocket_performance_dashboard.png`) after every run.

## Repository Layout

```
rocket_simulator/
├── src/
│   ├── __init__.py
│   ├── environment.py   # Atmosphere lookup & wind models
│   ├── vehicle.py       # Mass, geometry, and motor data
│   ├── physics.py       # Equations of motion & integration
│   └── dashboard.py     # Matplotlib plotting scripts
├── tests/
│   ├── __init__.py
│   └── test_atmosphere.py # Unit tests for atmospheric verification
├── main.py              # Entry point to run the full simulation
├── pyproject.toml       # Build system config
├── requirements.txt     # Python dependencies
└── README.md
```

## Running the Project

Clone the repo, install the requirements, and run the simulation:

```bash
git clone https://github.com/litter1809-tech/rocket-simulator-6dof.git
cd rocket-simulator-6dof
pip install -r requirements.txt
```

To run the unit tests:
```bash
python -m pytest
```

To run the full simulation pipeline:
```bash
python main.py
```

## Sample Output

When you run `main.py`, it prints the flight milestones to the terminal and saves the plots:

```
6-DoF flight integration: RK45 solution
status=1; events=1; final time=13.23 s
final position (x, y, z) = (-190.66, 3.87, -0.00) m

Performance metrics:
  Apogee altitude     : 191.8 m (629.3 ft)
  Max velocity        : 87.1 m/s (0.25 Mach)
  Burnout time        : 5.00 s
  Flight duration     : 13.23 s
  Downrange impact    : 190.7 m
