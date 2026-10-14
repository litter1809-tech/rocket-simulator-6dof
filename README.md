# 6-DOF Rocket Trajectory Simulation Engine

A modular, high-fidelity 6-Degrees-of-Freedom (6-DOF) flight dynamics simulator engineered in Python. Designed to model full three-dimensional rigid-body rocket flight—including variable mass depletion, aerodynamic moments, altitude-dependent atmospheric properties, and numerical integration—complete with automated test suites and performance visualization.

---

## Project Structure

rocket_simulator/
├── src/
│   ├── __init__.py
│   ├── environment.py    # 1976 US Standard Atmosphere & altitude-based wind modeling
│   ├── vehicle.py        # Rocket mass properties, geometry, and time-varying thrust profile
│   ├── physics.py        # 6-DOF equations of motion, force/moment summation, and kinematics
│   └── dashboard.py      # Automated trajectory plotting and performance metrics generation
├── tests/
│   └── test_atmosphere.py# Pytest suite verifying atmospheric reference values
├── main.py               # End-to-end simulation execution entry point
├── simulator.py          # Core simulation orchestrator, state wrapper, and event handling
├── pyproject.toml        # Project metadata and test runner configuration
├── requirements.txt      # Core runtime and development dependencies
├── rocket_sim_log.txt    # Text-based flight telemetry and event logs
└── rocket_performance_dashboard.png # Multi-panel performance and trajectory visualization

Technical Highlights

Atmospheric Modeling (environment.py): Implements the 1976 US Standard Atmosphere model across troposphere and stratosphere layers to compute temperature, pressure, density ($\rho$), and speed of sound ($a$), coupled with configurable wind profiles.

Flight Dynamics & Rigid-Body Physics (physics.py & vehicle.py): Solves coupled translational and rotational equations of motion, accounting for gravity vectors, aerodynamic forces/moments, and real-time variable mass depletion during motor burnout.

Simulation Orchestration (simulator.py): Manages state vector initialization, physical parameter lookups, and solver execution boundaries.Numerical Integration: Leverages scipy.integrate.solve_ivp utilizing an adaptive Runge-Kutta (RK45) scheme alongside root-finding event detection to precisely isolate critical flight milestones like apogee and ground impact.

Automated Validation (tests/test_atmosphere.py): Features a dedicated pytest suite verifying environmental calculations against standard physical benchmarks.

Diagnostics & Telemetry (dashboard.py): Computes key flight performance metrics (maximum velocity, peak Mach number, burnout time, apogee altitude, and downrange dispersion) while rendering a multi-panel diagnostic dashboard.

RequirementsPython: 3.10 or higher
NumPy: Vectorized mathematical operations and matrix manipulations
SciPy: ODE initial value problem solvers (solve_ivp)
Matplotlib: Trajectory plotting and performance visualization
Pytest: Automated unit testing framework

Installation & Usage
Clone the repository:Bashgit clone [https://github.com/litter1809-tech/rocket-simulator-6dof.git](https://github.com/litter1809-tech/rocket-simulator-6dof.git)
cd rocket-simulator-6dof
Install dependencies:Bashpip install -r requirements.txt
Run the test suite:Bashpytest
Execute the full simulation:Bashpython main.py
