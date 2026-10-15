[README.md](https://github.com/user-attachments/files/33411123/README.md)
# 6-DOF Rocket Trajectory Simulation

**Author:** Tegan Broody
**Institution:** Ridley High School

A Python simulation for modeling 6-DOF rocket flight dynamics. I built this to move away from messy single-script prototypes and structure it as a clean, tested engineering package. It handles altitude-dependent atmospheric physics, variable mass depletion during motor burn, and adaptive numerical integration.

## Features

- **Modular Layout:** Code is split into a dedicated `src/` directory separating the environment, vehicle specs, equations of motion, and plotting tools.
- **Atmospheric Model:** Uses the 1976 US Standard Atmosphere to calculate temperature, pressure, density, and speed of sound at varying altitudes.
- **Variable Mass & Thrust:** Tracks propellant burnout over time, switching cleanly from powered ascent to a coasting flight phase.
- **Numerical Integration:** Uses `scipy.integrate.solve_ivp` with RK45 to step through the trajectory equations.
- **Testing:** Includes basic unit tests via `pytest` to make sure the atmosphere calculations match reference values.
- **Telemetry Output:** Automatically plots and saves a performance dashboard (`rocket_performance_dashboard.png`) after every run.
- **Custom motor curves:** Upload RASP/RockSim `.eng` files in the Streamlit sidebar. Piecewise-linear thrust is integrated over time to model propellant depletion; header propellant mass and burn duration are used in place of the manual values.
- **Monte Carlo dispersion:** Run 50–100 randomized point-mass estimates with vectorized wind, thrust, and launch-angle variations. The nominal flight remains on the full 6-DOF RK45 engine.
- **Interactive engineering dashboard:** Streamlit charts show 3D flight, position/velocity/attitude components, flight loads, and optional real-flight CSV comparisons.
- **Ground Station flight replay:** View a 3D rocket model moving along the X/Y/Z position path, with red, green, and blue body-axis vectors rotating with the simulated attitude. Use Plotly's browser-native Play/Pause controls and frame scrubber, choose playback speed, or use the telemetry inspection slider to freeze a frame and inspect interpolated flight data.
- **Continuous testing:** GitHub Actions runs the pytest suite on pushes and pull requests.

## Model limitations

The attitude model uses a fixed aerodynamic moment arm, applies simplified
normal forces only during forward axial airflow, and does not include rotational
damping, launch-rail restraint, or recovery systems. Its attitude predictions
are educational estimates and have not been flight-validated.

## Repository Layout

```
rocket_simulator/
├── src/
│   ├── __init__.py
│   ├── environment.py   # Atmosphere lookup & wind models
│   ├── motor.py         # RASP/RockSim motor curves and parser
│   ├── dispersion.py    # Monte Carlo landing dispersion
│   ├── vehicle.py       # Mass, geometry, and motor data
│   ├── physics.py       # Equations of motion & integration
│   └── dashboard.py     # Matplotlib plotting scripts
├── .github/
│   └── workflows/
│       └── pytest.yml   # CI test workflow
├── app.py               # Streamlit engineering dashboard
├── tests/
│   ├── __init__.py
│   ├── test_atmosphere.py
│   ├── test_dispersion.py
│   ├── test_motor.py
│   └── test_physics.py
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

To launch the interactive Streamlit app:
```bash
streamlit run app.py
```

### Comparing a real flight

In the app sidebar, upload a UTF-8 CSV with `Time` (seconds) and `Altitude`
(metres) columns. `Velocity` (m/s), `X`/`Downrange` (m), and `Y`/`Crossrange`
(m) are optional. Time and altitude are rebased to the first valid sample;
if velocity is omitted, vertical velocity is estimated from altitude. Altimeter
files without X/Y coordinates are overlaid at the origin in the 3D view.

### Custom motor and dispersion analysis

Upload a standard RASP/RockSim `.eng` motor file in the sidebar. The file should
start with a motor header whose propellant mass is in grams, followed by rows of
time (seconds) and thrust (newtons); lines beginning with `;` are comments.
When accepted, the motor's header propellant mass and final curve time override
the manual propellant and burn-time inputs. The curve's integrated thrust
impulse determines the depletion history.

Enable **Monte Carlo analysis** and select 50–100 runs to sample wind magnitude,
wind direction, motor thrust, and launch-angle variation. The dispersion uses
a vectorized point-mass ballistic approximation for quick estimates; the nominal
flight remains a full 6-DOF RK45 simulation. The dispersion tab plots each
landing point and its apogee; a nominal landing is shown when a nominal
simulation is available.

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
