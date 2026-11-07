from model import execute_simulation
import random
import time

DIM = 50                # Dimensions of our 2D grid
SIMULATION_STEPS = 100  # Number of time steps
DENSITY = 0.8           # Density of houses
OCCUPANCY = 0.8         # Occupancy rate at initialisation
CAPITAL = [300000, 300001]
PRICE = [200000, 200001]
SEED = 11

start = time.time()
execute_simulation(
    extract_data=range(SIMULATION_STEPS),
    dimension=DIM,
    density=DENSITY,
    occupancy=OCCUPANCY,
    capital=CAPITAL,
    price=PRICE,
    seed=SEED,
    simulation_steps=SIMULATION_STEPS
    )
end = time.time()

print(f"Execution time: {end - start:.4f} seconds")