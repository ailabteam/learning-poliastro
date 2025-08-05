# project_ga_final_for_paper_english.py (FIXED r_a INDEXING)

import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.plotting import OrbitPlotter3D

import numpy as np
import random
from deap import base, creator, tools, algorithms
import os
import plotly.graph_objects as go

# ===================================================================
# --- EXPERIMENT CONFIGURATION ---
# ===================================================================
SCENARIO = 'FAR_ORBIT'
if not os.path.exists('results'):
    os.makedirs('results')
# ===================================================================

# --- Problem Setup ---
print(f"--- Starting experiment for scenario: {SCENARIO} ---")
leo_orbit = Orbit.circular(Earth, alt=400 * u.km)
if SCENARIO == 'GEO':
    r_target = Earth.R + 35786 * u.km
    target_orbit = Orbit.circular(Earth, r_target - Earth.R)
elif SCENARIO == 'FAR_ORBIT':
    r_target = 20 * leo_orbit.r_p
    target_orbit = Orbit.circular(Earth, r_target - Earth.R)
else:
    raise ValueError("Invalid scenario. Please choose 'GEO' or 'FAR_ORBIT'.")
print(f"Initial Orbit: LEO, radius {leo_orbit.r_p.to(u.km):.2f}")
print(f"Target Orbit: radius {r_target.to(u.km):.2f}")

# --- Fitness Function ---
def evaluate_bielliptic(individual):
    rb_ratio = individual[0]
    if rb_ratio <= 1.0: return 9999999,
    rb = rb_ratio * r_target
    try:
        maneuver = Maneuver.bielliptic(leo_orbit, rb, r_target)
        return maneuver.get_total_cost().to_value(u.m / u.s),
    except Exception: return 9999999,

# --- Genetic Algorithm Setup (DEAP) ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_rb_ratio", random.uniform, 1.01, 200.0) # Expanded range
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rb_ratio, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_bielliptic)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Main Function to Run and Analyze ---
def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    print("\nStarting GA evolution process...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=40,
                        stats=stats, halloffame=hof, verbose=True)
    print("Evolution process completed.")

    # --- Analysis and Results ---
    print("\n" + "="*20 + " ANALYSIS RESULTS " + "="*20)
    best_rb_ratio = hof[0][0]
    best_rb = best_rb_ratio * r_target
    ga_maneuver = Maneuver.bielliptic(leo_orbit, best_rb, r_target)
    ga_cost = ga_maneuver.get_total_cost()
    ga_time = ga_maneuver.get_total_time()
    hohmann_maneuver = Maneuver.hohmann(leo_orbit, r_target)
    hohmann_cost = hohmann_maneuver.get_total_cost()
    hohmann_time = hohmann_maneuver.get_total_time()
    print("\n--- Optimal Solution from Genetic Algorithm (Bi-elliptic) ---")
    print(f"Optimal intermediate radius ratio (rb / r_target): {best_rb_ratio:.4f}")
    print(f"Optimal intermediate apoapsis: {best_rb.to(u.km):.2f}")
    print(f"  - Delta-V Impulse 1: {np.linalg.norm(ga_maneuver.impulses[0][1]).to(u.m/u.s):.2f}")
    print(f"  - Delta-V Impulse 2: {np.linalg.norm(ga_maneuver.impulses[1][1]).to(u.m/u.s):.2f}")
    print(f"  - Delta-V Impulse 3: {np.linalg.norm(ga_maneuver.impulses[2][1]).to(u.m/u.s):.2f}")
    print(f"TOTAL DELTA-V (GA): {ga_cost.to(u.m/u.s):.2f}")
    print(f"TOTAL FLIGHT TIME (GA): {ga_time.to(u.day):.2f}")
    print("\n--- Classical Solution (Hohmann Transfer) ---")
    print(f"TOTAL DELTA-V (Hohmann): {hohmann_cost.to(u.m/u.s):.2f}")
    print(f"TOTAL FLIGHT TIME (Hohmann): {hohmann_time.to(u.day):.2f}")
    print("\n--- COMPARISON ---")
    if ga_cost < hohmann_cost:
        savings = hohmann_cost - ga_cost
        print(f"==> GA solution (Bi-elliptic) is superior, saving {savings.to(u.m/u.s):.2f}")
    else:
        savings = ga_cost - hohmann_cost
        print(f"==> Hohmann solution is superior, saving {savings.to(u.m/u.s):.2f}")

    # --- Plotting and Saving Figures ---
    print("\nGenerating and saving comparison plot...")
    ga_transfer1, ga_transfer2, _ = leo_orbit.apply_maneuver(ga_maneuver, intermediate=True)
    hoh_transfer, _ = leo_orbit.apply_maneuver(hohmann_maneuver, intermediate=True)

    # THAY ĐỔI CỐT LÕI Ở ĐÂY
    # Tính toán vị trí của điểm cú đẩy thứ 2
    apoapsis_orbit_1 = ga_transfer1.propagate_to_anomaly(180 * u.deg)
    apoapsis_position_vec = apoapsis_orbit_1.r

    # Bắt đầu vẽ
    plotter = OrbitPlotter3D()
    plotter.plot(leo_orbit, label="Initial Orbit (LEO)", color='gray')
    plotter.plot(target_orbit, label="Target Orbit", color='black')
    fig = plotter._figure

    # Hohmann Transfer
    hoh_coords = hoh_transfer.sample()
    fig.add_trace(go.Scatter3d(
        x=hoh_coords.x.to_value(u.km), y=hoh_coords.y.to_value(u.km), z=hoh_coords.z.to_value(u.km),
        mode='lines', line=dict(color='cyan', width=5, dash='dash'),
        name=f'Hohmann (ΔV: {hohmann_cost.to(u.km/u.s):.2f} km/s)'
    ))

    # GA Transfer Leg 1 & 2
    ga1_coords = ga_transfer1.sample()
    fig.add_trace(go.Scatter3d(
        x=ga1_coords.x.to_value(u.km), y=ga1_coords.y.to_value(u.km), z=ga1_coords.z.to_value(u.km),
        mode='lines', line=dict(color='magenta', width=8), name='GA Solution Leg 1'
    ))
    ga2_coords = ga_transfer2.sample()
    fig.add_trace(go.Scatter3d(
        x=ga2_coords.x.to_value(u.km), y=ga2_coords.y.to_value(u.km), z=ga2_coords.z.to_value(u.km),
        mode='lines', line=dict(color='red', width=8),
        name=f'GA Solution Leg 2 (Total ΔV: {ga_cost.to(u.km/u.s):.2f} km/s)'
    ))
    
    # Add markers for impulses, sử dụng vector vị trí đúng
    fig.add_trace(go.Scatter3d(
        x=[leo_orbit.r[0].to_value(u.km)], y=[leo_orbit.r[1].to_value(u.km)], z=[leo_orbit.r[2].to_value(u.km)],
        mode='markers', marker=dict(size=8, color='yellow', symbol='diamond'), name='Impulse 1'
    ))
    fig.add_trace(go.Scatter3d(
        x=[apoapsis_position_vec[0].to_value(u.km)], y=[apoapsis_position_vec[1].to_value(u.km)], z=[apoapsis_position_vec[2].to_value(u.km)],
        mode='markers', marker=dict(size=8, color='yellow', symbol='diamond'), name='Impulse 2'
    ))
    # Vị trí cú đẩy thứ 3 có thể lấy từ điểm bắt đầu của quỹ đạo đích
    # Giả sử nó cùng pha với điểm bắt đầu của LEO, chỉ khác bán kính
    r3_vec = target_orbit.r 
    fig.add_trace(go.Scatter3d(
        x=[r3_vec[0].to_value(u.km)], y=[r3_vec[1].to_value(u.km)], z=[r3_vec[2].to_value(u.km)],
        mode='markers', marker=dict(size=8, color='yellow', symbol='diamond'), name='Impulse 3'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=f"<b>Optimal Trajectory Comparison: {SCENARIO} Scenario</b>", x=0.5, font=dict(size=20)),
        legend=dict(title=dict(text="<b>Trajectories</b>"), font=dict(size=12)),
        scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    output_filename = os.path.join('results', f'ga_optimization_{SCENARIO}_high_res.png')
    fig.write_image(output_filename, width=1600, height=1200, scale=2)
    print(f"-> High-resolution plot saved to: {output_filename}")

if __name__ == "__main__":
    main()
