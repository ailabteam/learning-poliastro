# project_ga_simple.py (SỬA LỖI ĐÁNH MÁY)

import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
import numpy as np
import random
from deap import base, creator, tools, algorithms

print("--- Tối ưu hóa quỹ đạo 3-đẩy bằng GA (Phiên bản đơn giản) ---")

# --- Thiết lập vấn đề ---
leo_orbit = Orbit.circular(Earth, alt=400 * u.km)
r_geo = Earth.R + 35786 * u.km

# --- Hàm đánh giá mới (đơn giản hơn) ---
def evaluate_3_impulse(individual):
    rb_ratio = individual[0]
    if rb_ratio <= 1.0:
        return 9999999,
    rb = rb_ratio * r_geo
    try:
        maneuver = Maneuver.bielliptic(leo_orbit, rb, r_geo)
        total_dv = maneuver.get_total_cost()
        return total_dv.to_value(u.m / u.s),
    except Exception as e:
        return 9999999,

# --- Thiết lập DEAP ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_rb_ratio", random.uniform, 1.1, 30.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rb_ratio, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_3_impulse)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Chạy thuật toán ---
def main():
    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=20,
                        stats=stats, halloffame=hof, verbose=True)

    print("\n--- Kết quả tối ưu ---")
    best_individual = hof[0]
    best_rb_ratio = best_individual[0]
    best_fitness = best_individual.fitness.values[0]

    print(f"Tỷ lệ bán kính trung gian tối ưu (rb / r_geo): {best_rb_ratio:.4f}")
    print(f"Bán kính trung gian tối ưu: {best_rb_ratio * r_geo.to_value(u.km):.2f} km")
    print(f"Tổng Delta-V tối thiểu tìm được: {best_fitness:.2f} m/s")

    # So sánh với Hohmann
    hohmann_maneuver = Maneuver.hohmann(leo_orbit, r_geo)
    # SỬA LỖI Ở ĐÂY: u.m / u.s
    hohmann_cost = hohmann_maneuver.get_total_cost().to_value(u.m / u.s)
    print(f"Tổng Delta-V của Hohmann: {hohmann_cost:.2f} m/s")

    if best_fitness < hohmann_cost:
        savings = hohmann_cost - best_fitness
        print(f"=> Chuyển 3-đẩy tối ưu tốt hơn Hohmann, tiết kiệm {savings:.2f} m/s!")
    else:
        print("=> Hohmann vẫn là giải pháp tốt hơn cho trường hợp này.")

if __name__ == "__main__":
    main()
