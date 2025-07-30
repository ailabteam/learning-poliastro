# project_ga_final_for_paper.py

import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.plotting import OrbitPlotter3D

import numpy as np
import random
from deap import base, creator, tools, algorithms
import os # Dùng để tạo thư mục

# ===================================================================
# --- CẤU HÌNH THÍ NGHIỆM (CHỈ CẦN THAY ĐỔI Ở ĐÂY) ---
# ===================================================================
# Chọn kịch bản: 'GEO' hoặc 'FAR_ORBIT'
SCENARIO = 'FAR_ORBIT'  # <-- THAY ĐỔI 'GEO' THÀNH 'FAR_ORBIT' ĐỂ CHẠY THÍ NGHIỆM 2

# Tạo thư mục để lưu kết quả
if not os.path.exists('results'):
    os.makedirs('results')
# ===================================================================


# --- Thiết lập vấn đề ---
print(f"--- Bắt đầu thí nghiệm cho kịch bản: {SCENARIO} ---")
leo_orbit = Orbit.circular(Earth, alt=400 * u.km)

if SCENARIO == 'GEO':
    # Kịch bản 1: LEO -> GEO
    r_target = Earth.R + 35786 * u.km
    target_orbit = Orbit.circular(Earth, r_target - Earth.R)
elif SCENARIO == 'FAR_ORBIT':
    # Kịch bản 2: LEO -> Quỹ đạo rất xa
    r_target = 20 * leo_orbit.r_p
    target_orbit = Orbit.circular(Earth, r_target - Earth.R)
else:
    raise ValueError("Kịch bản không hợp lệ. Vui lòng chọn 'GEO' hoặc 'FAR_ORBIT'.")

print(f"Quỹ đạo ban đầu: LEO, bán kính {leo_orbit.r_p.to(u.km):.2f}")
print(f"Quỹ đạo đích: bán kính {r_target.to(u.km):.2f}")


# --- Hàm đánh giá (Fitness Function) ---
def evaluate_bielliptic(individual):
    """
    Đánh giá chi phí của một cú chuyển Bi-elliptic.
    Cá thể chứa 1 gen: rb_ratio (tỷ lệ bán kính trung gian so với bán kính đích).
    """
    rb_ratio = individual[0]

    # Bán kính quỹ đạo trung gian phải lớn hơn bán kính đích
    if rb_ratio <= 1.0:
        return 9999999,

    rb = rb_ratio * r_target

    try:
        maneuver = Maneuver.bielliptic(leo_orbit, rb, r_target)
        total_dv = maneuver.get_total_cost()
        return total_dv.to_value(u.m / u.s),
    except Exception as e:
        return 9999999,


# --- Thiết lập Thuật toán Di truyền (DEAP) ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Gen duy nhất là rb_ratio, cho phép tìm kiếm trong một khoảng rộng
toolbox.register("attr_rb_ratio", random.uniform, 1.01, 40.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_rb_ratio, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Các toán tử di truyền
toolbox.register("evaluate", evaluate_bielliptic)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# --- Hàm chính để chạy và phân tích ---
def main():
    # Chạy GA
    pop = toolbox.population(n=40)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    print("\nBắt đầu quá trình tiến hóa của GA...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=30,
                        stats=stats, halloffame=hof, verbose=True)
    print("Quá trình tiến hóa hoàn tất.")

    # --- Phân tích và in kết quả (METRICS FOR PAPER) ---
    print("\n" + "="*20 + " KẾT QUẢ PHÂN TÍCH " + "="*20)
    best_individual = hof[0]
    best_rb_ratio = best_individual[0]
    best_rb = best_rb_ratio * r_target
    
    # Tính toán lại maneuver tối ưu để lấy các thông số chi tiết
    ga_maneuver = Maneuver.bielliptic(leo_orbit, best_rb, r_target)
    ga_cost = ga_maneuver.get_total_cost()
    ga_time = ga_maneuver.get_total_time()
    
    # Tính toán maneuver Hohmann để so sánh
    hohmann_maneuver = Maneuver.hohmann(leo_orbit, r_target)
    hohmann_cost = hohmann_maneuver.get_total_cost()
    hohmann_time = hohmann_maneuver.get_total_time()

    print("\n--- Giải pháp tối ưu từ Thuật toán Di truyền (Bi-elliptic) ---")
    print(f"Tỷ lệ bán kính trung gian tối ưu (rb / r_target): {best_rb_ratio:.4f}")
    print(f"Bán kính trung gian tối ưu: {best_rb.to(u.km):.2f}")
    print(f"  - Chi phí Delta-V (cú đẩy 1): {ga_maneuver.impulses[0][1][1].to(u.m/u.s):.2f}") # Giả định đẩy theo trục y
    print(f"  - Chi phí Delta-V (cú đẩy 2): {abs(ga_maneuver.impulses[1][1][1]).to(u.m/u.s):.2f}")
    print(f"  - Chi phí Delta-V (cú đẩy 3): {abs(ga_maneuver.impulses[2][1][1]).to(u.m/u.s):.2f}")
    print(f"TỔNG CHI PHÍ DELTA-V (GA): {ga_cost.to(u.m/u.s):.2f}")
    print(f"TỔNG THỜI GIAN BAY (GA): {ga_time.to(u.day):.2f}")

    print("\n--- Giải pháp kinh điển (Hohmann Transfer) ---")
    print(f"TỔNG CHI PHÍ DELTA-V (Hohmann): {hohmann_cost.to(u.m/u.s):.2f}")
    print(f"TỔNG THỜI GIAN BAY (Hohmann): {hohmann_time.to(u.day):.2f}")

    print("\n--- SO SÁNH ---")
    if ga_cost < hohmann_cost:
        savings = hohmann_cost - ga_cost
        print(f"==> Giải pháp GA (Bi-elliptic) tốt hơn, tiết kiệm {savings.to(u.m/u.s):.2f}")
    else:
        savings = ga_cost - hohmann_cost
        print(f"==> Giải pháp Hohmann tốt hơn, tiết kiệm {savings.to(u.m/u.s):.2f}")

    # --- Lưu biểu đồ (FIGURES FOR PAPER) ---
    print("\nĐang tạo và lưu biểu đồ so sánh...")
    # Lấy các quỹ đạo chuyển tiếp
    ga_transfer1, ga_transfer2, _ = leo_orbit.apply_maneuver(ga_maneuver, intermediate=True)
    hoh_transfer, _ = leo_orbit.apply_maneuver(hohmann_maneuver, intermediate=True)

    plotter = OrbitPlotter3D()
    plotter.plot(leo_orbit, label="LEO (ban đầu)")
    plotter.plot(target_orbit, label="Đích")
    
    # Vẽ quỹ đạo của Hohmann (nét liền)
    plotter.plot(hoh_transfer, label=f"Hohmann ({hohmann_cost.to(u.km/u.s):.2f})", color="cyan")
    
    # Vẽ quỹ đạo tối ưu của GA (nét liền)
    plotter.plot(ga_transfer1, label=f"GA (leg 1)", color="magenta")
    plotter.plot(ga_transfer2, label=f"GA (leg 2, {ga_cost.to(u.km/u.s):.2f})", color="red")
    
    fig = plotter._figure
    
    # Chỉnh sửa đường nét đứt cho Hohmann SAU KHI VẼ một cách an toàn
    for trace in fig.data:
        # Chỉ xử lý các đối tượng có thuộc tính 'line' (tức là không phải Surface)
        if hasattr(trace, 'line'):
            if "Hohmann" in trace.name:
                trace.line.dash = 'dash'
    
    output_filename = os.path.join('results', f'ga_optimization_{SCENARIO}.png')
    fig.update_layout(
        width=1200, height=900,
        title_text=f"So sánh quỹ đạo tối ưu: {SCENARIO}",
        legend_title_text="Chiến lược (Tổng ΔV, km/s)"
    )
    fig.write_image(output_filename)
    print(f"-> Đã lưu biểu đồ thành công vào file: {output_filename}")


if __name__ == "__main__":
    main()
