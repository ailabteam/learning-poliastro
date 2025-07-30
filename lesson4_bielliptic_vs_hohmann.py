# lesson4_bielliptic_vs_hohmann.py (SỬA LỖI - Tìm kiếm và sửa đổi theo tên)

import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.plotting import OrbitPlotter3D

print("Poliastro - Bài học 4: So sánh Hohmann và Bi-elliptic")

# --- Các bước 1, 2, 3, 4 giữ nguyên ---
# ...
leo_altitude = 400 * u.km
leo_orbit = Orbit.circular(Earth, alt=leo_altitude)

geo_altitude = 35786 * u.km
r_geo = Earth.R + geo_altitude
geo_orbit = Orbit.circular(Earth, alt=geo_altitude)

print(f"\nNhiệm vụ: Đi từ LEO ({leo_altitude.value} km) đến GEO ({geo_altitude.value} km)")

hohmann = Maneuver.hohmann(leo_orbit, r_geo)
hohmann_cost = hohmann.get_total_cost()
print(f"\nChiến lược 1: Hohmann Transfer")
print(f"-> Tổng Delta-V: {hohmann_cost.to(u.km/u.s):.4f}")

rb = 2 * r_geo
bielliptic = Maneuver.bielliptic(leo_orbit, rb, r_geo)
bielliptic_cost = bielliptic.get_total_cost()
print(f"\nChiến lược 2: Bi-elliptic Transfer (với bán kính trung gian {rb.to(u.km):.0f})")
print(f"-> Tổng Delta-V: {bielliptic_cost.to(u.km/u.s):.4f}")

print("\n--- So sánh hiệu quả ---")
if hohmann_cost < bielliptic_cost:
    diff = (bielliptic_cost - hohmann_cost).to(u.m/u.s)
    print(f"Hohmann hiệu quả hơn, tiết kiệm được {diff:.2f}")
else:
    diff = (hohmann_cost - bielliptic_cost).to(u.m/u.s)
    print(f"Bi-elliptic hiệu quả hơn, tiết kiệm được {diff:.2f}")


# --- Bước 5: Vẽ và CHỈNH SỬA đồ thị (Cách làm đúng và bền vững) ---
print("\nĐang tạo biểu đồ 3D so sánh hai chiến lược...")
hoh_transfer_orbit, _ = leo_orbit.apply_maneuver(hohmann, intermediate=True)
bie_transfer1, bie_transfer2, _ = leo_orbit.apply_maneuver(bielliptic, intermediate=True)

plotter = OrbitPlotter3D()

# 1. Vẽ tất cả các quỹ đạo với các tham số cơ bản
plotter.plot(leo_orbit, label="LEO (ban đầu)")
plotter.plot(geo_orbit, label="GEO (đích)")
plotter.plot(hoh_transfer_orbit, label="Hohmann Transfer", color="red")
plotter.plot(bie_transfer1, label="Bi-elliptic (leg 1)", color="purple")
plotter.plot(bie_transfer2, label="Bi-elliptic (leg 2)", color="green")

# 2. Lấy figure và chỉnh sửa các "trace" sau khi vẽ
fig = plotter._figure

# 3. Lặp qua tất cả các đối tượng đồ họa trong figure
for trace in fig.data:
    # Nếu tên của đối tượng khớp với label chúng ta đã đặt cho đường Bi-elliptic...
    if trace.name in ["Bi-elliptic (leg 1)", "Bi-elliptic (leg 2)"]:
        # ...thì thay đổi kiểu đường kẻ của nó thành nét đứt.
        trace.line.dash = 'dash'

# Lưu lại figure đã được chỉnh sửa
output_filename = "bielliptic_vs_hohmann.png"
fig.update_layout(width=1200, height=900, title_text="So sánh chuyển quỹ đạo Hohmann và Bi-elliptic")
fig.write_image(output_filename)

print(f"-> Đã lưu biểu đồ thành công vào file: {output_filename}")
