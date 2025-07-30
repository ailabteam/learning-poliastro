# lesson3_hohmann_transfer.py (GIẢI PHÁP ĐƠN GIẢN VÀ ĐÚNG)

import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from poliastro.plotting import OrbitPlotter3D
# Import lại numpy
import numpy as np

print("Poliastro - Bài học 3: Chuyển quỹ đạo Hohmann")

# --- Bước 1: Giữ nguyên ---
leo_altitude = 400 * u.km
leo_orbit = Orbit.circular(Earth, alt=leo_altitude)
print("\n--- Quỹ đạo ban đầu (LEO) ---")
print(leo_orbit)

geo_altitude = 35786 * u.km
r_geo = Earth.R + geo_altitude
geo_orbit = Orbit.circular(Earth, alt=geo_altitude)
print("\n--- Quỹ đạo đích (GEO) ---")
print(geo_orbit)


# --- Bước 2: Tính toán và TỰ IN thông tin (Cách làm đúng) ---
hohmann_maneuver = Maneuver.hohmann(leo_orbit, r_geo)

print("\n--- Chi tiết thao tác chuyển Hohmann ---")
# Lặp qua các cú đẩy
for i, (time, dv) in enumerate(hohmann_maneuver.impulses):
    # CÁCH LÀM ĐÚNG: np.linalg.norm sẽ tự xử lý đơn vị của Quantity
    dv_magnitude = np.linalg.norm(dv)
    
    print(f"Cú đốt thứ {i+1}:")
    print(f"  - Thời gian thực hiện (so với ban đầu): {time.to(u.s):.2f}")
    print(f"  - Delta-V (vector): {dv.to(u.km/u.s)}")
    print(f"  - Chi phí Delta-V: {dv_magnitude.to(u.km/u.s).value:.4f} km/s")

# Lấy tổng chi phí
total_delta_v = hohmann_maneuver.get_total_cost()
print(f"\nTổng Delta-V cần thiết: {total_delta_v.to(u.km/u.s).value:.4f} km/s")


# --- Bước 3 và 4: Giữ nguyên ---
transfer_orbit, final_orbit = leo_orbit.apply_maneuver(hohmann_maneuver, intermediate=True)
print("\n--- Quỹ đạo chuyển tiếp ---")
print(transfer_orbit)

print("\nĐang tạo biểu đồ 3D cho quá trình chuyển quỹ đạo...")
plotter = OrbitPlotter3D()
plotter.plot(leo_orbit, label=f"LEO (ban đầu) - {leo_altitude.value} km")
plotter.plot(geo_orbit, label=f"GEO (đích) - {geo_altitude.value} km")
plotter.plot(transfer_orbit, label="Quỹ đạo chuyển tiếp Hohmann", color="red")

fig = plotter._figure
output_filename = "hohmann_transfer_3d.png"
fig.update_layout(width=1000, height=800)
fig.write_image(output_filename)

print(f"-> Đã lưu biểu đồ thành công vào file: {output_filename}")
