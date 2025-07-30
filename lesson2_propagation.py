# lesson2_propagation.py (SỬA LỖI TypeError: unexpected keyword 'marker_size')

import astropy.units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter3D

print("Poliastro - Bài học 2: Truyền bá quỹ đạo và Vẽ 3D")

# Bước 1 và 2 giữ nguyên...
# ---
epoch = Time("2024-01-01 12:00:00", scale="utc")
r = [-2384.46, 5729.01, 3050.46] * u.km
v = [-7.37, -2.98, 1.64] * u.km / u.s
iss_orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)
print("\n--- Quỹ đạo ISS ban đầu ---")
print(iss_orbit)

dt = 45 * u.min
print(f"\nĐang truyền bá quỹ đạo đi {dt}...")
iss_orbit_future = iss_orbit.propagate(dt)
print(f"Dị thường thực ban đầu: {iss_orbit.nu:.2f}")
print(f"Dị thường thực sau 45 phút: {iss_orbit_future.nu:.2f}")
# ---

# Bước 3: Vẽ và LƯU quỹ đạo 3D (Đã xóa marker_size)
print("\nĐang tạo biểu đồ 3D...")
plotter = OrbitPlotter3D()

# 1. Vẽ quỹ đạo đầy đủ
plotter.plot(iss_orbit, label=f"ISS Quỹ đạo (Epoch: {epoch.iso})")

# 2. Lấy mẫu (sample) quỹ đạo tại các điểm quan tâm
coords_initial = iss_orbit.sample(1)
coords_future = iss_orbit_future.sample(1)

# 3. Dùng plot_trajectory để vẽ các điểm (không có marker_size)
plotter.plot_trajectory(coords_initial, label="Vị trí ban đầu", color="blue")
plotter.plot_trajectory(coords_future, label=f"Vị trí sau {dt}", color="red")

# Lấy figure của Plotly và lưu lại
fig = plotter._figure
output_filename = "iss_propagation_3d.png"
fig.write_image(output_filename)

print(f"-> Đã lưu biểu đồ 3D thành công vào file: {output_filename}")
